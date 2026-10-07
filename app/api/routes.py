from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from logging import getLogger
from typing import Annotated
from uuid import uuid4, UUID
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.params import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import(
    BatchClassifyRequest,
    BatchExplainRequest,
    ClassificationDetailResponse,
    ClassificationListResponse,
    ClassifyBatchResponse,
    ClassifyRequest,
    ClassifyResponse,
    CorrectionRequest,
    CorrectionResponse,
    ExplainBatchJobsResponse,
    ExplainJobDetailResponse,
    ExplainJobResponse,
    ExplainRequest,
    ExplainResponse,
    ModelInfoResponse,
    ReadinessResponse,
    StatusResponse
)
from app.api.dependencies import (
    get_model_snapshot,
    get_runtime,
    get_executor,
    _call_runtime
)
from app.db.engine import async_session, get_db
from app.db.repository import (
    add_correction,
    find_explanation,
    get_classification,
    get_latest_correction,
    list_classifications,
    list_explanations,
    list_latest_corrections,
    save_classification,
    save_classifications_batch,
)
from app.metadata import ModelInfo
from app.runtime import ApplicationRuntime, RuntimeStatus

logger = getLogger(__name__)

# All routes in this api will automatically start with /v1
router = APIRouter(prefix="/v1")

# NOTE: we put no awaits here because these methods are strictly
# synchronous (standard def), as they are conditioned in the
# ApplicationRuntime logic.
@router.get(
    "/model_info",
    response_model=ModelInfoResponse,
    summary="Model metadata",
    description="Returns the loaded model type, device, labels, thresholds and limits. No GPU work, safe to poll.",
)
async def get_model_info(
    runtime: Annotated[ApplicationRuntime, Depends(get_runtime)],
    ):
    model_info = runtime.model_info()
    return _model_info_to_list(model_info)

@router.get(
    "/status",
    response_model=StatusResponse,
    summary="Runtime queue status",
    description="State plus waiting counts and queue capacities. Inline microsecond read; use for backpressure dashboards.",
)
async def get_runtime_status(
    runtime: Annotated[ApplicationRuntime, Depends(get_runtime)],
):
    status = runtime.status()
    return _status_to_list(status)

@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description="True when the model is loaded and accepting inference. For orchestrators, not moderators. Checks runtime only; DB outages do not fail readiness (return-first degraded mode).",
)
async def get_readiness(
    runtime: Annotated[ApplicationRuntime, Depends(get_runtime)],
):
    """
    /v1/ready checks only runtime readiness as it is the only current dependency
    it will later be expanded
    """
    is_ready = runtime.is_ready()
    return {"is_ready": is_ready}

@router.get(
    "/health/live",
    summary="Liveness probe",
    description="Process alive check. No runtime, no DB, always 200 when Python answers. Restart on failure.",
)
async def health_check() -> dict[str, str]:
    return {"status": "healthy"}

# NOTE: not including a db health check because persistence operations are not public
# and a down db must not pull traffic out.

# NOTE: we await _call_runtime because it is an async function as it 
# declared in dependencies.py, because we want to call runtime functions
# asynchronously, even they are synchronized by default.
@router.post(
    "/classify",
    response_model=ClassifyResponse,
    summary="Classify one comment",
    description="Scores text on all 6 labels. Returns immediately with a classification_id; row persists in background. Use the id for strict explain. Example: {\"text\": \"You are an idiot\"}.",
)
async def classify_text(
    background_tasks: BackgroundTasks,
    runtime: Annotated[ApplicationRuntime, Depends(get_runtime)],
    executor: Annotated[ThreadPoolExecutor, Depends(get_executor)],
    snapshot: Annotated[dict, Depends(get_model_snapshot)],
    payload: ClassifyRequest
):
    """
    classifies the input text to the target labels,
    client can use classification_id to explain the classified text
    turning one-shot inference into addressable explainable history
    """
    text = payload.text
    result = await _call_runtime(runtime.predict, executor, text) # NOTE: Next time figure this out BETTER
    classification_id = uuid4()

    # Snapshot plain data for background: no ORM objects cross threads.
    # Model metadata comes lock-free from lifespan state, not per-call.
    labels_plain = {
        label: {"probability": pred.probability, "predicted": pred.predicted}
        for label, pred in result.labels.items()
    }
    background_tasks.add_task(
        _persist_classification,
        id=classification_id,
        text=text,
        text_hash=sha256(text.encode("utf-8")).hexdigest(),
        model_version="v2.0.0",
        model_type=snapshot["model_type"],
        device=snapshot["device"],
        thresholds=snapshot["thresholds"],
        labels=labels_plain,
    )

    return {
        "text": text,
        "labels": result.labels,
        "classification_id": classification_id
    }

async def _persist_classification(
        *,
        id: UUID,
        text: str,
        text_hash: str,
        model_version: str,
        model_type: str,
        device: str,
        thresholds: list,
        labels: dict,
):
    # Own fresh session: request session would close before background runs.
    # Failure here only logs; response already returned (accepted loss window).
    # NOTE: logger kept intentionally minimal here. Full structured logging
    # is a separate project task; this exception line is mandatory so
    # background loss is never silent.
    try:
        async with async_session() as session:
            await save_classification(
                session,
                id=id,
                text=text,
                text_hash=text_hash,
                model_version=model_version,
                model_type=model_type,
                device=device,
                thresholds=thresholds,
                labels=labels,
            )
    except Exception:
        logger.exception("background persist failed for classification %s", id)

@router.post(
    "/classify_batch",
    response_model=ClassifyBatchResponse,
    summary="Classify a batch",
    description="Scores 1..N texts in input order, one transaction persisted in background. Example: {\"texts\": [\"You are an idiot\", \"Thanks for helping\"]}.",
)
async def classify_batch(
    background_tasks: BackgroundTasks,
    runtime: Annotated[ApplicationRuntime, Depends(get_runtime)],
    executor: Annotated[ThreadPoolExecutor, Depends(get_executor)],
    snapshot: Annotated[dict, Depends(get_model_snapshot)],
    payload: BatchClassifyRequest
):
    texts = payload.texts
    results = await _call_runtime(runtime.predict_batch, executor, texts)
    ids = [uuid4() for _ in texts]
    items = [
        {
            "id": cid,
            "text": text,
            "text_hash": sha256(text.encode("utf-8")).hexdigest(),
            "model_version": "v2.0.0",
            "model_type": snapshot["model_type"],
            "device": snapshot["device"],
            "thresholds": snapshot["thresholds"],
            "labels": {
                label: {"probability": pred.probability, "predicted": pred.predicted}
                for label, pred in r.labels.items()
            },
        }
        for cid, text, r in zip(ids, texts, results)
    ]
    background_tasks.add_task(_persist_classifications_batch, items=items)
    return {
        "results": [
            {"text": text, "labels": r.labels, "classification_id": cid}
            for text, r, cid in zip(texts, results, ids)
        ]
    }

async def _persist_classifications_batch(*, items: list[dict]):
    # Same return-first contract as single: own session, log-only on failure.
    try:
        async with async_session() as session:
            await save_classifications_batch(session, items=items)
    except Exception:
        logger.exception("background persist failed for %d classifications", len(items))

@router.post(
    "/explain",
    response_model=None,
    responses={
        200: {"model": ExplainResponse, "description": "Cache hit, spans without GPU"},
        202: {"model": ExplainJobResponse, "description": "Accepted, poll GET /jobs/{id}"},
    },
    summary="Explain one classification (async on miss)",
    description="Strict: requires a prior classification_id (404 if unknown). Cache hit returns 200 spans with zero GPU; miss creates a job, pushes to Redis, returns 202 with job_id. Poll GET /v1/jobs/{job_id}. Example: {\"classification_id\": \"<uuid from /classify>\", \"label\": \"toxic\"}.",
)
async def explain_text(
    db: Annotated[AsyncSession, Depends(get_db)],
    payload: ExplainRequest
):
    from fastapi.responses import JSONResponse

    classification_id = payload.classification_id
    label = payload.label
    model_version = "v2.0.0"

    # Strict: must have been classified before. Reuse exact stored text.
    classification = await get_classification(db, classification_id=classification_id)
    if classification is None:
        raise HTTPException(status_code=404, detail="classification not found")

    # Cache before GPU: same (id, label, version) returns stored spans.
    cached = await find_explanation(
        db, classification_id=classification_id, label=label, model_version=model_version
    )
    if cached is not None:
        body = ExplainResponse.model_validate(
            {
                "text": classification.text,
                "label": cached.label,
                "spans": cached.spans,
                "convergence_delta": cached.convergence_delta,
            }
        ).model_dump(mode="json")
        return JSONResponse(status_code=200, content=body)

    # Miss: idempotent create-or-return, then wake the worker via Redis.
    # The API never runs IG here, so classify stays hot under explain load.
    from app.db.repository import create_job

    job, _ = await create_job(
        db,
        classification_id=classification_id,
        label=label,
        model_version=model_version,
    )
    try:
        from app.db.queue import enqueue_explain_job

        await enqueue_explain_job(job.id)
    except Exception:
        logger.exception("redis enqueue failed for job %s", job.id)
        # Row is already queued in Postgres; worker rebuilds from it on boot.
    body = ExplainJobResponse.model_validate(
        {
            "job_id": job.id,
            "classification_id": job.classification_id,
            "label": job.label,
            "model_version": job.model_version,
            "status": job.status,
            "error": job.error,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        }
    ).model_dump(mode="json")
    # 202 when newly queued; repeat of a queued/running job returns 200
    # with the same body so clients can poll uniformly.
    from app.db.repository import find_explanation as _find

    fresh = await _find(
        db, classification_id=classification_id, label=label, model_version=model_version
    )
    if fresh is not None:
        ok = ExplainResponse.model_validate(
            {
                "text": classification.text,
                "label": fresh.label,
                "spans": fresh.spans,
                "convergence_delta": fresh.convergence_delta,
            }
        ).model_dump(mode="json")
        return JSONResponse(status_code=200, content=ok)
    return JSONResponse(status_code=202, content=body)


@router.get(
    "/jobs/{job_id}",
    response_model=ExplainJobDetailResponse,
    summary="Explanation job status",
    description="Postgres is truth. Queued/running poll again; done includes spans; failed includes error for retry visibility.",
)
async def get_explain_job(
    job_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    from app.db.repository import get_job

    job = await get_job(db, job_id=job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    result = None
    if job.status == "done":
        expl = await find_explanation(
            db,
            classification_id=job.classification_id,
            label=job.label,
            model_version=job.model_version,
        )
        if expl is not None:
            cls = await get_classification(db, classification_id=job.classification_id)
            result = {
                "text": cls.text if cls else "",
                "label": expl.label,
                "spans": expl.spans,
                "convergence_delta": expl.convergence_delta,
            }
    return {
        "job_id": job.id,
        "classification_id": job.classification_id,
        "label": job.label,
        "model_version": job.model_version,
        "status": job.status,
        "error": job.error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "result": result,
    }

@router.post(
    "/explain_batch",
    response_model=ExplainBatchJobsResponse,
    status_code=202,
    summary="Explain a batch for one label (async)",
    description="Strict per id with shared label. Validates all ids (404 on first unknown), serves cache as done jobs, creates queued jobs for misses and pushes each to Redis. Poll GET /v1/jobs/{id} per job.",
)
async def explain_batch(
    db: Annotated[AsyncSession, Depends(get_db)],
    payload: BatchExplainRequest
):
    from app.db.repository import create_job

    ids = payload.classification_ids
    label = payload.label
    model_version = "v2.0.0"

    # Strict: all rows must exist to keep order exact.
    for cid in ids:
        row = await get_classification(db, classification_id=cid)
        if row is None:
            raise HTTPException(status_code=404, detail=f"classification {cid} not found")

    jobs_out = []
    for cid in ids:
        # Served from cache: ensure a done job exists for uniform polling.
        cached = await find_explanation(
            db, classification_id=cid, label=label, model_version=model_version
        )
        job, created = await create_job(
            db, classification_id=cid, label=label, model_version=model_version
        )
        if cached is not None and job.status != "done":
            from app.db.repository import mark_job

            job = await mark_job(db, job_id=job.id, status="done")
        if created and cached is None:
            try:
                from app.db.queue import enqueue_explain_job

                await enqueue_explain_job(job.id)
            except Exception:
                logger.exception("redis enqueue failed for job %s", job.id)
        jobs_out.append(
            ExplainJobResponse.model_validate(
                {
                    "job_id": job.id,
                    "classification_id": job.classification_id,
                    "label": job.label,
                    "model_version": job.model_version,
                    "status": job.status,
                    "error": job.error,
                    "created_at": job.created_at,
                    "updated_at": job.updated_at,
                }
            )
        )
    return {"jobs": [j.model_dump(mode="json") for j in jobs_out]}

def _model_info_to_list(info: ModelInfo) -> dict:
    return {
        "model_type": info.model_type,
        "device": info.device,
        "labels": list(info.labels),
        "thresholds": list(info.thresholds),
        "max_length": info.max_length,
        "explanation_steps": info.explanation_steps,
        "inference_backend": info.inference_backend,
    }

def _status_to_list(status: RuntimeStatus):
    return {
        "state": status.state.name.lower(),
        "operation_active": status.operation_active,
        "waiting_classifications": status.waiting_classifications,
        "waiting_explanations": status.waiting_explanations,
        "classification_queue_capacity": status.classification_queue_capacity,
        "explanation_queue_capacity": status.explanation_queue_capacity,
    }


def _labels_to_response(labels: dict) -> dict:
    # DB stores plain {label: {probability, predicted}}; already response-shaped.
    return labels


def _correction_to_response(row) -> dict:
    return {
        "id": row.id,
        "classification_id": row.classification_id,
        "corrected_labels": row.corrected_labels,
        "moderator_id": row.moderator_id,
        "flagged_at": row.flagged_at,
        "created_at": row.created_at,
    }


@router.get(
    "/classifications",
    response_model=ClassificationListResponse,
    summary="List classifications",
    description="Newest-first feed with corrected flag. corrected=None shows all, false is the moderator queue (uncorrected only), true is audit (corrected only).",
)
async def list_classifications_view(
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    corrected: Annotated[bool | None, Query(description="None=all, true=only corrected, false=queue view")] = None,
):
    rows = await list_classifications(db, limit=limit, offset=offset, corrected=corrected)
    ids = [r.id for r in rows]
    latest = await list_latest_corrections(db, classification_ids=ids)
    items = []
    for r in rows:
        corr = latest.get(str(r.id))
        items.append(
            {
                "classification_id": r.id,
                "text": r.text,
                "labels": _labels_to_response(r.labels),
                "corrected": corr is not None,
                "created_at": r.created_at,
            }
        )
    return {"items": items, "limit": limit, "offset": offset}


@router.get(
    "/classifications/{classification_id}",
    response_model=ClassificationDetailResponse,
    summary="Classification detail",
    description="Immutable prediction plus corrected overlay (latest event wins) and all cached explanations. 404 if id unknown.",
)
async def get_classification_detail(
    classification_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    row = await get_classification(db, classification_id=classification_id)
    if row is None:
        raise HTTPException(status_code=404, detail="classification not found")
    corr = await get_latest_correction(db, classification_id=classification_id)
    expl_rows = await list_explanations(db, classification_id=classification_id)
    return {
        "classification_id": row.id,
        "text": row.text,
        "labels": _labels_to_response(row.labels),
        "model_version": row.model_version,
        "model_type": row.model_type,
        "created_at": row.created_at,
        "corrected": corr is not None,
        "corrected_labels": (corr.corrected_labels if corr is not None else None),
        "explanations": [
            {
                "text": row.text,
                "label": e.label,
                "spans": e.spans,
                "convergence_delta": e.convergence_delta,
            }
            for e in expl_rows
        ],
    }


@router.post(
    "/classifications/{classification_id}/corrections",
    response_model=CorrectionResponse,
    status_code=201,
    summary="Submit moderator correction",
    description="Appends a full 6-boolean verdict (even agreeing labels). Must match exactly the stored label set or 422. Latest event wins for display; history preserved. Example corrected_labels: {\"toxic\": true, \"severe_toxic\": false, \"obscene\": false, \"threat\": false, \"insult\": true, \"identity_hate\": false}.",
)
async def create_correction(
    classification_id: UUID,
    payload: CorrectionRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    from datetime import datetime, timezone

    row = await get_classification(db, classification_id=classification_id)
    if row is None:
        raise HTTPException(status_code=404, detail="classification not found")
    # Require full 6-label verdict so training rows stay unambiguous.
    if set(payload.corrected_labels.keys()) != set(row.labels.keys()):
        raise HTTPException(
            status_code=422,
            detail=f"corrected_labels must contain exactly {sorted(row.labels.keys())}",
        )
    flagged_at = datetime.now(timezone.utc)
    # Repository commits; re-fetch latest to return the stored row shape.
    await add_correction(
        db,
        classification_id=classification_id,
        corrected_labels=dict(payload.corrected_labels),
        flagged_at=flagged_at,
        moderator_id=payload.moderator_id,
    )
    latest = await get_latest_correction(db, classification_id=classification_id)
    assert latest is not None
    return _correction_to_response(latest)