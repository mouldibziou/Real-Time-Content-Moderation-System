from concurrent.futures import ThreadPoolExecutor
import asyncio
from hashlib import sha256
from logging import getLogger
import logging
from time import monotonic
from typing import Annotated
from uuid import uuid4, UUID
from fastapi import APIRouter, HTTPException, Query
from fastapi.params import Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text
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
from app.settings import load_model_version
from app.observability import bind_context, log_event
from app.metrics import current_api_metrics

MODEL_VERSION = load_model_version()

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
    description="Checks model readiness and PostgreSQL connectivity; returns 503 when unavailable.",
)
async def get_readiness(
    runtime: Annotated[ApplicationRuntime, Depends(get_runtime)],
):
    is_ready = runtime.is_ready()
    if is_ready and getattr(runtime, "remote_inference", False) is True:
        is_ready = await runtime.check_ready()
    if is_ready:
        async def check_database():
            async with async_session() as session:
                await session.execute(text("SELECT 1"))
        try:
            await asyncio.wait_for(check_database(), timeout=2)
        except Exception:
            is_ready = False
    return JSONResponse(status_code=200 if is_ready else 503, content={"is_ready": is_ready})

@router.get(
    "/health/live",
    summary="Liveness probe",
    description="Process alive check. No runtime, no DB, always 200 when Python answers. Restart on failure.",
)
async def health_check() -> dict[str, str]:
    return {"status": "healthy"}

# NOTE: we await _call_runtime because it is an async function as it 
# declared in dependencies.py, because we want to call runtime functions
# asynchronously, even they are synchronized by default.
@router.post(
    "/classify",
    response_model=ClassifyResponse,
    summary="Classify one comment",
    description="Scores text on all 6 labels. Returns a committed canonical classification_id; duplicate text reuses its stored ID. Use the id for strict explain. Example: {\"text\": \"You are an idiot\"}.",
)
async def classify_text(
    db: Annotated[AsyncSession, Depends(get_db)],
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
    started = monotonic()
    bind_context(stage="classification.runtime", model_version=MODEL_VERSION)
    result = await _call_runtime(runtime.predict, executor, text) # NOTE: Next time figure this out BETTER
    classification_id = uuid4()

    # Commit before responding so every returned ID resolves immediately.
    # Repository deduplication returns the original ID for repeated inputs.
    labels_plain = {
        label: {"probability": pred.probability, "predicted": pred.predicted}
        for label, pred in result.labels.items()
    }
    classification_id = await save_classification(
        db,
        id=classification_id,
        text=text,
        text_hash=sha256(text.encode("utf-8")).hexdigest(),
        model_version=MODEL_VERSION,
        model_type=snapshot["model_type"],
        device=snapshot["device"],
        thresholds=snapshot["thresholds"],
        labels=labels_plain,
    )

    bind_context(stage="response", classification_id=classification_id)
    if (metrics := current_api_metrics()) is not None:
        metrics.record_predictions([result])
    log_event(logger, "classification.completed", batch_size=1,
              duration_ms=round((monotonic() - started) * 1000, 2))
    return {
        "text": text,
        "labels": result.labels,
        "classification_id": classification_id
    }

@router.post(
    "/classify_batch",
    response_model=ClassifyBatchResponse,
    summary="Classify a batch",
    description="Scores 1..N texts in input order, returns committed canonical IDs, with duplicates preserved in input order. Example: {\"texts\": [\"You are an idiot\", \"Thanks for helping\"]}.",
)
async def classify_batch(
    db: Annotated[AsyncSession, Depends(get_db)],
    runtime: Annotated[ApplicationRuntime, Depends(get_runtime)],
    executor: Annotated[ThreadPoolExecutor, Depends(get_executor)],
    snapshot: Annotated[dict, Depends(get_model_snapshot)],
    payload: BatchClassifyRequest
):
    texts = payload.texts
    started = monotonic()
    bind_context(stage="classification.runtime", model_version=MODEL_VERSION, batch_size=len(texts))
    results = await _call_runtime(runtime.predict_batch, executor, texts)
    ids = [uuid4() for _ in texts]
    items = [
        {
            "id": cid,
            "text": text,
            "text_hash": sha256(text.encode("utf-8")).hexdigest(),
            "model_version": MODEL_VERSION,
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
    ids = await save_classifications_batch(db, items=items)
    if (metrics := current_api_metrics()) is not None:
        metrics.record_predictions(results)
    bind_context(stage="response")
    log_event(logger, "classification.batch_completed", batch_size=len(texts),
              duration_ms=round((monotonic() - started) * 1000, 2))
    return {
        "results": [
            {"text": text, "labels": r.labels, "classification_id": cid}
            for text, r, cid in zip(texts, results, ids)
        ]
    }

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
    model_version = MODEL_VERSION
    bind_context(stage="explanation.lookup", classification_id=classification_id,
                 label=label, model_version=model_version)

    # Strict: must have been classified before. Reuse exact stored text.
    classification = await get_classification(db, classification_id=classification_id)
    if classification is None:
        raise HTTPException(status_code=404, detail="classification not found")
    if classification.model_version != model_version:
        raise HTTPException(status_code=409, detail="classification model version unavailable")

    # Cache before GPU: same (id, label, version) returns stored spans.
    cached = await find_explanation(
        db, classification_id=classification_id, label=label, model_version=model_version
    )
    if cached is not None:
        bind_context(stage="response")
        log_event(logger, "explanation.cache_hit")
        body = ExplainResponse.model_validate(
            {
                "text": classification.text,
                "label": cached.label,
                "spans": cached.spans,
                "convergence_delta": cached.convergence_delta,
            }
        ).model_dump(mode="json")
        if (metrics := current_api_metrics()) is not None:
            metrics.cache_hits.inc()
        return JSONResponse(status_code=200, content=body)

    # Miss: idempotent create-or-return, then wake the worker via Redis.
    # The API never runs IG here, so classify stays hot under explain load.
    from app.db.repository import create_job

    bind_context(stage="explanation.job_creation")
    job, created = await create_job(
        db,
        classification_id=classification_id,
        label=label,
        model_version=model_version,
    )
    bind_context(job_id=job.id)
    if created and (metrics := current_api_metrics()) is not None:
        metrics.jobs_created.inc()
    log_event(logger, "explanation.job_created" if created else "explanation.job_reused",
              job_status=job.status)
    try:
        from app.db.queue import enqueue_explain_job

        if created:
            bind_context(stage="explanation.notification")
            await enqueue_explain_job(job.id)
            log_event(logger, "explanation.worker_notified", level=logging.DEBUG)
    except Exception:
        log_event(logger, "explanation.notification_failed", level=logging.WARNING,
                  exc_info=True, recovery="database_polling")
        # PostgreSQL polling recovers a missed notification while the worker runs.
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

    bind_context(stage="explanation.lookup")
    fresh = await _find(
        db, classification_id=classification_id, label=label, model_version=model_version
    )
    if fresh is not None:
        bind_context(stage="response")
        log_event(logger, "explanation.cache_hit", source="post_submission_check")
        ok = ExplainResponse.model_validate(
            {
                "text": classification.text,
                "label": fresh.label,
                "spans": fresh.spans,
                "convergence_delta": fresh.convergence_delta,
            }
        ).model_dump(mode="json")
        if (metrics := current_api_metrics()) is not None:
            metrics.cache_hits.inc()
        return JSONResponse(status_code=200, content=ok)
    bind_context(stage="response")
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

    bind_context(stage="explanation.job_lookup", job_id=job_id)
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
    model_version = MODEL_VERSION
    bind_context(stage="explanation.batch_lookup", model_version=model_version,
                 label=label, batch_size=len(ids))

    # Strict: all rows must exist to keep order exact.
    for cid in ids:
        row = await get_classification(db, classification_id=cid)
        if row is None:
            raise HTTPException(status_code=404, detail=f"classification {cid} not found")
        if row.model_version != model_version:
            raise HTTPException(status_code=409, detail="classification model version unavailable")

    jobs_out = []
    cached_count = 0
    created_count = 0
    for cid in ids:
        # Served from cache: ensure a done job exists for uniform polling.
        bind_context(stage="explanation.lookup")
        cached = await find_explanation(
            db, classification_id=cid, label=label, model_version=model_version
        )
        bind_context(stage="explanation.job_creation")
        job, created = await create_job(
            db, classification_id=cid, label=label, model_version=model_version
        )
        log_event(logger, "explanation.job_created" if created else "explanation.job_reused",
                  classification_id=cid, job_id=job.id, job_status=job.status)
        created_count += int(created)
        if created and (metrics := current_api_metrics()) is not None:
            metrics.jobs_created.inc()
        cached_count += int(cached is not None)
        if cached is not None and job.status != "done":
            from app.db.repository import mark_job

            bind_context(stage="explanation.persistence")
            job = await mark_job(db, job_id=job.id, status="done")
        if cached is not None:
            log_event(logger, "explanation.cache_hit", classification_id=cid, job_id=job.id)
        if created and cached is None:
            try:
                from app.db.queue import enqueue_explain_job

                bind_context(stage="explanation.notification")
                await enqueue_explain_job(job.id)
                log_event(logger, "explanation.worker_notified", level=logging.DEBUG,
                          classification_id=cid, job_id=job.id)
            except Exception:
                log_event(logger, "explanation.notification_failed", level=logging.WARNING,
                          exc_info=True, classification_id=cid, job_id=job.id,
                          recovery="database_polling")
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
    bind_context(stage="response")
    if (metrics := current_api_metrics()) is not None:
        metrics.cache_hits.inc(cached_count)
    log_event(logger, "explanation.batch_submitted", batch_size=len(ids),
              created_count=created_count, cached_count=cached_count)
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
        "model_version": MODEL_VERSION,
        "max_text_chars": info.max_text_chars,
        "max_prediction_batch_size": info.max_prediction_batch_size,
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
    bind_context(stage="correction.persistence", classification_id=classification_id)
    # Repository commits; re-fetch latest to return the stored row shape.
    await add_correction(
        db,
        classification_id=classification_id,
        corrected_labels=dict(payload.corrected_labels),
        flagged_at=flagged_at,
        moderator_id=payload.moderator_id,
    )
    log_event(logger, "correction.saved")
    latest = await get_latest_correction(db, classification_id=classification_id)
    assert latest is not None
    return _correction_to_response(latest)
