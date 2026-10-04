from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from logging import getLogger
from typing import Annotated
from uuid import uuid4, UUID
from fastapi import APIRouter, BackgroundTasks
from fastapi.params import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import(
    BatchClassifyRequest,
    BatchExplainRequest,
    ClassifyBatchResponse,
    ClassifyRequest,
    ClassifyResponse,
    ExplainBatchResponse,
    ExplainRequest,
    ExplainResponse,
    ModelInfoResponse,
    ReadinessResponse,
    StatusResponse
)
from app.api.dependencies import (
    get_runtime,
    get_executor,
    _call_runtime
)
from app.db.engine import async_session
from app.db.repository import save_classification, save_classifications_batch
from app.metadata import ModelInfo
from app.runtime import ApplicationRuntime, RuntimeStatus

logger = getLogger(__name__)

# All routes in this api will automatically start with /v1
router = APIRouter(prefix="/v1")

# NOTE: we put no awaits here because these methods are strictly
# synchronous (standard def), as they are conditioned in the
# ApplicationRuntime logic.
@router.get("/model_info", response_model=ModelInfoResponse)
async def get_model_info(
    runtime: Annotated[ApplicationRuntime, Depends(get_runtime)],
    ):
    model_info = runtime.model_info()
    return _model_info_to_list(model_info)

@router.get("/status", response_model=StatusResponse)
async def get_runtime_status(
    runtime: Annotated[ApplicationRuntime, Depends(get_runtime)],
):
    status = runtime.status()
    return _status_to_list(status)

@router.get("/health/ready", response_model=ReadinessResponse)
async def get_readiness(
    runtime: Annotated[ApplicationRuntime, Depends(get_runtime)],
):
    """
    /v1/ready checks only runtime readiness as it is the only current dependency
    it will later be expanded
    """
    is_ready = runtime.is_ready()
    return {"is_ready": is_ready}

@router.get("/health/live")
async def health_check() -> dict[str, str]:
    return {"status": "healthy"}

# NOTE: not including a db health check because persistence operations are not public
# and a down db must not pull traffic out.

# NOTE: we await _call_runtime because it is an async function as it 
# declared in dependencies.py, because we want to call runtime functions
# asynchronously, even they are synchronized by default.
@router.post("/classify", response_model=ClassifyResponse)
async def classify_text(
    background_tasks: BackgroundTasks,
    runtime: Annotated[ApplicationRuntime, Depends(get_runtime)],
    executor: Annotated[ThreadPoolExecutor, Depends(get_executor)],
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
    info = runtime.model_info()
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
        model_type=info.model_type,
        device=info.device,
        thresholds=list(info.thresholds),
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

@router.post("/classify_batch", response_model=ClassifyBatchResponse)
async def classify_batch(
    background_tasks: BackgroundTasks,
    runtime: Annotated[ApplicationRuntime, Depends(get_runtime)],
    executor: Annotated[ThreadPoolExecutor, Depends(get_executor)],
    payload: BatchClassifyRequest
):
    texts = payload.texts
    results = await _call_runtime(runtime.predict_batch, executor, texts)
    info = runtime.model_info()
    ids = [uuid4() for _ in texts]
    items = [
        {
            "id": cid,
            "text": text,
            "text_hash": sha256(text.encode("utf-8")).hexdigest(),
            "model_version": "v2.0.0",
            "model_type": info.model_type,
            "device": info.device,
            "thresholds": list(info.thresholds),
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

@router.post("/explain", response_model=ExplainResponse)
async def explain_text(
    runtime: Annotated[ApplicationRuntime, Depends(get_runtime)],
    executor: Annotated[ThreadPoolExecutor, Depends(get_executor)],
    payload: ExplainRequest
):
    text = payload.text
    label = payload.label

    result = await _call_runtime(runtime.explain, executor, text, label)
    return {
        "text": text,
        "label": result.label,
        "spans": list(result.spans),
        "convergence_delta": result.convergence_delta
    }

@router.post("/explain_batch", response_model=ExplainBatchResponse)
async def explain_batch(
    runtime: Annotated[ApplicationRuntime, Depends(get_runtime)],
    executor: Annotated[ThreadPoolExecutor, Depends(get_executor)],
    payload: BatchExplainRequest
):
    texts = payload.texts
    label = payload.label

    results = await _call_runtime(runtime.explain_batch, executor, texts, label)
    return {
        "results": [
            {"text": text, "label": r.label, "spans": list(r.spans), "convergence_delta": r.convergence_delta}
            for text, r in zip(texts, results)
        ]
    }

def _model_info_to_list(info: ModelInfo) -> dict:
    return {
        "model_type": info.model_type,
        "device": info.device,
        "labels": list(info.labels),
        "thresholds": list(info.thresholds),
        "max_length": info.max_length,
        "explanation_steps": info.explanation_steps,
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