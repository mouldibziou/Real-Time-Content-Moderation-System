from concurrent.futures import ThreadPoolExecutor
from typing import Annotated
from uuid import uuid4
from fastapi import APIRouter
from fastapi.params import Depends

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
from app.metadata import ModelInfo
from app.runtime import ApplicationRuntime, RuntimeStatus

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

# NOTE: we await _call_runtime because it is an async function as it 
# declared in dependencies.py, because we want to call runtime functions
# asynchronously, even they are synchronized by default.
@router.post("/classify", response_model=ClassifyResponse)
async def classify_text(
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
    return {
        "text": text,
        "labels": result.labels,
        "classification_id": classification_id
    }

@router.post("/classify_batch", response_model=ClassifyBatchResponse)
async def classify_batch(
    runtime: Annotated[ApplicationRuntime, Depends(get_runtime)],
    executor: Annotated[ThreadPoolExecutor, Depends(get_executor)],
    payload: BatchClassifyRequest
):
    texts = payload.texts
    results = await _call_runtime(runtime.predict_batch, executor, texts)
    return {
        "results": [
            {"text": text, "labels": r.labels, "classification_id": uuid4()}
            for text, r in zip(texts, results)
        ]
    }

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