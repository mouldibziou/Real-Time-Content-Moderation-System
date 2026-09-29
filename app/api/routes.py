from typing import Annotated
from fastapi import APIRouter
from fastapi.params import Depends

from app.api.schemas import(
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

def _model_info_to_list(info: ModelInfo) -> dict:
    """
    NOTE: If these helper functions exceed 4/5, write them to converters.py file
    also automate them
    """
    return {
        "model_type": info.model_type,
        "device": info.device,
        "labels": list(info.labels),
        "thresholds": list(info.thresholds),
        "max_length": info.max_length,
        "explanation_steps": info.explanation_steps,
    }

@router.get("/model_info", response_model=ModelInfoResponse)
async def get_model_info(
    runtime: Annotated[ApplicationRuntime, Depends(get_runtime)],
    ):
    model_info = runtime.model_info()
    return _model_info_to_list(model_info)


def _status_to_list(status: RuntimeStatus):
    return {
        "state": status.state.name.lower(),
        "operation_active": status.operation_active,
        "waiting_classifications": status.waiting_classifications,
        "waiting_explanations": status.waiting_explanations,
        "classification_queue_capacity": status.classification_queue_capacity,
        "explanation_queue_capacity": status.explanation_queue_capacity,
    }

@router.get("/status", response_model=StatusResponse)
async def get_runtime_status(
    runtime: Annotated[ApplicationRuntime, Depends(get_runtime)],
):
    status = runtime.status()
    return _status_to_list(status)

@router.get("/ready", response_model=ReadinessResponse)
async def get_readiness(
    runtime: Annotated[ApplicationRuntime, Depends(get_runtime)],
):
    is_ready = runtime.is_ready()
    return is_ready

