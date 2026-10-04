from typing import Annotated, Literal
from uuid import UUID
from pydantic import BaseModel, Field

ToxicLabel = Literal['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
CommentText = Annotated[str, Field(min_length=1, max_length=10_000)]


class AttributedSpanResponse(BaseModel):
    model_config = {"extra": "forbid"}

    text: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    score: float
    normalized_importance: float | None = Field(ge=0, le=1.0)

class LabelPredictionResponse(BaseModel):
    model_config = {"extra": "forbid"}

    probability: float = Field(ge=0, le=1.0)
    predicted: bool

class ClassifyRequest(BaseModel):
    model_config = {"extra": "forbid", "str_strip_whitespace": True}

    text: CommentText = Field(
        description="Single comment to classify",
        examples=["You are an idiot"],
    )

class BatchClassifyRequest(BaseModel):
    model_config = {"extra": "forbid", "str_strip_whitespace": True}

    texts: list[CommentText] = Field(
        min_length=1,
        description="Batch of comments to classify",
        examples=[["You are an idiot", "This comment is helpful"]],
    )

class ExplainRequest(BaseModel):
    model_config = {"extra": "forbid"}

    classification_id: UUID = Field(
        description="Prior classification to explain (strict: must exist)",
    )

    label: ToxicLabel = Field(
        description="Label to explain",
        examples=["toxic"],
    )

class BatchExplainRequest(BaseModel):
    model_config = {"extra": "forbid"}

    classification_ids: list[UUID] = Field(
        min_length=1,
        description="Prior classifications to explain for a shared label",
    )

    label: ToxicLabel = Field(
        description="Shared label to explain for the whole batch",
        examples=["toxic"],
    )

class ClassifyResponse(BaseModel):
    model_config = {"extra": "forbid"}

    text: str
    labels: dict[ToxicLabel, LabelPredictionResponse]
    classification_id: UUID

class ClassifyBatchResponse(BaseModel):
    model_config = {"extra": "forbid"}

    results: list[ClassifyResponse]

class ExplainResponse(BaseModel):
    model_config = {"extra": "forbid"}

    text: str
    label: ToxicLabel
    spans: list[AttributedSpanResponse]
    convergence_delta: float | None = None

class ExplainBatchResponse(BaseModel):
    model_config = {"extra": "forbid"}

    results: list[ExplainResponse]

class StatusResponse(BaseModel):
    model_config = {"extra": "forbid"}

    state: Literal["stopped", "starting", "ready", "stopping"]
    operation_active: bool
    waiting_classifications: int = Field(ge=0)
    waiting_explanations: int = Field(ge=0)
    classification_queue_capacity: int = Field(ge=1)
    explanation_queue_capacity: int = Field(ge=1)

class ModelInfoResponse(BaseModel):
    model_config = {"extra": "forbid"}

    model_type: str
    device: str
    labels: list[str]
    thresholds: list[float]
    max_length: int = Field(ge=1)
    explanation_steps: int = Field(ge=1)

class ReadinessResponse(BaseModel):
    model_config = {"extra": "forbid"}

    is_ready: bool