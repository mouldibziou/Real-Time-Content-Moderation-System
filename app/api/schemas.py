from typing import Annotated, Literal
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
    model_config = {"extra": "forbid", "str_strip_whitespace": True}

    text: CommentText = Field(
        description="Single comment to explain",
        examples=["You are an idiot"],
    )

    label: ToxicLabel = Field(
        description="Label to explain",
        examples=["toxic"],
    )

class BatchExplainRequest(BaseModel):
    model_config = {"extra": "forbid", "str_strip_whitespace": True}

    texts: list[CommentText] = Field(
        min_length=1,
        description="Batch of comments to explain for a shared label",
        examples=[["You are an idiot", "This comment is helpful"]],
    )

    label: ToxicLabel = Field(
        description="Shared label to explain for the whole batch",
        examples=["toxic"],
    )

class ClassifyResponse(BaseModel):
    model_config = {"extra": "forbid"}

    text: str
    labels: dict[ToxicLabel, LabelPredictionResponse]

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