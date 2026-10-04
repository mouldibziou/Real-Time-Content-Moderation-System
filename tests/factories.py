"""Small immutable metadata fixtures for runtime tests."""
from app.metadata import ModelInfo


def make_model_info() -> ModelInfo:
    return ModelInfo(
        model_type="distilbert", device="cpu",
        labels=("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"),
        thresholds=(0.74, 0.45, 0.74, 0.59, 0.61, 0.76),
        max_length=512, explanation_steps=32,
        inference_backend="onnx",
    )
