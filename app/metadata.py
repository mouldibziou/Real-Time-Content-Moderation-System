from dataclasses import dataclass

@dataclass(frozen=True)
class ModelInfo:
    model_type: str
    device: str
    labels: tuple[str, ...]
    thresholds: tuple[float, ...]
    max_length: int
    explanation_steps: int
    inference_backend: str = "onnx"
    max_text_chars: int = 10_000
    max_prediction_batch_size: int = 64
