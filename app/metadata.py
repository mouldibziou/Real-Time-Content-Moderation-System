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