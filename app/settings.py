from dataclasses import dataclass
from math import isfinite
from pathlib import Path

@dataclass(frozen=True)
class RuntimeSettings:
    max_length: int = 512
    model_dir: Path = Path("models/distilbert/v2.0.0")
    labels: tuple[str, ...] = ('toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate')
    thresholds: tuple[float, ...] = (0.74, 0.45, 0.74, 0.59, 0.61, 0.76)
    device: str = "mps"
    n_steps: int = 32
    max_text_chars: int = 10_000
    inference_backend: str = "onnx"
    onnx_model_dir: Path = Path("models/distilbert/v2.0.0-onnx")
    onnx_providers: tuple[str, ...] = ("CPUExecutionProvider",)

    """
    NOTE: the below values are provisional defaults, they will be altered accordingly with maximum acceptable latency.

    1. How many waiting requests can you accept before latency becomes unacceptable?
    2. How much optional explanation traffic are you willing to hold?
    3. How long can an explanation wait during sustained classification traffic?
    """
    max_waiting_classifications: int = 8 # maximum number of classification calls allowed to wait in the classification queue
    max_waiting_explanations: int = 2 # check right above
    classification_burst_limit: int = 5 # when both queues contain work, the maximum number of consecutive classification operations served before the scheduler must serve one explanation
    """
    NOTE: operation wait timeout in seconds, 
    max prediction ,and explanation batch sizes
    are provisional limits for deployment,
    these will be tuned later after real-world measurements.
    """
    operation_wait_timeout_seconds: float = 30.0 # NOTE: operation timeout should not heavily exceed the final expected latency (p99, p95)
    max_prediction_batch_size: int = 64
    max_explanation_batch_size: int = 8

    def __post_init__(self) -> None:
        if not isinstance(self.max_length, int) or type(self.max_length) is bool:
            raise TypeError("Maximum tokens length must be an integer")
        if self.max_length < 1:
            raise ValueError("Maximum tokens length must be positive")
        if not isinstance(self.n_steps, int) or type(self.n_steps) is bool:
            raise TypeError("Number of steps must be an integer")
        if self.n_steps < 1:
            raise ValueError("Number of steps must be positive")
        if not self.labels:
            raise ValueError("Labels should be nonempty, classification needs at least one label")
        if len(self.labels) != len(self.thresholds):
            raise ValueError("Every label needs exactly one threshold")
        if any (not label.strip() for label in self.labels):
            raise ValueError("Labels should not be empty")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("Labels should not contain duplicates")
        if any(not isfinite(threshold) or not (0 <= threshold <= 1) for threshold in self.thresholds):
            raise ValueError("Threshold values must be finite and between 0 and 1 inclusive")
        if not isinstance(self.max_text_chars, int) or type(self.max_text_chars) is bool:
            raise TypeError("Maximum text characters must be an integer")
        if self.max_text_chars < 1:
            raise ValueError("Maximum text characters must be positive")
        if not isinstance(self.max_prediction_batch_size, int) or type(self.max_prediction_batch_size) is bool:
            raise TypeError("Maximum prediction batch size must be an integer")
        if self.max_prediction_batch_size < 1:
            raise ValueError("Maximum prediction batch size must be positive")
        if not isinstance(self.max_explanation_batch_size, int) or type(self.max_explanation_batch_size) is bool:
            raise TypeError("Maximum explanation batch size must be an integer")
        if self.max_explanation_batch_size < 1:
            raise ValueError("Maximum explanation batch size must be positive")
        if not isinstance(self.operation_wait_timeout_seconds, (float, int)) or type(self.operation_wait_timeout_seconds) is bool:
            raise TypeError("Operation wait timeout must be a float or an integer")
        if not isfinite(self.operation_wait_timeout_seconds) or (self.operation_wait_timeout_seconds <= 0):
            raise ValueError("Operation wait timeout must be finite and geater than zero")
        if not isinstance(self.max_waiting_classifications, int) or type(self.max_waiting_classifications) is bool:
            raise TypeError("Maximum waiting classifications must be an integer")
        if self.max_waiting_classifications <= 0:
            raise ValueError("Maximum waiting classifications must be positive")
        if not isinstance(self.max_waiting_explanations, int) or type(self.max_waiting_explanations) is bool:
            raise TypeError("Maximum waiting explanations must be an integer")
        if self.max_waiting_explanations <= 0:
            raise ValueError("Maximum waiting explanations must be positive")
        if not isinstance(self.classification_burst_limit, int) or type(self.classification_burst_limit) is bool:
            raise TypeError("Classification burst limit must be an integer")
        if self.classification_burst_limit <= 0:
            raise ValueError("Classification burst limit must be positive")
        if not isinstance(self.inference_backend, str):
            raise TypeError("Inference backend must be a string")
        if self.inference_backend not in ("torch", "onnx"):
            raise ValueError("Inference backend must be either 'torch' or 'onnx'")
        if not isinstance(self.onnx_model_dir, Path):
            raise TypeError("ONNX model directory must be a pathlib.Path")
        if not isinstance(self.onnx_providers, (tuple, list)) or not self.onnx_providers:
            raise ValueError("ONNX providers must be a nonempty tuple or list")
        if any(not isinstance(provider, str) or not provider.strip() for provider in self.onnx_providers):
            raise ValueError("ONNX providers must be nonempty strings")