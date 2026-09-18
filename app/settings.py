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
    """
    NOTE: operation wait timeout in seconds, 
    max prediction ,and explanation batch sizes
    are provisional limits for deployment,
    these will be tuned later after real-world measurements.
    """
    operation_wait_timeout_seconds: float = 30.0
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