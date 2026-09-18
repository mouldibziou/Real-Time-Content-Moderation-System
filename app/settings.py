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
        if self.max_text_chars < 0:
            raise ValueError("Maximum tech characters must be positive")