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

    def __post_init__(self) -> None:
        if self.max_length < 1:
            raise ValueError("Maximum tokens length must be positive")
        if self.n_steps < 1:
            raise ValueError("Number of steps should be positive")
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