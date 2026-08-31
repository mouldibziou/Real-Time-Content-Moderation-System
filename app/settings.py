from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class RuntimeSettings:
    model_dir: Path
    labels: tuple[str, ...]
    thresholds: tuple[float, ...]
    device: str = "cpu"
    max_length: int = 512

    def __post_init__(self) -> None:
        if not self.model_dir.is_dir():
            raise ValueError(f"Model directory does not exist: {self.model_dir}")

        if self.max_length < 1:
            raise ValueError("max_length must be at least 1")

        if len(self.labels) == 0:
            raise ValueError("At least one label is required")

        if len(self.labels) != len(self.thresholds):
            raise ValueError("labels and thresholds must have the same length")

        if any(not 0.0 <= threshold <= 1.0 for threshold in self.thresholds):
            raise ValueError("Each threshold must be between 0 and 1")