from dataclasses import dataclass
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerFast
import torch

from app.settings import RuntimeSettings

"""
NOTE: I used a fast tokenizer as the base class for tokenizers.
"""
@dataclass(frozen=True)
class ModelResources:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerFast
    device: torch.device

def load_resources(settings: RuntimeSettings) -> ModelResources:
    if not settings.model_dir.is_dir():
        raise FileNotFoundError(
            f"Model directory does not exist or is not a directory: {settings.model_dir}"
        )

    device = resolve_device(settings.device)

    tokenizer = AutoTokenizer.from_pretrained(
        settings.model_dir,
        use_fast=True,
        local_files_only=True,
        )
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for explanation offsets.")

    if tokenizer.pad_token_id is None:
        raise ValueError("Explainer needs 'pad token' to construct its baseline," \
        "Tokenizer should return a 'pad_token_id' token")

    model = AutoModelForSequenceClassification.from_pretrained(
        settings.model_dir,
        local_files_only=True,
    )

    if model.config.num_labels != len(settings.labels):
        raise ValueError(f"Number of prediction labels: {len(settings.labels)}, \
                         is different from model's number of prediction labels: {model.config.num_labels}")

    if settings.max_length > model.config.max_position_embeddings:
        raise ValueError(f"Input's max token input: {settings.max_length}, \
                         exceeds {model.config.model_type}'s max positional embeddings: {model.config.max_position_embeddings}")


    model_labels = tuple(
        model.config.id2label[index]
        for index in range(model.config.num_labels)
    )
    if settings.labels != model_labels:
        raise ValueError(
    f"Label order mismatch: settings={settings.labels}, model={model_labels}"
)

    model.to(device)
    model.eval()

    return ModelResources(
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

def resolve_device(value: str) -> torch.device:
    """Resolve a supported device without silently falling back to CPU."""
    device = torch.device(value)
    if device.type not in {"cpu", "mps", "cuda"}:
        raise ValueError(f"Unsupported device type: {device.type}")

    if device.type in {"cpu", "mps"} and device.index is not None:
        raise ValueError(f"Device indices are not supported for {device.type}")

    if device.type == "cpu":
        return device

    if device.type == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS device is not available")
        return device

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is not available")

    index = torch.cuda.current_device() if device.index is None else device.index
    device_count = torch.cuda.device_count()
    if not 0 <= index < device_count:
        raise ValueError(
            f"CUDA device index {index} is out of range for {device_count} available devices"
        )
    return torch.device("cuda", index)
