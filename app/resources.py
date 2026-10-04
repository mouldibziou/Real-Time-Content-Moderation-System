from dataclasses import dataclass
from time import monotonic
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerFast
from onnxruntime import InferenceSession
import torch
import logging

from app.settings import RuntimeSettings

logger = logging.getLogger(__name__)

"""
NOTE: I used a fast tokenizer as the base class for tokenizers.
"""
@dataclass(frozen=True)
class ModelResources:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerFast
    device: torch.device
    ort_session: InferenceSession | None = None

def load_resources(settings: RuntimeSettings) -> ModelResources:
    loading_started = monotonic()
    logger.info("Model loading started")

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

    ort_session = None
    if settings.inference_backend == "onnx":
        ort_session = load_onnx_session(settings)

    elapsed_seconds = monotonic() - loading_started
    logger.info(
        "Model loading completed device=%s duration_seconds=%.3f",
        device,
        elapsed_seconds,
    )

    return ModelResources(
        model=model,
        tokenizer=tokenizer,
        device=device,
        ort_session=ort_session,
    )

def load_onnx_session(settings: RuntimeSettings) -> InferenceSession:
    """Load the exported ONNX model for classification inference.

    The torch model above is still required for explanations, which need
    gradients. The tokenizer always comes from settings.model_dir; only the
    classification forward pass moves to ONNX.
    """
    onnx_file = settings.onnx_model_dir / "model.onnx"
    if not onnx_file.is_file():
        raise FileNotFoundError(
            f"ONNX model file not found: {onnx_file}. "
            f"Export it from {settings.model_dir} "
            f"(e.g. optimum-cli export onnx --model {settings.model_dir} "
            f"--task text-classification --opset 14 --output {settings.onnx_model_dir}), "
            f"or set inference_backend='torch'."
        )

    session = InferenceSession(str(onnx_file), providers=list(settings.onnx_providers))

    input_names = {model_input.name for model_input in session.get_inputs()}
    if not {"input_ids", "attention_mask"} <= input_names:
        raise ValueError(
            f"ONNX model inputs {sorted(input_names)} must include "
            f"'input_ids' and 'attention_mask'."
        )

    return session

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
