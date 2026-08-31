from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

from app.classifier import ToxicityClassifier
from app.settings import RuntimeSettings

def build_classifier(settings: RuntimeSettings) -> ToxicityClassifier:

    tokenizer = AutoTokenizer.from_pretrained(settings.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(settings.model_dir)

    if model.config.num_labels != len(settings.labels):
        raise ValueError(
            "Model output count does not match configured labels: "
            f"{model.config.num_labels} != {len(settings.labels)}"
        )

    device = torch.device(settings.device)
    model.to(device)
    model.eval()

    return ToxicityClassifier(
        model=model,
        tokenizer=tokenizer,
        labels=settings.labels,
        thresholds=settings.thresholds,
        device=device,
        max_length=settings.max_length
    )