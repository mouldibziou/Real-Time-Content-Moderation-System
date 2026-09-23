from dataclasses import dataclass

from app.classifier import ToxicityClassifier
from app.explainer import ToxicityExplainer
from app.metadata import ModelInfo
from app.resources import load_resources
from app.settings import RuntimeSettings

@dataclass(frozen=True)
class ApplicationServices:
    classifier: ToxicityClassifier
    explainer: ToxicityExplainer
    model_info: ModelInfo

def build_services(settings: RuntimeSettings) -> ApplicationServices:
    """
    Enforcing that the two services share the same model and tokenizer
    """
    resources = load_resources(settings)

    model = resources.model
    tokenizer = resources.tokenizer
    device = resources.device

    labels = settings.labels
    max_length = settings.max_length
    thresholds = settings.thresholds
    n_steps = settings.n_steps

    classifier = ToxicityClassifier(
        model=model, 
        tokenizer=tokenizer, 
        thresholds=thresholds, 
        device=device, 
        max_length=max_length, 
        labels=labels
    )

    explainer = ToxicityExplainer(
        model=model, 
        tokenizer=tokenizer, 
        labels=labels, 
        device=device, 
        max_length=max_length, 
        n_steps=n_steps
    )

    model_info = ModelInfo(
        model_type=model.config.model_type,
        device=str(device),
        labels=settings.labels,
        thresholds=settings.thresholds,
        max_length=settings.max_length,
        explanation_steps=settings.n_steps
        )

    return ApplicationServices(
        classifier=classifier,
        explainer=explainer,
        model_info=model_info
    )