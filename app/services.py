from dataclasses import dataclass

from app.classifier import ToxicityClassifier, ToxicityOnnxClassifier
from app.explainer import ToxicityExplainer
from app.metadata import ModelInfo
from app.resources import load_resources
from app.settings import RuntimeSettings

@dataclass(frozen=True)
class ApplicationServices:
    classifier: ToxicityClassifier | ToxicityOnnxClassifier
    explainer: ToxicityExplainer
    model_info: ModelInfo

def build_services(settings: RuntimeSettings) -> ApplicationServices:
    """
    Enforcing that the two services share the same model and tokenizer.

    Classification runs on the configured inference backend (ONNX by
    default), while explanations always use the torch model, which provides
    the gradients ONNX cannot.
    """
    resources = load_resources(settings)

    model = resources.model
    tokenizer = resources.tokenizer
    device = resources.device

    labels = settings.labels
    max_length = settings.max_length
    thresholds = settings.thresholds
    n_steps = settings.n_steps

    if settings.inference_backend == "onnx":
        if resources.ort_session is None:
            raise RuntimeError("ONNX inference backend selected but no ONNX session was loaded")
        classifier = ToxicityOnnxClassifier(
            session=resources.ort_session,
            tokenizer=tokenizer,
            thresholds=thresholds,
            max_length=max_length,
            labels=labels,
            inference_batch_size=settings.inference_batch_size,
        )
    else:
        classifier = ToxicityClassifier(
            model=model,
            tokenizer=tokenizer,
            thresholds=thresholds,
            device=device,
            max_length=max_length,
            labels=labels,
            inference_batch_size=settings.inference_batch_size,
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
        explanation_steps=settings.n_steps,
        inference_backend=settings.inference_backend,
        max_text_chars=settings.max_text_chars,
        max_prediction_batch_size=settings.max_prediction_batch_size,
        )

    return ApplicationServices(
        classifier=classifier,
        explainer=explainer,
        model_info=model_info
    )
