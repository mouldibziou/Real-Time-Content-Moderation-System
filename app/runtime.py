from collections.abc import Callable

from app.classifier import ToxicityClassifier
from app.factory import build_classifier
from app.settings import RuntimeSettings

class ApplicationRuntime:
    def __init__(
            self,
            settings: RuntimeSettings,
            classifier_builder: Callable[
                [RuntimeSettings], ToxicityClassifier
            ] = build_classifier
    ) -> None:
        self._settings = settings
        self._classifier_builder = classifier_builder
        self._classifier: ToxicityClassifier | None = None

    def start(self) -> None:
        if self._classifier is None:
            self._classifier = self._classifier_builder(self._settings)

    def is_ready(self) -> bool:
        return self._classifier is not None

    def predict(self, text: str):
        return self._get_classifier().predict(text)

    def predict_batch(self, texts: list[str]):
        return self._get_classifier().predict_batch(texts)

    def _get_classifier(self) -> ToxicityClassifier:
        """The predicitions fail clearly before the startup"""
        if self._classifier is None:
            raise RuntimeError("Application runtime has not been started")

        return self._classifier