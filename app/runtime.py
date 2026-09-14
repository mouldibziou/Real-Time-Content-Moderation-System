from collections.abc import Callable

from app.classifier import ClassificationResult
from app.explainer import AttributionResult
from app.services import ApplicationServices, build_services
from app.settings import RuntimeSettings


class ApplicationRuntime:
    def __init__(
            self,
            settings: RuntimeSettings,
            *,
            services_builder: Callable[
                [RuntimeSettings], ApplicationServices
            ] = build_services
    ) -> None:
        self._settings = settings
        self._services_builder = services_builder
        self._services: ApplicationServices | None = None

    def is_ready(self) -> bool:
        """
        returns the status of the application runtime
        defaults to False, when the runtime is not yet started.
        """
        return self._services is not None

    def start(self) -> None:
        if self.is_ready():
            return

        self._services = self._services_builder(self._settings) 

    def stop(self) -> None:
        """
        Releases only the runtime's references and allows a later restart

        NOTE: During shutdown, services still exist for the active operation, 
        but the runtime should already be unready
        """
        self._services = None

    def predict(self, text: str) -> ClassificationResult:
        services = self._require_services()
        return services.classifier.predict(text)

    def predict_batch(self, texts: list[str]) -> list[ClassificationResult]:
        services = self._require_services()
        return services.classifier.predict_batch(texts)

    def explain(self, text: str, label: str) -> AttributionResult:
        services = self._require_services()
        return services.explainer.explain(text, label)

    def explain_batch(self, texts: list[str], label: str) -> list[AttributionResult]:
        services = self._require_services()
        return services.explainer.explain_batch(texts, label)
        
    def _require_services(self) -> ApplicationServices:
        if self._services is None:
            raise RuntimeError("Application runtime has not been started")

        return self._services