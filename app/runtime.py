from collections.abc import Callable
from enum import Enum, auto
from threading import Condition

from prompt_toolkit import Application

from app.classifier import ClassificationResult
from app.explainer import AttributionResult
from app.services import ApplicationServices, build_services
from app.settings import RuntimeSettings

class RuntimeState(Enum):
    STOPPED = auto()
    STARTING = auto()
    READY = auto()
    STOPPING = auto()

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

        self._condition = Condition()
        self._state = RuntimeState.STOPPED
        self._operation_active = False

    def is_ready(self) -> bool:
        with self._condition:
            return self._state is RuntimeState.READY

    def start(self) -> None:
        with self._condition:
            if self._state is RuntimeState.READY:
                return
            if self._state is not RuntimeState.STOPPED:
                raise RuntimeError("Startup cannot begin in the current state")
            self._state = RuntimeState.STARTING
            
        try:
            services = self._services_builder(self._settings)
        except Exception:
            with self._condition:
                self._state = RuntimeState.STOPPED
            raise
        else:
            with self._condition:
                self._services = services
                self._state = RuntimeState.READY

    def stop(self) -> None:
        with self._condition:
            if self._state is RuntimeState.STOPPED:
                return

            if self._state is not RuntimeState.READY:
                raise RuntimeError(f"Application runtime is already {self._state.name}")
            self._state = RuntimeState.STOPPING
            self._condition.notify_all()
            while self._operation_active:
                self._condition.wait()
            self._services = None
            self._state = RuntimeState.STOPPED

    def predict(self, text: str) -> ClassificationResult:
        services = self._begin_operation()
        try:
            return services.classifier.predict(text)
        finally:
            self._end_operation()

    def predict_batch(self, texts: list[str]) -> list[ClassificationResult]:
        services = self._begin_operation()
        try:
            return services.classifier.predict_batch(texts)
        finally:
            self._end_operation()

    def explain(self, text: str, label: str) -> AttributionResult:
        services = self._begin_operation()
        try:
            return services.explainer.explain(text, label)
        finally:
            self._end_operation()

    def explain_batch(self, texts: list[str], label: str) -> list[AttributionResult]:
        services = self._begin_operation()
        try:
            return services.explainer.explain_batch(texts, label)
        finally:
            self._end_operation()
        
    def _require_services(self) -> ApplicationServices:
        if self._services is None or self._state is not RuntimeState.READY:
            raise RuntimeError("Application runtime is not ready")

        return self._services

    def _begin_operation(self) -> ApplicationServices:
        with self._condition:
            while self._operation_active and self._state is RuntimeState.READY:
                self._condition.wait() #wait() blocks other caller(s), we treat one operation at a time

            services = self._require_services()
            self._operation_active = True
            return services

    def _end_operation(self) -> None:
        with self._condition:
            self._operation_active = False
            self._condition.notify_all() #notify_all() makes waiting callers eligible to resume
