from collections.abc import Callable
from enum import Enum, auto
from threading import Condition

from app.classifier import ClassificationResult
from app.explainer import AttributionResult
from app.services import ApplicationServices, build_services
from app.settings import RuntimeSettings
from app.scheduling import WaitingOperation, OperationKind, choose_next_kind

from time import monotonic
from collections import deque # sequence container optimized for fast appends and pops from both ends

class RuntimeState(Enum):
    STOPPED = auto()
    STARTING = auto()
    READY = auto()
    STOPPING = auto()

class RuntimeBusyError(RuntimeError):
    """Raised when the relevant waiting queue has no remaining capacity"""

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

        self._classification_queue: deque[WaitingOperation] = deque()
        self._explanation_queue: deque[WaitingOperation] = deque()
        self._consecutive_classifications = 0

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
        self._validate_text(text)
        services = self._begin_operation()
        try:
            return services.classifier.predict(text)
        finally:
            self._end_operation()

    def predict_batch(self, texts: list[str]) -> list[ClassificationResult]:
        self._validate_texts(texts, 
                             max_batch_size=self._settings.max_prediction_batch_size)
        services = self._begin_operation()
        try:
            return services.classifier.predict_batch(texts)
        finally:
            self._end_operation()

    def explain(self, text: str, label: str) -> AttributionResult:
        self._validate_text(text)
        self._validate_label(label)
        services = self._begin_operation()
        try:
            return services.explainer.explain(text, label)
        finally:
            self._end_operation()

    def explain_batch(self, texts: list[str], label: str) -> list[AttributionResult]:
        self._validate_texts(texts, 
                             max_batch_size=self._settings.max_explanation_batch_size)
        self._validate_label(label)
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
        deadline = monotonic() + self._settings.operation_wait_timeout_seconds
        with self._condition:
            while self._operation_active and self._state is RuntimeState.READY:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for model access")
                self._condition.wait(timeout=remaining) # suspends the calling thread while releasing the condition lock

            services = self._require_services()
            self._operation_active = True
            return services

    def _end_operation(self) -> None:
        with self._condition:
            self._operation_active = False
            self._condition.notify_all() #notify_all() makes waiting callers eligible to resume

    def _validate_text(self, text: str, idx: int = -1) -> None:
        index_info = f"Texts[{idx}]" if idx != -1 else "Text"
        if not isinstance(text, str):
            raise TypeError(f"{index_info} must be a string")

        if len(text) > self._settings.max_text_chars:
            raise ValueError(f"{index_info} length must be at most {self._settings.max_text_chars}")
        
        if text.strip() == "":
            raise ValueError(f"{index_info} must contain non-whitespace characters")

    def _validate_texts(self, texts: list[str], *, max_batch_size: int) -> None:
        if not isinstance(texts, list):
            raise TypeError("The input must be a list")
        
        if len(texts) == 0:
            raise ValueError("Input list should not be empty")

        if len(texts) > max_batch_size:
            raise ValueError(f"Input batch size must be at most {max_batch_size}")
        
        for idx ,text in enumerate(texts):
            self._validate_text(text, idx)

    def _validate_label(self, label: str) -> None:
        """
        Validates explanation labels before reservation
        """
        if not isinstance(label, str):
            raise TypeError("Explanation label must be a string")
        if not (label in self._settings.labels):
            raise ValueError(f"Provided explanation label {label},\
                              is not one of these labels {self._settings.labels}")

    def _next_waiting_operation(self) -> WaitingOperation | None:
        """
        examines the queues and scheduling states, then identifies the next ticket to serve
        callers must hold _condition when using this helper
        """
        if self._operation_active:
            return None
        next_kind = choose_next_kind(
            classification_waiting=bool(self._classification_queue),
            explanation_waiting=bool(self._explanation_queue),
            consecutive_classifications=self._consecutive_classifications,
            classification_burst_limit=self._settings.classification_burst_limit
        )
        if next_kind is None:
            return None
        else:
            return self._queue_for(next_kind)[0]

    def _queue_for(
            self, kind: OperationKind
    ) -> deque[WaitingOperation]:
        """
        maps an operation category to its queue
        callers must hold _condition when using this helper
        """
        if kind == OperationKind.CLASSIFICATION:
            return self._classification_queue
        elif kind == OperationKind.EXPLANATION:
            return self._explanation_queue
        else:
            raise ValueError(f"Operation kind is not Classification or Explanation")