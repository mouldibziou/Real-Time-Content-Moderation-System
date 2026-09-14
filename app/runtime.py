from collections.abc import Callable

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