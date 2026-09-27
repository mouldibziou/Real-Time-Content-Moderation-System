from typing import Callable

from fastapi import FastAPI
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from app.runtime import ApplicationRuntime
from app.services import ApplicationServices, build_services
from app.settings import RuntimeSettings

def create_app(
        settings: RuntimeSettings,
        *,
        services_builder: Callable[
            [RuntimeSettings], ApplicationServices
        ] = build_services
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = ApplicationRuntime(
            settings=settings,
            services_builder=services_builder
        )
        runtime.start()
        try:
            runtime_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="runtime")
        except Exception:
            runtime.stop()
            raise

        # runtime and executor should be published after runtime success
        app.state.runtime = runtime
        app.state.runtime_executor = runtime_executor

        try:
            yield
        finally:
            runtime.stop()
            runtime_executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(
        title="Toxicity Classification API",
        version="1.0.0",
        lifespan=lifespan
    )

    return app