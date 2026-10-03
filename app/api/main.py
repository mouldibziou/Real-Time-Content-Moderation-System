from typing import Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from app.runtime import ApplicationRuntime
from app.services import ApplicationServices, build_services
from app.settings import RuntimeSettings
from app.api.routes import router
from app.errors import (
    RuntimeNotReadyError,
    RuntimeBusyError,
    OperationWaitTimeoutError,
)

"""
NOTE: reminder

>>> The database design is return-first-persist-after for p95, strict explain-by-ID
"""

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

    # Centralized domain -> HTTP mapping. Routes let these bubble;
    # handlers translate to retryable vs client-error signals.
    @app.exception_handler(RuntimeNotReadyError)
    async def not_ready_handler(request: Request, exc: RuntimeNotReadyError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(RuntimeBusyError)
    async def busy_handler(request: Request, exc: RuntimeBusyError):
        return JSONResponse(status_code=429, content={"detail": str(exc)})

    @app.exception_handler(OperationWaitTimeoutError)
    async def wait_timeout_handler(request: Request, exc: OperationWaitTimeoutError):
        return JSONResponse(status_code=504, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        # Runtime validation (bad text, bad label, oversized batch).
        # Matches Pydantic's 422 envelope so clients handle one shape.
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(TypeError)
    async def type_error_handler(request: Request, exc: TypeError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    app.include_router(
        router
    )

    return app

settings = RuntimeSettings()

# main:app (module-level function) is needed by uvicorn to start the ASGI server
# to start the server run "uvicorn main:app"
app = create_app(settings)