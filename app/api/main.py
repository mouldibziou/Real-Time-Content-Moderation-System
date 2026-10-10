from typing import Callable
from logging import getLogger
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.runtime import ApplicationRuntime
from app.services import ApplicationServices, build_services
from app.settings import RuntimeSettings, load_runtime_settings
from app.api.routes import router
from app.errors import (
    RuntimeNotReadyError,
    RuntimeBusyError,
    OperationWaitTimeoutError,
)
from app.db.engine import engine
from app.db.queue import close_redis
from app.observability import configure_logging, log_event, RequestLoggingMiddleware

def create_app(
        settings: RuntimeSettings,
        *,
        services_builder: Callable[
            [RuntimeSettings], ApplicationServices
        ] = build_services
) -> FastAPI:
    logger = getLogger(__name__)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging("api")
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
        app.state.engine = engine
        # Immutable snapshot: model info never changes per boot, so request
        # handlers read this dict instead of locking the runtime per call.
        # Saves one Condition acquire on every classify under load.
        info = runtime.model_info()
        app.state.model_snapshot = {
            "model_type": info.model_type,
            "device": info.device,
            "thresholds": list(info.thresholds),
        }

        # Boot can finish during a database outage; readiness gates traffic.
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            log_event(logger, "database.ready")
        except Exception:
            log_event(logger, "database.unavailable", level=logging.WARNING, exc_info=True)

        try:
            yield
        finally:
            runtime.stop()
            runtime_executor.shutdown(wait=True, cancel_futures=True)
            await close_redis()
            await engine.dispose() # We await because it returns a coroutine

    app = FastAPI(
        title="Toxicity Classification API",
        version="1.0.0",
        lifespan=lifespan,
        # GZip cuts spans/list bodies on the wire for clients sending
        # Accept-Encoding: gzip.
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(RequestLoggingMiddleware)

    @app.exception_handler(SQLAlchemyError)
    async def database_error_handler(request: Request, exc: SQLAlchemyError):
        log_event(logger, "database.request_failed", level=logging.ERROR, exc_info=True)
        return JSONResponse(status_code=503, content={"detail": "Database temporarily unavailable"})

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception):
        # Middleware already logged the error. Starlette renders this response
        # outside that middleware, so attach its correlation ID here as well.
        headers = {"X-Request-ID": request.state.request_id} if hasattr(request.state, "request_id") else {}
        return PlainTextResponse("Internal Server Error", status_code=500, headers=headers)

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

settings = load_runtime_settings()

# Uvicorn imports this application once per worker: app.api.main:app.
app = create_app(settings)
