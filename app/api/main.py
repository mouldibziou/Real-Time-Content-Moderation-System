from typing import Callable
from logging import getLogger

from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from sqlalchemy import text

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

        # Soft DB check: log-only, never block boot. Return-first serves
        # 200s while Postgres blips; background persists retry/log.
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            logger.info("database connectivity check passed")
        except Exception:
            logger.warning("database unavailable at startup, serving without persistence", exc_info=True)

        try:
            yield
        finally:
            runtime.stop()
            runtime_executor.shutdown(wait=True, cancel_futures=True)
            await engine.dispose() # We await because it returns a coroutine

    app = FastAPI(
        title="Toxicity Classification API",
        version="1.0.0",
        lifespan=lifespan,
        # GZip cuts spans/list bodies on the wire for clients sending
        # Accept-Encoding: gzip.
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

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
