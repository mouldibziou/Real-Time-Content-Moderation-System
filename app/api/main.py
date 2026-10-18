from typing import Callable
from logging import getLogger
import logging
import asyncio
import os

from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, start_http_server
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
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
from app.metrics import ApiMetrics, MetricsMiddleware, read_queue_counts, render_api_metrics, api_export_registry, refresh_queue_metrics

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
        remote = settings.inference_backend == "triton"
        if remote:
            from app.triton_runtime import TritonRuntime
        runtime = TritonRuntime(settings) if remote else ApplicationRuntime(
            settings=settings, services_builder=services_builder)
        if remote:
            await runtime.start()
        else:
            runtime.start()
        try:
            runtime_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="runtime")
        except Exception:
            if remote:
                await runtime.stop()
            else:
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

        metrics_server = metrics_thread = queue_task = None
        try:
            port = int(os.getenv("API_METRICS_PORT", "0"))
            if not 0 <= port <= 65535:
                raise ValueError("API_METRICS_PORT must be between 0 and 65535")
            if port:
                metrics_server, metrics_thread = start_http_server(
                    port, addr=os.getenv("API_METRICS_HOST", "0.0.0.0"),
                    registry=api_export_registry(metrics, metrics.queue_counts))
                queue_task = asyncio.create_task(refresh_queue_metrics(metrics))
                log_event(logger, "metrics.server_started", port=port)
            yield
        finally:
            if queue_task is not None:
                queue_task.cancel()
                with suppress(asyncio.CancelledError):
                    await queue_task
            if metrics_server is not None:
                await asyncio.to_thread(metrics_server.shutdown)
                metrics_server.server_close()
                await asyncio.to_thread(metrics_thread.join)
            if remote:
                await runtime.stop()
            else:
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
    metrics = ApiMetrics()
    app.state.metrics = metrics
    app.add_middleware(MetricsMiddleware, metrics=metrics)

    @app.get("/metrics/", include_in_schema=False)
    async def prometheus_metrics():
        counts = await read_queue_counts()
        return Response(render_api_metrics(metrics, counts), headers={"Content-Type": CONTENT_TYPE_LATEST})

    @app.exception_handler(SQLAlchemyError)
    async def database_error_handler(request: Request, exc: SQLAlchemyError):
        log_event(logger, "database.request_failed", level=logging.ERROR, exc_info=True)
        return JSONResponse(status_code=503, content={"detail": "Database temporarily unavailable"})

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception):
        # Middleware already logged the error. Starlette renders this response
        # outside that middleware, so attach its correlation ID here as well.
        headers = {"X-Request-ID": request.state.request_id} if hasattr(request.state, "request_id") else {}
        origin = request.headers.get("origin")
        if origin in settings.cors_origins:
            headers.update({"Access-Control-Allow-Origin": origin, "Vary": "Origin",
                            "Access-Control-Expose-Headers": "X-Request-ID"})
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
    # Add CORS outside the request middlewares for translated API errors.
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=list(settings.cors_origins),
            allow_credentials=False, allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type"], expose_headers=["X-Request-ID"],
        )

    return app

settings = load_runtime_settings()

# Uvicorn imports this application once per worker: app.api.main:app.
app = create_app(settings)
