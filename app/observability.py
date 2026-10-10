"""Small, shared JSON logging setup for the API and explanation worker."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import json
import logging
import os
import sys
from time import monotonic
import traceback
from uuid import uuid4


_context: ContextVar[dict] = ContextVar("log_context", default={})


def bind_context(**fields):
    """Add safe identifiers/stage information to this task's subsequent logs."""
    _context.set({**_context.get(), **fields})


@contextmanager
def log_context(**fields):
    token = _context.set({**_context.get(), **fields})
    try:
        yield
    finally:
        _context.reset(token)


def log_event(logger, event, *, level=logging.INFO, exc_info=False, **fields):
    logger.log(level, event, extra={"fields": {**_context.get(), **fields}, "event": event},
               exc_info=exc_info)


class JsonFormatter(logging.Formatter):
    """One JSON object per line; never render exception messages or SQL params."""

    def __init__(self, service="api"):
        super().__init__()
        self.service = service

    def format(self, record):
        data = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "service": self.service,
            "logger": record.name,
            "event": getattr(record, "event", "application.message"),
            "message": record.getMessage(),
            **_context.get(),
            **getattr(record, "fields", {}),
        }
        if record.name == "uvicorn.access" and isinstance(record.args, tuple) and len(record.args) == 5:
            _, method, path, _, status = record.args
            data.update(event="http.access", message="HTTP request", method=method,
                        path=path.split("?", 1)[0], status_code=status)
        if record.exc_info and record.exc_info[0] is not None:
            data["error_type"] = record.exc_info[0].__name__
            # Keep debugging locations without exception values, source, or locals.
            data["error_stack"] = [
                {"file": frame.filename, "line": frame.lineno, "function": frame.name}
                for frame in traceback.extract_tb(record.exc_info[2])
            ]
        return json.dumps(data, default=str, ensure_ascii=True)


def configure_logging(service):
    """Called once per process at startup, before loading models."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(service))
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(),
                        handlers=[handler], force=True)


class RequestLoggingMiddleware:
    """Correlate concurrent requests and log once after response handling ends."""

    def __init__(self, app):
        self.app = app
        self.logger = logging.getLogger("app.http")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        # Generate our own ID rather than trusting an arbitrary client header.
        request_id = str(uuid4())
        started = monotonic()
        status = None
        response_complete = False
        error = None

        async def logged_send(message):
            nonlocal status, response_complete
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = [(k, v) for k, v in message.get("headers", [])
                           if k.lower() != b"x-request-id"]
                message = {**message, "headers": [*headers, (b"x-request-id", request_id.encode())]}
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                response_complete = True

        with log_context(request_id=request_id, method=scope["method"], stage="request"):
            scope.setdefault("state", {})["request_id"] = request_id
            try:
                await self.app(scope, receive, logged_send)
            except Exception:
                error = sys.exc_info()
                raise
            finally:
                route = getattr(scope.get("route"), "path", "unmatched")
                # Quiet successful probes, but keep unhealthy probes visible.
                quiet = route in {"/v1/health/live", "/v1/health/ready"} and status == 200
                if not quiet or error or not response_complete:
                    failed = error is not None
                    event = "request.failed" if failed else (
                        "request.completed" if response_complete else "request.aborted")
                    level = logging.ERROR if failed or (status is not None and status >= 500) else (
                        logging.WARNING if not response_complete else logging.INFO)
                    log_event(self.logger, event, level=level, exc_info=error or False,
                              route=route, status_code=status if status is not None else (500 if failed else None),
                              response_complete=response_complete,
                              duration_ms=round((monotonic() - started) * 1000, 2))
