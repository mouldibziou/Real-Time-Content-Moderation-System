import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor
from fastapi import Request
from app.runtime import ApplicationRuntime
from app.errors import RuntimeNotReadyError
from functools import partial
from contextvars import copy_context
from app.metrics import current_api_metrics

"""
For learning purposes:

we use the dependencies because they attached to the FastAPI application state,
and do no exist as random variables.
"""

def get_runtime(request: Request) -> ApplicationRuntime:
    # Lifespan publishes runtime to app.state only after start() succeeds.
    # If tests skip lifespan, startup failed, or a request races startup,
    # the attribute is missing. That is "temporarily not ready" (retryable
    # 503), not a code bug (500), so translate it to the domain error that
    # our centralized handler maps to 503.
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise RuntimeNotReadyError("Application runtime is not ready")
    return runtime

def get_executor(request: Request) -> ThreadPoolExecutor:
    # Same reasoning as get_runtime: the pool is published only after both
    # start() and pool creation succeed. Missing pool also means not ready.
    executor = getattr(request.app.state, "runtime_executor", None)
    if executor is None:
        raise RuntimeNotReadyError("Application runtime is not ready")
    return executor


def get_model_snapshot(request: Request) -> dict:
    # Immutable per boot, published by lifespan. No lock, no 503: if the
    # model loaded, the snapshot exists; background writes degrade to
    # logged-only when the DB is down, never to missing metadata.
    snapshot = getattr(request.app.state, "model_snapshot", None)
    if snapshot is None:
        raise RuntimeNotReadyError("Application runtime is not ready")
    return snapshot

async def _call_runtime(
        func,
        executor: ThreadPoolExecutor,
        *args,
        **kwargs
        ):
    """
    Non-blocking bridge between FastAPI's event loop and sync model code.

    runtime.predict/explain are sync and heavy (seconds on torch). Calling
    them directly on the event loop would freeze it: no healthchecks, no
    status, no other requests until they return.

    This helper hands the blocking call to the dedicated single-worker pool
    via an executor submission and awaits its wrapped future. Awaiting suspends only
    this request's coroutine and gives the loop back, so fast paths stay
    responsive while the pool thread runs inference.

    Sync model code stays sync for torch/GIL simplicity; this async wrapper
    preserves server concurrency.
    """
    bound_func = partial(func, *args, **kwargs)
    if inspect.iscoroutinefunction(func):
        return await bound_func()
    loop = asyncio.get_running_loop()
    # Executor submissions do not propagate contextvars automatically.
    metrics = current_api_metrics()
    ticket = metrics.classification_ticket() if metrics is not None else None

    def execute():
        if ticket is not None:
            ticket.start()
        try:
            return bound_func()
        finally:
            if ticket is not None:
                ticket.finish()

    try:
        future = executor.submit(copy_context().run, execute)
    except BaseException:
        if ticket is not None:
            ticket.cancelled()
        raise
    if ticket is not None:
        future.add_done_callback(lambda completed: ticket.cancelled() if completed.cancelled() else None)
    return await asyncio.wrap_future(future, loop=loop)
