import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import Request
from app.runtime import ApplicationRuntime
from app.errors import RuntimeNotReadyError
from functools import partial

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

async def _call_runtime(
        func, 
        executor: ThreadPoolExecutor, 
        *args,
        **kwargs
        ):
    """
    helper to call runtime functions asynchronously
    """
    bound_func = partial(func, *args, **kwargs)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, bound_func)