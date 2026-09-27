import asyncio

from app.classifier import ClassificationResult
from app.explainer import AttributionResult 
"""
asyncio is single-threaded (doesn't spawn new threads)
each coroutine that runs uses the same main thread the event loop lives in
"""
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI
from app.runtime import ApplicationRuntime
from app.settings import RuntimeSettings
from app.services import build_services

app = FastAPI(title="Toxicty Classification API")

runtime_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="runtime")
"""
not FastAPI's generic default pool, which we will want free for our blocking operations

ApplicationRuntime already serializes with _begin_operations / _end_operations + Condition,,
1 worker for model operations is correct. Adding more workers wouldn't give parallelism.
So, predict/explain are synchronous, CPU/GIL + torch heavy, must not run on the event loop,
and must go to run_in_executor.
"""

runtime = ApplicationRuntime(
    settings=RuntimeSettings,
    services_builder=build_services
) # BUG 1


"""
model_info, status, is_ready are microsecond, lock-only operations, they should be fast
and not hang behind long explanation, prediction queues.

problem remains how to make sync runtime operations run in event loop
"""
@app.get("/model_info")
async def get_model_info():
    return await _call_runtime(runtime.model_info)

@app.get("/status")
async def get_runtime_status():
    return await _call_runtime(runtime.status)

@app.get("/is_ready")
async def get_runtime_readiness():
    return await _call_runtime(runtime.is_ready)

@app.post("/predict")
async def predict_sentence(text: str) -> ClassificationResult:
    # NOTE: Q1 -> should I build a new pydantic model that validates the input and output of the function ?
    # NOTE: Q2 -> now I am sure, I need a new I/O pydantic models that return JSON formats maybe
    # NOTE: Q3 -> should I add database writing now ?
    return await _call_runtime(runtime.predict(text))

@app.post("/predict_batch")
async def predict_batch(texts: list[str]) -> list[ClassificationResult]:
    # NOTE: -> I need I/O models + db writing
    return await _call_runtime(runtime.predict_batch(texts))

@app.post("/explain")
async def explain(text: str) -> AttributionResult:
    # NOTE: -> I need I/O models + db writing
    return await _call_runtime(runtime.explain(text))

@app.post("/explain_batch")
async def explain_batch(texts: list[str]) -> list[AttributionResult]:
    # NOTE: -> I need I/O models + db writing
    return await _call_runtime(runtime.explain_batch(texts))


async def _call_runtime(func, *args):
    """
    helper function to call runtime functions asynchronously

    asyncio.to_thread(...) is the same idea, but always uses the default executor

    loop = asyncio.get_running_loop() is the current running event loop
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(runtime_executor, func, *args)

"""
NOTE: I need another update endpoint, so when a mod updates a wrongly classified text,
its result gets updated in the API and used later for retraining.

NOTE: I need another endpoint to save every predicted sentence and its prediction in the database,
so when the admin opens its dashboard, he gets everything available to its eyesight
                                            |
                                            |
                                            \/
So we can modify every prediction endpoint to save the predictions on the way to our database

NOTE: I need to design the database table so that it has the input sentence, its prediction,
explanation, timestamp, model used, metadata, and the new mod corrected labels. And if we reach 1000
wrongly classified labels we retrain the model, and insert the new version automatically.

NOTE: I need a database reading endpoint that moderators will extract sentences from , we can
use filtering too here (based on labels, or other business logic filters)
"""
