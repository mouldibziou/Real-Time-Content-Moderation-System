"""Redis wakes the worker; PostgreSQL owns recoverable job state."""
import asyncio
from collections import defaultdict
import logging
import os
import signal
from uuid import UUID

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("expl_worker")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v2.0.0")
LEASE_SECONDS = int(os.getenv("WORKER_LEASE_SECONDS", "60"))
if LEASE_SECONDS < 6:
    raise ValueError("WORKER_LEASE_SECONDS must be at least 6")


async def _heartbeat(jobs, finished):
    from app.db.engine import async_session
    from app.db.jobs import renew_job

    while True:
        try:
            await asyncio.wait_for(finished.wait(), timeout=LEASE_SECONDS / 3)
            return
        except asyncio.TimeoutError:
            async with async_session() as session:
                for job in jobs:
                    if not await renew_job(session, job.id, job.lease_token, lease_seconds=LEASE_SECONDS):
                        logger.warning("job ownership lost job_id=%s", job.id)


async def _process_batch(job_ids, runtime):
    from app.db.engine import async_session
    from app.db.jobs import claim_job, fail_job, finish_job
    from app.db.repository import get_classification

    groups = defaultdict(list)
    for raw in dict.fromkeys(job_ids):
        try:
            jid = UUID(raw)
        except (ValueError, TypeError):
            logger.warning("dropping malformed job id %r", raw)
            continue
        async with async_session() as session:
            job = await claim_job(session, jid, lease_seconds=LEASE_SECONDS)
            if job is None:
                continue
            classification = await get_classification(session, classification_id=job.classification_id)
            if classification is None or job.model_version != MODEL_VERSION:
                await fail_job(session, jid, job.lease_token,
                               "Classification missing or model version unavailable", permanent=True)
                continue
            groups[job.label].append((job, classification.text))

    active = [job for items in groups.values() for job, _ in items]
    finished = asyncio.Event()
    heartbeat = asyncio.create_task(_heartbeat(active, finished))
    try:
        for label, items in groups.items():
            jobs = [job for job, _ in items]
            try:
                # Leave the event loop available for signals and lease renewal.
                if len(items) == 1:
                    results = [await asyncio.to_thread(runtime.explain, items[0][1], label)]
                else:
                    results = await asyncio.to_thread(runtime.explain_batch, [text for _, text in items], label)
                if len(results) != len(items):
                    raise RuntimeError("Explainer returned an incomplete batch")
                async with async_session() as session:
                    for (job, _), result in zip(items, results):
                        await finish_job(session, job.id, job.lease_token, result)
            except Exception as exc:
                logger.exception("explanation processing failed")
                async with async_session() as session:
                    for job in jobs:
                        await fail_job(session, job.id, job.lease_token, str(exc),
                                       permanent=isinstance(exc, (ValueError, TypeError)))
            finally:
                for job in jobs:
                    active.remove(job)
    finally:
        finished.set()
        await heartbeat


async def _process_one(job_id, runtime):
    await _process_batch([job_id], runtime)


async def _pending_jobs():
    from sqlalchemy import select
    from app.db.engine import async_session
    from app.db.jobs import recover_jobs
    from app.db.models import ExplanationJob

    async with async_session() as session:
        await recover_jobs(session)
        return list((await session.execute(
            select(ExplanationJob.id).where(ExplanationJob.status == "queued")
            .order_by(ExplanationJob.created_at).limit(4)
        )).scalars())


async def main():
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    from app.db.engine import engine
    from app.db.queue import QUEUE_KEY, REDIS_URL
    from app.worker_redis import BLOCK_TIMEOUT_SECONDS, WorkerRedis, create_worker_redis
    from app.runtime import ApplicationRuntime
    from app.services import build_services
    from app.settings import load_runtime_settings

    client = create_worker_redis(REDIS_URL)
    runtime = ApplicationRuntime(settings=load_runtime_settings(), services_builder=build_services)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    redis = WorkerRedis(client, stop)
    try:
        runtime.start()
        logger.info("worker model ready (version %s)", MODEL_VERSION)
        while not stop.is_set():
            # Recovers expired claims and missed Redis publications every cycle.
            pending = await _pending_jobs()
            if pending:
                await _process_batch([str(jid) for jid in pending], runtime)
                continue
            popped = await redis.call("blpop", QUEUE_KEY, timeout=BLOCK_TIMEOUT_SECONDS)
            if popped is None:
                continue
            batch = [popped[1]]
            for _ in range(3):
                raw = await redis.call("lpop", QUEUE_KEY)
                if raw is None:
                    break
                batch.append(raw)
            await _process_batch(batch, runtime)
    finally:
        await client.aclose()
        runtime.stop()
        await engine.dispose()
        logger.info("worker_health=stopped worker shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
