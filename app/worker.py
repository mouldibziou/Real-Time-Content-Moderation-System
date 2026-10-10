"""Redis wakes the worker; PostgreSQL owns recoverable job state."""
import asyncio
from collections import defaultdict
import logging
import os
import signal
import sys
from time import monotonic
from uuid import UUID

from dotenv import load_dotenv
from app.settings import load_model_version
from app.observability import configure_logging, log_event
from app.metrics import WORKER_METRICS
from prometheus_client import start_http_server

load_dotenv()
logger = logging.getLogger("expl_worker")
MODEL_VERSION = load_model_version()
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
                        log_event(logger, "explanation.ownership_lost", level=logging.WARNING,
                                  job_id=job.id, attempt=job.attempts)


async def _process_batch(job_ids, runtime):
    attempt_starts = {}
    try:
        await _process_batch_with_timing(job_ids, runtime, attempt_starts)
    finally:
        # Include abandoned/cancelled attempts, not just successful ones.
        for started in attempt_starts.values():
            WORKER_METRICS.attempt_duration.observe(monotonic() - started)


async def _process_batch_with_timing(job_ids, runtime, attempt_starts):
    from app.db.engine import async_session
    from app.db.jobs import claim_job, fail_job, finish_job
    from app.db.repository import get_classification

    groups = defaultdict(list)
    for raw in dict.fromkeys(job_ids):
        try:
            jid = UUID(raw)
        except (ValueError, TypeError):
            log_event(logger, "explanation.invalid_notification", level=logging.WARNING)
            continue
        async with async_session() as session:
            job = await claim_job(session, jid, lease_seconds=LEASE_SECONDS)
            if job is None:
                continue
            attempt_starts[job.id] = monotonic()
            log_event(logger, "explanation.job_claimed", job_id=job.id,
                      classification_id=job.classification_id, label=job.label,
                      model_version=job.model_version, attempt=job.attempts)
            classification = await get_classification(session, classification_id=job.classification_id)
            if classification is None or job.model_version != MODEL_VERSION:
                status = await fail_job(session, jid, job.lease_token,
                                        "Classification missing or model version unavailable", permanent=True)
                WORKER_METRICS.attempt_duration.observe(monotonic() - attempt_starts.pop(job.id))
                log_event(logger, "explanation.job_failed" if status else "explanation.ownership_lost",
                          level=logging.ERROR if status else logging.WARNING,
                          job_id=job.id, classification_id=job.classification_id,
                          attempt=job.attempts, reason="classification_or_model_unavailable")
                continue
            groups[job.label].append((job, classification.text))

    active = [job for items in groups.values() for job, _ in items]
    finished = asyncio.Event()
    heartbeat = asyncio.create_task(_heartbeat(active, finished))
    try:
        for label, items in groups.items():
            jobs = [job for job, _ in items]
            started = monotonic()
            stage = "explanation.runtime"
            finalized = set()
            try:
                # Leave the event loop available for signals and lease renewal.
                if len(items) == 1:
                    results = [await asyncio.to_thread(runtime.explain, items[0][1], label)]
                else:
                    results = await asyncio.to_thread(runtime.explain_batch, [text for _, text in items], label)
                if len(results) != len(items):
                    raise RuntimeError("Explainer returned an incomplete batch")
                stage = "explanation.persistence"
                async with async_session() as session:
                    for (job, _), result in zip(items, results):
                        committed = await finish_job(session, job.id, job.lease_token, result)
                        WORKER_METRICS.attempt_duration.observe(monotonic() - attempt_starts.pop(job.id))
                        finalized.add(job.id)
                        log_event(logger, "explanation.job_completed" if committed else "explanation.ownership_lost",
                                  level=logging.INFO if committed else logging.WARNING,
                                  job_id=job.id, classification_id=job.classification_id,
                                  model_version=job.model_version, label=label, attempt=job.attempts,
                                  batch_size=len(items), duration_ms=round((monotonic() - started) * 1000, 2))
            except Exception as exc:
                jobs = [job for job in jobs if job.id not in finalized]
                # One diagnostic entry per affected job; no exception values.
                for job in jobs:
                    log_event(logger, "explanation.attempt_failed", level=logging.ERROR, exc_info=True,
                              job_id=job.id, classification_id=job.classification_id,
                              attempt=job.attempts, stage=stage, model_version=job.model_version,
                              duration_ms=round((monotonic() - started) * 1000, 2))
                async with async_session() as session:
                    for job in jobs:
                        status = await fail_job(session, job.id, job.lease_token, str(exc),
                                                permanent=isinstance(exc, (ValueError, TypeError)))
                        WORKER_METRICS.attempt_duration.observe(monotonic() - attempt_starts.pop(job.id))
                        event = {"queued": "explanation.retry_scheduled", "failed": "explanation.job_failed"}.get(
                            status, "explanation.ownership_lost")
                        log_event(logger, event, level=logging.ERROR if status == "failed" else logging.WARNING,
                                  job_id=job.id, attempt=job.attempts, job_status=status)
            finally:
                for job, _ in items:
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
    configure_logging("worker")
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
    metrics_server = None
    try:
        metrics_port = int(os.getenv("WORKER_METRICS_PORT", "9101"))
        if not 1 <= metrics_port <= 65535:
            raise ValueError("WORKER_METRICS_PORT must be between 1 and 65535")
        metrics_server, metrics_thread = start_http_server(
            metrics_port, addr=os.getenv("WORKER_METRICS_HOST", "0.0.0.0"), registry=WORKER_METRICS.registry
        )
        runtime.start()
        log_event(logger, "worker.ready", model_version=MODEL_VERSION)
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
        if metrics_server is not None:
            await asyncio.to_thread(metrics_server.shutdown)
            metrics_server.server_close()
            await asyncio.to_thread(metrics_thread.join)
        await client.aclose()
        runtime.stop()
        await engine.dispose()
        log_event(logger, "worker.stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        log_event(logger, "worker.failed", level=logging.ERROR, exc_info=True)
        sys.exit(1)
