"""Explanation worker: own process, own model, own thread.

API never runs IG on a miss; it inserts a queued Postgres row and pushes
the id to Redis. This loop pops ids, marks running, loads the exact
stored text, runs the explainer synchronously, then saves spans and marks
done in one transaction. Postgres is truth, Redis is a doorbell.
"""

import asyncio
import logging
import os
import signal
from uuid import UUID

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("expl_worker")

MODEL_VERSION = os.getenv("MODEL_VERSION", "v2.0.0")


async def _process_one(job_id: str, runtime) -> None:
    from app.db.engine import async_session
    from app.db.repository import find_explanation, get_classification, get_job, mark_job, save_explanation

    try:
        jid = UUID(job_id)
    except ValueError:
        logger.warning("dropping malformed job id %r", job_id)
        return

    async with async_session() as session:
        job = await get_job(session, job_id=jid)
        if job is None:
            logger.warning("job %s not found, skipping", jid)
            return
        if job.status == "done":
            return
        await mark_job(session, job_id=jid, status="running")

    # Load text outside the mark transaction; short separate session.
    async with async_session() as session:
        job = await get_job(session, job_id=jid)
        cls = await get_classification(session, classification_id=job.classification_id)
        if cls is None:
            await mark_job(session, job_id=jid, status="failed", error="classification not found")
            return
        text, label, version = cls.text, job.label, job.model_version

    # Synchronous IG on the worker's own thread; no event-loop blocking
    # concern here since this process serves no HTTP.
    try:
        result = runtime.explain(text, label)
    except Exception as exc:  # noqa: BLE001 - must mark failed, never crash loop
        logger.exception("explain failed for job %s", jid)
        async with async_session() as session:
            await mark_job(session, job_id=jid, status="failed", error=str(exc)[:500])
        return

    spans_plain = [
        {
            "text": s.text,
            "start": s.start,
            "end": s.end,
            "score": s.score,
            "normalized_importance": s.normalized_importance,
        }
        for s in result.spans
    ]
    async with async_session() as session:
        # Idempotent: concurrent duplicate inserts keep the first winner.
        from sqlalchemy.exc import IntegrityError

        from app.db.models import Explanation

        # Skip if another worker already finished this triple.
        existing = await find_explanation(
            session, classification_id=job.classification_id, label=label, model_version=version
        )
        if existing is None:
            session.add(
                Explanation(
                    classification_id=job.classification_id,
                    label=label,
                    spans=spans_plain,
                    convergence_delta=result.convergence_delta,
                    model_version=version,
                )
            )
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
        await mark_job(session, job_id=jid, status="done")


async def _requeue_orphans(redis) -> int:
    """On boot, re-push queued rows so crashes lose no work."""
    from sqlalchemy import select

    from app.db.engine import async_session
    from app.db.models import ExplanationJob
    from app.db.queue import QUEUE_KEY

    async with async_session() as session:
        rows = (
            await session.execute(select(ExplanationJob.id).where(ExplanationJob.status == "queued"))
        ).scalars().all()
    if not rows:
        return 0
    for jid in rows:
        await redis.call("rpush", QUEUE_KEY, str(jid))
    return len(rows)


async def _process_batch(job_ids: list[str], runtime) -> None:
    """Coalesced path: one pooled explain_batch per (label, version) group.

    Same tables and polling contract as singles; ~3-4x fewer IG passes
    on full-miss batches. Per-job failures mark only that job.
    """
    from app.db.engine import async_session
    from app.db.repository import find_explanation, get_classification, get_job, mark_job, save_explanation

    # Load all jobs + texts first; 404-equivalents mark failed, never crash.
    items: list[dict] = []
    async with async_session() as session:
        for raw in job_ids:
            try:
                jid = UUID(raw)
            except ValueError:
                logger.warning("dropping malformed job id %r", raw)
                continue
            job = await get_job(session, job_id=jid)
            if job is None or job.status == "done":
                continue
            await mark_job(session, job_id=jid, status="running")
            cls = await get_classification(session, classification_id=job.classification_id)
            if cls is None:
                await mark_job(session, job_id=jid, status="failed", error="classification not found")
                continue
            items.append({"job": job, "text": cls.text})

    # Group misses by (label, version); cache hits already done at enqueue.
    from collections import defaultdict

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for it in items:
        groups[(it["job"].label, it["job"].model_version)].append(it)

    for (label, version), group in groups.items():
        texts = [it["text"] for it in group]
        try:
            results = runtime.explain_batch(texts, label)
        except Exception as exc:  # noqa: BLE001 - per-group failure, others proceed
            logger.exception("explain_batch failed for label %s", label)
            async with async_session() as session:
                for it in group:
                    await mark_job(session, job_id=it["job"].id, status="failed", error=str(exc)[:500])
            continue
        async with async_session() as session:
            for it, result in zip(group, results):
                spans_plain = [
                    {
                        "text": s.text,
                        "start": s.start,
                        "end": s.end,
                        "score": s.score,
                        "normalized_importance": s.normalized_importance,
                    }
                    for s in result.spans
                ]
                existing = await find_explanation(
                    session,
                    classification_id=it["job"].classification_id,
                    label=label,
                    model_version=version,
                )
                if existing is None:
                    import uuid as _uuid

                    await save_explanation(
                        session,
                        id=_uuid.uuid4(),
                        classification_id=it["job"].classification_id,
                        label=label,
                        spans=spans_plain,
                        convergence_delta=result.convergence_delta,
                        model_version=version,
                    )
                await mark_job(session, job_id=it["job"].id, status="done")


async def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    from app.db.queue import QUEUE_KEY, REDIS_URL
    from app.worker_redis import BLOCK_TIMEOUT_SECONDS, WorkerRedis, create_worker_redis
    from app.runtime import ApplicationRuntime
    from app.services import build_services
    from app.settings import load_runtime_settings

    # Validate Redis configuration before loading the model.
    client = create_worker_redis(REDIS_URL)
    # Own model instance: loaded once, lives for the process lifetime.
    runtime = ApplicationRuntime(settings=load_runtime_settings(), services_builder=build_services)
    runtime.start()
    logger.info("worker model ready (version %s)", MODEL_VERSION)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass

    redis = WorkerRedis(client, stop)
    try:
        n = await _requeue_orphans(redis)
        if n:
            logger.info("re-queued %d orphan jobs from Postgres", n)
        logger.info("worker listening on redis queue %r", QUEUE_KEY)
        while not stop.is_set():
            popped = await redis.call("blpop", QUEUE_KEY, timeout=BLOCK_TIMEOUT_SECONDS)
            if popped is None:
                continue
            _, first_id = popped
            # Drain up to 4 total without blocking: one IG batch per group.
            batch_ids = [first_id]
            for _ in range(3):
                nxt = await redis.call("lpop", QUEUE_KEY)
                if nxt is None:
                    break
                batch_ids.append(nxt)
            try:
                if len(batch_ids) == 1:
                    await _process_one(batch_ids[0], runtime)
                else:
                    await _process_batch(batch_ids, runtime)
            except Exception:  # noqa: BLE001 - loop must never die on one batch
                logger.exception("worker loop error on batch of %d", len(batch_ids))
    finally:
        try:
            await client.aclose()
        finally:
            runtime.stop()
            logger.info("worker_health=stopped worker shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
