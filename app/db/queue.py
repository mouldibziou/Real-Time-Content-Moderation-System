import os
import asyncio
from uuid import UUID

from dotenv import load_dotenv
from redis.asyncio import Redis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_KEY = os.getenv("EXPLAIN_QUEUE_KEY", "expl_jobs")
PUBLISH_TIMEOUT_SECONDS = 3

_client: Redis | None = None


def get_redis() -> Redis:
    global _client
    if _client is None:
        _client = Redis.from_url(REDIS_URL, decode_responses=True,
                                 socket_connect_timeout=2, socket_timeout=2,
                                 retry=Retry(NoBackoff(), 0))
    return _client


async def enqueue_explain_job(job_id: UUID) -> None:
    # Dumb pipe only: Postgres row is truth, Redis just wakes the worker.
    await asyncio.wait_for(get_redis().rpush(QUEUE_KEY, str(job_id)), PUBLISH_TIMEOUT_SECONDS)


async def close_redis() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
