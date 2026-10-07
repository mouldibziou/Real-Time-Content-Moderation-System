import os
from uuid import UUID

from dotenv import load_dotenv
from redis.asyncio import Redis

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_KEY = os.getenv("EXPLAIN_QUEUE_KEY", "expl_jobs")

_client: Redis | None = None


def get_redis() -> Redis:
    global _client
    if _client is None:
        _client = Redis.from_url(REDIS_URL, decode_responses=True)
    return _client


async def enqueue_explain_job(job_id: UUID) -> None:
    # Dumb pipe only: Postgres row is truth, Redis just wakes the worker.
    await get_redis().rpush(QUEUE_KEY, str(job_id))
