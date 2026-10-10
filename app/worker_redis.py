"""Explicit Redis timeouts and bounded, shutdown-aware worker reconnects."""
import asyncio
import logging
from math import isfinite

from redis.asyncio import Redis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff
from redis.exceptions import ConnectionError, TimeoutError

logger = logging.getLogger("expl_worker")
BLOCK_TIMEOUT_SECONDS = 5


def create_worker_redis(url: str) -> Redis:
    client = Redis.from_url(
        url,
        decode_responses=True,
        socket_timeout=15,
        socket_connect_timeout=3,
        # This module owns retries; do not stack implicit client retries.
        retry=Retry(NoBackoff(), 0),
    )
    # URL query parameters override kwargs in redis-py. Validate the effective
    # values, so a URL cannot reintroduce the idle BLPOP/read-timeout race.
    options = client.connection_pool.connection_kwargs
    read_timeout = options.get("socket_timeout")
    connect_timeout = options.get("socket_connect_timeout")
    if read_timeout is None or not isfinite(read_timeout) or read_timeout <= BLOCK_TIMEOUT_SECONDS:
        raise ValueError("Worker Redis socket_timeout must be finite and greater than 5 seconds")
    if connect_timeout is None or not isfinite(connect_timeout) or connect_timeout <= 0:
        raise ValueError("Worker Redis socket_connect_timeout must be finite and positive")
    return client


class WorkerRedis:
    """Reconnect transient failures; exit visibly after five retries.

    Health transitions are logged as worker_health=ready/degraded/failed.
    This protects the connection loop, not delivery guarantees for destructive
    Redis pops. Job leases and abandoned-job recovery remain separate work.
    """

    def __init__(self, client: Redis, stop: asyncio.Event, *, max_retries: int = 5):
        self.client = client
        self.stop = stop
        self.max_retries = max_retries
        self.healthy = False

    async def call(self, command: str, *args, **kwargs):
        for attempt in range(self.max_retries + 1):
            if self.stop.is_set():
                return None
            try:
                result = await getattr(self.client, command)(*args, **kwargs)
            except (ConnectionError, TimeoutError) as exc:
                self.healthy = False
                if attempt == self.max_retries:
                    logger.error(
                        "worker_health=failed redis_command=%s retries=%d error_type=%s",
                        command, attempt, type(exc).__name__,
                        extra={"event": "worker.redis_failed", "fields": {
                            "worker_health": "failed", "redis_command": command,
                            "retries": attempt, "error_type": type(exc).__name__,
                        }},
                    )
                    raise
                delay = min(0.5 * 2 ** attempt, 8)
                logger.warning(
                    "worker_health=degraded redis_command=%s retry=%d delay_seconds=%s error_type=%s",
                    command, attempt + 1, delay, type(exc).__name__,
                    extra={"event": "worker.redis_retry", "fields": {
                        "worker_health": "degraded", "redis_command": command,
                        "attempt": attempt + 1, "delay_seconds": delay,
                        "error_type": type(exc).__name__,
                    }},
                )
                # Event.wait lets SIGTERM interrupt reconnect backoff.
                try:
                    await asyncio.wait_for(self.stop.wait(), timeout=delay)
                    return None
                except asyncio.TimeoutError:
                    continue
            else:
                if not self.healthy:
                    logger.info("worker_health=ready redis_connection=available",
                                extra={"event": "worker.redis_ready", "fields": {"worker_health": "ready"}})
                self.healthy = True
                return result
