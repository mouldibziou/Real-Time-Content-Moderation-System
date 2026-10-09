"""Worker Redis failure handling without models, a database, or a server."""
import asyncio
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, patch

from redis.exceptions import ConnectionError, ResponseError, TimeoutError

from app.worker_redis import WorkerRedis, create_worker_redis


class WorkerRedisConfigurationTests(TestCase):
    def test_default_timeout_exceeds_blocking_pop_and_client_retries_are_disabled(self):
        client = create_worker_redis("redis://localhost:6379/0")
        options = client.connection_pool.connection_kwargs
        self.assertEqual(options["socket_timeout"], 15)
        self.assertEqual(options["socket_connect_timeout"], 3)
        self.assertEqual(options["retry"]._retries, 0)

    def test_url_timeout_overrides_are_validated(self):
        for query in ("socket_timeout=5", "socket_timeout=0", "socket_timeout=nan",
                      "socket_connect_timeout=0", "socket_connect_timeout=inf"):
            with self.subTest(query=query), self.assertRaises(ValueError):
                create_worker_redis("redis://localhost:6379/0?" + query)

    def test_valid_url_timeout_override_is_preserved(self):
        client = create_worker_redis("redis://localhost:6379/0?socket_timeout=20")
        self.assertEqual(client.connection_pool.connection_kwargs["socket_timeout"], 20)


class WorkerRedisReconnectTests(IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = AsyncMock()
        self.stop = asyncio.Event()
        self.queue = WorkerRedis(self.client, self.stop)
        self.delays = []
        self.real_wait_for = asyncio.wait_for

        async def skip_delay(waiter, *, timeout):
            self.delays.append(timeout)
            waiter.close()
            raise asyncio.TimeoutError

        self.enterContext(patch("app.worker_redis.asyncio.wait_for", skip_delay))

    async def test_idle_pop_is_healthy_without_retries(self):
        self.client.blpop.return_value = None
        self.assertIsNone(await self.queue.call("blpop", "jobs", timeout=5))
        self.assertTrue(self.queue.healthy)
        self.client.blpop.assert_awaited_once_with("jobs", timeout=5)
        self.assertEqual(self.delays, [])

    async def test_connection_errors_and_timeouts_recover_and_report_health(self):
        self.client.blpop.side_effect = [ConnectionError(), TimeoutError(), ("jobs", "id")]
        with self.assertLogs("expl_worker", level="INFO") as logs:
            result = await self.queue.call("blpop", "jobs", timeout=5)
        self.assertEqual(result, ("jobs", "id"))
        self.assertTrue(self.queue.healthy)
        self.assertEqual(self.delays, [0.5, 1])
        self.assertTrue(any("worker_health=degraded" in line for line in logs.output))
        self.assertTrue(any("worker_health=ready" in line for line in logs.output))

    async def test_persistent_failure_exhausts_five_retries_and_reports_failed(self):
        self.client.lpop.side_effect = ConnectionError()
        with self.assertLogs("expl_worker") as logs, self.assertRaises(ConnectionError):
            await self.queue.call("lpop", "jobs")
        self.assertEqual(self.client.lpop.await_count, 6)
        self.assertEqual(self.delays, [0.5, 1, 2, 4, 8])
        self.assertFalse(self.queue.healthy)
        self.assertIn("worker_health=failed", logs.output[-1])

    async def test_startup_push_uses_the_same_recovery_policy(self):
        self.client.rpush.side_effect = [ConnectionError(), 1]
        with self.assertLogs("expl_worker"):
            self.assertEqual(await self.queue.call("rpush", "jobs", "id"), 1)
        self.assertEqual(self.client.rpush.await_count, 2)

    async def test_shutdown_prevents_another_command(self):
        self.stop.set()
        self.assertIsNone(await self.queue.call("blpop", "jobs", timeout=5))
        self.client.blpop.assert_not_awaited()

    async def test_shutdown_interrupts_real_backoff(self):
        async def fail_and_stop(*args, **kwargs):
            self.stop.set()
            raise ConnectionError()

        self.client.blpop.side_effect = fail_and_stop
        # Restore actual wait_for to verify Event.wait wakes the backoff.
        with patch("app.worker_redis.asyncio.wait_for", self.real_wait_for), self.assertLogs("expl_worker"):
            self.assertIsNone(await asyncio.wait_for(self.queue.call("blpop", "jobs"), 0.1))
        self.client.blpop.assert_awaited_once()

    async def test_non_transient_errors_are_not_retried(self):
        self.client.blpop.side_effect = ResponseError("invalid command")
        with self.assertRaises(ResponseError):
            await self.queue.call("blpop", "jobs")
        self.client.blpop.assert_awaited_once()
        self.assertEqual(self.delays, [])

    async def test_cancellation_is_not_swallowed(self):
        self.client.blpop.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await self.queue.call("blpop", "jobs")
        self.client.blpop.assert_awaited_once()
