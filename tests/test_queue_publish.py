import asyncio
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch
from uuid import uuid4
from app.db import queue


class QueuePublishTests(IsolatedAsyncioTestCase):
    async def test_unresponsive_redis_has_a_bounded_publish_deadline(self):
        cancelled = asyncio.Event()
        async def hang(*args):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        client = AsyncMock()
        client.rpush.side_effect = hang
        with patch.object(queue, "get_redis", return_value=client), \
             patch.object(queue, "PUBLISH_TIMEOUT_SECONDS", 0.02):
            with self.assertRaises(TimeoutError):
                await queue.enqueue_explain_job(uuid4())
        self.assertTrue(cancelled.is_set())

    async def test_shutdown_closes_and_clears_the_client(self):
        client = AsyncMock()
        with patch.object(queue, "_client", client):
            await queue.close_redis()
            client.aclose.assert_awaited_once()
            self.assertIsNone(queue._client)
