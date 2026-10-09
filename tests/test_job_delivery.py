import asyncio
from datetime import timedelta
from hashlib import sha256
import os
from unittest import IsolatedAsyncioTestCase, skipUnless
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from sqlalchemy import func, select, text, update
from sqlalchemy.engine import make_url


@skipUnless(os.getenv("TOXICITY_DB_TESTS") == "1", "requires disposable PostgreSQL")
class JobDeliveryTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from app.db.engine import engine
        if not make_url(os.environ['DATABASE_URL']).database.endswith('_tests'):
            raise RuntimeError('Integration tests require a database ending in _tests')
        self.engine = engine
        async with engine.begin() as conn:
            await conn.execute(text('TRUNCATE classifications CASCADE'))

    async def asyncTearDown(self):
        await self.engine.dispose()

    async def seed(self):
        from app.db.engine import async_session
        from app.db.repository import create_job, save_classification
        cid = uuid4(); content = 'Test '+str(uuid4())
        async with async_session() as s:
            await save_classification(s, id=cid, text=content, text_hash=sha256(content.encode()).hexdigest(),
                                      model_version='v2.0.0', model_type='distilbert', device='cpu', thresholds=[], labels={})
            job, _ = await create_job(s, classification_id=cid, label='toxic', model_version='v2.0.0')
            return job.id, cid

    async def claim(self, jid):
        from app.db.engine import async_session
        from app.db.jobs import claim_job
        async with async_session() as s:
            return await claim_job(s, jid)

    async def expire(self, jid):
        from app.db.models import ExplanationJob
        async with self.engine.begin() as conn:
            await conn.execute(update(ExplanationJob).where(ExplanationJob.id == jid).values(
                lease_expires_at=func.now()-timedelta(seconds=1)))

    async def test_concurrent_delivery_has_one_owner(self):
        jid, _ = await self.seed()
        claims = await asyncio.gather(self.claim(jid), self.claim(jid))
        self.assertEqual(sum(c is not None for c in claims), 1)

    async def test_expired_claim_is_recovered_and_old_owner_cannot_complete(self):
        from app.db.engine import async_session
        from app.db.jobs import finish_job
        from app.explainer import AttributionResult
        from app.worker import _pending_jobs
        jid, _ = await self.seed(); old = await self.claim(jid)
        await self.expire(jid)
        self.assertIn(jid, await _pending_jobs())
        new = await self.claim(jid)
        self.assertNotEqual(old.lease_token, new.lease_token)
        async with async_session() as s:
            self.assertFalse(await finish_job(s, jid, old.lease_token, AttributionResult('toxic', ())))

    async def test_heartbeat_preserves_active_claim(self):
        from app.db.engine import async_session
        from app.db.jobs import renew_job, recover_jobs
        from app.db.repository import get_job
        jid, _ = await self.seed(); job = await self.claim(jid)
        await self.expire(jid)
        async with async_session() as s:
            self.assertTrue(await renew_job(s, jid, job.lease_token))
            await recover_jobs(s)
            self.assertEqual((await get_job(s, job_id=jid)).status, 'running')

    async def test_result_and_done_commit_together(self):
        from app.db.engine import async_session
        from app.db.jobs import finish_job
        from app.db.repository import get_job, find_explanation
        from app.explainer import AttributionResult, AttributedSpan
        jid, cid = await self.seed(); job = await self.claim(jid)
        result = AttributionResult('toxic', (AttributedSpan('Test',0,4,1.,1.),))
        async with async_session() as s:
            with patch.object(s, 'commit', new=AsyncMock(side_effect=OSError('injected commit failure'))):
                with self.assertRaises(OSError): await finish_job(s, jid, job.lease_token, result)
        async with async_session() as s:
            self.assertIsNone(await find_explanation(s, classification_id=cid,label='toxic',model_version='v2.0.0'))
            self.assertEqual((await get_job(s, job_id=jid)).status, 'running')
            self.assertTrue(await finish_job(s, jid, job.lease_token, result))
        async with async_session() as s:
            self.assertEqual((await get_job(s, job_id=jid)).status, 'done')
            self.assertIsNotNone(await find_explanation(s, classification_id=cid,label='toxic',model_version='v2.0.0'))

    async def test_retries_end_in_terminal_failure(self):
        from app.db.engine import async_session
        from app.db.jobs import fail_job
        from app.db.repository import get_job
        jid, _ = await self.seed()
        for _ in range(3):
            job = await self.claim(jid)
            async with async_session() as s: await fail_job(s,jid,job.lease_token,'temporary failure')
        self.assertIsNone(await self.claim(jid))
        async with async_session() as s: self.assertEqual((await get_job(s,job_id=jid)).status,'failed')

    async def test_expired_final_attempt_becomes_failed(self):
        from app.worker import _pending_jobs
        from app.db.engine import async_session
        from app.db.jobs import fail_job
        from app.db.repository import get_job
        jid, _ = await self.seed()
        for _ in range(2):
            job = await self.claim(jid)
            async with async_session() as s: await fail_job(s,jid,job.lease_token,'retry')
        await self.claim(jid); await self.expire(jid)
        self.assertNotIn(jid, await _pending_jobs())
        async with async_session() as s: self.assertEqual((await get_job(s,job_id=jid)).status,'failed')

    async def test_missing_redis_notification_is_discovered(self):
        from app.worker import _pending_jobs
        jid, _ = await self.seed()
        self.assertIn(jid, await _pending_jobs())

    async def test_duplicate_batch_ids_compute_once(self):
        from app.worker import _process_batch
        from app.explainer import AttributionResult
        from unittest.mock import Mock
        jid, _ = await self.seed()
        runtime = Mock(); runtime.explain.return_value = AttributionResult('toxic', ())
        await _process_batch([str(jid),str(jid)],runtime)
        runtime.explain.assert_called_once()
        runtime.explain_batch.assert_not_called()

    async def test_slow_compute_renews_lease_without_blocking_event_loop(self):
        import time
        from app.worker import _process_one, _pending_jobs
        from app.db.engine import async_session
        from app.db.repository import get_job
        from app.explainer import AttributionResult
        from unittest.mock import Mock
        jid, _ = await self.seed()
        def compute(*args):
            time.sleep(7)
            return AttributionResult('toxic', ())
        runtime = Mock(); runtime.explain.side_effect = compute
        with patch('app.worker.LEASE_SECONDS', 6):
            task = asyncio.create_task(_process_one(str(jid), runtime))
            try:
                await asyncio.sleep(6.4)
                self.assertNotIn(jid, await _pending_jobs())
                async with async_session() as s:
                    self.assertEqual((await get_job(s, job_id=jid)).status, 'running')
            finally:
                await task

    async def test_repeated_request_publishes_only_new_job(self):
        from app.api.routes import explain_text
        from app.api.schemas import ExplainRequest
        from app.db.engine import async_session
        from app.db.models import ExplanationJob
        from sqlalchemy import delete
        jid, cid = await self.seed()
        async with self.engine.begin() as conn:
            await conn.execute(delete(ExplanationJob).where(ExplanationJob.id == jid))
        with patch('app.db.queue.enqueue_explain_job', new=AsyncMock()) as publish:
            async with async_session() as s:
                payload = ExplainRequest(classification_id=cid, label='toxic')
                first = await explain_text(s, payload)
                second = await explain_text(s, payload)
                import json
                self.assertEqual(json.loads(first.body)['job_id'], json.loads(second.body)['job_id'])
                publish.assert_awaited_once()
