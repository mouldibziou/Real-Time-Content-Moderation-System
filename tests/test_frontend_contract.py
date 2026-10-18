"""Public frontend contracts: browser access and committed, addressable results."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
from unittest import IsolatedAsyncioTestCase, TestCase, skipUnless
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID
from sqlalchemy import func, select, text
from sqlalchemy.exc import SQLAlchemyError
from tests.test_application_logging import ASGIClient
from tests.factories import make_model_info
from app.settings import RuntimeSettings, load_runtime_settings

class BrowserContractTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from app.api.main import create_app
        self.app = create_app(RuntimeSettings(cors_origins=('https://flagged-example.pages.dev',)))
        self.client = ASGIClient(self.app)
        self.headers = {'Origin': 'https://flagged-example.pages.dev'}

    async def test_exact_origin_preflight_and_no_credentials(self):
        response = await self.client.request('OPTIONS', '/v1/classify', headers={**self.headers,
            'Access-Control-Request-Method':'POST', 'Access-Control-Request-Headers':'content-type'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['access-control-allow-origin'], self.headers['Origin'])
        self.assertNotIn('access-control-allow-credentials', response.headers)
        denied = await self.client.request('OPTIONS', '/v1/classify', headers={
            'Origin':'https://unrelated.pages.dev','Access-Control-Request-Method':'POST'})
        self.assertEqual(denied.status_code, 400)
        self.assertNotIn('access-control-allow-origin', denied.headers)

    async def test_model_metadata_includes_release_and_input_limits(self):
        self.app.state.runtime = Mock(model_info=Mock(return_value=make_model_info()))
        response = await self.client.request('GET', '/v1/model_info', headers=self.headers)
        body = json.loads(response.body)
        self.assertEqual(body['model_version'], 'v2.0.0')
        self.assertEqual(body['max_text_chars'], 10000)
        self.assertEqual(body['max_prediction_batch_size'], 64)
        self.assertEqual(response.headers['access-control-allow-origin'], self.headers['Origin'])

    async def test_unexpected_errors_have_cors_without_private_details(self):
        self.app.state.runtime = Mock(model_info=Mock(side_effect=RuntimeError("private failure")))
        response = await self.client.request('GET', '/v1/model_info', headers=self.headers)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.headers['access-control-allow-origin'], self.headers['Origin'])
        self.assertNotIn(b'private failure', response.body)

    async def test_translated_database_errors_have_cors(self):
        from app.db.engine import get_db
        async def unavailable():
            raise SQLAlchemyError('private details')
            yield
        self.app.dependency_overrides[get_db] = unavailable
        response = await self.client.request('GET','/v1/classifications/00000000-0000-0000-0000-000000000001',headers=self.headers)
        self.assertEqual(response.status_code,503)
        self.assertEqual(response.headers['access-control-allow-origin'],self.headers['Origin'])
        self.assertNotIn(b'private details',response.body)

class CorsSettingsTests(TestCase):
    def test_origins_are_explicit_and_disabled_by_default(self):
        self.assertEqual(load_runtime_settings({}).cors_origins, ())
        self.assertEqual(load_runtime_settings({'TOXICITY_CORS_ORIGINS':'http://localhost:5173, https://example.pages.dev'}).cors_origins,('http://localhost:5173','https://example.pages.dev'))
        for invalid in ('*','https://*.pages.dev','https://example.com/path','https://user:password@example.com'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                RuntimeSettings(cors_origins=(invalid,))

@skipUnless(os.getenv('TOXICITY_DB_TESTS') == '1', 'requires disposable PostgreSQL')
class CommittedClassificationTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from app.api.main import create_app
        from app.classifier import ClassificationResult, LabelPrediction
        from app.db.engine import engine
        from sqlalchemy.engine import make_url
        if not make_url(os.environ['DATABASE_URL']).database.endswith('_tests'):
            raise RuntimeError('Only a disposable _tests database is allowed')
        self.engine=engine
        async with engine.begin() as conn:
            await conn.execute(text('TRUNCATE classifications CASCADE'))
        self.app=create_app(RuntimeSettings())
        self.executor=ThreadPoolExecutor(max_workers=1)
        self.app.state.runtime=Mock()
        prediction=ClassificationResult({label:LabelPrediction(.9,True) for label in RuntimeSettings().labels})
        self.app.state.runtime.predict.return_value=prediction
        self.app.state.runtime.predict_batch.side_effect=lambda texts:[prediction for _ in texts]
        self.app.state.runtime_executor=self.executor
        self.app.state.model_snapshot={'model_type':'distilbert','device':'cpu','thresholds':list(RuntimeSettings().thresholds)}
        self.client=ASGIClient(self.app)

    async def asyncTearDown(self):
        self.executor.shutdown(wait=True)
        await self.engine.dispose()

    async def classify(self, content):
        response=await self.client.post('/v1/classify',json={'text':content})
        self.assertEqual(response.status_code,200)
        return json.loads(response.body)

    async def test_duplicate_concurrent_results_resolve_immediately(self):
        results=await asyncio.gather(*[self.classify('Repeated preset') for _ in range(8)])
        ids={result['classification_id'] for result in results}
        self.assertEqual(len(ids),1)
        response=await self.client.get('/v1/classifications/'+results[0]['classification_id'])
        self.assertEqual(response.status_code,200)
        self.assertEqual(json.loads(response.body)['text'],'Repeated preset')

    async def test_overlapping_batches_keep_existing_ids_and_input_order(self):
        existing=await self.classify('Existing')
        async def batch(texts):
            response=await self.client.post('/v1/classify_batch',json={'texts':texts})
            self.assertEqual(response.status_code,200)
            return json.loads(response.body)['results']
        first,second=await asyncio.gather(batch(['New','Existing','Last','New']),batch(['Last','New','Existing']))
        self.assertEqual(first[0]['classification_id'],first[3]['classification_id'])
        self.assertEqual(first[1]['classification_id'],existing['classification_id'])
        self.assertEqual(second[0]['classification_id'],first[2]['classification_id'])
        for item in first+second:
            self.assertEqual((await self.client.get('/v1/classifications/'+item['classification_id'])).status_code,200)

    async def test_explanation_job_and_correction_work_immediately_after_response(self):
        result=await self.classify('A moderation example')
        with patch('app.db.queue.enqueue_explain_job',AsyncMock()):
            response=await self.client.post('/v1/explain',json={'classification_id':result['classification_id'],'label':'toxic'})
        self.assertEqual(response.status_code,202)
        job=json.loads(response.body)
        self.assertEqual((await self.client.get('/v1/jobs/'+job['job_id'])).status_code,200)
        labels={label:False for label in RuntimeSettings().labels}
        response=await self.client.post('/v1/classifications/'+result['classification_id']+'/corrections',json={'corrected_labels':labels,'moderator_id':'public-demo'})
        self.assertEqual(response.status_code,201)
        detail=json.loads((await self.client.get('/v1/classifications/'+result['classification_id'])).body)
        self.assertTrue(detail['corrected'])
        self.assertEqual(detail['corrected_labels'],labels)
        self.assertTrue(detail['labels']['toxic']['predicted'])

    async def test_commit_failure_returns_503_and_rolls_back(self):
        from sqlalchemy.ext.asyncio import AsyncSession
        from app.db.models import Classification
        from app.db.engine import async_session
        with patch.object(AsyncSession,'commit',AsyncMock(side_effect=SQLAlchemyError('injected failure'))):
            response=await self.client.post('/v1/classify',json={'text':'Should roll back'})
        self.assertEqual(response.status_code,503)
        async with async_session() as session:
            self.assertEqual(await session.scalar(select(func.count()).select_from(Classification)),0)
        self.assertEqual(self.app.state.metrics.registry.get_sample_value('classification_items_total'),0)

    async def test_batch_failure_leaves_no_partial_records(self):
        from app.db import repository
        from app.db.engine import async_session
        from app.db.models import Classification
        original=repository._insert_classification
        calls=0
        async def fail_second(session,item):
            nonlocal calls
            calls+=1
            if calls==2: raise SQLAlchemyError('injected batch failure')
            return await original(session,item)
        with patch('app.db.repository._insert_classification',side_effect=fail_second):
            response=await self.client.post('/v1/classify_batch',json={'texts':['first','second']})
        self.assertEqual(response.status_code,503)
        async with async_session() as session:
            self.assertEqual(await session.scalar(select(func.count()).select_from(Classification)),0)
