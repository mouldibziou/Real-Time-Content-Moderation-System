"""Prometheus exposition, aggregation, and committed job lifecycle regressions."""
import asyncio
from datetime import datetime, timezone
import os
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase, skipUnless
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from prometheus_client import generate_latest
from prometheus_client.parser import text_string_to_metric_families
from sqlalchemy.exc import SQLAlchemyError

from app.metrics import ApiMetrics, WorkerMetrics, current_api_metrics, render_api_metrics, read_queue_counts
from tests import test_application_logging as logging_tests
from tests import test_job_delivery as job_tests


class MetricExpositionTests(TestCase):
    def test_initial_values_and_buckets_are_valid_prometheus_text(self):
        metrics = ApiMetrics()
        output = render_api_metrics(metrics, {"queued": 3, "running": 1}).decode()
        families = {family.name: family for family in text_string_to_metric_families(output)}
        self.assertEqual(families["classification_items"].type, "counter")
        self.assertEqual(len([s for s in families["classification_predictions"].samples if s.name.endswith("_total")]), 12)
        self.assertIn('explanation_jobs_current{status="queued"} 3.0', output)
        self.assertIn("explanation_queue_collection_success 1.0", output)
        worker_output = generate_latest(WorkerMetrics().registry).decode()
        for name in ("explanation_job_attempt_duration_seconds", "explanation_job_queue_wait_seconds",
                     "explanation_job_end_to_end_duration_seconds"):
            self.assertIn(name + '_bucket{le="+Inf"} 0.0', worker_output)

    def test_failed_queue_collection_omits_queue_counts_instead_of_reporting_zero(self):
        output = render_api_metrics(ApiMetrics(), None).decode()
        self.assertIn("explanation_queue_collection_success 0.0", output)
        self.assertNotIn("explanation_jobs_current", output)
        self.assertIn("classification_items_total 0.0", output)


class ApiMetricsTests(IsolatedAsyncioTestCase):
    asyncSetUp = logging_tests.ApiWorkflowLoggingTests.asyncSetUp
    asyncTearDown = logging_tests.ApiWorkflowLoggingTests.asyncTearDown

    def sample(self, name, labels=None):
        return self.app.state.metrics.registry.get_sample_value(name, labels or {})

    async def test_endpoint_is_text_and_scrapes_do_not_clear_or_count_themselves(self):
        with patch("app.api.main.read_queue_counts", AsyncMock(return_value={"queued": 2, "running": 0})):
            first = await self.client.get("/metrics/")
            second = await self.client.get("/metrics/")
        self.assertEqual(first.status_code, 200)
        self.assertTrue(first.headers["content-type"].startswith("text/plain"))
        self.assertIn(b"explanation_jobs_current", first.body)
        self.assertNotIn(b"http_requests_total{", second.body)
        self.assertIsNone(current_api_metrics())

    async def test_single_and_batch_count_items_decisions_and_one_request_each(self):
        from app.classifier import ClassificationResult, LabelPrediction
        self.runtime.predict_batch.return_value = [self.runtime.predict.return_value,
                                                   ClassificationResult({"toxic": LabelPrediction(.1, False)})]
        with patch("app.api.routes.save_classification", AsyncMock(return_value=uuid4())), \
                patch("app.api.routes.save_classifications_batch", AsyncMock(return_value=[uuid4(), uuid4()])):
            self.assertEqual((await self.client.post("/v1/classify", json={"text": "PRIVATE ONE"})).status_code, 200)
            self.assertEqual((await self.client.post("/v1/classify_batch", json={"texts": ["PRIVATE TWO", "PRIVATE THREE"]})).status_code, 200)
        self.assertEqual(self.sample("classification_items_total"), 3)
        self.assertEqual(self.sample("classification_predictions_total", {"label": "toxic", "decision": "positive"}), 2)
        self.assertEqual(self.sample("classification_predictions_total", {"label": "toxic", "decision": "negative"}), 1)
        for route in ("/v1/classify", "/v1/classify_batch"):
            self.assertEqual(self.sample("http_requests_total", {"method": "POST", "route": route, "status_code": "200"}), 1)
            self.assertEqual(self.sample("http_request_duration_seconds_count", {"method": "POST", "route": route}), 1)
        self.assertNotIn(b"PRIVATE", generate_latest(self.app.state.metrics.registry))

    async def test_failed_persistence_and_unhandled_error_do_not_count_predictions(self):
        with patch("app.api.routes.save_classification", AsyncMock(side_effect=SQLAlchemyError("private parameters"))):
            self.assertEqual((await self.client.post("/v1/classify", json={"text": "hello"})).status_code, 503)
        self.runtime.predict.side_effect = RuntimeError("private input")
        self.assertEqual((await self.client.post("/v1/classify", json={"text": "hello"})).status_code, 500)
        self.assertEqual(self.sample("classification_items_total"), 0)
        for status in ("503", "500"):
            self.assertEqual(self.sample("http_requests_total", {"method": "POST", "route": "/v1/classify", "status_code": status}), 1)

    async def test_route_templates_and_unknown_methods_have_bounded_labels(self):
        with patch("app.db.repository.get_job", AsyncMock(return_value=None)):
            for _ in range(2):
                await self.client.get("/v1/jobs/" + str(uuid4()))
        for _ in range(2):
            await self.client.request("ARBITRARY_" + str(uuid4()), "/unknown/" + str(uuid4()))
        self.assertEqual(self.sample("http_requests_total", {"method": "GET", "route": "/v1/jobs/{job_id}", "status_code": "404"}), 2)
        self.assertEqual(self.sample("http_requests_total", {"method": "OTHER", "route": "unmatched", "status_code": "404"}), 2)
        await self.client.get("/v1/health/live")
        self.assertNotIn(b'/v1/health/live', generate_latest(self.app.state.metrics.registry))

    async def test_concurrent_requests_and_separate_app_instances_do_not_mix_counts(self):
        with patch("app.api.routes.save_classification", AsyncMock(return_value=uuid4())):
            responses = await asyncio.gather(*[self.client.post("/v1/classify", json={"text": "hello"}) for _ in range(8)])
        self.assertTrue(all(r.status_code == 200 for r in responses))
        self.assertEqual(self.sample("classification_items_total"), 8)
        self.assertEqual(ApiMetrics().registry.get_sample_value("classification_items_total"), 0)

    async def test_reused_job_is_not_counted_as_new_and_redis_failure_keeps_count(self):
        from app.api.routes import MODEL_VERSION
        cid = uuid4()
        job = SimpleNamespace(id=uuid4(), classification_id=cid, label="toxic", model_version=MODEL_VERSION,
                              status="queued", error=None, created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc))
        with patch("app.api.routes.get_classification", AsyncMock(return_value=SimpleNamespace(model_version=MODEL_VERSION))), \
                patch("app.api.routes.find_explanation", AsyncMock(return_value=None)), \
                patch("app.db.repository.find_explanation", AsyncMock(return_value=None)), \
                patch("app.db.repository.create_job", AsyncMock(side_effect=[(job, True), (job, False)])), \
                patch("app.db.queue.enqueue_explain_job", AsyncMock(side_effect=OSError("private redis address"))):
            for _ in range(2):
                self.assertEqual((await self.client.post("/v1/explain", json={"classification_id": str(cid), "label": "toxic"})).status_code, 202)
        self.assertEqual(self.sample("explanation_jobs_created_total"), 1)

    async def test_cached_explanations_count_results_including_batch_hits(self):
        from app.api.routes import MODEL_VERSION
        cid = uuid4()
        job = SimpleNamespace(id=uuid4(), classification_id=cid, label="toxic", model_version=MODEL_VERSION,
                              status="done", error=None, created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc))
        cached = SimpleNamespace(label="toxic", spans=[], convergence_delta=0.)
        with patch("app.api.routes.get_classification", AsyncMock(return_value=SimpleNamespace(model_version=MODEL_VERSION, text="private"))), \
                patch("app.api.routes.find_explanation", AsyncMock(return_value=cached)), \
                patch("app.db.repository.create_job", AsyncMock(return_value=(job, False))):
            self.assertEqual((await self.client.post("/v1/explain", json={"classification_id": str(cid), "label": "toxic"})).status_code, 200)
            await self.client.post("/v1/explain_batch", json={"classification_ids": [str(cid)], "label": "toxic"})
        self.assertEqual(self.sample("explanation_cache_hits_total"), 2)
        self.assertEqual(self.sample("explanation_jobs_created_total"), 0)

    async def test_failed_batch_does_not_count_cache_results_as_served(self):
        from app.api.routes import MODEL_VERSION
        first, second = uuid4(), uuid4()
        job = SimpleNamespace(id=uuid4(), classification_id=first, label="toxic", model_version=MODEL_VERSION,
                              status="done", error=None, created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc))
        with patch("app.api.routes.get_classification", AsyncMock(return_value=SimpleNamespace(model_version=MODEL_VERSION))), \
                patch("app.api.routes.find_explanation", AsyncMock(return_value=Mock())), \
                patch("app.db.repository.create_job", AsyncMock(side_effect=[(job, False), SQLAlchemyError("failed second item")])):
            response = await self.client.post("/v1/explain_batch", json={"classification_ids": [str(first), str(second)], "label": "toxic"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.sample("explanation_cache_hits_total"), 0)


class QueueCollectionTests(IsolatedAsyncioTestCase):
    async def test_failed_database_collection_is_explicit(self):
        with patch("app.db.engine.async_session", side_effect=OSError("private database address")), \
                self.assertLogs("app.metrics", level="WARNING"):
            self.assertIsNone(await read_queue_counts())


class WorkerMetricsTests(IsolatedAsyncioTestCase):
    process = logging_tests.WorkerLoggingTests.process

    async def test_attempt_durations_include_success_failure_and_lost_ownership(self):
        metrics = WorkerMetrics()
        with patch("app.worker.WORKER_METRICS", metrics):
            await self.process()
            await self.process(committed=False)
            await self.process(failure=RuntimeError("failed attempt"))
        self.assertEqual(metrics.registry.get_sample_value("explanation_job_attempt_duration_seconds_count"), 3)
        self.assertGreater(metrics.registry.get_sample_value("explanation_job_attempt_duration_seconds_sum"), 0)

    async def test_worker_http_endpoint_exposes_only_its_own_registry_and_closes(self):
        from prometheus_client import start_http_server
        from urllib.request import urlopen
        metrics = WorkerMetrics()
        metrics.attempts.inc()
        server, thread = start_http_server(0, addr="127.0.0.1", registry=metrics.registry)
        try:
            def fetch():
                with urlopen(f"http://127.0.0.1:{server.server_port}/metrics/", timeout=3) as response:
                    return response.read()
            body = await asyncio.to_thread(fetch)
            self.assertIn(b"explanation_job_attempts_total 1.0", body)
            self.assertNotIn(b"http_requests_total", body)
            self.assertNotIn(b"classification_predictions_total", body)
        finally:
            await asyncio.to_thread(server.shutdown)
            server.server_close()
            await asyncio.to_thread(thread.join)
        self.assertFalse(thread.is_alive())


@skipUnless(os.getenv("TOXICITY_DB_TESTS") == "1", "requires disposable PostgreSQL")
class JobMetricsTests(IsolatedAsyncioTestCase):
    asyncSetUp = job_tests.JobDeliveryTests.asyncSetUp
    asyncTearDown = job_tests.JobDeliveryTests.asyncTearDown
    seed = job_tests.JobDeliveryTests.seed
    claim = job_tests.JobDeliveryTests.claim
    expire = job_tests.JobDeliveryTests.expire

    async def test_retries_duplicate_delivery_and_completion_measure_actual_transitions(self):
        from app.db.engine import async_session
        from app.db.jobs import fail_job, finish_job
        from app.explainer import AttributionResult
        metrics = WorkerMetrics()
        with patch("app.db.jobs.WORKER_METRICS", metrics):
            jid, _ = await self.seed()
            first = await self.claim(jid)
            self.assertIsNone(await self.claim(jid))
            self.assertEqual(metrics.registry.get_sample_value("explanation_job_attempts_total"), 1)
            async with async_session() as session:
                await fail_job(session, jid, first.lease_token, "retry")
            second = await self.claim(jid)
            async with async_session() as session:
                self.assertFalse(await finish_job(session, jid, first.lease_token, AttributionResult("toxic", ())))
                self.assertTrue(await finish_job(session, jid, second.lease_token, AttributionResult("toxic", ())))
                self.assertFalse(await finish_job(session, jid, second.lease_token, AttributionResult("toxic", ())))
            self.assertEqual(metrics.registry.get_sample_value("explanation_job_attempts_total"), 2)
            self.assertEqual(metrics.registry.get_sample_value("explanation_job_queue_wait_seconds_count"), 1)
            self.assertEqual(metrics.registry.get_sample_value("explanation_job_outcomes_total", {"outcome": "retry"}), 1)
            self.assertEqual(metrics.registry.get_sample_value("explanation_job_outcomes_total", {"outcome": "completed"}), 1)
            self.assertEqual(metrics.registry.get_sample_value("explanation_job_end_to_end_duration_seconds_count"), 1)

    async def test_failed_commit_does_not_increment_completion_or_end_to_end(self):
        from app.db.engine import async_session
        from app.db.jobs import finish_job
        from app.explainer import AttributionResult
        metrics = WorkerMetrics()
        jid, _ = await self.seed()
        job = await self.claim(jid)
        with patch("app.db.jobs.WORKER_METRICS", metrics):
            async with async_session() as session:
                with patch.object(session, "commit", AsyncMock(side_effect=OSError("injected failure"))):
                    with self.assertRaises(OSError):
                        await finish_job(session, jid, job.lease_token, AttributionResult("toxic", ()))
        self.assertEqual(metrics.registry.get_sample_value("explanation_job_outcomes_total", {"outcome": "completed"}), 0)
        self.assertEqual(metrics.registry.get_sample_value("explanation_job_end_to_end_duration_seconds_count"), 0)

    async def test_queue_counts_and_lease_recovery(self):
        from app.db.engine import async_session
        from app.db.jobs import recover_jobs
        metrics = WorkerMetrics()
        with patch("app.db.jobs.WORKER_METRICS", metrics):
            jid, _ = await self.seed()
            self.assertEqual(await read_queue_counts(), {"queued": 1, "running": 0})
            await self.claim(jid)
            self.assertEqual(await read_queue_counts(), {"queued": 0, "running": 1})
            await self.expire(jid)
            async with async_session() as session:
                await recover_jobs(session)
            self.assertEqual(await read_queue_counts(), {"queued": 1, "running": 0})
            self.assertEqual(metrics.registry.get_sample_value("explanation_job_outcomes_total", {"outcome": "retry"}), 1)
