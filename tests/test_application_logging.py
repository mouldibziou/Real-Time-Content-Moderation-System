"""Logging regressions: concurrent clients, truthful outcomes, and safe output."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID, uuid4
from urllib.parse import urlsplit

from fastapi import FastAPI
from sqlalchemy.exc import SQLAlchemyError

from app.observability import JsonFormatter, RequestLoggingMiddleware, log_context, log_event


def formatted(records):
    return [json.loads(JsonFormatter().format(record)) for record in records]


class ASGIClient:
    """Exercise complete ASGI responses without an extra test dependency."""

    def __init__(self, app):
        self.app = app

    async def get(self, path):
        return await self.request("GET", path)

    async def post(self, path, *, json):
        return await self.request("POST", path, body=json)

    async def request(self, method, path, body=None):
        url = urlsplit(path)
        payload = b"" if body is None else json.dumps(body).encode()
        messages = []

        async def receive():
            return {"type": "http.request", "body": payload, "more_body": False}

        async def send(message):
            messages.append(message)

        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
                 "http_version": "1.1", "method": method, "scheme": "http",
                 "path": url.path, "raw_path": url.path.encode(), "query_string": url.query.encode(),
                 "root_path": "", "headers": [(b"content-type", b"application/json")],
                 "server": ("test", 80), "client": ("127.0.0.1", 1234)}
        try:
            await self.app(scope, receive, send)
        except Exception:
            # Starlette re-raises errors after sending its 500 response.
            if not any(m["type"] == "http.response.start" for m in messages):
                raise
        start = next(m for m in messages if m["type"] == "http.response.start")
        return SimpleNamespace(status_code=start["status"],
                               headers={k.decode().lower(): v.decode() for k, v in start["headers"]},
                               body=b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body"))


class FormatterTests(TestCase):
    def test_exception_does_not_expose_message_sql_or_chained_error(self):
        secret = "private comment and password"
        try:
            try:
                raise ValueError(secret)
            except ValueError as exc:
                raise SQLAlchemyError("SELECT " + secret) from exc
        except SQLAlchemyError:
            with self.assertLogs("privacy") as captured:
                log_event(logging.getLogger("privacy"), "database.request_failed",
                          level=logging.ERROR, exc_info=True)
        output = JsonFormatter().format(captured.records[0])
        self.assertNotIn(secret, output)
        self.assertNotIn("SELECT", output)
        data = json.loads(output)
        self.assertEqual(data["error_type"], "SQLAlchemyError")
        self.assertTrue(data["error_stack"])

    def test_configured_worker_writes_json_to_stdout_and_filters_debug(self):
        script = """
import logging
from app.observability import configure_logging, log_event
configure_logging('worker')
log_event(logging.getLogger('worker'), 'worker.ready')
log_event(logging.getLogger('worker'), 'hidden', level=logging.DEBUG)
"""
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                                env={**os.environ, "LOG_LEVEL": "INFO"}, check=True)
        self.assertEqual(result.stderr, "")
        lines = result.stdout.splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["service"], "worker")

    def test_uvicorn_config_and_access_logs_are_valid_json_without_query_string(self):
        script = """
import json, logging, logging.config
with open('docker/logging.json') as file:
    logging.config.dictConfig(json.load(file))
logging.getLogger('uvicorn.access').info('%s - "%s %s HTTP/%s" %d',
    '127.0.0.1:1234', 'GET', '/v1/status?token=secret', '1.1', 200)
"""
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        self.assertEqual(data["event"], "http.access")
        self.assertEqual(data["path"], "/v1/status")
        self.assertNotIn("secret", result.stdout)


class RequestLoggingTests(IsolatedAsyncioTestCase):
    async def test_concurrent_requests_keep_separate_contexts_and_headers(self):
        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)
        arrived = 0
        both_arrived = asyncio.Event()

        @app.get("/work/{item}")
        async def work(item: int):
            nonlocal arrived
            arrived += 1
            if arrived == 2:
                both_arrived.set()
            await both_arrived.wait()
            log_event(logging.getLogger("app.work"), "work.completed", item=item)
            return {"item": item}

        client = ASGIClient(app)
        with self.assertLogs("app", level="INFO") as captured:
            responses = await asyncio.gather(client.get("/work/1?secret=private"), client.get("/work/2"))
        records = formatted(captured.records)
        ids = [response.headers["x-request-id"] for response in responses]
        self.assertNotEqual(*ids)
        for request_id in ids:
            UUID(request_id)
            matching = [r for r in records if r["request_id"] == request_id]
            self.assertEqual(len(matching), 2)
            summary = next(r for r in matching if r["event"] == "request.completed")
            self.assertEqual(summary["route"], "/work/{item}")
            self.assertEqual(summary["status_code"], 200)
        self.assertNotIn("private", json.dumps(records))
        with self.assertLogs("app.work", level="INFO") as captured:
            log_event(logging.getLogger("app.work"), "outside.request")
        self.assertNotIn("request_id", formatted(captured.records)[0])

    async def test_successful_health_checks_are_quiet_but_failures_are_visible(self):
        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)

        @app.get("/v1/health/live")
        async def live():
            return {"status": "healthy"}

        @app.get("/v1/health/ready", status_code=503)
        async def ready():
            return {"is_ready": False}

        client = ASGIClient(app)
        with self.assertNoLogs("app.http"):
            response = await client.get("/v1/health/live")
        self.assertIn("x-request-id", response.headers)
        with self.assertLogs("app.http") as captured:
            await client.get("/v1/health/ready")
        self.assertEqual(formatted(captured.records)[0]["status_code"], 503)


class ApiWorkflowLoggingTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from app.api.main import create_app
        from app.db.engine import get_db
        from app.settings import RuntimeSettings
        from app.classifier import ClassificationResult, LabelPrediction
        self.app = create_app(RuntimeSettings(device="cpu"))
        self.runtime = Mock()
        self.runtime.predict.return_value = ClassificationResult({"toxic": LabelPrediction(0.9, True)})
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.app.state.runtime = self.runtime
        self.app.state.runtime_executor = self.executor
        self.app.state.model_snapshot = {"model_type": "distilbert", "device": "cpu", "thresholds": []}

        async def fake_db():
            yield Mock()

        self.app.dependency_overrides[get_db] = fake_db
        self.client = ASGIClient(self.app)

    async def asyncTearDown(self):
        self.executor.shutdown(wait=True)

    async def test_classification_logs_committed_id_and_never_input_text(self):
        # Return-first: the logged id is pre-generated and echoed in the
        # response; persistence happens in background with its own session.
        with patch("app.api.routes.save_classification", AsyncMock()) as persisted, \
                self.assertLogs("app", level="INFO") as captured:
            response = await self.client.post("/v1/classify", json={"text": "PRIVATE CLIENT COMMENT"})
        self.assertEqual(response.status_code, 200)
        records = formatted(captured.records)
        completed = next(r for r in records if r["event"] == "classification.completed")
        self.assertEqual(completed["classification_id"], json.loads(response.body)["classification_id"])
        self.assertEqual(completed["request_id"], response.headers["x-request-id"])
        self.assertNotIn("PRIVATE CLIENT COMMENT", json.dumps(records))
        persisted.assert_awaited_once()

    async def test_database_failure_still_returns_but_never_false_success(self):
        # Return-first: a persistence failure logs in background and never
        # fails the already-returned 200, and no input text leaks.
        with patch("app.api.routes.save_classification", AsyncMock(side_effect=SQLAlchemyError("PRIVATE SQL PARAMS"))), \
                self.assertLogs("app", level="INFO") as captured:
            response = await self.client.post("/v1/classify", json={"text": "private text"})
        self.assertEqual(response.status_code, 200)
        records = formatted(captured.records)
        self.assertIn("classification.completed", [r["event"] for r in records])
        self.assertNotIn("PRIVATE SQL PARAMS", json.dumps(records))

    async def test_classification_batch_reports_size_without_texts(self):
        self.runtime.predict_batch.return_value = [self.runtime.predict.return_value] * 2
        with patch("app.api.routes.save_classifications_batch", AsyncMock(return_value=[uuid4(), uuid4()])), \
                self.assertLogs("app", level="INFO") as captured:
            response = await self.client.post("/v1/classify_batch", json={"texts": ["PRIVATE ONE", "PRIVATE TWO"]})
        self.assertEqual(response.status_code, 200)
        records = formatted(captured.records)
        completed = next(r for r in records if r["event"] == "classification.batch_completed")
        self.assertEqual(completed["batch_size"], 2)
        self.assertNotIn("PRIVATE", json.dumps(records))

    async def test_cached_explanation_batch_logs_hits_for_already_done_jobs(self):
        from datetime import datetime, timezone
        from app.api.routes import MODEL_VERSION
        cid, jid = uuid4(), uuid4()
        job = SimpleNamespace(id=jid, classification_id=cid, label="toxic", model_version=MODEL_VERSION,
                              status="done", error=None, created_at=datetime.now(timezone.utc),
                              updated_at=datetime.now(timezone.utc))
        with patch("app.api.routes.get_classification", AsyncMock(return_value=SimpleNamespace(model_version=MODEL_VERSION))), \
                patch("app.api.routes.find_explanation", AsyncMock(return_value=Mock())), \
                patch("app.db.repository.create_job", AsyncMock(return_value=(job, False))), \
                patch("app.db.queue.enqueue_explain_job", AsyncMock()) as publish, \
                self.assertLogs("app", level="INFO") as captured:
            response = await self.client.post("/v1/explain_batch", json={"classification_ids": [str(cid)], "label": "toxic"})
        self.assertEqual(response.status_code, 202)
        publish.assert_not_awaited()
        records = formatted(captured.records)
        self.assertIn("explanation.cache_hit", [r["event"] for r in records])
        summary = next(r for r in records if r["event"] == "explanation.batch_submitted")
        self.assertEqual(summary["cached_count"], 1)
        self.assertEqual(summary["created_count"], 0)

    async def test_unexpected_failure_has_request_id_and_safe_stack(self):
        self.runtime.predict.side_effect = RuntimeError("PRIVATE INPUT")
        with self.assertLogs("app", level="INFO") as captured:
            response = await self.client.post("/v1/classify", json={"text": "private text"})
        self.assertEqual(response.status_code, 500)
        records = formatted(captured.records)
        failure = next(r for r in records if r["event"] == "request.failed")
        self.assertEqual(failure["request_id"], response.headers["x-request-id"])
        self.assertEqual(failure["stage"], "classification.runtime")
        self.assertEqual(failure["error_type"], "RuntimeError")
        self.assertNotIn("PRIVATE INPUT", json.dumps(records))

    async def test_validation_failure_is_visible_without_body(self):
        with self.assertLogs("app.http", level="INFO") as captured:
            response = await self.client.post("/v1/classify", json={"text": ""})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(formatted(captured.records)[0]["status_code"], 422)

    async def test_redis_failure_still_accepts_durable_job_and_logs_recovery(self):
        from datetime import datetime, timezone
        from app.api.routes import MODEL_VERSION
        cid, jid = uuid4(), uuid4()
        job = SimpleNamespace(id=jid, classification_id=cid, label="toxic", model_version=MODEL_VERSION,
                              status="queued", error=None, created_at=datetime.now(timezone.utc),
                              updated_at=datetime.now(timezone.utc))
        with patch("app.api.routes.get_classification", AsyncMock(return_value=SimpleNamespace(model_version=MODEL_VERSION))), \
                patch("app.api.routes.find_explanation", AsyncMock(return_value=None)), \
                patch("app.db.repository.find_explanation", AsyncMock(return_value=None)), \
                patch("app.db.repository.create_job", AsyncMock(return_value=(job, True))), \
                patch("app.db.queue.enqueue_explain_job", AsyncMock(side_effect=OSError("PRIVATE REDIS URL"))), \
                self.assertLogs("app", level="INFO") as captured:
            response = await self.client.post("/v1/explain", json={"classification_id": str(cid), "label": "toxic"})
        self.assertEqual(response.status_code, 202)
        records = formatted(captured.records)
        warning = next(r for r in records if r["event"] == "explanation.notification_failed")
        self.assertEqual(warning["job_id"], str(jid))
        self.assertEqual(warning["level"], "WARNING")
        self.assertEqual(warning["recovery"], "database_polling")
        self.assertNotIn("PRIVATE REDIS URL", json.dumps(records))

    async def test_runtime_thread_inherits_request_id(self):
        def predict(text):
            log_event(logging.getLogger("app.inference"), "test.inference")
            return SimpleNamespace(labels={})
        self.runtime.predict.side_effect = predict
        with patch("app.api.routes.save_classification", AsyncMock(return_value=uuid4())), \
                self.assertLogs("app", level="INFO") as captured:
            response = await self.client.post("/v1/classify", json={"text": "hello"})
        inference = next(r for r in formatted(captured.records) if r["event"] == "test.inference")
        self.assertEqual(inference["request_id"], response.headers["x-request-id"])


class WorkerLoggingTests(IsolatedAsyncioTestCase):
    async def process(self, *, committed=True, failure=None, failure_status="queued"):
        from app.worker import _process_batch, MODEL_VERSION
        self.job = SimpleNamespace(id=uuid4(), classification_id=uuid4(), label="toxic",
                                   model_version=MODEL_VERSION, lease_token=uuid4(), attempts=1)
        session = AsyncMock()
        session_factory = Mock()
        session_factory.return_value.__aenter__ = AsyncMock(return_value=session)
        session_factory.return_value.__aexit__ = AsyncMock(return_value=False)
        runtime = Mock()
        runtime.explain.return_value = SimpleNamespace(spans=[])
        if failure:
            runtime.explain.side_effect = failure
        with patch("app.db.engine.async_session", session_factory), \
                patch("app.db.jobs.claim_job", AsyncMock(return_value=self.job)), \
                patch("app.db.repository.get_classification", AsyncMock(return_value=SimpleNamespace(text="PRIVATE TEXT"))), \
                patch("app.db.jobs.finish_job", AsyncMock(return_value=committed)), \
                patch("app.db.jobs.fail_job", AsyncMock(return_value=failure_status)), \
                self.assertLogs("expl_worker", level="INFO") as captured:
            await _process_batch([str(self.job.id)], runtime)
        return formatted(captured.records)

    async def test_completed_job_is_logged_after_successful_persistence(self):
        records = await self.process()
        self.assertEqual([r["event"] for r in records], ["explanation.job_claimed", "explanation.job_completed"])
        self.assertEqual(records[-1]["job_id"], str(self.job.id))
        self.assertNotIn("PRIVATE TEXT", json.dumps(records))

    async def test_lost_ownership_never_logs_success(self):
        records = await self.process(committed=False)
        self.assertEqual(records[-1]["event"], "explanation.ownership_lost")
        self.assertNotIn("explanation.job_completed", [r["event"] for r in records])

    async def test_attempt_failure_distinguishes_retry_from_terminal_failure(self):
        for status, event in [("queued", "explanation.retry_scheduled"), ("failed", "explanation.job_failed")]:
            with self.subTest(status=status):
                records = await self.process(failure=RuntimeError("PRIVATE ERROR"), failure_status=status)
                self.assertEqual(records[-1]["event"], event)
                self.assertEqual(records[1]["stage"], "explanation.runtime")
                self.assertNotIn("PRIVATE ERROR", json.dumps(records))
