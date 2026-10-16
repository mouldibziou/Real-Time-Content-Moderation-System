import asyncio
import json
import os
from dataclasses import replace
from unittest import IsolatedAsyncioTestCase, TestCase, skipUnless
from unittest.mock import AsyncMock, Mock, patch

try:
    import httpx
except ImportError:
    from unittest import SkipTest
    raise SkipTest("Install requirements/triton-client.txt to test the optional Triton path")
import numpy as np

from app.api.dependencies import _call_runtime
from app.errors import OperationWaitTimeoutError, RuntimeBusyError, RuntimeNotReadyError
from app.settings import RuntimeSettings, load_runtime_settings
from app.triton_runtime import TritonRuntime


def model_config(settings):
    return {"max_batch_size": 8, "parameters": {
        key: {"string_value": value} for key, value in {
            "model_version": "v2.0.0", "model_type": "distilbert",
            "labels": json.dumps(settings.labels), "max_length": str(settings.max_length),
        }.items()}}


def inference_reply(request):
    texts = json.loads(request.content)["inputs"][0]["data"]
    return httpx.Response(200, json={"outputs": [{"name": "LOGITS", "datatype": "FP32",
        "shape": [len(texts), 6], "data": [float(len(t) % 5) for t in texts for _ in range(6)]}]})


class TritonSettingsTests(TestCase):
    def test_environment_and_invalid_origins(self):
        settings = load_runtime_settings({"TOXICITY_INFERENCE_BACKEND": "triton",
            "TOXICITY_TRITON_URL": "http://localhost:8009", "TOXICITY_TRITON_MAX_INFLIGHT": "8"})
        self.assertEqual(settings.triton_max_inflight, 8)
        self.assertEqual(settings.triton_url, "http://localhost:8009")
        for origin in ("file:///tmp/socket", "http://user:password@triton", "http://triton?token=secret", "http://triton/path"):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                replace(settings, triton_url=origin)
        with self.assertRaises(ValueError):
            replace(settings, device="cuda")
        with self.assertRaises(ValueError):
            replace(settings, triton_max_inflight=0)


class TritonRuntimeTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.settings = RuntimeSettings(inference_backend="triton", inference_batch_size=8)
        self.infer = AsyncMock(side_effect=inference_reply)

        async def handler(request):
            if request.url.path.endswith("/ready"):
                return httpx.Response(200)
            if request.url.path.endswith("/config"):
                return httpx.Response(200, json=model_config(self.settings))
            return await self.infer(request)

        self.client = httpx.AsyncClient(base_url="http://triton:8000", transport=httpx.MockTransport(handler))
        self.runtime = TritonRuntime(self.settings, client=self.client)
        await self.runtime.start()

    async def asyncTearDown(self):
        await self.runtime.stop()

    async def test_eight_independent_calls_reach_triton_concurrently(self):
        all_arrived = asyncio.Event()
        arrived = 0

        async def respond(request):
            nonlocal arrived
            arrived += 1
            if arrived == 8:
                all_arrived.set()
            await asyncio.wait_for(all_arrived.wait(), 2)
            return inference_reply(request)

        self.infer.side_effect = respond
        executor = Mock()
        texts = ["x" * i for i in range(1, 9)]
        results = await asyncio.gather(*[_call_runtime(self.runtime.predict, executor, t) for t in texts])
        executor.submit.assert_not_called()
        self.assertEqual(arrived, 8)
        for text, result in zip(texts, results):
            self.assertAlmostEqual(result.labels["toxic"].probability, 1 / (1 + np.exp(-(len(text) % 5))))
        self.assertFalse(self.runtime.status().operation_active)

    async def test_large_public_batch_is_split_and_order_is_preserved(self):
        texts = ["x" * i for i in range(1, 18)]
        results = await self.runtime.predict_batch(texts)
        calls = [json.loads(call.args[0].content)["inputs"][0] for call in self.infer.call_args_list]
        self.assertEqual([x["shape"] for x in calls], [[8, 1], [8, 1], [1, 1]])
        self.assertEqual([t for x in calls for t in x["data"]], texts)
        self.assertEqual(len(results), 17)

    async def test_capacity_and_cancellation_release_the_slot(self):
        self.runtime._settings = replace(self.settings, triton_max_inflight=1)
        entered = asyncio.Event()

        async def block(request):
            entered.set()
            await asyncio.Event().wait()

        self.infer.side_effect = block
        task = asyncio.create_task(self.runtime.predict("hello"))
        await asyncio.wait_for(entered.wait(), 2)
        try:
            with self.assertRaises(RuntimeBusyError):
                await self.runtime.predict("other")
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.infer.side_effect = inference_reply
        await self.runtime.predict("works again")
        self.assertEqual(self.runtime._active, 0)

    async def test_timeout_and_service_failure_do_not_leak_capacity_or_details(self):
        for exception, expected in ((httpx.ReadTimeout("PRIVATE"), OperationWaitTimeoutError),
                                     (httpx.ConnectError("PRIVATE"), RuntimeNotReadyError)):
            self.infer.side_effect = exception
            with self.assertRaises(expected) as caught:
                await self.runtime.predict("private input")
            self.assertNotIn("PRIVATE", str(caught.exception))
            self.assertEqual(self.runtime._active, 0)
        self.infer.side_effect = lambda request: httpx.Response(503, text="PRIVATE")
        with self.assertRaises(RuntimeNotReadyError):
            await self.runtime.predict("hello")

    async def test_malformed_output_and_oversized_input_are_rejected(self):
        for output in ({}, {"outputs": [{"name": "LOGITS", "shape": [1, 5], "datatype": "FP32", "data": [0]*5}]}):
            self.infer.side_effect = lambda request: httpx.Response(200, json=output)
            with self.assertRaises(RuntimeNotReadyError):
                await self.runtime.predict("hello")
        before = self.infer.call_count
        with self.assertRaises(ValueError):
            await self.runtime.predict_batch(["hello"] * 65)
        self.assertEqual(self.infer.call_count, before)

    async def test_shutdown_drains_active_call_and_rejects_new_work(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def wait(request):
            entered.set()
            await release.wait()
            return inference_reply(request)

        self.infer.side_effect = wait
        task = asyncio.create_task(self.runtime.predict("hello"))
        await asyncio.wait_for(entered.wait(), 2)
        stop = asyncio.create_task(self.runtime.stop())
        await asyncio.sleep(0)
        try:
            self.assertFalse(stop.done())
            with self.assertRaises(RuntimeNotReadyError):
                await self.runtime.predict("new")
        finally:
            release.set()
            await task
            await stop

    async def test_metadata_mismatch_closes_client(self):
        async def wrong(request):
            if request.url.path.endswith("/ready"):
                return httpx.Response(200)
            config = model_config(self.settings)
            config["parameters"]["model_version"]["string_value"] = "wrong"
            return httpx.Response(200, json=config)
        client = httpx.AsyncClient(base_url="http://triton", transport=httpx.MockTransport(wrong))
        with self.assertRaisesRegex(RuntimeError, "metadata"):
            await TritonRuntime(self.settings, client=client).start()
        self.assertTrue(client.is_closed)

    async def test_api_keeps_independent_ids_and_background_writes(self):
        from app.api.main import create_app
        from concurrent.futures import ThreadPoolExecutor
        from tests.test_application_logging import ASGIClient

        app = create_app(self.settings)
        app.state.runtime = self.runtime
        app.state.runtime_executor = ThreadPoolExecutor(max_workers=1)
        app.state.model_snapshot = {"model_type": "distilbert", "device": "cpu", "thresholds": list(self.settings.thresholds)}
        sent_ids = set()

        async def observe_response(scope, receive, send):
            async def observe(message):
                await send(message)
                if message["type"] == "http.response.body" and not message.get("more_body", False):
                    sent_ids.add(json.loads(message["body"])["classification_id"])
            await app(scope, receive, observe)

        async def persisted(**item):
            self.assertIn(str(item["id"]), sent_ids, "Persistence began before the response body was sent")

        client = ASGIClient(observe_response)
        persist = AsyncMock(side_effect=persisted)
        try:
            with patch("app.api.routes._persist_classification", persist):
                responses = await asyncio.gather(*[client.post("/v1/classify", json={"text": "x" * i}) for i in range(1, 9)])
            payloads = [json.loads(r.body) for r in responses]
            self.assertTrue(all(r.status_code == 200 for r in responses))
            self.assertEqual(len({p["classification_id"] for p in payloads}), 8)
            self.assertEqual(persist.await_count, 8)
            by_id = {p["classification_id"]: p for p in payloads}
            for call in persist.await_args_list:
                # Completion order may differ: compare IDs rather than assuming
                # the persistence tasks finish in arrival order.
                self.assertEqual(call.kwargs["text"], by_id[str(call.kwargs["id"])]["text"])
        finally:
            app.state.runtime_executor.shutdown()

    async def test_readiness_detects_triton_outage_before_database_check(self):
        from app.api.main import create_app
        from tests.test_application_logging import ASGIClient
        app = create_app(self.settings)
        app.state.runtime = self.runtime
        with patch.object(self.runtime, "check_ready", AsyncMock(return_value=False)), \
                patch("app.api.routes.async_session") as database:
            response = await ASGIClient(app).get("/v1/health/ready")
        self.assertEqual(response.status_code, 503)
        database.assert_not_called()

    async def test_lifespan_does_not_load_the_local_model(self):
        from app.api.main import create_app
        builder = Mock(side_effect=AssertionError("Must not load a local inference model"))
        app = create_app(self.settings, services_builder=builder)
        database = Mock()
        database.connect.return_value.__aenter__ = AsyncMock(side_effect=OSError("offline test database"))
        database.connect.return_value.__aexit__ = AsyncMock()
        database.dispose = AsyncMock()
        with patch("app.triton_runtime.TritonRuntime", return_value=self.runtime), \
                patch("app.api.main.engine", database), \
                patch("app.api.main.close_redis", AsyncMock()), \
                patch.dict("os.environ", {"API_METRICS_PORT": "0"}):
            # asyncSetUp already started this runtime; skip the second startup.
            with patch.object(self.runtime, "start", AsyncMock()):
                async with app.router.lifespan_context(app):
                    self.assertEqual(app.state.model_snapshot["model_type"], "distilbert")
        builder.assert_not_called()


@skipUnless(os.getenv("TOXICITY_TRITON_INTEGRATION_URL"), "Opt-in live Triton container test")
class LiveTritonTests(IsolatedAsyncioTestCase):
    async def test_full_length_eight_text_batch(self):
        runtime = TritonRuntime(RuntimeSettings(inference_backend="triton", inference_batch_size=8,
            operation_wait_timeout_seconds=60, triton_url=os.environ["TOXICITY_TRITON_INTEGRATION_URL"]))
        await runtime.start()
        try:
            results = await runtime.predict_batch(["word " * 600] * 8)
            self.assertEqual(len(results), 8)
            self.assertTrue(all(len(result.labels) == 6 for result in results))
            self.assertTrue(await runtime.check_ready())
        finally:
            await runtime.stop()

    async def test_concurrent_requests_are_batched_and_match_local_fp32(self):
        from app.resources import load_onnx_session
        from app.classifier import ToxicityOnnxClassifier
        from transformers import AutoTokenizer

        settings = RuntimeSettings(inference_backend="triton", cpu_threads=1,
            inference_batch_size=8, operation_wait_timeout_seconds=60,
            triton_url=os.environ["TOXICITY_TRITON_INTEGRATION_URL"])
        runtime = TritonRuntime(settings)
        await runtime.start()
        texts = ["You are an idiot.", "Thanks for helping.", "Good morning!", "An ordinary sentence.",
                 "Stop insulting people.", "Bonjour, merci.", "مرحبا بك", "A longer sentence with some additional words."]
        async with httpx.AsyncClient(base_url=settings.triton_url, trust_env=False) as client:
            before = (await client.get("/v2/models/toxicity/versions/1/stats")).json()["model_stats"][0]
            try:
                results = await asyncio.gather(*[runtime.predict(t) for t in texts])
                after = (await client.get("/v2/models/toxicity/versions/1/stats")).json()["model_stats"][0]
                local = replace(settings, inference_backend="onnx")
                classifier = ToxicityOnnxClassifier(load_onnx_session(local),
                    AutoTokenizer.from_pretrained(local.model_dir, local_files_only=True),
                    local.thresholds, local.labels, local.max_length, inference_batch_size=8)
                expected = classifier.predict_batch(texts)
                for actual, reference in zip(results, expected):
                    for label in settings.labels:
                        self.assertAlmostEqual(actual.labels[label].probability,
                                               reference.labels[label].probability, delta=1e-5)
                        self.assertEqual(actual.labels[label].predicted, reference.labels[label].predicted)
                items = int(after["inference_count"]) - int(before["inference_count"])
                executions = int(after["execution_count"]) - int(before["execution_count"])
                self.assertEqual(items, 8)
                self.assertLess(executions, items, "No requests were combined into an inference batch")
                print(json.dumps({"live_triton_texts": items, "model_executions": executions,
                                  "mean_executed_batch": items / executions}))
            finally:
                await runtime.stop()
