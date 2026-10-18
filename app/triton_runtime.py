"""Concurrent classification client; Triton owns model scheduling and execution."""
import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import numpy as np

from app.classifier import ToxicityOnnxClassifier
from app.errors import OperationWaitTimeoutError, RuntimeBusyError, RuntimeNotReadyError
from app.metadata import ModelInfo
from app.metrics import current_api_metrics
from app.runtime import ApplicationRuntime, RuntimeState, RuntimeStatus
from app.settings import MODEL_VERSION, RuntimeSettings


class TritonRuntime:
    remote_inference = True
    # Reuse the public input rules, without entering the local inference gate.
    _validate_text = ApplicationRuntime._validate_text
    _validate_texts = ApplicationRuntime._validate_texts

    def __init__(self, settings: RuntimeSettings, *, client=None):
        self._settings = settings
        self._state = RuntimeState.STOPPED
        self._active = 0
        self._drained = asyncio.Event()
        self._drained.set()
        self._client = client
        self._max_batch = 0
        self._info = None

    async def start(self):
        self._state = RuntimeState.STARTING
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._settings.triton_url, trust_env=False,
                timeout=self._settings.operation_wait_timeout_seconds,
                limits=httpx.Limits(max_connections=self._settings.triton_max_inflight,
                                    max_keepalive_connections=self._settings.triton_max_inflight),
            )
        try:
            if not await self.check_ready():
                raise RuntimeNotReadyError("Triton model is not ready")
            response = await self._client.get("/v2/models/toxicity/versions/1/config")
            response.raise_for_status()
            config = response.json()
            parameters = {k: v["string_value"] for k, v in config["parameters"].items()}
            if parameters["model_version"] != MODEL_VERSION or json.loads(parameters["labels"]) != list(self._settings.labels) or int(parameters["max_length"]) != self._settings.max_length:
                raise RuntimeError("Triton model metadata does not match API configuration")
            self._max_batch = min(config["max_batch_size"], self._settings.inference_batch_size)
            if self._max_batch < 1:
                raise RuntimeError("Triton model must support batching")
            self._info = ModelInfo(
                model_type=parameters["model_type"], device="cpu",
                labels=self._settings.labels, thresholds=self._settings.thresholds,
                max_length=self._settings.max_length, explanation_steps=self._settings.n_steps,
                inference_backend="triton",
                max_text_chars=self._settings.max_text_chars,
                max_prediction_batch_size=self._settings.max_prediction_batch_size,
            )
        except BaseException:
            self._state = RuntimeState.STOPPED
            await self._client.aclose()
            raise
        self._state = RuntimeState.READY

    async def stop(self):
        self._state = RuntimeState.STOPPING
        await self._drained.wait()
        await self._client.aclose()
        self._state = RuntimeState.STOPPED

    async def check_ready(self):
        try:
            async with asyncio.timeout(2):
                response = await self._client.get("/v2/models/toxicity/versions/1/ready")
                return response.status_code == 200
        except (httpx.HTTPError, TimeoutError):
            return False

    def is_ready(self):
        return self._state is RuntimeState.READY

    def model_info(self):
        if not self.is_ready():
            raise RuntimeNotReadyError("Triton runtime is not ready")
        return self._info

    def status(self):
        # Waiting inside Triton is measured by Triton's queue metrics. The API
        # admits calls immediately or rejects them; it has no second wait queue.
        return RuntimeStatus(self._state, self._active > 0, 0, 0,
                             self._settings.triton_max_inflight,
                             self._settings.max_waiting_explanations)

    @asynccontextmanager
    async def _operation(self):
        if not self.is_ready():
            raise RuntimeNotReadyError("Triton runtime is not ready")
        if self._active >= self._settings.triton_max_inflight:
            raise RuntimeBusyError("Classification capacity reached")
        # No await between capacity check and reservation: all calls use the
        # same event loop, so admission is atomic without a blocking mutex.
        self._active += 1
        self._drained.clear()
        metrics = current_api_metrics()
        ticket = metrics.classification_ticket() if metrics else None
        if ticket:
            ticket.start()
        try:
            async with asyncio.timeout(self._settings.operation_wait_timeout_seconds):
                yield
        except (TimeoutError, httpx.TimeoutException):
            raise OperationWaitTimeoutError("Triton inference timed out") from None
        except httpx.HTTPError:
            raise RuntimeNotReadyError("Triton inference unavailable") from None
        finally:
            if ticket:
                ticket.finish()
            self._active -= 1
            if not self._active:
                self._drained.set()

    async def predict(self, text):
        self._validate_text(text)
        async with self._operation():
            return (await self._infer([text]))[0]

    async def predict_batch(self, texts):
        self._validate_texts(texts, max_batch_size=self._settings.max_prediction_batch_size)
        async with self._operation():
            results = []
            # A public batch may contain 64 texts. Keep internal calls <= the
            # model's ceiling, preserving order without spawning unbounded tasks.
            for offset in range(0, len(texts), self._max_batch):
                results.extend(await self._infer(texts[offset:offset + self._max_batch]))
            return results

    async def _infer(self, texts):
        response = await self._client.post("/v2/models/toxicity/versions/1/infer", json={
            "inputs": [{"name": "TEXT", "shape": [len(texts), 1], "datatype": "BYTES", "data": texts}],
            "outputs": [{"name": "LOGITS"}],
        })
        if response.status_code == 400:
            # Triton reports scheduler queue rejection as a protocol error;
            # public inputs have already been validated. Never expose its body.
            raise RuntimeNotReadyError("Triton rejected inference")
        response.raise_for_status()
        try:
            outputs = response.json()["outputs"]
            output = next(x for x in outputs if x["name"] == "LOGITS")
            expected = [len(texts), len(self._settings.labels)]
            if output["shape"] != expected or output["datatype"] != "FP32":
                raise ValueError("Unexpected output shape or datatype")
            logits = np.asarray(output["data"], dtype=np.float64).reshape(expected)
            if not np.isfinite(logits).all():
                raise ValueError("Nonfinite logits")
        except (ValueError, TypeError, KeyError, StopIteration):
            raise RuntimeNotReadyError("Invalid Triton inference response") from None
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        decisions = probabilities >= np.asarray(self._settings.thresholds).reshape(1, -1)
        return ToxicityOnnxClassifier._to_results(texts, self._settings.labels, probabilities, decisions)
