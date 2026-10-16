"""Triton gathers requests; one tokenizer/ONNX call executes their combined texts."""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        config = json.loads(args["model_config"])
        parameters = {key: value["string_value"] for key, value in config["parameters"].items()}
        self.max_batch_size = config["max_batch_size"]
        self.max_length = int(parameters["max_length"])
        root = Path(parameters["artifact_root"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            root / "models/distilbert/v2.0.0", use_fast=True, local_files_only=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(parameters["cpu_threads"])
        self.session = ort.InferenceSession(
            str(root / "models/distilbert/v2.0.0-onnx/model.onnx"),
            sess_options=options, providers=["CPUExecutionProvider"])
        self.label_count = len(json.loads(parameters["labels"]))

    def execute(self, requests):
        responses = [None] * len(requests)
        texts, slices = [], []
        for index, request in enumerate(requests):
            if request.is_cancelled():
                responses[index] = self._error("Inference cancelled", pb_utils.TritonError.CANCELLED)
                continue
            try:
                tensor = pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy()
                if tensor.ndim != 2 or tensor.shape[1] != 1 or not 1 <= tensor.shape[0] <= self.max_batch_size:
                    raise ValueError("Invalid input shape")
                items = [x.decode("utf-8") if isinstance(x, bytes) else x for x in tensor.reshape(-1)]
                if any(not isinstance(x, str) or not x.strip() or len(x) > 10_000 for x in items):
                    raise ValueError("Invalid text")
                slices.append((index, len(texts), len(texts) + len(items)))
                texts.extend(items)
            except (ValueError, TypeError, AttributeError, UnicodeError):
                responses[index] = self._error("Invalid classification input", pb_utils.TritonError.INVALID_ARG)
        if not texts:
            return responses
        try:
            if len(texts) > self.max_batch_size:
                raise RuntimeError("Scheduler exceeded the model batch limit")
            encoded = self.tokenizer(texts, return_tensors="np", truncation=True,
                                     padding=True, max_length=self.max_length)
            feed = {name: np.asarray(encoded[name], dtype=np.int64)
                    for name in ("input_ids", "attention_mask")}
            logits = np.asarray(self.session.run(None, feed)[0], dtype=np.float32)
            if logits.shape != (len(texts), self.label_count) or not np.isfinite(logits).all():
                raise RuntimeError("Invalid model output")
            for index, start, stop in slices:
                responses[index] = pb_utils.InferenceResponse(output_tensors=[
                    pb_utils.Tensor("LOGITS", logits[start:stop].copy())])
        except Exception:
            # Inputs and backend exception messages may contain private text.
            pb_utils.Logger.log_error("Classification batch inference failed")
            for index, _, _ in slices:
                responses[index] = self._error("Classification inference failed", pb_utils.TritonError.INTERNAL)
        return responses

    @staticmethod
    def _error(message, code):
        return pb_utils.InferenceResponse(error=pb_utils.TritonError(message, code))
