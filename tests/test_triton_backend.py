import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, skipUnless
from unittest.mock import Mock, patch

import numpy as np


def load_backend():
    class Error:
        CANCELLED, INVALID_ARG, INTERNAL = "cancelled", "invalid", "internal"
        def __init__(self, message, code):
            self.message, self.code = message, code

    class Tensor:
        def __init__(self, name, data):
            self.name, self.data = name, data
        def as_numpy(self):
            return self.data

    utils = SimpleNamespace(TritonError=Error, Tensor=Tensor, Logger=SimpleNamespace(log_error=Mock()),
        get_input_tensor_by_name=lambda request, name: Tensor(name, request.texts),
        InferenceResponse=lambda **kwargs: SimpleNamespace(**kwargs))
    spec = importlib.util.spec_from_file_location("toxicity_triton_backend", "docker/triton/model_repository/toxicity/1/model.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", {"triton_python_backend_utils": utils}):
        spec.loader.exec_module(module)
    return module, utils


class TritonBackendTests(TestCase):
    def setUp(self):
        self.module, self.utils = load_backend()
        self.model = self.module.TritonPythonModel()
        self.model.max_batch_size = 8
        self.model.max_length = 512
        self.model.label_count = 6
        self.model.tokenizer = Mock(return_value={"input_ids": np.arange(12).reshape(3, 4),
                                                 "attention_mask": np.ones((3, 4))})
        self.logits = np.arange(18, dtype=np.float32).reshape(3, 6)
        self.model.session = Mock()
        self.model.session.run.return_value = [self.logits]

    def request(self, texts, *, cancelled=False):
        return SimpleNamespace(texts=np.asarray(texts, dtype=object).reshape(-1, 1), is_cancelled=lambda: cancelled)

    def test_multiple_requests_share_one_inference_and_receive_their_own_rows(self):
        responses = self.model.execute([self.request([b"one"]), self.request([b"two", b"three"])])
        self.model.tokenizer.assert_called_once_with(["one", "two", "three"], return_tensors="np",
                                                     truncation=True, padding=True, max_length=512)
        self.model.session.run.assert_called_once()
        np.testing.assert_array_equal(responses[0].output_tensors[0].data, self.logits[:1])
        np.testing.assert_array_equal(responses[1].output_tensors[0].data, self.logits[1:])
        self.assertEqual(self.model.session.run.call_args.args[1]["input_ids"].dtype, np.int64)

    def test_invalid_and_cancelled_requests_do_not_poison_valid_requests(self):
        responses = self.model.execute([self.request([b" "]), self.request([b"secret"], cancelled=True),
                                        self.request([b"one", b"two", b"three"])])
        self.assertEqual(responses[0].error.code, "invalid")
        self.assertEqual(responses[1].error.code, "cancelled")
        self.assertEqual(responses[2].output_tensors[0].data.shape, (3, 6))
        self.model.session.run.assert_called_once()

    def test_errors_never_expose_text_or_backend_exception(self):
        self.model.session.run.side_effect = RuntimeError("PRIVATE")
        responses = self.model.execute([self.request([b"one", b"two", b"three"])])
        self.assertNotIn("PRIVATE", responses[0].error.message)
        self.assertEqual(responses[0].error.code, "internal")

    def test_scheduler_ceiling_is_enforced(self):
        result = self.model.execute([self.request([b"one"] * 5), self.request([b"two"] * 4)])
        self.assertTrue(all(r.error.code == "internal" for r in result))
        self.model.session.run.assert_not_called()

    def test_startup_config_defaults_and_overrides(self):
        spec = importlib.util.spec_from_file_location("triton_entrypoint", "docker/triton/entrypoint.py")
        entrypoint = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entrypoint)
        with patch.dict("os.environ", {}, clear=True):
            text = entrypoint.render_config(Path.cwd())
            self.assertIn("max_batch_size: 8", text)
            self.assertIn("max_queue_delay_microseconds: 0", text)
            self.assertIn("count: 1 kind: KIND_CPU", text)
        with patch.dict("os.environ", {"TRITON_MAX_BATCH_SIZE": "4", "TRITON_QUEUE_DELAY_US": "2000"}):
            text = entrypoint.render_config(Path.cwd())
            self.assertIn("max_batch_size: 4", text)
            self.assertIn("max_queue_delay_microseconds: 2000", text)
        with patch.dict("os.environ", {"TRITON_MAX_BATCH_SIZE": "0"}), self.assertRaises(ValueError):
            entrypoint.render_config(Path.cwd())


@skipUnless(os.getenv("TOXICITY_REAL_MODEL_TESTS") == "1", "Opt-in real model validation")
class RealTritonAdapterTests(TestCase):
    def test_fp32_results_match_existing_classifier_for_mixed_length_texts(self):
        from app.resources import load_onnx_session
        from app.classifier import ToxicityOnnxClassifier
        from app.settings import RuntimeSettings
        from transformers import AutoTokenizer
        from tests.test_triton_runtime import model_config
        settings = RuntimeSettings(cpu_threads=1, inference_batch_size=8)
        config = model_config(settings)
        config["parameters"].update({"artifact_root": {"string_value": str(Path.cwd())},
                                      "cpu_threads": {"string_value": "1"}})
        module, _ = load_backend()
        adapter = module.TritonPythonModel()
        adapter.initialize({"model_config": json.dumps(config)})
        texts = ["Thanks for helping.", "You are an idiot.", "word " * 600,
                 "Bonjour, merci beaucoup.", " short ", "مرحبا بك", "An ordinary sentence.", "Stop insulting people."]
        requests = [SimpleNamespace(texts=np.asarray([[t.encode()]], dtype=object), is_cancelled=lambda: False)
                    for t in texts]
        responses = adapter.execute(requests)
        actual = np.concatenate([r.output_tensors[0].data for r in responses])
        classifier = ToxicityOnnxClassifier(load_onnx_session(settings),
            AutoTokenizer.from_pretrained(settings.model_dir, local_files_only=True),
            settings.thresholds, settings.labels, settings.max_length, inference_batch_size=8)
        expected = classifier._logits(texts)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
        actual_probabilities = 1 / (1 + np.exp(-actual.astype(np.float64)))
        for row, text in zip(actual_probabilities, texts):
            individual = classifier.predict(text)
            for label, threshold, probability in zip(settings.labels, settings.thresholds, row):
                self.assertAlmostEqual(probability, individual.labels[label].probability, delta=1e-5)
                self.assertEqual(probability >= threshold, individual.labels[label].predicted)
