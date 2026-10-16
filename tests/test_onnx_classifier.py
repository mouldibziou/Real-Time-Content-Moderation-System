"""Tests for ONNX classification inference and backend wiring.

The torch path is covered by the pre-existing suite; these tests pin the new
behavior: the ONNX classifier contract, the token_type_ids fix on the torch
classifier, settings validation, and ONNX session loading.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
import torch

from app.classifier import ToxicityClassifier, ToxicityOnnxClassifier
from app.settings import RuntimeSettings


LABELS = ("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")
THRESHOLDS = (0.74, 0.45, 0.74, 0.59, 0.61, 0.76)
# Fixed logits chosen so each label lands on a known side of its threshold.
LOGITS = [[2.0, -2.0, 0.0, 1.0, -0.5, 3.0]]
EXPECTED_PREDICTED = (True, False, False, True, False, True)


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=np.float64)))


class FakeSession:
    """Minimal onnxruntime stand-in: records feeds, replays fixed logits."""

    def __init__(self, logits):
        self.logits = np.asarray(logits, dtype=np.float32)
        self.feeds = []

    def run(self, output_names, feed):
        self.feeds.append(feed)
        rows = len(next(iter(feed.values())))
        return [np.tile(self.logits, (rows, 1))]


def numpy_tokenizer(texts, **kwargs):
    if isinstance(texts, str):
        texts = [texts]
    width = max(len(text) for text in texts)
    return {
        "input_ids": np.zeros((len(texts), width), dtype=np.int64),
        "attention_mask": np.ones((len(texts), width), dtype=np.int64),
    }


class OnnxClassifierTests(TestCase):
    def setUp(self):
        self.session = FakeSession(LOGITS)
        self.classifier = ToxicityOnnxClassifier(
            session=self.session,
            tokenizer=numpy_tokenizer,
            thresholds=THRESHOLDS,
            labels=LABELS,
            max_length=512,
        )

    def test_predict_returns_probabilities_and_thresholded_labels(self):
        result = self.classifier.predict("hello world")
        self.assertEqual(tuple(result.labels), LABELS)
        expected_probs = sigmoid(LOGITS)[0]
        for label, expected_prob, expected_pred in zip(LABELS, expected_probs, EXPECTED_PREDICTED):
            prediction = result.labels[label]
            self.assertAlmostEqual(prediction.probability, float(expected_prob), places=6)
            self.assertIs(prediction.predicted, expected_pred)

    def test_predict_batch_preserves_input_order(self):
        results = self.classifier.predict_batch(["first", "second"])
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertEqual(
                tuple(pred.predicted for pred in result.labels.values()),
                EXPECTED_PREDICTED,
            )

    def test_session_feed_contains_only_int64_inputs(self):
        self.classifier.predict("hello")
        feed = self.session.feeds[-1]
        self.assertEqual(set(feed), {"input_ids", "attention_mask"})
        for value in feed.values():
            self.assertEqual(value.dtype, np.int64)

    def test_configured_batch_size_controls_actual_session_feeds(self):
        texts = [str(index) for index in range(13)]
        for size in (1, 2, 4, 8, 64):
            with self.subTest(size=size):
                session = FakeSession(LOGITS)
                classifier = ToxicityOnnxClassifier(
                    session=session, tokenizer=numpy_tokenizer,
                    thresholds=THRESHOLDS, labels=LABELS, max_length=512,
                    inference_batch_size=size,
                )
                results = classifier.predict_batch(texts)
                self.assertEqual(len(results), len(texts))
                self.assertEqual(
                    [feed["input_ids"].shape[0] for feed in session.feeds],
                    [min(size, len(texts) - offset) for offset in range(0, len(texts), size)],
                )

    def test_chunked_batch_matches_individual_results_in_order(self):
        texts = [str(i) for i in range(13)]
        def logits(batch):
            return np.asarray([[float(text)] * len(LABELS) for text in batch])
        with patch.object(self.classifier, "_logits", side_effect=logits) as model:
            actual = self.classifier.predict_batch(texts)
            self.assertTrue(all(len(call.args[0]) <= 4 for call in model.call_args_list))
            expected = [self.classifier.predict(text) for text in texts]
        self.assertEqual(actual, expected)

    def test_parity_with_torch_classifier_on_same_logits(self):
        torch_logits = torch.tensor(LOGITS, dtype=torch.float32)
        model = Mock()
        model.return_value = SimpleNamespace(logits=torch_logits)

        def torch_tokenizer(texts, **kwargs):
            rows = 1 if isinstance(texts, str) else len(texts)
            return {
                "input_ids": torch.zeros(rows, 4, dtype=torch.long),
                "attention_mask": torch.ones(rows, 4, dtype=torch.long),
                # The saved tokenizer emits these; DistilBERT must not receive them.
                "token_type_ids": torch.zeros(rows, 4, dtype=torch.long),
            }

        torch_classifier = ToxicityClassifier(
            model=model,
            tokenizer=torch_tokenizer,
            thresholds=THRESHOLDS,
            labels=LABELS,
            device=torch.device("cpu"),
            max_length=512,
        )
        torch_result = torch_classifier.predict("hello world")
        onnx_result = self.classifier.predict("hello world")
        for label in LABELS:
            self.assertAlmostEqual(
                torch_result.labels[label].probability,
                onnx_result.labels[label].probability,
                places=6,
            )
            self.assertEqual(
                torch_result.labels[label].predicted,
                onnx_result.labels[label].predicted,
            )


class TorchTokenTypeIdsTests(TestCase):
    def test_configured_batch_size_preserves_torch_predictions_and_order(self):
        texts = [str(index) for index in range(13)]
        for size in (1, 2, 4, 8, 64):
            with self.subTest(size=size):
                batches = []

                def tokenizer(texts, **kwargs):
                    texts = [texts] if isinstance(texts, str) else texts
                    ids = torch.tensor([[int(text)] for text in texts])
                    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

                def model(**kwargs):
                    ids = kwargs["input_ids"]
                    batches.append(ids.shape[0])
                    return SimpleNamespace(logits=ids.float().repeat(1, len(LABELS)))

                classifier = ToxicityClassifier(
                    model=model, tokenizer=tokenizer, thresholds=THRESHOLDS,
                    labels=LABELS, device=torch.device("cpu"), max_length=512,
                    inference_batch_size=size,
                )
                actual = classifier.predict_batch(texts)
                self.assertEqual(batches, [min(size, len(texts) - offset)
                                          for offset in range(0, len(texts), size)])
                self.assertEqual(actual, [classifier.predict(text) for text in texts])

    def test_classifiers_reject_invalid_batch_sizes(self):
        for classifier_type, kwargs in (
            (ToxicityClassifier, {"model": Mock(), "device": torch.device("cpu")}),
            (ToxicityOnnxClassifier, {"session": FakeSession(LOGITS)}),
        ):
            for value in (0, -1, True, 2.5, "8", None):
                with self.subTest(classifier=classifier_type, value=value):
                    with self.assertRaisesRegex(ValueError, "Inference batch size"):
                        classifier_type(tokenizer=Mock(), thresholds=THRESHOLDS,
                                        labels=LABELS, max_length=512,
                                        inference_batch_size=value, **kwargs)

    def test_torch_classifier_drops_token_type_ids(self):
        received = {}

        def fake_model(**kwargs):
            received.update(kwargs)
            batch = kwargs["input_ids"].shape[0]
            return SimpleNamespace(logits=torch.zeros(batch, len(LABELS)))

        def tokenizer(texts, **kwargs):
            rows = 1 if isinstance(texts, str) else len(texts)
            return {
                "input_ids": torch.zeros(rows, 4, dtype=torch.long),
                "attention_mask": torch.ones(rows, 4, dtype=torch.long),
                "token_type_ids": torch.zeros(rows, 4, dtype=torch.long),
            }

        classifier = ToxicityClassifier(
            model=fake_model,
            tokenizer=tokenizer,
            thresholds=THRESHOLDS,
            labels=LABELS,
            device=torch.device("cpu"),
            max_length=512,
        )
        classifier.predict("hello")
        self.assertEqual(set(received), {"input_ids", "attention_mask"})
        classifier.predict_batch(["hello", "world"])
        self.assertEqual(set(received), {"input_ids", "attention_mask"})


class BackendSettingsTests(TestCase):
    def test_onnx_is_the_default_backend(self):
        settings = RuntimeSettings(device="cpu")
        self.assertEqual(settings.inference_backend, "onnx")
        self.assertEqual(settings.onnx_model_dir, Path("models/distilbert/v2.0.0-onnx"))
        self.assertEqual(settings.onnx_providers, ("CPUExecutionProvider",))

    def test_invalid_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Inference backend"):
            RuntimeSettings(device="cpu", inference_backend="tensorrt")

    def test_nonstring_backend_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "must be a string"):
            RuntimeSettings(device="cpu", inference_backend=None)

    def test_empty_providers_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            RuntimeSettings(device="cpu", onnx_providers=())

    def test_non_path_onnx_dir_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "pathlib.Path"):
            RuntimeSettings(device="cpu", onnx_model_dir="models/distilbert/v2.0.0-onnx")


class OnnxResourceLoadingTests(TestCase):
    def _torch_mocks(self):
        tokenizer = Mock(is_fast=True, pad_token_id=0)
        model = Mock()
        model.config.num_labels = len(LABELS)
        model.config.max_position_embeddings = 512
        model.config.model_type = "distilbert"
        model.config.id2label = dict(enumerate(LABELS))
        return tokenizer, model

    def test_missing_onnx_file_raises_helpful_error(self):
        from app.resources import load_resources

        directory = self.enterContext(TemporaryDirectory())
        settings = RuntimeSettings(
            model_dir=Path(directory),
            device="cpu",
            inference_backend="onnx",
            onnx_model_dir=Path(directory) / "missing-onnx",
        )
        tokenizer, model = self._torch_mocks()
        with patch("app.resources.AutoTokenizer.from_pretrained", return_value=tokenizer), \
                patch("app.resources.AutoModelForSequenceClassification.from_pretrained", return_value=model):
            with self.assertRaisesRegex(FileNotFoundError, "inference_backend='torch'"):
                load_resources(settings)

    def test_unexpected_session_inputs_are_rejected(self):
        from app.resources import load_onnx_session

        directory = self.enterContext(TemporaryDirectory())
        onnx_dir = Path(directory)
        (onnx_dir / "model.onnx").write_bytes(b"fake")
        settings = RuntimeSettings(
            device="cpu",
            inference_backend="onnx",
            onnx_model_dir=onnx_dir,
        )
        bad_input = SimpleNamespace(name="token_type_ids")
        session = Mock()
        session.get_inputs.return_value = [bad_input]
        with patch("app.resources.InferenceSession", return_value=session) as loader:
            with self.assertRaisesRegex(ValueError, "must include 'input_ids' and 'attention_mask'"):
                load_onnx_session(settings)
        loader.assert_called_once()

    def test_successful_load_returns_session(self):
        from app.resources import load_onnx_session

        directory = self.enterContext(TemporaryDirectory())
        onnx_dir = Path(directory)
        (onnx_dir / "model.onnx").write_bytes(b"fake")
        settings = RuntimeSettings(
            device="cpu",
            inference_backend="onnx",
            onnx_model_dir=onnx_dir,
        )
        session = Mock()
        session.get_inputs.return_value = [
            SimpleNamespace(name="input_ids"),
            SimpleNamespace(name="attention_mask"),
        ]
        with patch("app.resources.InferenceSession", return_value=session) as loader, \
             patch("app.resources.available_cpu_count", return_value=4):
            self.assertIs(load_onnx_session(settings), session)
            self.assertEqual(loader.call_args.kwargs["sess_options"].intra_op_num_threads, 4)
        settings = RuntimeSettings(device="cpu", onnx_model_dir=onnx_dir, cpu_threads=2)
        with patch("app.resources.InferenceSession", return_value=session) as loader:
            load_onnx_session(settings)
            self.assertEqual(loader.call_args.kwargs["sess_options"].intra_op_num_threads, 2)
