from concurrent.futures import Future
from dataclasses import FrozenInstanceError, asdict
from threading import Event, Thread
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import torch

from app.metadata import ModelInfo
from app.runtime import ApplicationRuntime, RuntimeState
from app.services import ApplicationServices, build_services
from app.settings import RuntimeSettings
from tests.factories import make_model_info


class ModelInfoTests(TestCase):
    def setUp(self):
        self.info = make_model_info()
        self.services = ApplicationServices(Mock(), Mock(), self.info)
        self.builder = Mock(return_value=self.services)
        self.runtime = ApplicationRuntime(RuntimeSettings(device="cpu"), services_builder=self.builder)

    def test_before_start_and_after_stop_raise_not_ready(self):
        for phase in ("before", "after"):
            with self.subTest(phase=phase):
                if phase == "after":
                    self.runtime.start()
                    self.runtime.stop()
                with self.assertRaisesRegex(RuntimeError, "not ready"):
                    self.runtime.model_info()

    def test_starting_and_stopping_reject_even_when_services_exist(self):
        self.runtime.start()
        for state in (RuntimeState.STARTING, RuntimeState.STOPPING):
            with self.subTest(state=state):
                with self.runtime._condition:
                    self.runtime._state = state
                with self.assertRaisesRegex(RuntimeError, "not ready"):
                    self.runtime.model_info()

    def test_ready_returns_frozen_metadata_without_reservation(self):
        self.runtime.start()
        with patch.object(self.runtime, '_begin_operation', side_effect=AssertionError("Must not reserve model")):
            result = self.runtime.model_info()
        self.assertIs(result, self.info)
        with self.assertRaises(FrozenInstanceError):
            result.device = "cuda:0"
        self.runtime.stop()
        self.assertEqual(result.device, "cpu")
        self.assertEqual(set(asdict(result)), {
            "model_type", "device", "labels", "thresholds", "max_length", "explanation_steps",
            "inference_backend", "max_text_chars", "max_prediction_batch_size",
        })

    def test_metadata_returns_while_explanation_is_running(self):
        entered, release = Event(), Event()
        active_result, info_result = Future(), Future()

        def explain(*args):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("Test release missing")

        def capture(future, function, *args):
            try:
                future.set_result(function(*args))
            except BaseException as error:
                future.set_exception(error)

        self.services.explainer.explain.side_effect = explain
        self.runtime.start()
        active = Thread(target=capture, args=(active_result, self.runtime.explain, "hello", "toxic"), daemon=True)
        reader = Thread(target=capture, args=(info_result, self.runtime.model_info), daemon=True)
        try:
            active.start()
            self.assertTrue(entered.wait(5))
            reader.start()
            self.assertIs(info_result.result(timeout=2), self.info)
            self.assertFalse(active_result.done())
            release.set()
            active_result.result(timeout=5)
        finally:
            release.set()
            active.join(5)
            if reader.ident is not None:
                reader.join(5)
            self.assertFalse(active.is_alive())
            self.assertFalse(reader.is_alive())

    def test_builder_uses_loaded_model_type_and_resolved_device(self):
        settings = RuntimeSettings(device="cuda", labels=("a", "b"), thresholds=(0.2, 0.8),
                                   max_length=128, n_steps=7, inference_backend="torch",
                                   inference_batch_size=2)
        resources = SimpleNamespace(
            model=SimpleNamespace(config=SimpleNamespace(model_type="distilbert")),
            tokenizer=object(), device=torch.device("cuda:1"), ort_session=None,
        )
        with patch('app.services.load_resources', return_value=resources) as loader, \
                patch('app.services.ToxicityClassifier') as classifier, \
                patch('app.services.ToxicityExplainer') as explainer:
            services = build_services(settings)
        loader.assert_called_once_with(settings)
        self.assertEqual(services.model_info, ModelInfo(
            model_type="distilbert", device="cuda:1", labels=("a", "b"),
            thresholds=(0.2, 0.8), max_length=128, explanation_steps=7,
            inference_backend="torch",
        ))
        self.assertIs(services.classifier, classifier.return_value)
        self.assertIs(services.explainer, explainer.return_value)
        self.assertIs(classifier.call_args.kwargs['model'], resources.model)
        self.assertEqual(classifier.call_args.kwargs['inference_batch_size'], 2)
        self.assertIs(explainer.call_args.kwargs['model'], resources.model)

    def test_builder_wires_onnx_classifier_by_default(self):
        settings = RuntimeSettings(device="cpu", inference_batch_size=8)
        self.assertEqual(settings.inference_backend, "onnx")
        session = object()
        resources = SimpleNamespace(
            model=SimpleNamespace(config=SimpleNamespace(model_type="distilbert")),
            tokenizer=object(), device=torch.device("cpu"), ort_session=session,
        )
        with patch('app.services.load_resources', return_value=resources), \
                patch('app.services.ToxicityOnnxClassifier') as onnx_classifier, \
                patch('app.services.ToxicityExplainer') as explainer:
            services = build_services(settings)
        self.assertIs(services.classifier, onnx_classifier.return_value)
        self.assertIs(services.explainer, explainer.return_value)
        self.assertIs(onnx_classifier.call_args.kwargs['session'], session)
        self.assertIs(onnx_classifier.call_args.kwargs['tokenizer'], resources.tokenizer)
        self.assertEqual(onnx_classifier.call_args.kwargs['inference_batch_size'], 8)
        # The explainer always keeps the torch model, which ONNX cannot replace.
        self.assertIs(explainer.call_args.kwargs['model'], resources.model)
        self.assertEqual(services.model_info.inference_backend, "onnx")

    def test_builder_rejects_onnx_without_loaded_session(self):
        settings = RuntimeSettings(device="cpu", inference_backend="onnx")
        resources = SimpleNamespace(
            model=SimpleNamespace(config=SimpleNamespace(model_type="distilbert")),
            tokenizer=object(), device=torch.device("cpu"), ort_session=None,
        )
        with patch('app.services.load_resources', return_value=resources):
            with self.assertRaisesRegex(RuntimeError, "no ONNX session was loaded"):
                build_services(settings)
