"""Configuration and public runtime input contracts, without model loading."""
from unittest import TestCase
from unittest.mock import Mock

from app.classifier import ToxicityClassifier
from app.explainer import ToxicityExplainer
from tests.factories import make_model_info
from app.runtime import ApplicationRuntime
from app.services import ApplicationServices
from app.settings import RuntimeSettings


class IntegerSettingsTests(TestCase):
    fields = ("max_length", "n_steps", "max_text_chars",
              "max_prediction_batch_size", "max_explanation_batch_size")

    def test_integer_settings_reject_wrong_types_including_booleans(self):
        for field in self.fields:
            for value in (True, False, 1.5, "2", None):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(TypeError):
                        RuntimeSettings(**{field: value})

    def test_integer_settings_reject_zero_and_negative_values(self):
        for field in self.fields:
            for value in (0, -1):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        RuntimeSettings(**{field: value})

    def test_integer_settings_accept_one(self):
        settings = RuntimeSettings(**{field: 1 for field in self.fields})
        for field in self.fields:
            self.assertEqual(getattr(settings, field), 1)


class RuntimeLimitAndLabelTests(TestCase):
    def setUp(self):
        self.settings = RuntimeSettings(
            device="cpu", max_text_chars=5,
            max_prediction_batch_size=3, max_explanation_batch_size=2,
        )
        self.services = ApplicationServices(Mock(spec=ToxicityClassifier), Mock(spec=ToxicityExplainer), model_info=make_model_info())
        self.runtime = ApplicationRuntime(self.settings, services_builder=Mock(return_value=self.services))
        self.runtime.start()
        # Keep the real reservation behavior, while checking invalid inputs never reach it.
        self.runtime._begin_operation = Mock(wraps=self.runtime._begin_operation)

    def assert_rejected(self, name, args, error_type, pattern=None):
        begin = self.runtime._begin_operation
        begin.reset_mock()
        self.services.classifier.reset_mock()
        self.services.explainer.reset_mock()
        if pattern is None:
            with self.assertRaises(error_type):
                getattr(self.runtime, name)(*args)
        else:
            with self.assertRaisesRegex(error_type, pattern):
                getattr(self.runtime, name)(*args)
        begin.assert_not_called()
        self.assertEqual(self.services.classifier.mock_calls, [])
        self.assertEqual(self.services.explainer.mock_calls, [])
        self.assertFalse(self.runtime._operation_active)

    def test_character_limit_accepts_boundary_without_modifying_text(self):
        text = " é🙂  "  # Five Python code points, including surrounding spaces.
        cases = (("predict", (text,), self.services.classifier),
                 ("explain", (text, "toxic"), self.services.explainer),
                 ("predict_batch", ([text],), self.services.classifier),
                 ("explain_batch", ([text], "toxic"), self.services.explainer))
        for name, args, service in cases:
            with self.subTest(operation=name):
                method = getattr(service, name)
                self.assertIs(getattr(self.runtime, name)(*args), method.return_value)
                method.assert_called_once_with(*args)

    def test_character_limit_rejects_oversize_before_reservation(self):
        for name in ("predict", "explain", "predict_batch", "explain_batch"):
            with self.subTest(operation=name):
                value = ["ok", "abcdef"] if name.endswith("batch") else "abcdef"
                args = (value, "toxic") if name.startswith("explain") else (value,)
                self.assert_rejected(name, args, ValueError, "at most 5")

    def test_separate_batch_limits_accept_boundary_and_reject_one_more(self):
        for name, limit, service in (("predict_batch", 3, self.services.classifier),
                                      ("explain_batch", 2, self.services.explainer)):
            with self.subTest(operation=name):
                texts = ["ok"] * limit
                args = (texts, "toxic") if name == "explain_batch" else (texts,)
                method = getattr(service, name)
                self.assertIs(getattr(self.runtime, name)(*args), method.return_value)
                method.assert_called_once_with(*args)
                oversized = ["ok"] * (limit + 1)
                args = (oversized, "toxic") if name == "explain_batch" else (oversized,)
                self.assert_rejected(name, args, ValueError, f"at most {limit}")

    def test_indexed_item_errors_preserve_exception_type(self):
        for name in ("predict_batch", "explain_batch"):
            for invalid, error in ((None, TypeError), (123, TypeError), (" ", ValueError)):
                with self.subTest(operation=name, invalid=invalid):
                    texts = ["ok", invalid]
                    args = (texts, "toxic") if name == "explain_batch" else (texts,)
                    self.assert_rejected(name, args, error, r"(?i)texts\[1\]")

    def test_invalid_labels_are_rejected_before_reservation(self):
        for name in ("explain", "explain_batch"):
            for label, error in ((None, TypeError), (1, TypeError), ([], TypeError),
                                 ("", ValueError), ("TOXIC", ValueError),
                                 (" toxic ", ValueError), ("unknown", ValueError)):
                with self.subTest(operation=name, label=label):
                    value = ["hello"] if name.endswith("batch") else "hello"
                    self.assert_rejected(name, (value, label), error)

    def test_valid_labels_reach_services_unchanged(self):
        for name in ("explain", "explain_batch"):
            for label in self.settings.labels:
                with self.subTest(operation=name, label=label):
                    method = getattr(self.services.explainer, name)
                    method.reset_mock()
                    value = ["hello"] if name.endswith("batch") else "hello"
                    self.assertIs(getattr(self.runtime, name)(value, label), method.return_value)
                    method.assert_called_once_with(value, label)


class OperationWaitTimeoutSettingsTests(TestCase):
    def test_positive_finite_durations_are_accepted(self):
        for value in (0.5, 1, 30.0):
            with self.subTest(value=value):
                settings = RuntimeSettings(operation_wait_timeout_seconds=value)
                self.assertEqual(settings.operation_wait_timeout_seconds, value)

    def test_invalid_duration_values_are_rejected(self):
        for value in (0, -1, float('nan'), float('inf'), float('-inf')):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    RuntimeSettings(operation_wait_timeout_seconds=value)

    def test_wrong_duration_types_are_rejected(self):
        for value in (True, False, None, '30', []):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    RuntimeSettings(operation_wait_timeout_seconds=value)
