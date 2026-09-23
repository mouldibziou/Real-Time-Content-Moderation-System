"""Runtime contract tests; no model artifacts or accelerator are required."""
from concurrent.futures import Future
from threading import Condition, Event, Thread, current_thread
from unittest import TestCase
from unittest.mock import Mock

from app.classifier import ClassificationResult, ToxicityClassifier
from app.explainer import AttributionResult, ToxicityExplainer
from tests.factories import make_model_info
from app.runtime import ApplicationRuntime, RuntimeState
from app.services import ApplicationServices
from app.settings import RuntimeSettings


TIMEOUT = 5  # Failure bound, never used as a scheduling delay.
OPERATIONS = (
    ("predict", ("hello",), "classifier"),
    ("predict_batch", (["hello", "world"],), "classifier"),
    ("explain", ("hello", "toxic"), "explainer"),
    ("explain_batch", (["hello", "world"], "toxic"), "explainer"),
)


class ObservedCondition(Condition):
    """Expose wait entry for deterministic scheduling, retaining real locking."""

    def __init__(self):
        super().__init__()
        self.waiting = {name: Event() for name in ("queued", "shutdown")}

    def wait(self, timeout=None):
        event = self.waiting.get(current_thread().name)
        if event is not None:
            event.set()
        return super().wait(timeout)


class ApplicationRuntimeTests(TestCase):
    def setUp(self):
        self.settings = RuntimeSettings(device="cpu", max_prediction_batch_size=4, max_explanation_batch_size=4)
        self.classifier = Mock(spec=ToxicityClassifier)
        self.explainer = Mock(spec=ToxicityExplainer)
        self.classifier.predict.return_value = ClassificationResult(labels={})
        self.classifier.predict_batch.return_value = [ClassificationResult(labels={})]
        self.explainer.explain.return_value = AttributionResult(label="toxic", spans=())
        self.explainer.explain_batch.return_value = [AttributionResult(label="toxic", spans=())]
        self.services = ApplicationServices(self.classifier, self.explainer, model_info=make_model_info())
        self.builder = Mock(return_value=self.services)
        self.runtime = ApplicationRuntime(self.settings, services_builder=self.builder)
        self.condition = ObservedCondition()
        self.runtime._condition = self.condition
        self.threads = []
        self.releases = []
        self.addCleanup(self.cleanup_threads)

    def cleanup_threads(self):
        # Always release controlled service calls, including on assertion failure.
        for event in self.releases:
            event.set()
        for thread in self.threads:
            thread.join(TIMEOUT)
        self.assertFalse(any(thread.is_alive() for thread in self.threads), "Worker did not finish")

    def launch(self, name, function, *args):
        future = Future()

        def run():
            try:
                future.set_result(function(*args))
            except BaseException as error:
                future.set_exception(error)

        thread = Thread(target=run, name=name, daemon=True)
        self.threads.append(thread)
        thread.start()
        return future

    def block_call(self, method, error=None):
        entered, release = Event(), Event()
        self.releases.append(release)
        result = method.return_value

        def blocked(*args):
            entered.set()
            if not release.wait(TIMEOUT):
                raise TimeoutError("Test did not release the service call")
            if error is not None:
                raise error
            return result

        method.side_effect = blocked
        return entered, release

    def test_construction_is_lazy_and_start_is_idempotent(self):
        self.builder.assert_not_called()
        self.assertFalse(self.runtime.is_ready())
        self.runtime.start()
        self.runtime.start()
        self.builder.assert_called_once_with(self.settings)
        self.assertTrue(self.runtime.is_ready())

    def test_failed_start_preserves_error_and_allows_retry(self):
        error = FileNotFoundError("missing artifact")
        self.builder.side_effect = [error, self.services]
        with self.assertRaises(FileNotFoundError) as caught:
            self.runtime.start()
        self.assertIs(caught.exception, error)
        self.assertFalse(self.runtime.is_ready())
        self.assertIsNone(self.runtime._services)
        self.runtime.start()
        self.assertTrue(self.runtime.is_ready())
        self.assertEqual(self.builder.call_count, 2)

    def test_operations_reject_before_start_and_after_stop(self):
        for phase in ("before start", "after stop"):
            if phase == "after stop":
                self.runtime.start()
                self.runtime.stop()
            for name, args, owner in OPERATIONS:
                with self.subTest(phase=phase, operation=name):
                    with self.assertRaisesRegex(RuntimeError, "not ready"):
                        getattr(self.runtime, name)(*args)
                    getattr(getattr(self, owner), name).assert_not_called()
                    self.assertFalse(self.runtime._operation_active)

    def test_all_operations_forward_arguments_and_preserve_results(self):
        self.runtime.start()
        for name, args, owner in OPERATIONS:
            with self.subTest(operation=name):
                method = getattr(getattr(self, owner), name)
                self.assertIs(getattr(self.runtime, name)(*args), method.return_value)
                method.assert_called_once_with(*args)
                self.assertFalse(self.runtime._operation_active)
        self.builder.assert_called_once_with(self.settings)

    def test_all_service_failures_release_reservation(self):
        self.runtime.start()
        for name, args, owner in OPERATIONS:
            with self.subTest(operation=name):
                method = getattr(getattr(self, owner), name)
                error = ValueError("service failure")
                method.side_effect = error
                with self.assertRaises(ValueError) as caught:
                    getattr(self.runtime, name)(*args)
                self.assertIs(caught.exception, error)
                self.assertFalse(self.runtime._operation_active)
                self.assertTrue(self.runtime.is_ready())
                method.side_effect = None
                self.assertIs(getattr(self.runtime, name)(*args), method.return_value)

    def test_stop_is_idempotent_and_restart_uses_new_services(self):
        self.runtime.stop()
        self.builder.assert_not_called()
        self.runtime.start()
        self.runtime.stop()
        self.runtime.stop()
        self.assertFalse(self.runtime.is_ready())
        self.assertIsNone(self.runtime._services)
        replacement = ApplicationServices(Mock(spec=ToxicityClassifier), Mock(spec=ToxicityExplainer), model_info=make_model_info())
        self.builder.return_value = replacement
        self.runtime.start()
        self.assertIs(self.runtime.predict("hello"), replacement.classifier.predict.return_value)
        self.classifier.predict.assert_not_called()
        self.assertEqual(self.builder.call_count, 2)

    def test_starting_rejects_overlapping_lifecycle_calls(self):
        entered, release = self.block_call(self.builder)
        startup = self.launch("startup", self.runtime.start)
        self.assertTrue(entered.wait(TIMEOUT))
        # Complete this inspection on another worker to bound a lock regression.
        inspection = self.launch("inspection", self.runtime.is_ready)
        self.assertFalse(inspection.result(TIMEOUT))
        for name in ("start", "stop"):
            with self.assertRaises(RuntimeError):
                self.launch(name, getattr(self.runtime, name)).result(TIMEOUT)
        self.builder.assert_called_once_with(self.settings)
        release.set()
        startup.result(TIMEOUT)
        self.assertTrue(self.runtime.is_ready())

    def test_every_pair_of_operations_shares_one_reservation(self):
        self.runtime.start()
        for first_name, first_args, first_owner in OPERATIONS:
            for next_name, next_args, next_owner in OPERATIONS:
                with self.subTest(active=first_name, queued=next_name):
                    first_method = getattr(getattr(self, first_owner), first_name)
                    next_method = getattr(getattr(self, next_owner), next_name)
                    first_method.reset_mock()
                    if next_method is not first_method:
                        next_method.reset_mock()
                    self.condition.waiting["queued"].clear()
                    entered, release = self.block_call(first_method)
                    active = self.launch("active", getattr(self.runtime, first_name), *first_args)
                    self.assertTrue(entered.wait(TIMEOUT))
                    queued = self.launch("queued", getattr(self.runtime, next_name), *next_args)
                    self.assertTrue(self.condition.waiting["queued"].wait(TIMEOUT))
                    with self.condition:
                        self.assertEqual(next_method.call_count, int(next_method is first_method))
                    release.set()
                    active.result(TIMEOUT)
                    queued.result(TIMEOUT)
                    first_method.side_effect = None

    def test_shutdown_drains_active_work_and_rejects_waiting_and_new_work(self):
        self.runtime.start()
        entered, release = self.block_call(self.explainer.explain)
        active = self.launch("active", self.runtime.explain, "hello", "toxic")
        self.assertTrue(entered.wait(TIMEOUT))
        queued = self.launch("queued", self.runtime.predict, "queued text")
        self.assertTrue(self.condition.waiting["queued"].wait(TIMEOUT))
        shutdown = self.launch("shutdown", self.runtime.stop)
        self.assertTrue(self.condition.waiting["shutdown"].wait(TIMEOUT))
        with self.condition:
            self.assertIs(self.runtime._state, RuntimeState.STOPPING)
            self.assertIs(self.runtime._services, self.services)
        self.assertFalse(self.runtime.is_ready())
        self.assertFalse(shutdown.done())
        # The waiting request must be rejected before active work is released.
        with self.assertRaisesRegex(RuntimeError, "not ready"):
            queued.result(TIMEOUT)
        self.classifier.predict.assert_not_called()
        for name, args, _ in OPERATIONS:
            with self.subTest(operation=name):
                with self.assertRaisesRegex(RuntimeError, "not ready"):
                    getattr(self.runtime, name)(*args)
        for name in ("start", "stop"):
            with self.assertRaises(RuntimeError):
                getattr(self.runtime, name)()
        release.set()
        active.result(TIMEOUT)
        shutdown.result(TIMEOUT)
        self.assertFalse(self.runtime.is_ready())
        self.assertIsNone(self.runtime._services)

    def test_shutdown_finishes_even_when_active_operation_fails(self):
        self.runtime.start()
        error = ValueError("explanation failed")
        entered, release = self.block_call(self.explainer.explain, error=error)
        active = self.launch("active", self.runtime.explain, "hello", "toxic")
        self.assertTrue(entered.wait(TIMEOUT))
        shutdown = self.launch("shutdown", self.runtime.stop)
        self.assertTrue(self.condition.waiting["shutdown"].wait(TIMEOUT))
        release.set()
        with self.assertRaises(ValueError) as caught:
            active.result(TIMEOUT)
        self.assertIs(caught.exception, error)
        shutdown.result(TIMEOUT)
        self.assertFalse(self.runtime._operation_active)
        self.assertFalse(self.runtime.is_ready())


class SingleTextValidationTests(TestCase):
    def make_runtime(self):
        classifier = Mock(spec=ToxicityClassifier)
        explainer = Mock(spec=ToxicityExplainer)
        services = ApplicationServices(classifier, explainer, model_info=make_model_info())
        builder = Mock(return_value=services)
        return ApplicationRuntime(RuntimeSettings(device="cpu", max_prediction_batch_size=4, max_explanation_batch_size=4), services_builder=builder), services, builder

    def test_invalid_text_does_not_leave_a_reservation_or_call_services(self):
        invalid_inputs = ((None, TypeError), (123, TypeError), ([], TypeError),
                          ("", ValueError), (" \t\n", ValueError))
        for name in ("predict", "explain"):
            for text, expected_error in invalid_inputs:
                with self.subTest(operation=name, text=text):
                    runtime, services, _ = self.make_runtime()
                    runtime.start()
                    args = (text,) if name == "predict" else (text, "toxic")
                    with self.assertRaises(expected_error):
                        getattr(runtime, name)(*args)
                    services.classifier.predict.assert_not_called()
                    services.explainer.explain.assert_not_called()
                    # Do not call stop on a leaked reservation: it would hang.
                    self.assertFalse(runtime._operation_active)

    def test_text_validation_precedes_readiness(self):
        for name in ("predict", "explain"):
            for text, expected_error in ((None, TypeError), (" \t", ValueError)):
                with self.subTest(operation=name, text=text):
                    runtime, _, builder = self.make_runtime()
                    args = (text,) if name == "predict" else (text, "toxic")
                    with self.assertRaises(expected_error):
                        getattr(runtime, name)(*args)
                    builder.assert_not_called()

    def test_valid_text_is_forwarded_without_normalization(self):
        for name, owner in (("predict", "classifier"), ("explain", "explainer")):
            with self.subTest(operation=name):
                runtime, services, _ = self.make_runtime()
                runtime.start()
                text = "  café\nhello 🙂  "
                args = (text,) if name == "predict" else (text, "toxic")
                method = getattr(getattr(services, owner), name)
                self.assertIs(getattr(runtime, name)(*args), method.return_value)
                method.assert_called_once_with(*args)
                self.assertFalse(runtime._operation_active)


class BatchTextValidationTests(TestCase):
    def make_runtime(self):
        services = ApplicationServices(Mock(spec=ToxicityClassifier), Mock(spec=ToxicityExplainer), model_info=make_model_info())
        builder = Mock(return_value=services)
        return ApplicationRuntime(RuntimeSettings(device="cpu", max_prediction_batch_size=4, max_explanation_batch_size=4), services_builder=builder), services

    def test_invalid_batches_are_rejected_before_model_reservation(self):
        cases = (
            (None, TypeError), ("hello", TypeError), (("hello",), TypeError),
            ([], ValueError), (["valid", None], TypeError),
            (["valid", 123], TypeError), (["valid", ""], ValueError),
            (["valid", " \t\n"], ValueError),
        )
        for name in ("predict_batch", "explain_batch"):
            for texts, expected in cases:
                with self.subTest(operation=name, texts=texts):
                    runtime, services = self.make_runtime()
                    runtime.start()
                    # A sentinel proves validation runs before any reservation.
                    runtime._begin_operation = Mock(side_effect=AssertionError("Reserved invalid input"))
                    args = (texts,) if name == "predict_batch" else (texts, "toxic")
                    with self.assertRaises(expected):
                        getattr(runtime, name)(*args)
                    runtime._begin_operation.assert_not_called()
                    services.classifier.predict_batch.assert_not_called()
                    services.explainer.explain_batch.assert_not_called()

    def test_empty_batch_validation_precedes_readiness(self):
        for name in ("predict_batch", "explain_batch"):
            with self.subTest(operation=name):
                runtime, _ = self.make_runtime()
                args = ([],) if name == "predict_batch" else ([], "toxic")
                with self.assertRaises(ValueError):
                    getattr(runtime, name)(*args)

    def test_valid_batches_preserve_order_text_and_duplicates(self):
        for name, owner in (("predict_batch", "classifier"), ("explain_batch", "explainer")):
            with self.subTest(operation=name):
                runtime, services = self.make_runtime()
                runtime.start()
                texts = ["  café  ", "hello\n🙂", "  café  "]
                original = texts.copy()
                args = (texts,) if name == "predict_batch" else (texts, "toxic")
                method = getattr(getattr(services, owner), name)
                self.assertIs(getattr(runtime, name)(*args), method.return_value)
                method.assert_called_once_with(*args)
                self.assertEqual(texts, original)
                self.assertFalse(runtime._operation_active)
