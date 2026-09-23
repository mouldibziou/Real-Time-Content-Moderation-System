"""Read-only status snapshots, including during active model execution."""
from concurrent.futures import Future
from dataclasses import FrozenInstanceError
from threading import Event, Thread
from unittest import TestCase
from unittest.mock import Mock

from tests.factories import make_model_info
from app.runtime import ApplicationRuntime, RuntimeState
from app.scheduling import OperationKind, WaitingOperation
from app.services import ApplicationServices
from app.settings import RuntimeSettings


class RuntimeStatusTests(TestCase):
    def setUp(self):
        self.services = ApplicationServices(Mock(), Mock(), model_info=make_model_info())
        self.builder = Mock(return_value=self.services)
        self.runtime = ApplicationRuntime(
            RuntimeSettings(device="cpu", max_waiting_classifications=3, max_waiting_explanations=2),
            services_builder=self.builder,
        )

    def test_stopped_status_needs_no_services_and_reports_capacities(self):
        status = self.runtime.status()
        self.builder.assert_not_called()
        self.assertIs(status.state, RuntimeState.STOPPED)
        self.assertFalse(status.operation_active)
        self.assertEqual(status.waiting_classifications, 0)
        self.assertEqual(status.waiting_explanations, 0)
        self.assertEqual(status.classification_queue_capacity, 3)
        self.assertEqual(status.explanation_queue_capacity, 2)

    def test_snapshot_is_frozen_and_independent_of_later_changes(self):
        old = self.runtime.status()
        with self.assertRaises(FrozenInstanceError):
            old.operation_active = True
        self.runtime.start()
        with self.runtime._condition:
            self.runtime._classification_queue.append(WaitingOperation(OperationKind.CLASSIFICATION, 100))
        new = self.runtime.status()
        self.assertIs(old.state, RuntimeState.STOPPED)
        self.assertEqual(old.waiting_classifications, 0)
        self.assertIs(new.state, RuntimeState.READY)
        self.assertEqual(new.waiting_classifications, 1)

    def test_status_reports_every_state_without_requiring_services(self):
        for state in RuntimeState:
            with self.subTest(state=state):
                with self.runtime._condition:
                    self.runtime._state = state
                self.assertIs(self.runtime.status().state, state)

    def test_status_completes_while_explanation_is_still_running(self):
        entered, release = Event(), Event()
        active_result, status_result = Future(), Future()

        def explain(text, label):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("Test did not release explanation")
            return "done"

        self.services.explainer.explain.side_effect = explain
        self.runtime.start()

        def capture(future, function, *args):
            try:
                future.set_result(function(*args))
            except BaseException as error:
                future.set_exception(error)

        active = Thread(target=capture, args=(active_result, self.runtime.explain, "hello", "toxic"), daemon=True)
        reader = Thread(target=capture, args=(status_result, self.runtime.status), daemon=True)
        try:
            active.start()
            self.assertTrue(entered.wait(5))
            reader.start()
            status = status_result.result(timeout=2)
            self.assertFalse(active_result.done())
            self.assertIs(status.state, RuntimeState.READY)
            self.assertTrue(status.operation_active)
            self.assertEqual(status.waiting_explanations, 0)
            release.set()
            self.assertEqual(active_result.result(timeout=5), "done")
            self.assertTrue(status.operation_active)  # Earlier observation remains unchanged.
            self.assertFalse(self.runtime.status().operation_active)
        finally:
            release.set()
            active.join(5)
            if reader.ident is not None:
                reader.join(5)
            self.assertFalse(active.is_alive())
            self.assertFalse(reader.is_alive())
