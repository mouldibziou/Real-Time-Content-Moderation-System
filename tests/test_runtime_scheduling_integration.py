"""Ticket lifecycle checks with controlled clock/wakeups, without model loading."""
from unittest import TestCase
from unittest.mock import Mock, patch

from app.runtime import ApplicationRuntime, RuntimeBusyError, RuntimeState
from app.scheduling import OperationKind, WaitingOperation
from app.services import ApplicationServices
from app.settings import RuntimeSettings


class TicketIntegrationTests(TestCase):
    def setUp(self):
        self.services = ApplicationServices(Mock(), Mock())
        self.runtime = ApplicationRuntime(
            RuntimeSettings(device="cpu", max_waiting_classifications=2,
                            max_waiting_explanations=1, classification_burst_limit=2),
            services_builder=Mock(return_value=self.services),
        )
        self.runtime.start()
        self.enterContext(patch('app.runtime.monotonic', return_value=100.0))

    def test_free_model_claims_and_removes_ticket(self):
        for kind in OperationKind:
            with self.subTest(kind=kind):
                with patch.object(self.runtime._condition, 'wait') as wait:
                    self.assertIs(self.runtime._begin_operation(kind), self.services)
                wait.assert_not_called()
                self.assertTrue(self.runtime._operation_active)
                self.assertFalse(self.runtime._classification_queue)
                self.assertFalse(self.runtime._explanation_queue)
                self.runtime._end_operation()
                self.assertEqual(self.runtime._consecutive_classifications, 0)

    def test_full_queue_rejection_preserves_existing_ticket(self):
        with self.runtime._condition:
            existing = self.runtime._enqueue_operation(OperationKind.EXPLANATION, deadline=130)
            self.runtime._operation_active = True
        with self.assertRaises(RuntimeBusyError):
            self.runtime._begin_operation(OperationKind.EXPLANATION)
        self.assertEqual(list(self.runtime._explanation_queue), [existing])
        self.assertTrue(self.runtime._operation_active)

    def test_wait_failure_removes_own_ticket_only_and_notifies(self):
        self.runtime._operation_active = True
        earlier = WaitingOperation(OperationKind.CLASSIFICATION, 130)
        self.runtime._classification_queue.append(earlier)
        error = RuntimeError('wait interrupted')
        with patch.object(self.runtime._condition, 'wait', side_effect=error), \
                patch.object(self.runtime._condition, 'notify_all') as notify:
            with self.assertRaises(RuntimeError) as caught:
                self.runtime._begin_operation(OperationKind.CLASSIFICATION)
        self.assertIs(caught.exception, error)
        self.assertEqual(list(self.runtime._classification_queue), [earlier])
        self.assertTrue(self.runtime._operation_active)
        self.assertGreaterEqual(notify.call_count, 2)  # Admission and removal.

    def test_stop_waits_for_each_kind_of_outstanding_work(self):
        for outstanding in ('active', 'classification', 'explanation'):
            with self.subTest(outstanding=outstanding):
                self.runtime._state = RuntimeState.READY
                self.runtime._services = self.services
                self.runtime._operation_active = outstanding == 'active'
                self.runtime._classification_queue.clear()
                self.runtime._explanation_queue.clear()
                if outstanding == 'classification':
                    self.runtime._classification_queue.append(WaitingOperation(OperationKind.CLASSIFICATION, 130))
                if outstanding == 'explanation':
                    self.runtime._explanation_queue.append(WaitingOperation(OperationKind.EXPLANATION, 130))

                def finish_outstanding_work():
                    self.assertIs(self.runtime._state, RuntimeState.STOPPING)
                    self.assertIs(self.runtime._services, self.services)
                    self.runtime._operation_active = False
                    self.runtime._classification_queue.clear()
                    self.runtime._explanation_queue.clear()

                with patch.object(self.runtime._condition, 'wait', side_effect=finish_outstanding_work) as wait:
                    self.runtime.stop()
                wait.assert_called_once_with()
                self.assertIs(self.runtime._state, RuntimeState.STOPPED)
                self.assertIsNone(self.runtime._services)

    def test_stop_resets_burst_counter(self):
        self.runtime._consecutive_classifications = 2
        self.runtime.stop()
        self.assertEqual(self.runtime._consecutive_classifications, 0)


class ThreadedSchedulingOrderTests(TestCase):
    def test_fifo_and_classification_burst_through_public_operations(self):
        from concurrent.futures import Future
        from threading import Event, Thread

        entered, release = Event(), Event()
        execution_order = []
        classifier, explainer = Mock(), Mock()

        def predict(text):
            execution_order.append(text)
            if text == 'active':
                entered.set()
                if not release.wait(5):
                    raise TimeoutError('Test did not release active operation')
            return text

        def explain(text, label):
            execution_order.append(text)
            return text

        classifier.predict.side_effect = predict
        explainer.explain.side_effect = explain
        runtime = ApplicationRuntime(
            RuntimeSettings(device='cpu', classification_burst_limit=2,
                            max_waiting_classifications=3, max_waiting_explanations=2),
            services_builder=Mock(return_value=ApplicationServices(classifier, explainer)),
        )
        runtime.start()
        threads, futures = [], []

        def launch(function, *args):
            future = Future()

            def run():
                try:
                    future.set_result(function(*args))
                except BaseException as error:
                    future.set_exception(error)

            thread = Thread(target=run, daemon=True)
            threads.append(thread)
            futures.append(future)
            thread.start()

        try:
            launch(runtime.predict, 'active')
            self.assertTrue(entered.wait(5))
            # Admit explanations first; classification priority must still apply.
            for index in range(1, 3):
                launch(runtime.explain, f'E{index}', 'toxic')
                with runtime._condition:
                    self.assertTrue(runtime._condition.wait_for(
                        lambda: len(runtime._explanation_queue) == index, timeout=5))
            for index in range(1, 4):
                launch(runtime.predict, f'C{index}')
                with runtime._condition:
                    self.assertTrue(runtime._condition.wait_for(
                        lambda: len(runtime._classification_queue) == index, timeout=5))
            release.set()
            for future in futures:
                future.result(timeout=5)
            # The initial classification has already consumed one burst turn.
            self.assertEqual(execution_order, ['active', 'C1', 'E1', 'C2', 'C3', 'E2'])
            with runtime._condition:
                self.assertFalse(runtime._classification_queue)
                self.assertFalse(runtime._explanation_queue)
                self.assertFalse(runtime._operation_active)
                self.assertEqual(runtime._consecutive_classifications, 0)
            runtime.stop()
        finally:
            release.set()
            with runtime._condition:
                runtime._state = RuntimeState.STOPPING
                runtime._condition.notify_all()
            for thread in threads:
                thread.join(5)
            self.assertFalse(any(thread.is_alive() for thread in threads))
