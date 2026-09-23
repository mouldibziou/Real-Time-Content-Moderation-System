"""Queue lookup and selection before admission is integrated."""
from unittest import TestCase
from unittest.mock import Mock

from tests.factories import make_model_info
from app.runtime import ApplicationRuntime
from app.scheduling import OperationKind, WaitingOperation
from app.settings import RuntimeSettings


class RuntimeQueueHelpersTests(TestCase):
    def setUp(self):
        self.runtime = ApplicationRuntime(
            RuntimeSettings(device="cpu", classification_burst_limit=2),
            services_builder=Mock(),
        )

    def test_lookup_returns_existing_queue_objects(self):
        with self.runtime._condition:
            self.assertIs(self.runtime._queue_for(OperationKind.CLASSIFICATION),
                          self.runtime._classification_queue)
            self.assertIs(self.runtime._queue_for(OperationKind.EXPLANATION),
                          self.runtime._explanation_queue)

    def test_lookup_rejects_unsupported_kind(self):
        with self.runtime._condition:
            with self.assertRaises(ValueError):
                self.runtime._queue_for("classification")

    def test_empty_queues_have_no_selected_ticket(self):
        with self.runtime._condition:
            self.assertIsNone(self.runtime._next_waiting_operation())

    def test_each_queue_returns_its_oldest_ticket_without_removing_it(self):
        for kind, queue in ((OperationKind.CLASSIFICATION, self.runtime._classification_queue),
                            (OperationKind.EXPLANATION, self.runtime._explanation_queue)):
            with self.subTest(kind=kind), self.runtime._condition:
                first = WaitingOperation(kind, 100.0)
                second = WaitingOperation(kind, 100.0)
                queue.extend([first, second])
                try:
                    self.assertIs(self.runtime._next_waiting_operation(), first)
                    self.assertIs(self.runtime._next_waiting_operation(), first)
                    self.assertEqual(list(queue), [first, second])
                    self.assertEqual(self.runtime._consecutive_classifications, 0)
                finally:
                    queue.clear()

    def test_burst_boundary_selects_explanation_without_consuming_turn(self):
        classification = WaitingOperation(OperationKind.CLASSIFICATION, 100.0)
        explanation = WaitingOperation(OperationKind.EXPLANATION, 100.0)
        with self.runtime._condition:
            self.runtime._classification_queue.append(classification)
            self.runtime._explanation_queue.append(explanation)
            self.runtime._consecutive_classifications = 1
            self.assertIs(self.runtime._next_waiting_operation(), classification)
            self.runtime._consecutive_classifications = 2
            self.assertIs(self.runtime._next_waiting_operation(), explanation)
            self.assertEqual(self.runtime._consecutive_classifications, 2)
            self.assertEqual(len(self.runtime._classification_queue), 1)
            self.assertEqual(len(self.runtime._explanation_queue), 1)

    def test_active_operation_prevents_ticket_selection(self):
        with self.runtime._condition:
            self.runtime._classification_queue.append(
                WaitingOperation(OperationKind.CLASSIFICATION, 100.0))
            self.runtime._operation_active = True
            self.assertIsNone(self.runtime._next_waiting_operation())
            self.assertEqual(len(self.runtime._classification_queue), 1)


class RuntimeAdmissionAndClaimTests(TestCase):
    def setUp(self):
        from app.services import ApplicationServices
        from unittest.mock import patch

        self.services = ApplicationServices(Mock(), Mock(), model_info=make_model_info())
        self.runtime = ApplicationRuntime(
            RuntimeSettings(device="cpu", classification_burst_limit=2,
                            max_waiting_classifications=2, max_waiting_explanations=1),
            services_builder=Mock(return_value=self.services),
        )
        self.runtime.start()
        self.enterContext(patch("app.runtime.monotonic", return_value=90.0))

    def snapshot(self):
        return (list(self.runtime._classification_queue),
                list(self.runtime._explanation_queue),
                self.runtime._operation_active,
                self.runtime._consecutive_classifications,
                self.runtime._services)

    def enqueue(self, kind, deadline=100.0):
        return self.runtime._enqueue_operation(kind, deadline=deadline)

    def test_admission_preserves_fifo_and_notifies_without_reserving(self):
        from unittest.mock import patch

        with self.runtime._condition, patch.object(
            self.runtime._condition, "notify_all", wraps=self.runtime._condition.notify_all
        ) as notify:
            first = self.enqueue(OperationKind.CLASSIFICATION)
            second = self.enqueue(OperationKind.CLASSIFICATION)
            self.assertIsNot(first, second)
            self.assertEqual(list(self.runtime._classification_queue), [first, second])
            self.assertEqual(first.deadline, 100.0)
            self.assertIs(first.kind, OperationKind.CLASSIFICATION)
            self.assertFalse(self.runtime._operation_active)
            self.assertEqual(self.runtime._consecutive_classifications, 0)
            self.assertEqual(notify.call_count, 2)

    def test_queue_capacities_are_independent_and_full_rejection_is_atomic(self):
        from app.runtime import RuntimeBusyError

        with self.runtime._condition:
            self.enqueue(OperationKind.CLASSIFICATION)
            self.enqueue(OperationKind.CLASSIFICATION)
            before = self.snapshot()
            with self.assertRaises(RuntimeBusyError):
                self.enqueue(OperationKind.CLASSIFICATION)
            self.assertEqual(self.snapshot(), before)
            self.enqueue(OperationKind.EXPLANATION)
            before = self.snapshot()
            with self.assertRaises(RuntimeBusyError):
                self.enqueue(OperationKind.EXPLANATION)
            self.assertEqual(self.snapshot(), before)

    def test_admission_rejects_unready_runtime_and_invalid_kind(self):
        from app.runtime import RuntimeState

        with self.runtime._condition:
            with self.assertRaises(ValueError):
                self.enqueue("classification")
            for state in (RuntimeState.STOPPED, RuntimeState.STARTING, RuntimeState.STOPPING):
                with self.subTest(state=state):
                    self.runtime._state = state
                    before = self.snapshot()
                    with self.assertRaisesRegex(RuntimeError, "not ready"):
                        self.enqueue(OperationKind.CLASSIFICATION)
                    self.assertEqual(self.snapshot(), before)

    def test_classification_claim_removes_only_selected_ticket_and_caps_counter(self):
        with self.runtime._condition:
            first = self.enqueue(OperationKind.CLASSIFICATION)
            second = self.enqueue(OperationKind.CLASSIFICATION)
            self.runtime._consecutive_classifications = 2
            result = self.runtime._claim_operation(first)
            self.assertIs(result, self.services)
            self.assertEqual(list(self.runtime._classification_queue), [second])
            self.assertTrue(self.runtime._operation_active)
            self.assertEqual(self.runtime._consecutive_classifications, 2)

    def test_classification_claim_increments_counter(self):
        with self.runtime._condition:
            ticket = self.enqueue(OperationKind.CLASSIFICATION)
            self.runtime._claim_operation(ticket)
            self.assertEqual(self.runtime._consecutive_classifications, 1)

    def test_explanation_claim_at_burst_boundary_resets_counter(self):
        with self.runtime._condition:
            classification = self.enqueue(OperationKind.CLASSIFICATION)
            explanation = self.enqueue(OperationKind.EXPLANATION)
            self.runtime._consecutive_classifications = 2
            self.assertIs(self.runtime._claim_operation(explanation), self.services)
            self.assertEqual(self.runtime._consecutive_classifications, 0)
            self.assertTrue(self.runtime._operation_active)
            self.assertEqual(list(self.runtime._classification_queue), [classification])
            self.assertFalse(self.runtime._explanation_queue)

    def test_expired_claim_preserves_queue_reservation_and_counter(self):
        for deadline in (89.0, 90.0):
            with self.subTest(deadline=deadline), self.runtime._condition:
                self.runtime._classification_queue.clear()
                ticket = self.enqueue(OperationKind.CLASSIFICATION, deadline=deadline)
                before = self.snapshot()
                with self.assertRaises(TimeoutError):
                    self.runtime._claim_operation(ticket)
                self.assertEqual(self.snapshot(), before)

    def test_nonselected_ticket_cannot_claim_or_mutate_state(self):
        with self.runtime._condition:
            first = self.enqueue(OperationKind.CLASSIFICATION)
            second = self.enqueue(OperationKind.CLASSIFICATION)
            explanation = self.enqueue(OperationKind.EXPLANATION)
            duplicate = WaitingOperation(first.kind, first.deadline)
            for ticket in (second, explanation, duplicate):
                with self.subTest(ticket=ticket):
                    before = self.snapshot()
                    with self.assertRaisesRegex(RuntimeError, "not next"):
                        self.runtime._claim_operation(ticket)
                    self.assertEqual(self.snapshot(), before)

    def test_active_operation_prevents_claim_without_mutation(self):
        with self.runtime._condition:
            ticket = self.enqueue(OperationKind.CLASSIFICATION)
            self.runtime._operation_active = True
            before = self.snapshot()
            with self.assertRaises(RuntimeError):
                self.runtime._claim_operation(ticket)
            self.assertEqual(self.snapshot(), before)

    def test_shutdown_prevents_claim_without_mutation(self):
        from app.runtime import RuntimeState

        with self.runtime._condition:
            ticket = self.enqueue(OperationKind.CLASSIFICATION)
            self.runtime._state = RuntimeState.STOPPING
            before = self.snapshot()
            with self.assertRaisesRegex(RuntimeError, "not ready"):
                self.runtime._claim_operation(ticket)
            self.assertEqual(self.snapshot(), before)
