"""Queue lookup and selection before admission is integrated."""
from unittest import TestCase
from unittest.mock import Mock

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
