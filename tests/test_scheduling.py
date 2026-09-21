from dataclasses import FrozenInstanceError
from unittest import TestCase

from app.scheduling import OperationKind, WaitingOperation, choose_next_kind
from app.settings import RuntimeSettings


class SchedulingPolicyTests(TestCase):
    def test_selection_table_including_burst_boundary(self):
        cases = (
            (False, False, 0, None),
            (False, False, 5, None),
            (True, False, 0, OperationKind.CLASSIFICATION),
            (True, False, 8, OperationKind.CLASSIFICATION),
            (False, True, 0, OperationKind.EXPLANATION),
            (False, True, 8, OperationKind.EXPLANATION),
            (True, True, 0, OperationKind.CLASSIFICATION),
            (True, True, 4, OperationKind.CLASSIFICATION),
            (True, True, 5, OperationKind.EXPLANATION),
            (True, True, 8, OperationKind.EXPLANATION),
        )
        for classification, explanation, count, expected in cases:
            with self.subTest(classification=classification, explanation=explanation, count=count):
                self.assertIs(choose_next_kind(
                    classification_waiting=classification,
                    explanation_waiting=explanation,
                    consecutive_classifications=count,
                    classification_burst_limit=5,
                ), expected)

    def test_selection_is_repeatable_without_consuming_a_turn(self):
        arguments = dict(classification_waiting=True, explanation_waiting=True,
                         consecutive_classifications=2, classification_burst_limit=3)
        for _ in range(5):
            self.assertIs(choose_next_kind(**arguments), OperationKind.CLASSIFICATION)

    def test_burst_limit_of_one_selects_explanation_after_one_classification(self):
        self.assertIs(choose_next_kind(
            classification_waiting=True, explanation_waiting=True,
            consecutive_classifications=1, classification_burst_limit=1,
        ), OperationKind.EXPLANATION)


class WaitingOperationTests(TestCase):
    def test_equal_fields_still_identify_distinct_requests(self):
        first = WaitingOperation(OperationKind.CLASSIFICATION, 100.0)
        second = WaitingOperation(OperationKind.CLASSIFICATION, 100.0)
        self.assertIsNot(first, second)
        self.assertNotEqual(first, second)
        self.assertEqual(first, first)
        pending = [first, second]
        pending.remove(second)
        self.assertEqual(len(pending), 1)
        self.assertIs(pending[0], first)

    def test_ticket_fields_cannot_be_reassigned(self):
        ticket = WaitingOperation(OperationKind.EXPLANATION, 100.0)
        for field, value in (("kind", OperationKind.CLASSIFICATION), ("deadline", 200.0)):
            with self.subTest(field=field):
                with self.assertRaises(FrozenInstanceError):
                    setattr(ticket, field, value)


class SchedulingSettingsTests(TestCase):
    fields = ("max_waiting_classifications", "max_waiting_explanations", "classification_burst_limit")

    def test_scheduling_settings_require_positive_integers(self):
        for field in self.fields:
            for value, error in ((True, TypeError), (False, TypeError), (1.5, TypeError),
                                 (None, TypeError), ("2", TypeError), (0, ValueError), (-1, ValueError)):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(error):
                        RuntimeSettings(**{field: value})

    def test_scheduling_settings_accept_one(self):
        settings = RuntimeSettings(**{field: 1 for field in self.fields})
        for field in self.fields:
            self.assertEqual(getattr(settings, field), 1)
