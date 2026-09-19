"""Deadline arithmetic tests use a controlled clock; thread coordination is
covered by the real-condition tests in test_runtime.py.
"""
from unittest import TestCase
from unittest.mock import Mock, call, patch

from app.classifier import ToxicityClassifier
from app.explainer import ToxicityExplainer
from app.runtime import ApplicationRuntime, RuntimeState
from app.services import ApplicationServices
from app.settings import RuntimeSettings


class ReservationTimeoutTests(TestCase):
    def setUp(self):
        self.settings = RuntimeSettings(device="cpu", operation_wait_timeout_seconds=10)
        self.services = ApplicationServices(Mock(spec=ToxicityClassifier), Mock(spec=ToxicityExplainer))
        self.runtime = ApplicationRuntime(self.settings, services_builder=Mock(return_value=self.services))
        self.runtime.start()

    def test_free_model_does_not_wait(self):
        with patch('app.runtime.monotonic', return_value=100), \
                patch.object(self.runtime._condition, 'wait') as wait:
            result = self.runtime.predict('hello')
        self.assertIs(result, self.services.classifier.predict.return_value)
        wait.assert_not_called()
        self.assertFalse(self.runtime._operation_active)

    def test_repeated_wakeups_use_one_deadline_and_preserve_active_reservation(self):
        operations = [('predict', ('hello',)), ('predict_batch', (['hello'],)),
                      ('explain', ('hello', 'toxic')), ('explain_batch', (['hello'], 'toxic'))]
        for name, args in operations:
            with self.subTest(operation=name):
                # Represent a reservation already held by another caller.
                self.runtime._operation_active = True
                with patch('app.runtime.monotonic', side_effect=[100, 100, 103, 110]), \
                        patch.object(self.runtime._condition, 'wait') as wait:
                    with self.assertRaisesRegex(TimeoutError, 'waiting for model access'):
                        getattr(self.runtime, name)(*args)
                self.assertEqual(wait.call_args_list, [call(timeout=10), call(timeout=7)])
                self.assertTrue(self.runtime._operation_active)
                self.assertTrue(self.runtime.is_ready())
                self.assertEqual(self.services.classifier.mock_calls, [])
                self.assertEqual(self.services.explainer.mock_calls, [])
        self.runtime._end_operation()
        self.runtime.predict('later')
        self.services.classifier.predict.assert_called_once_with('later')

    def test_expired_deadline_does_not_wait_again(self):
        self.runtime._operation_active = True
        with patch('app.runtime.monotonic', side_effect=[100, 111]), \
                patch.object(self.runtime._condition, 'wait') as wait:
            with self.assertRaises(TimeoutError):
                self.runtime.predict('hello')
        wait.assert_not_called()
        self.assertTrue(self.runtime._operation_active)

    def test_model_becoming_available_allows_waiting_call(self):
        self.runtime._operation_active = True

        def release_reservation(*, timeout):
            self.runtime._operation_active = False

        with patch('app.runtime.monotonic', side_effect=[100, 102]), \
                patch.object(self.runtime._condition, 'wait', side_effect=release_reservation) as wait:
            result = self.runtime.predict('hello')
        wait.assert_called_once_with(timeout=8)
        self.assertIs(result, self.services.classifier.predict.return_value)
        self.assertFalse(self.runtime._operation_active)

    def test_shutdown_during_wait_rejects_without_releasing_active_reservation(self):
        self.runtime._operation_active = True

        def begin_shutdown(*, timeout):
            self.runtime._state = RuntimeState.STOPPING

        with patch('app.runtime.monotonic', side_effect=[100, 100]), \
                patch.object(self.runtime._condition, 'wait', side_effect=begin_shutdown):
            with self.assertRaisesRegex(RuntimeError, 'not ready'):
                self.runtime.predict('hello')
        self.services.classifier.predict.assert_not_called()
        self.assertTrue(self.runtime._operation_active)
        self.assertIs(self.runtime._state, RuntimeState.STOPPING)
