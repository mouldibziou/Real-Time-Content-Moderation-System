"""Important lifecycle events, without configuring application log handlers."""
import logging
from threading import Condition
from unittest import TestCase
from unittest.mock import Mock, patch

from app.runtime import ApplicationRuntime, RuntimeState
from app.services import ApplicationServices
from app.settings import RuntimeSettings
from tests.factories import make_model_info


class TrackingCondition(Condition):
    """Track explicit lock scopes in these single-threaded lifecycle tests."""
    def __init__(self):
        super().__init__()
        self.depth = 0

    def __enter__(self):
        result = super().__enter__()
        self.depth += 1
        return result

    def __exit__(self, *args):
        self.depth -= 1
        return super().__exit__(*args)


class StartupLoggingTests(TestCase):
    def setUp(self):
        self.services = ApplicationServices(Mock(), Mock(), make_model_info())
        self.builder = Mock(return_value=self.services)
        self.runtime = ApplicationRuntime(RuntimeSettings(device="cpu"), services_builder=self.builder)
        self.runtime._condition = TrackingCondition()

    def test_success_logs_start_then_completion_at_info_level(self):
        with self.assertLogs('app.runtime', level='INFO') as captured:
            self.runtime.start()
        self.assertEqual([record.getMessage() for record in captured.records], [
            'Runtime startup started', 'Runtime startup completed',
        ])
        self.assertEqual([record.levelno for record in captured.records], [logging.INFO, logging.INFO])
        self.assertTrue(all(record.exc_info is None for record in captured.records))
        self.assertTrue(self.runtime.is_ready())

    def test_failure_logs_original_traceback_and_propagates_error(self):
        error = FileNotFoundError('Test model artifact missing')
        self.builder.side_effect = error
        with self.assertLogs('app.runtime', level='INFO') as captured:
            with self.assertRaises(FileNotFoundError) as raised:
                self.runtime.start()
        self.assertIs(raised.exception, error)
        self.assertEqual([record.getMessage() for record in captured.records], [
            'Runtime startup started', 'Runtime startup failed',
        ])
        failure = captured.records[-1]
        self.assertEqual(failure.levelno, logging.ERROR)
        self.assertIs(failure.exc_info[1], error)
        self.assertIsNotNone(failure.exc_info[2])
        self.assertIs(self.runtime.status().state, RuntimeState.STOPPED)

    def test_repeated_start_emits_no_extra_events(self):
        with self.assertLogs('app.runtime', level='INFO'):
            self.runtime.start()
        with self.assertNoLogs('app.runtime', level='DEBUG'):
            self.runtime.start()
        self.builder.assert_called_once()

    def test_rejected_start_emits_no_started_event(self):
        for state in (RuntimeState.STARTING, RuntimeState.STOPPING):
            with self.subTest(state=state):
                self.runtime._state = state
                with self.assertNoLogs('app.runtime', level='DEBUG'):
                    with self.assertRaises(RuntimeError):
                        self.runtime.start()
        self.builder.assert_not_called()

    def test_logging_occurs_outside_lock_after_corresponding_state_changes(self):
        events = []

        def record(message):
            self.assertEqual(self.runtime._condition.depth, 0)
            events.append((message, self.runtime.status().state))

        with patch('app.runtime.logger.info', side_effect=record), \
                patch('app.runtime.logger.exception', side_effect=record):
            self.runtime.start()
            self.runtime.stop()
            self.builder.side_effect = ValueError('Test startup failure')
            with self.assertRaises(ValueError):
                self.runtime.start()
        self.assertEqual(events, [
            ('Runtime startup started', RuntimeState.STARTING),
            ('Runtime startup completed', RuntimeState.READY),
            ('Runtime shutdown completed', RuntimeState.STOPPED),
            ('Runtime startup started', RuntimeState.STARTING),
            ('Runtime startup failed', RuntimeState.STOPPED),
        ])

    def test_shutdown_logs_completion_after_releasing_services(self):
        self.runtime.start()
        with self.assertLogs('app.runtime', level='INFO') as captured:
            self.runtime.stop()
        self.assertEqual(len(captured.records), 1)
        record = captured.records[0]
        self.assertEqual(record.getMessage(), 'Runtime shutdown completed')
        self.assertEqual(record.levelno, logging.INFO)
        self.assertIsNone(record.exc_info)
        self.assertIsNone(self.runtime._services)
        self.assertIs(self.runtime.status().state, RuntimeState.STOPPED)

    def test_stopping_an_already_stopped_runtime_emits_no_event(self):
        with self.assertNoLogs('app.runtime', level='DEBUG'):
            self.runtime.stop()
        self.runtime.start()
        self.runtime.stop()
        with self.assertNoLogs('app.runtime', level='DEBUG'):
            self.runtime.stop()

    def test_rejected_shutdown_emits_no_completion_event(self):
        for state in (RuntimeState.STARTING, RuntimeState.STOPPING):
            with self.subTest(state=state):
                self.runtime._state = state
                with self.assertNoLogs('app.runtime', level='DEBUG'):
                    with self.assertRaises(RuntimeError):
                        self.runtime.stop()
                self.assertIs(self.runtime.status().state, state)
