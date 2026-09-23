"""Opt-in integration check: TOXICITY_REAL_MODEL_TESTS=1 enables local weights.

Run from the repository root. TOXICITY_SMOKE_DEVICE defaults to cpu.
This checks execution and result structure, not predictive accuracy or capacity.
"""
import math
import os
from pathlib import Path
from time import perf_counter
from unittest import TestCase, skipUnless

from app.runtime import ApplicationRuntime, RuntimeState
from app.settings import RuntimeSettings


@skipUnless(os.environ.get('TOXICITY_REAL_MODEL_TESTS') == '1', 'Real model smoke test is opt-in')
class RealModelSmokeTests(TestCase):
    def test_prediction_explanation_prediction_and_shutdown(self):
        settings = RuntimeSettings(
            model_dir=Path(__file__).resolve().parents[1] / 'models/distilbert/v2.0.0',
            device=os.environ.get('TOXICITY_SMOKE_DEVICE', 'cpu'),
        )
        runtime = ApplicationRuntime(settings)
        text = 'You are being rude and unhelpful.'
        timings = {}
        try:
            started = perf_counter()
            runtime.start()
            timings['startup_seconds'] = perf_counter() - started
            self.assertTrue(runtime.is_ready())
            metadata = runtime.model_info()
            self.assertEqual(metadata.labels, settings.labels)
            self.assertEqual(metadata.max_length, 512)
            self.assertEqual(metadata.explanation_steps, settings.n_steps)

            started = perf_counter()
            before = runtime.predict(text)
            timings['prediction_seconds'] = perf_counter() - started
            self.assertEqual(tuple(before.labels), settings.labels)
            for prediction in before.labels.values():
                self.assertTrue(math.isfinite(prediction.probability))
                self.assertGreaterEqual(prediction.probability, 0)
                self.assertLessEqual(prediction.probability, 1)

            started = perf_counter()
            explanation = runtime.explain(text, 'toxic')
            timings['explanation_seconds'] = perf_counter() - started
            self.assertEqual(explanation.label, 'toxic')
            self.assertTrue(explanation.spans)
            for span in explanation.spans:
                self.assertTrue(0 <= span.start < span.end <= len(text))
                self.assertEqual(span.text, text[span.start:span.end])
                self.assertTrue(math.isfinite(span.score))
            if explanation.convergence_delta is not None:
                self.assertTrue(math.isfinite(explanation.convergence_delta))

            after = runtime.predict(text)
            for label in settings.labels:
                self.assertAlmostEqual(before.labels[label].probability, after.labels[label].probability, places=6)
                self.assertEqual(before.labels[label].predicted, after.labels[label].predicted)
            status = runtime.status()
            self.assertFalse(status.operation_active)
            self.assertEqual(status.waiting_classifications, 0)
            self.assertEqual(status.waiting_explanations, 0)
            print(f'\nReal model smoke: device={metadata.device}, timings={timings}', flush=True)
        finally:
            runtime.stop()
        self.assertIs(runtime.status().state, RuntimeState.STOPPED)
        with self.assertRaisesRegex(RuntimeError, 'not ready'):
            runtime.model_info()
