from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock

from app.runtime import ApplicationRuntime
from app.settings import RuntimeSettings

class FakeClassifier:
    def predict(self, text: str):
        return {"toxic": 1}

    def predict_batch(self, texts: list[str]):
        return [{"toxic": 1} for _ in texts]

class ApplicationRuntimeTests(TestCase):
    def setUp(self) -> None:
        self.settings = RuntimeSettings(
            model_dir=Path("."),
            labels=("toxic",),
            thresholds=(0.5,)
        )
        self.classifier = FakeClassifier()
        self.builder = Mock(return_value=self.classifier)
        self.runtime = ApplicationRuntime(
            settings=self.settings,
            classifier_builder=self.builder,
        )

    def test_is_not_ready_before_start(self) -> None:
        self.assertFalse(self.runtime.is_ready())

    def test_start_builds_classifier_once(self) -> None:
        self.runtime.start()
        self.runtime.start()

        self.builder.assert_called_once_with(self.settings)
        self.assertTrue(self.runtime.is_ready())

    def test_predict_requires_startup(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "has not been started"):
            self.runtime.predict("hello")

    def test_predict_delegates_to_classifier(self) -> None:
        self.runtime.start()

        result = self.runtime.predict("hello")

        self.assertEqual(result, {"toxic": 1})