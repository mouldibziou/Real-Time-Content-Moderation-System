"""Deployment configuration contracts; no real models or database required."""
import os
from pathlib import Path
import subprocess
import sys
from unittest import TestCase
from unittest.mock import patch

from app.settings import RuntimeSettings, load_runtime_settings, load_model_version


class RuntimeConfigurationTests(TestCase):
    def test_model_version_matches_packaged_release(self):
        self.assertEqual(load_model_version({}), "v2.0.0")
        self.assertEqual(load_model_version({"MODEL_VERSION": "v2.0.0"}), "v2.0.0")
        for value in ("", "v3.0.0"):
            with self.assertRaisesRegex(ValueError, "packaged model"):
                load_model_version({"MODEL_VERSION": value})

    def test_defaults_select_cpu_and_onnx(self):
        settings = load_runtime_settings({})
        self.assertEqual(settings, RuntimeSettings())
        self.assertEqual(settings.device, "cpu")
        self.assertEqual(settings.inference_backend, "onnx")
        self.assertEqual(settings.inference_batch_size, 4)
        self.assertEqual(settings.model_dir, Path("models/distilbert/v2.0.0"))
        self.assertEqual(settings.onnx_model_dir, Path("models/distilbert/v2.0.0-onnx"))

    def test_explicit_overrides_are_converted_and_keep_model_semantics(self):
        settings = load_runtime_settings({
            "TOXICITY_DEVICE": "cuda:1",
            "TOXICITY_INFERENCE_BACKEND": "torch",
            "TOXICITY_MODEL_DIR": "/opt/model release/torch",
            "TOXICITY_ONNX_MODEL_DIR": "/opt/model release/onnx",
            "TOXICITY_INFERENCE_BATCH_SIZE": "8",
        })
        self.assertEqual(settings.device, "cuda:1")
        self.assertEqual(settings.inference_backend, "torch")
        self.assertEqual(settings.model_dir, Path("/opt/model release/torch"))
        self.assertEqual(settings.onnx_model_dir, Path("/opt/model release/onnx"))
        self.assertEqual(settings.inference_batch_size, 8)
        self.assertEqual(settings.labels, RuntimeSettings().labels)
        self.assertEqual(settings.thresholds, RuntimeSettings().thresholds)

    def test_process_environment_is_read_at_call_time(self):
        with patch.dict(os.environ, {"TOXICITY_DEVICE": "mps"}, clear=True):
            self.assertEqual(load_runtime_settings().device, "mps")
            os.environ["TOXICITY_DEVICE"] = "cpu"
            self.assertEqual(load_runtime_settings().device, "cpu")

    def test_explicit_empty_environment_does_not_use_process_environment(self):
        with patch.dict(os.environ, {"TOXICITY_DEVICE": "mps"}, clear=True):
            self.assertEqual(load_runtime_settings({}).device, "cpu")

    def test_blank_values_are_rejected_with_variable_name(self):
        for variable in ("TOXICITY_DEVICE", "TOXICITY_INFERENCE_BACKEND",
                         "TOXICITY_MODEL_DIR", "TOXICITY_ONNX_MODEL_DIR",
                         "TOXICITY_INFERENCE_BATCH_SIZE"):
            for value in ("", " ", "\t\n"):
                with self.subTest(variable=variable, value=value):
                    with self.assertRaisesRegex(ValueError, variable):
                        load_runtime_settings({variable: value})

    def test_invalid_inference_batch_sizes_are_rejected(self):
        for value in ("0", "-1", "1.5", "eight"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                load_runtime_settings({"TOXICITY_INFERENCE_BATCH_SIZE": value})
        for value in (0, -1, True, 2.5, "8", None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Inference batch size"):
                RuntimeSettings(inference_batch_size=value)

    def test_invalid_backends_are_rejected(self):
        for value in ("tensorrt", "ONNX", " onnx "):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "Inference backend"):
                    load_runtime_settings({"TOXICITY_INFERENCE_BACKEND": value})

    def test_invalid_device_syntax_is_rejected(self):
        for value in ("meta", "cuda:N", "cuda:-1", "cpu:0", "mps:0", " cpu "):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "Device must"):
                    load_runtime_settings({"TOXICITY_DEVICE": value})

    def test_supported_devices_are_configurable_without_hardware_checks(self):
        for value in ("cpu", "mps", "cuda", "cuda:0", "cuda:12"):
            with self.subTest(value=value):
                self.assertEqual(load_runtime_settings({"TOXICITY_DEVICE": value}).device, value)

    def test_direct_settings_reject_invalid_types(self):
        for field, value in (("device", None), ("model_dir", "models/torch")):
            with self.subTest(field=field):
                with self.assertRaises(TypeError):
                    RuntimeSettings(**{field: value})

    def test_unknown_variables_and_database_credentials_are_not_consumed(self):
        self.assertEqual(load_runtime_settings({
            "DATABASE_URL": "postgresql+asyncpg://user:private-password@db/app",
            "TOXICITY_THRESHOLDS": "0",
        }), RuntimeSettings())

    def test_loading_configuration_does_not_import_model_or_database_modules(self):
        # A fresh interpreter prevents earlier tests from hiding heavy imports.
        result = subprocess.run(
            [sys.executable, "-c", """
import sys
from app.settings import load_runtime_settings
load_runtime_settings({
    'TOXICITY_MODEL_DIR': '/missing/torch',
    'TOXICITY_ONNX_MODEL_DIR': '/missing/onnx',
    'TOXICITY_DEVICE': 'cuda',
})
for name in ('torch', 'transformers', 'onnxruntime', 'sqlalchemy', 'dotenv',
             'app.resources', 'app.db.engine'):
    assert name not in sys.modules, name
"""],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_api_entry_point_loads_configuration_once_without_loading_resources(self):
        result = subprocess.run(
            [sys.executable, "-c", """
import importlib
import os
from unittest.mock import patch
from app.settings import load_runtime_settings

environment = {
    'DATABASE_URL': 'postgresql+asyncpg://test:test@localhost/test',
    'TOXICITY_DEVICE': 'cpu',
    'TOXICITY_INFERENCE_BACKEND': 'torch',
    'TOXICITY_MODEL_DIR': '/missing/torch',
    'TOXICITY_ONNX_MODEL_DIR': '/missing/onnx',
}
with patch.dict(os.environ, environment), \\
     patch('dotenv.load_dotenv'), \\
     patch('app.settings.load_runtime_settings', wraps=load_runtime_settings) as load, \\
     patch('app.resources.load_resources', side_effect=AssertionError('model load')) as resources, \\
     patch('sqlalchemy.ext.asyncio.AsyncEngine.connect', side_effect=AssertionError('database connection')) as connect:
    main = importlib.import_module('app.api.main')
    assert main.settings.device == 'cpu'
    assert main.settings.inference_backend == 'torch'
    assert str(main.settings.model_dir) == '/missing/torch'
    assert str(main.settings.onnx_model_dir) == '/missing/onnx'
    assert main.app.title == 'Toxicity Classification API'
    load.assert_called_once_with()
    resources.assert_not_called()
    connect.assert_not_called()
"""],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
