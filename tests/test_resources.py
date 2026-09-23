from unittest import TestCase
from unittest.mock import patch
from unittest.mock import Mock
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from app.resources import resolve_device, load_resources
from app.settings import RuntimeSettings


class ResourceLoadingLoggingTests(TestCase):
    def setUp(self):
        directory = self.enterContext(TemporaryDirectory())
        self.settings = RuntimeSettings(model_dir=Path(directory), device="cpu")
        self.tokenizer = Mock(is_fast=True, pad_token_id=0)
        self.model = Mock()
        self.model.config.num_labels = len(self.settings.labels)
        self.model.config.max_position_embeddings = 512
        self.model.config.id2label = dict(enumerate(self.settings.labels))
        self.enterContext(patch('app.resources.AutoTokenizer.from_pretrained', return_value=self.tokenizer))
        self.loader = self.enterContext(patch(
            'app.resources.AutoModelForSequenceClassification.from_pretrained', return_value=self.model))

    def test_success_logs_elapsed_time_and_returns_loaded_resources(self):
        with patch('app.resources.monotonic', side_effect=[100.0, 102.5]), \
                self.assertLogs('app.resources', level='INFO') as captured:
            resources = load_resources(self.settings)
        self.assertIs(resources.model, self.model)
        self.assertIs(resources.tokenizer, self.tokenizer)
        self.model.to.assert_called_once_with(torch.device('cpu'))
        self.model.eval.assert_called_once_with()
        self.assertEqual([record.getMessage() for record in captured.records], [
            'Model loading started',
            'Model loading completed device=cpu duration_seconds=2.500',
        ])
        self.assertTrue(all(record.levelname == 'INFO' for record in captured.records))

    def test_loading_failure_propagates_without_completion_or_duplicate_traceback(self):
        error = OSError('Test artifact could not be loaded')
        self.loader.side_effect = error
        with self.assertLogs('app.resources', level='INFO') as captured:
            with self.assertRaises(OSError) as raised:
                load_resources(self.settings)
        self.assertIs(raised.exception, error)
        self.assertEqual([record.getMessage() for record in captured.records], ['Model loading started'])
        self.assertIsNone(captured.records[0].exc_info)
        self.model.eval.assert_not_called()


class ResolveDeviceTests(TestCase):
    def setUp(self) -> None:
        # Simulate hardware so these tests run on CPU-only machines too.
        self.cuda_available = self.enterContext(
            patch("app.resources.torch.cuda.is_available", return_value=False)
        )
        self.mps_available = self.enterContext(
            patch("app.resources.torch.backends.mps.is_available", return_value=False)
        )
        self.current_device = self.enterContext(
            patch("app.resources.torch.cuda.current_device", return_value=1)
        )
        self.device_count = self.enterContext(
            patch("app.resources.torch.cuda.device_count", return_value=2)
        )

    def test_cpu_requires_no_accelerator(self) -> None:
        self.assertEqual(resolve_device("cpu"), torch.device("cpu"))
        self.cuda_available.assert_not_called()
        self.mps_available.assert_not_called()

    def test_available_mps(self) -> None:
        self.mps_available.return_value = True
        self.assertEqual(resolve_device("mps"), torch.device("mps"))

    def test_unavailable_mps_raises(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "MPS device is not available"):
            resolve_device("mps")

    def test_cpu_and_mps_indices_are_rejected(self) -> None:
        for value in ("cpu:0", "mps:0", "mps:1"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "indices are not supported"):
                    resolve_device(value)

    def test_bare_cuda_resolves_current_device(self) -> None:
        self.cuda_available.return_value = True
        self.assertEqual(resolve_device("cuda"), torch.device("cuda:1"))
        self.current_device.assert_called_once_with()

    def test_explicit_cuda_index_including_zero_is_preserved(self) -> None:
        self.cuda_available.return_value = True
        for value in ("cuda:0", "cuda:1"):
            with self.subTest(value=value):
                self.assertEqual(resolve_device(value), torch.device(value))
        self.current_device.assert_not_called()

    def test_unavailable_cuda_raises_before_querying_indices(self) -> None:
        for value in ("cuda", "cuda:0"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(RuntimeError, "CUDA device is not available"):
                    resolve_device(value)
        self.current_device.assert_not_called()
        self.device_count.assert_not_called()

    def test_out_of_range_cuda_index_is_rejected(self) -> None:
        self.cuda_available.return_value = True
        with self.assertRaisesRegex(ValueError, "index 2 is out of range"):
            resolve_device("cuda:2")

    def test_resolved_current_index_is_validated(self) -> None:
        self.cuda_available.return_value = True
        self.current_device.return_value = 2
        with self.assertRaisesRegex(ValueError, "index 2 is out of range"):
            resolve_device("cuda")

    def test_unsupported_device_type_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported device type: meta"):
            resolve_device("meta")

    def test_malformed_device_strings_raise_parsing_errors(self) -> None:
        for value in ("", "cuda:N", "cuda:-1"):
            with self.subTest(value=value):
                with self.assertRaises(RuntimeError):
                    resolve_device(value)
