from unittest import TestCase
from unittest.mock import patch

import torch

from app.resources import resolve_device


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
