from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
from app.cpu import available_cpu_count
from app.settings import RuntimeSettings, load_runtime_settings


class CpuBudgetTests(TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.enterContext(patch("app.cpu.os.sched_getaffinity", return_value=set(range(8))))

    def test_v2_quota_limits_visible_host_cpus(self):
        (self.root / "cpu.max").write_text("200000 100000")
        self.assertEqual(available_cpu_count(self.root), 2)

    def test_unlimited_quota_uses_affinity(self):
        (self.root / "cpu.max").write_text("max 100000")
        self.assertEqual(available_cpu_count(self.root), 8)

    def test_fractional_quota_has_at_least_one_thread(self):
        (self.root / "cpu.max").write_text("50000 100000")
        self.assertEqual(available_cpu_count(self.root), 1)

    def test_affinity_can_be_smaller_than_quota(self):
        (self.root / "cpu.max").write_text("1200000 100000")
        self.assertEqual(available_cpu_count(self.root), 8)

    def test_v1_quota(self):
        directory = self.root / "cpu"
        directory.mkdir()
        (directory / "cpu.cfs_quota_us").write_text("300000")
        (directory / "cpu.cfs_period_us").write_text("100000")
        self.assertEqual(available_cpu_count(self.root), 3)

    def test_missing_cgroups_and_affinity_fall_back_to_cpu_count(self):
        with patch("app.cpu.os.sched_getaffinity", side_effect=OSError), \
             patch("app.cpu.os.cpu_count", return_value=4):
            self.assertEqual(available_cpu_count(self.root), 4)

    def test_thread_override_validation(self):
        self.assertEqual(load_runtime_settings({"TOXICITY_CPU_THREADS": "4"}).cpu_threads, 4)
        self.assertEqual(load_runtime_settings({}).cpu_threads, 0)
        for value in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                RuntimeSettings(cpu_threads=value)
        for value in ("", "abc", "-1"):
            with self.assertRaises(ValueError):
                load_runtime_settings({"TOXICITY_CPU_THREADS": value})
