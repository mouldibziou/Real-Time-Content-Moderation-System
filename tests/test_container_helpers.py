"""Container probes and build-time artifact integrity contracts."""
import hashlib
import io
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock
from urllib.error import URLError


ROOT = Path(__file__).resolve().parents[1]
check_readiness = runpy.run_path(str(ROOT / "docker/healthcheck.py"))["check_readiness"]
verify_artifacts = runpy.run_path(str(ROOT / "docker/verify_artifacts.py"))["verify_artifacts"]


class ContainerReadinessTests(TestCase):
    def probe(self, payload, status=200):
        response = io.BytesIO(json.dumps(payload).encode())
        response.status = status
        opener = Mock()
        opener.open.return_value = response
        result = check_readiness(opener)
        opener.open.assert_called_once_with(
            "http://127.0.0.1:8000/v1/health/ready", timeout=2,
        )
        return result

    def test_ready_runtime_is_healthy(self):
        self.assertTrue(self.probe({"is_ready": True}))

    def test_http_success_without_true_readiness_is_unhealthy(self):
        for payload in ({"is_ready": False}, {"is_ready": "true"},
                        {"is_ready": 1}, {}, [], None):
            with self.subTest(payload=payload):
                self.assertFalse(self.probe(payload))

    def test_unsuccessful_http_status_is_unhealthy(self):
        self.assertFalse(self.probe({"is_ready": True}, status=503))

    def test_invalid_json_is_unhealthy(self):
        opener = Mock()
        response = io.BytesIO(b"not JSON")
        response.status = 200
        opener.open.return_value = response
        self.assertFalse(check_readiness(opener))

    def test_connection_failure_and_timeout_are_unhealthy(self):
        for error in (URLError("connection refused"), TimeoutError("timed out")):
            with self.subTest(error=type(error).__name__):
                opener = Mock()
                opener.open.side_effect = error
                self.assertFalse(check_readiness(opener))


class ServingArtifactIntegrityTests(TestCase):
    def setUp(self):
        self.directory = self.enterContext(TemporaryDirectory())
        self.root = Path(self.directory)
        self.artifact = self.root / "weights.bin"
        self.artifact.write_bytes(b"good")
        self.manifest = {"files": [{
            "path": "weights.bin", "size_bytes": 4,
            "sha256": hashlib.sha256(b"good").hexdigest(),
        }]}

    def test_matching_artifact_passes(self):
        verify_artifacts(self.root, self.manifest)

    def test_same_size_corruption_fails_checksum(self):
        self.artifact.write_bytes(b"evil")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            verify_artifacts(self.root, self.manifest)

    def test_truncated_artifact_fails_size_check(self):
        self.artifact.write_bytes(b"bad")
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            verify_artifacts(self.root, self.manifest)

    def test_missing_artifact_fails(self):
        self.artifact.unlink()
        with self.assertRaises(FileNotFoundError):
            verify_artifacts(self.root, self.manifest)

    def test_empty_manifest_fails(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            verify_artifacts(self.root, {"files": []})

    def test_artifact_paths_cannot_escape_application_directory(self):
        for path in ("../weights.bin", "/weights.bin"):
            with self.subTest(path=path):
                self.manifest["files"][0]["path"] = path
                with self.assertRaisesRegex(ValueError, "within the application"):
                    verify_artifacts(self.root, self.manifest)
