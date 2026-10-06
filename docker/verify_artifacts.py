"""Fail image construction when serving artifacts differ from the manifest."""
import hashlib
import json
from pathlib import Path


def verify_artifacts(root: Path, manifest: dict) -> None:
    entries = manifest["files"]
    if not entries:
        raise ValueError("Serving artifact manifest must not be empty")
    for entry in entries:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Artifact paths must stay within the application directory")
        artifact = root / relative
        if artifact.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"Artifact size mismatch: {relative}")
        with artifact.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"Artifact SHA-256 mismatch: {relative}")


if __name__ == "__main__":
    manifest = json.loads(Path(__file__).with_name("model-manifest.json").read_text())
    verify_artifacts(Path.cwd(), manifest)
    print(f"Verified {len(manifest['files'])} serving artifacts for {manifest['release']}")
