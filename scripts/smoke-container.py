"""Exercise a running API with migrated PostgreSQL; creates disposable test rows."""
import argparse
import json
from time import monotonic, sleep
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener
from uuid import UUID, uuid4


LABELS = ("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")


def smoke(base_url: str) -> None:
    opener = build_opener(ProxyHandler({}))

    def request(method, path, payload=None, expected=200):
        data = None if payload is None else json.dumps(payload).encode()
        headers = {} if payload is None else {"Content-Type": "application/json"}
        req = Request(base_url.rstrip("/") + path, data=data, headers=headers, method=method)
        try:
            with opener.open(req, timeout=120) as response:
                status, body = response.status, json.load(response)
        except HTTPError as error:
            status, body = error.code, json.load(error)
        allowed = expected if isinstance(expected, tuple) else (expected,)
        if status not in allowed:
            raise RuntimeError(f"{method} {path}: expected HTTP {expected}, got {status}")
        return body

    deadline = monotonic() + 120
    while True:
        try:
            if request("GET", "/v1/health/ready", expected=(200, 503)).get("is_ready") is True:
                break
        except (URLError, OSError):
            pass
        if monotonic() >= deadline:
            raise TimeoutError("API did not become ready within 120 seconds")
        sleep(1)

    assert request("GET", "/v1/health/live") == {"status": "healthy"}
    info = request("GET", "/v1/model_info")
    assert info["device"] == "cpu"
    assert info["inference_backend"] == "onnx"
    assert tuple(info["labels"]) == LABELS
    print("Readiness, liveness, and CPU/ONNX metadata passed.", flush=True)

    text = f"Smoke check {uuid4()}: You are being rude and unhelpful."
    classified = request("POST", "/v1/classify", {"text": text})
    UUID(classified["classification_id"])
    assert classified["text"] == text
    assert tuple(classified["labels"]) == LABELS
    for label in classified["labels"].values():
        assert 0 <= label["probability"] <= 1
        assert type(label["predicted"]) is bool
    request("POST", "/v1/classify", {"text": " "}, expected=422)

    detail_path = f"/v1/classifications/{classified['classification_id']}"
    detail = request("GET", detail_path)
    repeated = request("POST", "/v1/classify", {"text": text})
    assert repeated["classification_id"] == classified["classification_id"]
    assert detail["text"] == text
    assert detail["model_version"] == "v2.0.0"
    print("Classification, validation, and persisted lookup passed.", flush=True)

    explanation = request("POST", "/v1/explain", {
        "classification_id": classified["classification_id"], "label": "toxic",
    }, expected=(200, 202))
    if "job_id" in explanation:
        job_path = f"/v1/jobs/{explanation['job_id']}"
        deadline = monotonic() + 120
        while True:
            job = request("GET", job_path)
            if job["status"] == "done":
                explanation = job["result"]
                break
            if job["status"] == "failed":
                raise RuntimeError(f"Explanation job failed: {job['error']}")
            if monotonic() >= deadline:
                raise TimeoutError("Worker did not complete the explanation")
            sleep(0.2)
    assert explanation["spans"]
    assert explanation["label"] == "toxic"
    for span in explanation["spans"]:
        assert 0 <= span["start"] < span["end"] <= len(text)
        assert span["text"] == text[span["start"]:span["end"]]
    cached = request("POST", "/v1/explain", {
        "classification_id": classified["classification_id"], "label": "toxic",
    })
    assert cached == explanation

    correction = {label: False for label in LABELS}
    corrected = request("POST", detail_path + "/corrections", {
        "corrected_labels": correction, "moderator_id": "container-smoke",
    }, expected=201)
    assert corrected["corrected_labels"] == correction
    detail = request("GET", detail_path)
    assert detail["corrected"] is True
    assert detail["corrected_labels"] == correction
    assert request("GET", "/v1/status")["state"] == "ready"
    print("Real Torch explanation, correction, and runtime status passed.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", help="API URL; use only against a disposable test database")
    smoke(parser.parse_args().base_url)
