"""Generate synthetic metrics traffic against a disposable local demo database."""
import argparse
import json
from time import monotonic, sleep
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4


def demo(base_url, rounds, interval):
    opener = build_opener(ProxyHandler({}))

    def request(method, path, payload=None, expected=(200,)):
        data = None if payload is None else json.dumps(payload).encode()
        headers = {} if payload is None else {"Content-Type": "application/json"}
        req = Request(base_url.rstrip("/") + path, data=data, headers=headers, method=method)
        try:
            with opener.open(req, timeout=120) as response:
                status, body = response.status, json.load(response)
        except HTTPError as error:
            status, body = error.code, json.load(error)
        if status not in expected:
            raise RuntimeError(f"{method} {path}: expected {expected}, got {status}")
        return body

    for round_number in range(1, rounds + 1):
        tag = str(uuid4())
        examples = (
            "Thank you for your thoughtful help.",
            "You are an idiot and your comments are stupid.",
            "I will kill you, you disgusting piece of shit.",
        )
        classified = [request("POST", "/v1/classify", {"text": f"{text} Demo {tag}"}) for text in examples]
        request("POST", "/v1/classify", {"text": " "}, expected=(422,))
        payload = {"classification_id": classified[1]["classification_id"], "label": "toxic"}
        result = request("POST", "/v1/explain", payload, expected=(200, 202))
        if "job_id" in result:
            deadline = monotonic() + 120
            while True:
                job = request("GET", f"/v1/jobs/{result['job_id']}")
                if job["status"] == "done":
                    break
                if job["status"] == "failed":
                    raise RuntimeError("Demo explanation job failed")
                if monotonic() >= deadline:
                    raise TimeoutError("Demo explanation did not finish within 120 seconds")
                sleep(0.5)
        request("POST", "/v1/explain", payload)  # Exercise the cache after completion.
        print(f"Round {round_number}/{rounds}: 3 predictions, 1 validation error, 1 completed explanation, 1 cache hit", flush=True)
        if round_number < rounds:
            sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--interval", type=float, default=10)
    args = parser.parse_args()
    if args.rounds < 1 or args.interval < 0:
        parser.error("rounds must be positive and interval must be non-negative")
    demo(args.base_url, args.rounds, args.interval)
