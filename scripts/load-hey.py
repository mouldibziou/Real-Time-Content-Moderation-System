"""Concurrent hey workloads for a disposable API database; never use on production."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from threading import Event
from time import monotonic, sleep
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4


LABELS = ("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")


def http(base, method, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = Request(base + path, data=data, method=method,
                  headers={"Content-Type": "application/json"})
    start = monotonic()
    try:
        with build_opener(ProxyHandler({})).open(req, timeout=120) as response:
            status, body = response.status, json.load(response)
    except HTTPError as error:
        status, body = error.code, error.read().decode(errors="replace")
    except (URLError, OSError) as error:
        status, body = 0, str(error)
    return status, body, monotonic() - start


def parse_report(report):
    codes = {code: int(count) for code, count in
             re.findall(r"\[(\d{3})\]\s+(\d+) responses", report)}
    def number(pattern):
        match = re.search(pattern, report)
        if not match:
            return None
        value = float(match.group(1))
        return value if math.isfinite(value) else None
    elapsed = number(r"Total:\s+([\d.]+) secs")
    succeeded = sum(count for code, count in codes.items() if code.startswith("2"))
    return {"status_counts": codes,
            "elapsed_seconds": elapsed,
            "successful_responses": succeeded,
            "successful_rps": succeeded / elapsed if elapsed else None,
            "rps": number(r"Requests/sec:\s+([\d.]+)"),
            "mean_seconds": number(r"Average:\s+([\d.]+)"),
            "p95_seconds": number(r"95%+ in\s+([\d.]+)"),
            "p99_seconds": number(r"99%+ in\s+([\d.]+)"),
            "transport_errors_reported": "Error distribution:" in report}


def hey(base, method, path, payload, options, output):
    command = ["hey", "-t", "120", "-m", method, *options]
    if payload is not None:
        command += ["-T", "application/json", "-d", json.dumps(payload)]
    command.append(base + path)
    result = subprocess.run(command, capture_output=True, text=True, timeout=450)
    report = result.stdout + result.stderr
    with output.open("a") as stream:
        stream.write(report + "\n")
    parsed = parse_report(report)
    parsed["exit_code"] = result.returncode
    return parsed


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * fraction) - 1]


def observations(out):
    queues = {"samples": 0, "http_statuses": Counter(), "active_samples": 0,
              "max_waiting_classifications": 0, "max_waiting_explanations": 0}
    for line in (out / "runtime-status.jsonl").read_text().splitlines():
        sample = json.loads(line)
        queues["samples"] += 1
        queues["http_statuses"][str(sample["http_status"])] += 1
        body = sample["body"]
        if sample["http_status"] == 200 and isinstance(body, dict):
            queues["active_samples"] += int(body.get("operation_active", False))
            for category in ("classifications", "explanations"):
                key = f"max_waiting_{category}"
                queues[key] = max(queues[key], body.get(f"waiting_{category}", 0))
    queues["http_statuses"] = dict(queues["http_statuses"])
    resources = {}
    sampling_errors = []
    for line in (out / "docker-stats.jsonl").read_text().splitlines():
        sample = json.loads(line)
        if sample["stderr"]:
            sampling_errors.append(sample["stderr"])
        for raw in sample["stdout"].splitlines():
            item = json.loads(raw)
            name = item["Name"]
            entry = resources.setdefault(name, {"peak_cpu_percent": 0,
                                                "peak_memory_percent": 0})
            entry["peak_cpu_percent"] = max(entry["peak_cpu_percent"],
                                            float(item["CPUPerc"].rstrip("%")))
            memory_percent = float(item["MemPerc"].rstrip("%"))
            if memory_percent >= entry["peak_memory_percent"]:
                entry["peak_memory_percent"] = memory_percent
                entry["memory_at_peak"] = item["MemUsage"]
            entry["last_network_io"] = item["NetIO"]
            entry["last_block_io"] = item["BlockIO"]
    native_file = out / "native-process.jsonl"
    if native_file.exists():
        entry = {"peak_rss_mib": 0, "peak_ps_cpu_percent": 0}
        for line in native_file.read_text().splitlines():
            sample = json.loads(line)
            if "rss_kib" in sample:
                entry["peak_rss_mib"] = max(entry["peak_rss_mib"], sample["rss_kib"] / 1024)
                entry["peak_ps_cpu_percent"] = max(entry["peak_ps_cpu_percent"], sample["cpu_percent"])
        resources["native_api"] = entry
    return {"runtime_queues": queues, "resources": resources,
            "resource_sampling_errors": sampling_errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--duration", type=int, default=180)
    parser.add_argument("--classify-clients", type=int, default=2)
    parser.add_argument("--api-container", default="toxicity-load-api")
    parser.add_argument("--db-container", default="toxicity-load-db")
    parser.add_argument("--no-docker-stats", action="store_true")
    parser.add_argument("--server-pid", type=int, help="Sample a native server with ps instead of Docker")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--disposable-db", action="store_true", required=True,
                        help="Confirm that creating classifications/corrections is safe")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    if urlparse(base).hostname not in ("localhost", "127.0.0.1"):
        parser.error("This local test only accepts localhost/127.0.0.1")
    if args.duration < 1 or args.classify_clients < 1:
        parser.error("Duration and client count must be positive")
    if not shutil.which("hey"):
        parser.error("hey is required (brew install hey)")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.output or Path("/tmp") / f"toxicity-load-{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    seed_tag = str(uuid4())
    status, readiness, _ = http(base, "GET", "/v1/health/ready")
    if status != 200 or readiness.get("is_ready") is not True:
        raise RuntimeError("API is not ready")

    def seed(count, record=None):
        texts = [f"Load {seed_tag} {uuid4()}: You are rude and unhelpful. "
                 "Please contribute respectfully to this discussion." for _ in range(count)]
        endpoint = "/v1/classify" if count == 1 else "/v1/classify_batch"
        payload = {"text": texts[0]} if count == 1 else {"texts": texts}
        code, body, elapsed = http(base, "POST", endpoint, payload)
        if record:
            record.write(json.dumps({"path": endpoint, "status": code,
                                     "seconds": elapsed}) + "\n")
            record.flush()
        if code != 200:
            return []
        items = [body] if count == 1 else body["results"]
        ids = [item["classification_id"] for item in items]
        deadline = monotonic() + 30
        for cid in ids:
            while True:
                code, detail, _ = http(base, "GET", f"/v1/classifications/{cid}")
                if code == 200:
                    break
                if code not in (404, 503) or monotonic() >= deadline:
                    raise RuntimeError(f"Stored ID {cid} unavailable: HTTP {code}")
                sleep(0.2)
        return ids

    ids = seed(3)
    if len(ids) != 3:
        raise RuntimeError("Unable to seed verified classification IDs")
    # Warm only these IDs; fresh-workflow lanes use unique IDs on every iteration.
    for path, payload in (
        ("/v1/explain", {"classification_id": ids[0], "label": "toxic"}),
        ("/v1/explain_batch", {"classification_ids": ids[1:], "label": "toxic"}),
    ):
        code, _, _ = http(base, "POST", path, payload)
        if code not in (200, 202):
            raise RuntimeError(f"Preflight {path}: HTTP {code}")
    # Wait for worker completion before measuring the cached lanes.
    for cid in ids:
        deadline = monotonic() + 120
        while True:
            code, detail, _ = http(base, "GET", f"/v1/classifications/{cid}")
            if code == 200 and detail.get("explanations"):
                break
            if monotonic() >= deadline:
                raise RuntimeError("Explanation cache failed to persist")
            sleep(0.2)

    # q is requests per second PER hey client. All lanes run simultaneously.
    lanes = [
        ("live", "GET", "/v1/health/live", None, 1, 1),
        ("ready", "GET", "/v1/health/ready", None, 1, 1),
        ("model_info", "GET", "/v1/model_info", None, 1, 1),
        ("status", "GET", "/v1/status", None, 1, 1),
        ("classify", "POST", "/v1/classify", {"text": f"Repeated load {seed_tag}"},
         args.classify_clients, 0),
        ("classify_batch", "POST", "/v1/classify_batch",
         {"texts": [f"Batch load {seed_tag} {i}" for i in range(8)]}, 1, 0),
        ("explain_cached", "POST", "/v1/explain",
         {"classification_id": ids[0], "label": "toxic"}, 1, 3),
        ("explain_batch_cached", "POST", "/v1/explain_batch",
         {"classification_ids": ids[1:], "label": "toxic"}, 1, 3),
        ("list", "GET", "/v1/classifications?limit=100&offset=0", None, 2, 20),
        ("detail", "GET", f"/v1/classifications/{ids[0]}", None, 1, 10),
        ("corrections", "POST", f"/v1/classifications/{ids[0]}/corrections",
         {"corrected_labels": dict.fromkeys(LABELS, False),
          "moderator_id": "local-load-test"}, 1, 10),
    ]
    results = {}
    stopped = Event()
    start = monotonic()
    deadline = start + args.duration
    print(f"Reports: {out.resolve()}", flush=True)
    print(f"Starting {args.duration}s simultaneous load on all 11 API operations; "
          "cached and fresh explanations are measured separately.", flush=True)

    def static(lane):
        name, method, path, payload, clients, rate = lane
        options = ["-z", f"{max(0.1, deadline - monotonic()):.3f}s", "-c", str(clients)]
        if rate:
            options += ["-q", str(rate)]
        results[name] = hey(base, method, path, payload, options, out / f"{name}.txt")

    def fresh(count):
        name = "explain_fresh" if count == 1 else "explain_batch_fresh"
        codes = Counter()
        successful_times = []
        attempts = 0
        failures = []
        with (out / f"{name}-provisioning.jsonl").open("w") as record:
            while monotonic() < deadline and not stopped.is_set():
                try:
                    new_ids = seed(count, record)
                    if not new_ids:
                        sleep(0.5)
                        continue
                    if monotonic() >= deadline:
                        break
                    payload = ({"classification_id": new_ids[0]} if count == 1
                               else {"classification_ids": new_ids})
                    payload["label"] = "toxic"
                    path = "/v1/explain" if count == 1 else "/v1/explain_batch"
                    one = hey(base, "POST", path, payload,
                              ["-n", "1", "-c", "1"], out / f"{name}.txt")
                    attempts += 1
                    codes.update(one["status_counts"])
                    if sum(one["status_counts"].get(code, 0) for code in ("200", "202")) == 1:
                        successful_times.append(one["mean_seconds"])
                    elif not one["status_counts"]:
                        codes["transport_error"] += 1
                except Exception as error:
                    failures.append(str(error))
                    record.write(json.dumps({"workflow_error": str(error)}) + "\n")
                    sleep(0.5)
        results[name] = {"status_counts": dict(codes), "attempts": attempts,
                         "rps": (codes["200"] + codes["202"]) / args.duration,
                         "successful_rps": (codes["200"] + codes["202"]) / args.duration,
                         "p95_seconds": percentile(successful_times, .95),
                         "p99_seconds": percentile(successful_times, .99),
                         "workflow_errors": failures,
                         "percentiles": "Accepted explanation requests; measures submission, not worker completion"}

    def monitor():
        with (out / "runtime-status.jsonl").open("w") as status_file, \
             (out / "docker-stats.jsonl").open("w") as stats_file, \
             (out / "native-process.jsonl").open("w") as native_file:
            while monotonic() < deadline and not stopped.is_set():
                code, body, elapsed = http(base, "GET", "/v1/status")
                status_file.write(json.dumps({"offset": monotonic() - start,
                                              "http_status": code, "body": body,
                                              "seconds": elapsed}) + "\n")
                status_file.flush()
                if args.server_pid:
                    sample = subprocess.run(
                        ["ps", "-p", str(args.server_pid), "-o", "pid=,%cpu=,rss="],
                        capture_output=True, text=True, timeout=5)
                    values = sample.stdout.strip().split()
                    if len(values) == 3:
                        native_file.write(json.dumps({"offset": monotonic() - start,
                            "pid": int(values[0]), "cpu_percent": float(values[1]),
                            "rss_kib": int(values[2])}) + "\n")
                    else:
                        native_file.write(json.dumps({"sampling_error": sample.stderr,
                                                       "stdout": sample.stdout}) + "\n")
                    native_file.flush()
                elif not args.no_docker_stats and shutil.which("docker"):
                    sample = subprocess.run(
                        ["docker", "stats", "--no-stream", "--format", "{{json .}}",
                         args.api_container, args.db_container],
                        capture_output=True, text=True, timeout=15)
                    stats_file.write(json.dumps({"offset": monotonic() - start,
                                                 "stdout": sample.stdout,
                                                 "stderr": sample.stderr}) + "\n")
                    stats_file.flush()
                stopped.wait(2)

    try:
        with ThreadPoolExecutor(max_workers=len(lanes) + 3) as pool:
            futures = [pool.submit(static, lane) for lane in lanes]
            futures += [pool.submit(fresh, 1), pool.submit(fresh, 2), pool.submit(monitor)]
            while monotonic() < deadline:
                sleep(min(10, max(0, deadline - monotonic())))
                print(f"Elapsed {min(args.duration, round(monotonic() - start))}s / "
                      f"{args.duration}s", flush=True)
            stopped.set()
            for future in futures:
                future.result()
    finally:
        stopped.set()
    results["metadata"] = {"base_url": base, "duration_seconds": args.duration,
        "elapsed_including_drain_seconds": monotonic() - start,
        "classify_clients": args.classify_clients,
        "server_pid": args.server_pid,
        "notes": ["Static hey percentiles include non-success HTTP responses.",
                  "Repeated classification payloads exercise duplicate handling.",
                  "Fresh lanes create unique stored IDs; provisioning is separately logged.",
                  "Fresh hey subprocesses add client-side gaps, not measured HTTP latency.",
                  "This closed-loop workload is not a fixed-arrival-rate capacity test."]}
    results["observations"] = observations(out)
    (out / "summary.json").write_text(json.dumps(results, indent=2))
    print("\nLane                       statuses                     p95(ms) success/s")
    for name, result in results.items():
        if name in ("metadata", "observations"):
            continue
        latency = result.get("p95_seconds")
        p95 = "n/a" if latency is None else f"{latency * 1000:.1f}"
        rps = result.get("successful_rps")
        rate = "n/a" if rps is None else f"{rps:.2f}"
        print(f"{name:26} {str(result['status_counts']):28} {p95:>8} {rate:>7}")
    print(f"\nSaved results to {out.resolve()}")


if __name__ == "__main__":
    main()
