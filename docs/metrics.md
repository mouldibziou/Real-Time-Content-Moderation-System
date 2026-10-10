# Prometheus metrics

The API exposes `GET /metrics/` on its existing port 8000. The worker starts an
independent metrics HTTP server on port 9101. Both return Prometheus text, not
JSON. No application metrics are saved in Postgres or Redis; counters and
histograms are process-local. Postgres is queried for current job state.

## Exposed measurements

| Target | Metric | Type | Labels |
| --- | --- | --- | --- |
| API | `http_requests_total` | Counter | `method`, `route`, `status_code` |
| API | `http_request_duration_seconds` | Histogram | `method`, `route` |
| API | `classification_items_total` | Counter | None |
| API | `classification_predictions_total` | Counter | `label`, `decision` |
| API | `explanation_jobs_created_total` | Counter | None |
| API | `explanation_cache_hits_total` | Counter | None |
| API | `explanation_jobs_current` | Gauge | `status`: `queued`, `running` |
| API | `explanation_queue_collection_success` | Gauge | None |
| Worker | `explanation_job_attempts_total` | Counter | None |
| Worker | `explanation_job_attempt_duration_seconds` | Histogram | None |
| Worker | `explanation_job_queue_wait_seconds` | Histogram | None |
| Worker | `explanation_job_end_to_end_duration_seconds` | Histogram | None |
| Worker | `explanation_job_outcomes_total` | Counter | `outcome`: `completed`, `retry`, `failed` |

Histograms expose `_bucket`, `_count`, and `_sum`; each includes the automatic
`+Inf` bucket. The official client also exports metric `_created` timestamps.
The registries deliberately include only these application measurements, not
default Python/process collectors. Labelled HTTP series appear on first use;
the twelve prediction label/decision pairs start at zero.

## Counting and timing rules

- HTTP counts and durations cover completed responses and unexpected exceptions
  producing 500s. If a response already started before an exception, its actual
  status is retained. Requests cancelled without a response are not counted.
  Scrapes, the `/metrics` redirect and health probes are excluded. Route labels
  use templates, unmatched paths use `unmatched`, and unrecognized methods use
  `OTHER`. No text, identifiers, query strings or user labels are exported.
- Classification items and decisions are recorded after successful database
  commit. Batches count every input text, including duplicates; repeated
  submissions count again. Each real prediction contributes one decision for
  each of the six toxicity labels. `decision` is `positive` or `negative`.
- Newly committed jobs count once when `created=True`. Reusing a job adds no
  creation. A Redis notification failure does not undo the durable job count.
  Cache hits count returned explanation items, including batch cache hits and
  results discovered during the post-submission cache check.
- Worker claims increment attempts only after commit. Duplicate notifications
  and failed claims do not increment them. Attempt duration starts immediately
  after the claim returns and includes classification lookup, local batching
  waits, computation and persistence. It is observed once per owned attempt,
  including errors, lost ownership and cancellation. Process termination can
  prevent observations from being recorded.
- First-claim queue wait uses PostgreSQL `updated_at - created_at` when
  `attempts == 1`. It does not measure subsequent retry waits.
- Successful end-to-end duration runs from job creation to the successful
  result commit, including retries. It compares the stored creation timestamp
  with the worker's UTC clock, so keep database and worker clocks synchronized.
  Network latency and client polling intervals are excluded.
- Outcome counters increment only after database transitions commit. Lease
  expiry recovery also counts a `retry` or terminal `failed` transition.
  Lost ownership and failed commits do not produce a completed outcome.

HTTP buckets in seconds: `0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30`.
Job buckets: `0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600, 1800`.

## Queue collection and replicas

The API reads queued/running job counts during each scrape, with a two-second
timeout. Successful empty queues export zero. If the database is unavailable,
the endpoint still returns HTTP 200 and all in-memory measurements, but exports
`explanation_queue_collection_success 0` and omits queue-depth series. It never
reports a failed collection as an empty queue.

These gauges describe a shared database, not independent per-replica queues.
For the initial single-API deployment, scrape that API. With multiple replicas,
designate one queue collector or use `max by (status)` instead of summing these
gauges. In-memory counters must be scraped separately on every replica. The
current deployment uses one Uvicorn process per API container; this does not
enable Python client's multiprocess mode.

All counters reset on process restart. Prometheus's `rate`/`increase` functions
handle observed resets. Metrics are best-effort telemetry: a process can crash
between a database commit and a counter increment. They are not an audit ledger.

## Scraping on the Docker network

Rebuild the API and worker images before starting them. The worker's internal
port is exposed in both Compose files but is not published on the host.
`WORKER_METRICS_PORT` defaults to `9101`; `WORKER_METRICS_HOST` defaults to
`0.0.0.0` inside the container. The HTTP server is closed during worker shutdown.

Connect your Prometheus container to the same Compose `svc` network and use:

```yaml
global:
  scrape_interval: 15s
  scrape_timeout: 5s
scrape_configs:
  - job_name: toxicity_api
    metrics_path: /metrics/
    static_configs:
      - targets: ["api:8000"]
  - job_name: toxicity_worker
    metrics_path: /metrics/
    static_configs:
      - targets: ["worker:9101"]
```

The example assumes the default worker metrics port. Keep `/metrics` and
`/metrics/` private at your public reverse proxy. Prometheus storage, retention,
Grafana and EC2 log collection remain deployment configuration, separate from
these endpoint changes. This change does not start a Prometheus or Grafana server.

For a local API, inspect the response with:

```sh
curl http://127.0.0.1:8000/metrics/
```

## Example queries

Request rate:

```promql
sum by (route) (rate(http_requests_total[5m]))
```

HTTP p95 latency in seconds:

```promql
histogram_quantile(0.95, sum by (le, route) (rate(http_request_duration_seconds_bucket[5m])))
```

Positive fraction for each toxicity label (multilabel fractions do not sum to 1):

```promql
sum by (label) (rate(classification_predictions_total{decision="positive"}[5m]))
/
sum by (label) (rate(classification_predictions_total[5m]))
```

Explanation p95 time to availability, including queue wait and retries:

```promql
histogram_quantile(0.95, sum by (le) (rate(explanation_job_end_to_end_duration_seconds_bucket[5m])))
```

References: [Python client](https://prometheus.github.io/client_python/),
[histograms](https://prometheus.io/docs/practices/histograms/),
[instrumentation](https://prometheus.io/docs/practices/instrumentation/).

## Verification performed

Both API and worker targets built with the pinned, hash-verified client and
passed dependency checks. All 209 tests passed in the Linux/Python 3.12 runtime
under a read-only filesystem, including real-model inference and disposable
PostgreSQL lifecycle tests. No tests were skipped.

A real HTTP smoke test with the new API/worker images, PostgreSQL and Redis
verified one classification, all six prediction decisions, explanation creation,
worker claim and completion, a subsequent cache hit, and observations in HTTP,
attempt, first-claim queue-wait and end-to-end latency histograms. Compose
configuration validation passed for local and cloud configurations.

## Try the local dashboard

See [Local Prometheus and Grafana](local-monitoring.md) for a ready-to-run Compose overlay, a provisioned dashboard, and synthetic traffic generation.
