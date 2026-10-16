# Cloud traffic measurements

Run `scripts/cloud-load.js` using k6 on a separate client (initially the Mac), not
on the application EC2 host. This creates synthetic classification records and
explanation jobs in the deployed database, and does not delete or correct data.
The existing `load-hey.py` remains for disposable local stress tests.

Install k6 with `brew install k6`. Keep the SSH tunnel open and run from the
repository root:

```sh
mkdir -p /tmp/toxicity-load-results
k6 run -e BASE_URL=http://localhost:18000 -e RATE=1 -e DURATION=2m \
  -e REPORT_PATH=/tmp/toxicity-load-results/rate-1.json scripts/cloud-load.js
```

This schedules one **journey**, not one HTTP request, per second regardless of
response speed. The assumed mix is 70% single classification, 10% four-item
batch, and 20% reads. Five percent of single classifications request a fresh
explanation; these poll every two seconds for up to 120 seconds and then verify
a cache hit. Text samples include short/long and multilingual comments. These
are workload assumptions, not an observed distribution of real clients. The
synthetic suffix makes inputs distinct. Change `EXPLAIN_SHARE` (0 to 1) for a
different explanation mix; `EXPLAIN_SHARE=0` isolates synchronous API capacity.

After an initial successful test, compare separate runs at RATE=2, RATE=5, and
RATE=10. Keep the same duration and workload mix, save each report separately,
and wait for the explanation backlog to drain between tests. Do not jump to the
next rate when errors, sustained queue growth, or memory pressure occur. Two
minutes is an initial probe; validate sustainable capacity with longer tests
after selecting a load level. Small explanation samples cannot establish a
reliable p99. No response-time SLO is assumed by the script's thresholds: the
`avg>=0` thresholds only expose per-operation latency submetrics.

Read the saved report:

* `successful_request_ms`: client wall-clock latency for successful HTTP
  responses, including connection work; compare per-operation avg/p50/p95/p99.
* `client_wall_ms`: includes errors, useful when investigating fast rejections.
* `successful_requests`: successful HTTP count/rate, including job polling.
* `classified_items`: texts classified (a four-item batch contributes four).
* `application_request_failed`, `http_req_failed`, `response_statuses`: errors
  and status codes. Status 0 indicates a client/transport failure.
* `dropped_iterations`: scheduled journeys k6 could not start because all VUs
  were busy. The requested rate was not fully delivered when this rises.
* `explanation_jobs_submitted`, `observed_explanation_completed`, and
  `observed_explanation_completion_ms`: accepted jobs vs observed completions.
  Timeouts may leave jobs queued/running; check worker metrics afterward.
* Built-in HTTP timings separate connecting, TLS handshake, sending, waiting
  (TTFB), and receiving. HTTP duration excludes initial connection work. Under
  an SSH tunnel, k6 sees HTTP to localhost, so TLS handshake measures no public
  HTTPS handshake and TCP connecting measures the local tunnel connection.

Use Grafana's matching time range for HTTP/server p95, classification queue,
explanation queued/running counts, worker wait/runtime/end-to-end histograms,
and monitoring target health. Uvicorn admission 503s may occur before application
instrumentation; the client's status counts are essential. Do not infer network
latency by subtracting two percentiles: these are different samples and scopes.

In a second EC2 SSH terminal, observe `docker stats` while testing. Save a
baseline and post-test `docker compose --env-file .env.cloud -f compose.cloud.yaml
ps -a` snapshot to detect restarts and health changes. Inspect RDS CPU, connection
count, free memory, and CPU credits in CloudWatch; also check ElastiCache memory,
connections and evictions. These are separate from application counters.

The initial results describe **Mac -> SSH tunnel -> API -> managed services**.
They do not establish direct public-internet/TLS performance. Once HTTPS is
configured, rerun the same test with the public URL from Tunisia and relevant
European client locations. Avoid exposing raw port 8000 just to run a benchmark.
