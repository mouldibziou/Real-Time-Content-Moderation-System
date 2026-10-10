# Local Prometheus and Grafana

Run the API, explanation worker, PostgreSQL, Redis, Prometheus, and Grafana together:

```sh
docker compose -p toxicity-observability -f compose.yaml -f compose.monitoring.yaml up -d --build
```

The separate `toxicity-observability` project gets its own database and Redis volumes. This is a local demo; its services listen on localhost. The monitoring overlay is not used by `compose.cloud.yaml`.

Open:

- Dashboard: http://localhost:3000/d/toxicity-local
- Prometheus targets: http://localhost:9090/targets — both targets should be UP.
- Raw API measurements: http://localhost:8000/metrics/

Grafana permits anonymous **viewing** locally, so the dashboard opens without a login. To administer Grafana, use `admin` / `local-demo-only`, or set `GRAFANA_ADMIN_PASSWORD` before the first startup. Grafana stores the initialized password in its volume; changing this environment variable afterward does not reset it.

Prometheus scrapes the API and worker every five seconds. Grafana's datasource and dashboard are provisioned from `monitoring/`, so no manual setup is necessary. The worker metrics port remains private to the Docker network. The local demo uses worker port 9101.

## Generate traffic

Wait for the API's `/v1/health/ready` endpoint to return HTTP 200, then run:

```sh
python3 scripts/demo-metrics.py
```

This creates synthetic classification and explanation rows in the demo database. Each of six rounds classifies three texts, sends one intentionally invalid request (422), queues and waits for an explanation, and then requests its cached result. It creates real model work, not mocked metrics. Use `--rounds 12 --interval 10` for a longer live demonstration. If `API_PORT` differs, pass `--base-url http://127.0.0.1:YOUR_PORT`.

Allow at least two scrapes for rate charts to populate. When traffic stops, request rates fall to zero and recent latency charts become empty; lifetime counters and earlier history remain available. Histogram quantiles are estimates, especially for this small sample. Predictions are multi-label, so positive percentages across labels need not total 100%.

The dashboard distinguishes API response time, first queue wait, worker attempt duration, and successful explanation end-to-end duration. HTTP `/explain` latency measures submission or cache access; the worker measures the background job.

## Storage and shutdown

Prometheus writes its time-series database to the `prometheus_data` named volume. Retention is 15 days or 1 GB, whichever is reached first. Retention is applied to storage blocks, so allow some additional disk space for current data and the write-ahead log. Grafana configuration is stored in `grafana_data`. These volumes survive container recreation.

Stop the demo while preserving all its data:

```sh
docker compose -p toxicity-observability -f compose.yaml -f compose.monitoring.yaml down
```

Use the startup command again to resume. Adding `--volumes` to shutdown permanently deletes this demo's database, Redis data, Prometheus history, and Grafana settings.

Dashboard edits belong in `monitoring/grafana/dashboards/toxicity.json`; the provisioned dashboard is read-only in Grafana. These metrics describe application behavior. Host CPU and memory monitoring can be added separately when deploying.

For the five deployable images and cloud Compose setup, see [monitoring-deployment.md](monitoring-deployment.md).
