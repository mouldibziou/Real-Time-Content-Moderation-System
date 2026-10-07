# Local concurrent container load test

Run only against the disposable test stack. This generates real correction rows
and classifications, not merely read requests. The migration image is a one-off
job; the API image is the HTTP server. Neither image is GPU-enabled.

From the repository root, with the API ready at port 8000 and `hey` installed:

```sh
python3 scripts/load-hey.py --duration 180 --disposable-db
```

All eleven application operations run together for a three-minute traffic window.
Seeding and cache warmup happen beforehand; in-flight requests may finish afterward
(HTTP timeout: 120 seconds). Reports go into a new timestamped directory under
`/tmp`, whose exact path is printed. Choose another unused location with `--output`.

The test uses `hey` for HTTP load, with a Python standard-library coordinator to
create verified IDs, rotate fresh explanation inputs, sample status/resources,
and collect reports. Static request bodies alone would turn explanation load
into cache hits after the first response.

| Workload | Concurrent clients | Rate per client |
| --- | ---: | ---: |
| Single classification | 2 | Unthrottled |
| Classification batch (8 texts) | 1 | Unthrottled |
| Cached explanation | 1 | 3 requests/second |
| Cached explanation batch (2 IDs) | 1 | 3 requests/second |
| Fresh explanation workflow | 1 | Sequential, unthrottled |
| Fresh explanation batch workflow (2 IDs) | 1 | Sequential, unthrottled |
| Classification list (100 records) | 2 | 20 requests/second |
| Classification detail | 1 | 10 requests/second |
| Committed moderator correction | 1 | 10 requests/second |
| Liveness, readiness, metadata, status | 1 each | 1 request/second |

Actual rates may be lower when responses are slow. Fresh workflows classify new
unique texts, verify persistence, then launch a single-request `hey` process for
the explanation. Provisioning results are logged separately. Subprocess startup
adds gaps, so this is not a maximum-throughput or fixed-arrival-rate benchmark.
Repeated classification bodies intentionally exercise duplicate handling; they
do not create a new stored row on every request. Correction requests do commit
new rows, and fresh workflows exercise successful classification writes.

The summary contains per-lane statuses, throughput, and latency percentiles.
Static `hey` percentiles include all HTTP statuses, so always inspect rejection
and error counts alongside latency. `successful_rps` counts only 2xx responses;
the raw `hey` requests/second figure also includes rejections and transport errors.
Fresh explanation percentiles use successful
requests only; their sample size may be small. Raw reports retain transport-error
details. Expected successful statuses are 200, except corrections use 201.

Additional output:

- `runtime-status.jsonl`: operation-active state and runtime queue counters.
- `docker-stats.jsonl`: API/database CPU, memory, network and block-I/O snapshots.
- `*-provisioning.jsonl`: fresh classification requests and workflow failures.
- `summary.json`: machine-readable per-lane results and limitations.

Docker access is required for resource sampling. The test stack uses PostgreSQL
on temporary memory-backed storage; these results do not measure persistent cloud
disk performance or database durability. The API is restricted to two CPUs.
The initial 2 GiB test was killed by Docker for exceeding its memory limit; the
repeat uses 4 GiB. Record resource limits with every result. Linux amd64 emulation
on Apple Silicon is not production hardware.

The current single-thread HTTP executor queues model requests before the runtime
scheduler. Runtime waiting counters can therefore remain zero despite substantial
HTTP waiting. This test observes that behavior; it does not prove runtime queue
capacity, priority, or wait timeout enforcement through HTTP. Uvicorn's separate
concurrency limit may produce 503 rejections before requests reach the routes.

After the baseline, increase classification load in a separate run:

```sh
python3 scripts/load-hey.py --duration 180 --classify-clients 8 --disposable-db
```

This deliberately probes overload; compare successes, 503/429/504 counts, latency,
and memory. Keep hardware and container limits fixed. Do not interpret short error
responses as fast successful inference. Database pool limits, executor waiting,
model execution, and server admission are distinct possible bottlenecks.

The stack containers are `toxicity-load-api`, `toxicity-load-db`, and the exited
`toxicity-load-migrations` job, on network `toxicity-load-test`. PostgreSQL has no
host-published port. Stopping or removing the memory-backed database container
loses its data; restarting it requires running migrations again.

```sh
docker stop --timeout 75 toxicity-load-api
docker stop toxicity-load-db
docker rm toxicity-load-api toxicity-load-db toxicity-load-migrations
docker network rm toxicity-load-test
```

Reference: [hey command options](https://github.com/rakyll/hey).

## Native local server

The same workload can target a native Uvicorn server using a separate test
database. Supply its process ID to sample native RSS and `ps` CPU usage without
querying Docker:

```sh
python3 scripts/load-hey.py --duration 180 --disposable-db --no-docker-stats --server-pid SERVER_PID
```

Replace `SERVER_PID` with the actual Uvicorn process ID. Native RSS is not the
same measurement as Docker cgroup memory; `ps` CPU sampling also differs from
Docker's interval-based CPU measurement. Native execution has no container CPU
or memory cap. Compare request behavior while accounting for architecture,
Python/dependency versions, database storage, resources, and test duration.
