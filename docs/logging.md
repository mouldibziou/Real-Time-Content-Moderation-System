# Application logging

The API and worker write one JSON object per line to stdout. No new service,
logging dependency, database table, or log file is required. During deployment,
configure the hosting platform to collect both containers' stdout/stderr and set
its retention period. Container output alone is not permanent storage.

## Read the logs locally

```sh
docker compose logs --follow api worker
```

Existing images need rebuilding to include these changes. `LOG_LEVEL` defaults
to `INFO`. Set it in the API or worker environment to `DEBUG` when investigating
a problem; restore `INFO` afterwards. The API sets its application logging level
at lifespan startup; Uvicorn's own loggers use `docker/logging.json`.

Every API response carries a generated `X-Request-ID`, including handled errors
and unexpected 500 responses. Use this ID to find the associated request and
application events. Client-provided request IDs are not trusted or reused.

Example (IDs shortened for readability):

```json
{"timestamp":"2026-10-10T14:05:00+00:00","level":"INFO","service":"api","logger":"app.api.routes","event":"classification.completed","message":"classification.completed","request_id":"request-123","classification_id":"classification-456","model_version":"v2.0.0","stage":"response","batch_size":1,"duration_ms":182.5}
```

## What gets recorded

| Event | Meaning |
| --- | --- |
| `request.completed` | Response handling ended; includes HTTP method, route template, status, duration and request ID |
| `request.failed` | An unexpected exception escaped request handling; includes stage, error type and code locations |
| `request.aborted` | Handling ended without a complete response, for example cancellation |
| `classification.completed` | Single prediction finished and its classification was committed |
| `classification.batch_completed` | Batch predictions and their transaction completed; includes batch size |
| `explanation.cache_hit` | An existing explanation was available |
| `explanation.job_created` / `explanation.job_reused` | The database returned a new or existing durable job |
| `explanation.notification_failed` | Redis notification failed; database polling can still discover the job |
| `explanation.worker_notified` | Redis accepted the notification (`DEBUG`) |
| `explanation.batch_submitted` | Batch submission finished; includes created and cached counts |
| `explanation.job_claimed` | A worker obtained ownership of a job attempt |
| `explanation.job_completed` | The result and successful job status were committed by the current owner |
| `explanation.attempt_failed` | Processing or persistence raised an exception |
| `explanation.retry_scheduled` / `explanation.job_failed` | The database committed a retry or terminal failure |
| `explanation.job_recovered` | An expired worker lease was recovered for another attempt |
| `explanation.ownership_lost` | The worker could no longer renew or finalize its ownership |
| `correction.saved` | A moderator correction was committed; verdict and moderator identity are omitted |
| `worker.ready` / `worker.stopped` / `worker.failed` | Worker lifecycle |
| `worker.redis_ready` / `worker.redis_retry` / `worker.redis_failed` | Redis availability and reconnect attempts |
| `database.ready` / `database.unavailable` / `database.request_failed` | Database startup and request failures |

Existing runtime lifecycle messages also use the JSON formatter. Uvicorn access
logs remain enabled and are emitted as `http.access`; application request summaries
add timing and context. Successful liveness/readiness probes do not produce
application summaries, but unhealthy probes do. Uvicorn can still log probes.

Expected client errors (400/404/409/422) produce an `INFO` request summary, rather
than a server error with a traceback. Unexpected exceptions and 5xx summaries
are `ERROR`. Recovery conditions such as lost ownership and failed Redis
notifications are `WARNING`.

## Follow a client's explanation

1. Find the API event using the response's `X-Request-ID`.
2. Find `explanation.job_created` or `explanation.job_reused` and its `job_id`.
3. Search that `job_id` across API and worker logs to follow attempts and results.

The worker identifies work by job ID and attempt; it does not claim to share the
originating HTTP request ID. This supports both Redis notifications and recovery
through database polling without changing the job schema. Polling requests have
their own request IDs and include the requested job ID in their summaries.

`duration_ms` uses a monotonic clock. A request summary measures application
response handling. Classification events measure runtime waiting/execution and
persistence, starting inside the route. Worker completion events measure batch
execution and persistence up to that job's commit; all jobs in a batch share its
start time. None of these timings measures time on the client's network or the
explanation job's wait before being claimed. They are log fields; application
metrics and distributed tracing are separate follow-up work.

## Privacy and scope

Application events use selected IDs, stages, counts and durations. They do not
include submitted comments, explanation spans, prediction bodies, headers,
credentials, moderator identities or query strings. Request summaries use route
templates rather than arbitrary URL paths. Uvicorn access records omit query
strings and client addresses but retain the request path.

Exception output includes the exception class and stack frame locations, not
exception messages, chained exception values, source lines, local variables or
SQL parameters. This intentionally trades some diagnostic detail for protection
against copying client text and credentials into logs. Third-party/native library
output and independently configured cloud access logs need their own review.

Job status in Postgres remains authoritative. A crash can occur between a commit
and its log entry; logs are diagnostic evidence, not an audit ledger or proof of
delivery to the client's device.

## Verification

```sh
venv/bin/python -m unittest tests.test_application_logging -v
venv/bin/python -m unittest discover -s tests -v
```

The focused tests exercise concurrent request isolation, request-ID headers,
executor context propagation, privacy, persistence failures, Redis fallback,
worker retries and ownership loss. Live database delivery tests remain opt-in
through `TOXICITY_DB_TESTS=1` with a disposable database ending in `_tests`.
