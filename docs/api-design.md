# API design

## Layers

- `app/api/main.py`: `create_app` factory with injectable settings and
  services builder. Inner lifespan owns timing: build runtime, `start()`,
  build single-worker pool, publish to `app.state`, yield, then `stop()`
  and pool shutdown. Soft Postgres check (log-only) and centralized
  error handlers live here. No business logic, no routing.
- `app/api/routes.py`: thin HTTP adapter. Declares versioned paths,
  adapts Pydantic in and out, splits fast vs model lanes. No ML, no SQL
  except through repository calls.
- `app/api/schemas.py`: transport contract. Requests validate shape
  early, responses serialize domain dataclasses to JSON. `ToxicLabel` is
  intentionally duplicated from settings for v1 (documented).
- `app/api/dependencies.py`: `get_runtime`/`get_executor` translate
  missing state to `RuntimeNotReadyError` (503, not 500). `_call_runtime`
  is the non-blocking bridge: sync torch to the pool, await the future
  so the loop stays free.
- Domain (`runtime`, `classifier`, `explainer`, `services`): pure ML,
  no HTTP or DB knowledge. `ApplicationRuntime` serializes model access
  with a `Condition`; inference releases the lock while running.
- Persistence (`app/db`): `engine` pool + per-request sessions,
  `models` mirroring DDL, `repository` stateless async functions,
  `queue` Redis doorbell. Alembic versions everything; hand SQL retired.
- Worker (`app/worker.py`): separate process, own runtime and model,
  `BLPOP` loop with batch coalescing. Postgres is truth, Redis wakes.

## Concurrency

Two threads per API process: event-loop thread for fast paths and async
Postgres (`asyncpg` releases the GIL on socket wait), single pool thread
for blocking torch (C extensions release the GIL while computing).
Fast paths (`health`, `ready`, `status`, `model_info`) run inline in
microseconds and never enter the operation queue. Model paths offload
via `_call_runtime`. One uvicorn worker per pod; scale by pods, never by
workers, because queues and model weights are per-process.

## Endpoints (all under `/v1` except docs)

- `GET /health/live`: process alive, no deps, always 200 when Python
  answers. Restart on failure.
- `GET /health/ready`: model ready boolean. Orchestrator traffic gate.
  Checks the model/Triton and PostgreSQL; outages return 503.
- `GET /status`: queue depths and capacities for backpressure dashboards.
- `GET /model_info`: type, device, labels, thresholds snapshot.
- `POST /classify {text}`: sync pool inference (~50 ms isolated),
  returns `{text, labels, classification_id}` after commit. Repeated text
  reuses the canonical stored ID. Database failures return 503.
- `POST /classify_batch {texts[]}`: same in input order, one committed
  transaction with canonical duplicate IDs.
- `POST /explain {classification_id, label}`: strict 404 if unknown.
  Cache hit returns 200 spans with zero GPU. Miss creates-or-returns an
  idempotent job and returns 202 `{job_id}`; worker computes.
- `POST /explain_batch {classification_ids[], label}`: validates all ids
  (404 on first unknown), returns 202 `{jobs[]}` with per-id done/queued
  states for uniform polling.
- `GET /jobs/{job_id}`: Postgres truth with `result` spans on done,
  `error` on failed.
- `GET /classifications?limit&offset&corrected=`: newest-first feed.
  `corrected` None shows all, false is the moderator queue, true audit.
  SQL `EXISTS` keeps pages exact; batched latest-correction map avoids
  N+1, each item carries the `corrected` flag.
- `GET /classifications/{id}`: immutable prediction plus corrected
  overlay (latest event wins) and all cached explanations. 404 if missing.
- `POST /classifications/{id}/corrections {6 booleans, moderator_id?}`:
  201 append-only. Requires exactly the stored label set (422
  otherwise). History preserved for audit and training.

## Persistence and jobs

`classifications` immutable, `correction_events` append-only with latest
wins for display, `explanations` one row per `(classification_id, label,
model_version)` as a GPU cache, `explanation_jobs` tracking
`queued|running|done|failed` with the same triple unique for idempotent
repeats. Classification responses wait for repository commit, so returned IDs are
immediately addressable; batch failure rolls back the whole transaction. Explains are
fully async: API enqueues, worker drains up to 4 per wake grouped by
label, one pooled batch per group, per-job failure isolation, boot
rebuild from `queued` rows. Redis holds no truth.

## Errors

`NotReady -> 503`, `Busy -> 429`, `Timeout -> 504`, runtime
`ValueError/TypeError -> 422` in the Pydantic envelope. Unknown ids 404.
Duplicates never error. Background and lifespan soft-check failures log
with tracebacks and keep serving.

## Deployment topology

One image, two roles: API pods (classify hot, explains enqueue only)
and worker pods (`python -m app.worker`, own models). Postgres and Redis
shared. Ingress routes by path; HPA scales pools independently. Lockfile
regen required before image builds for new deps.

## Public frontend

`frontend/README.md` documents the Cloudflare Pages build, exact-origin CORS via
`TOXICITY_CORS_ORIGINS`, and the restricted Caddy demo route allowlist.
`model_info` includes `model_version`, `max_text_chars`, and
`max_prediction_batch_size`. The previously measured return-first latency is
historical until the commit-before-response route is rebenchmarked.
