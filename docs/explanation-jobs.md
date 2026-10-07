# Async explanation jobs

## Why async

Single-worker IG holds seconds of GPU per miss. Mixed classify+explain at
full concurrency saturated one worker (classify p95 52 ms isolated vs
~2 s mixed), so the API no longer runs IG on the request path. Classify
stays synchronous with return-first background writes; explains go
through jobs. Same contract for single and batch.

## Components

- Postgres (truth): `classifications`, `explanations`, `explanation_jobs`.
  `explanation_jobs` has UUID PK, `classification_id FK cascade`, `label`,
  `model_version`, `status queued|running|done|failed`, nullable `error`,
  timestamps. Unique `(classification_id, label, model_version)` makes
  repeats return the job instead of doubling GPU.
- Redis (doorbell): list `expl_jobs`. API pushes ids after inserting the
  row; worker `BLPOP`s. Crash loses nothing: rows rebuild the queue.
- API (`app/api/routes.py`): strict load by id (404 if unknown), cache
  check first. Hit returns 200 spans with zero GPU. Miss creates-or-returns
  the job, pushes Redis (log-only on Redis failure), returns 202 with the
  job body. Repeat misses return the same job (200 if already done).
- Worker (`app/worker.py`): own process, own `ApplicationRuntime` loaded
  once, own thread. Loop blocks on `BLPOP`, drains up to 4 ids without
  blocking, groups by `(label, model_version)`, one pooled `explain_batch`
  per group, per-job `failed` isolation. Writes spans idempotently, marks
  `done`. Re-pushes `queued` rows on boot. SIGTERM/SIGINT drain via
  5 s pop timeout, then `runtime.stop()`.

## Request flow

1. `POST /classify` returns `{text, labels, classification_id}`, row
   persists in background (duplicate hashes ignored via unique).
2. `POST /explain {classification_id, label}` validates existence (404),
   checks `(id, label, version)` cache: hit 200 spans, else 202 `{job_id}`.
3. `POST /explain_batch {classification_ids[], label}` validates all ids
   (404 on first unknown), returns 202 `{jobs[]}` with per-id done/queued
   states for uniform polling.
4. Client polls `GET /v1/jobs/{job_id}`: `queued|running` wait, `done`
   embeds `result` spans from `explanations`, `failed` carries `error`.
   Clients never touch Postgres or Redis directly.

## Failure contract

- Unknown classification or job: 404.
- Bad label or shape: 422 (existing handlers).
- Redis down at enqueue: 202 anyway; row stays `queued`, worker rebuilds
  on boot. Logged, never 500s the explain.
- Worker error (missing row, IG fault): job `failed` with truncated
  message, loop survives, batch-mates unaffected.
- Duplicate pushes or concurrent misses: first winner kept via uniques,
  losers ignored at info level.

## Local run

1. Postgres up with `alembic upgrade head`, Redis on
   `redis://localhost:6379/0`.
2. Terminal A: `python -m uvicorn app.api.main:app --port 8000 --reload`.
3. Terminal B: `python -m app.worker`.
4. Classify, explain (202), poll job until `done`.
