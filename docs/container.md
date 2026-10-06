# Container build and operation

The API image targets **Linux amd64 / Python 3.12 / CPU**. Build from the repository
root, explicitly selecting the architecture even on an Apple Silicon Mac:

```sh
docker build --platform linux/amd64 --target runtime -t toxicity-api:local .
docker build --platform linux/amd64 --target migrations -t toxicity-api-migrations:local .
```

These commands build local images; they do not publish or deploy them.

## What the Dockerfile packages

The digest-pinned Python base matches the dependency verification environment.
Dependency installation happens in separate stages with required hashes and
binary wheels. Only the resulting virtual environment is copied into each final
image. BuildKit's pip cache speeds subsequent builds without becoming part of the
runtime filesystem. Source edits reuse the dependency layer.

The default `runtime` target includes the API and exactly five model files:
Torch weights, model configuration, two tokenizer files, and ONNX weights.
`docker/model-manifest.json` fingerprints both formats. Image construction verifies
their sizes and SHA-256 hashes as the same non-root user that serves requests.
This verifies artifact identity, not training provenance or predictive accuracy.

The `migrations` target contains Alembic, its database driver, and schema definitions.
It has no models, ML libraries, or API server. Run it once per deployment before
admitting traffic; API worker startup does not run migrations.

`.dockerignore` uses an allowlist. Secrets, Git metadata, training data, notebooks,
other models, optimizer state, tests, and the Mac virtual environment never enter
the build context. Tests can be mounted read-only for verification.

## Model artifacts in a clean checkout

Git ignores the model directories. Before building in CI or on another machine,
obtain the five files listed in `docker/model-manifest.json` from your approved
artifact storage and restore their repository-relative paths. The build fails
if files are absent or do not match the recorded release. It does not download or
export models. A new release requires intentionally updating the manifest, model
paths, serving metadata, and validation evidence together.

## Configuration and networking

Copy `docker/runtime.env.example` to `.env.docker`, replace its placeholder
credentials, and use the hostname of PostgreSQL on your Docker network. The example
uses `db`; a database on another network or hosting platform needs its actual host.
Keep the file private. `.env.docker` is Git-ignored and excluded from builds.

After creating a network and provisioning PostgreSQL on it, run migrations:

```sh
docker run --rm --platform linux/amd64 --network toxicity \
  --read-only --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --cap-drop ALL --security-opt no-new-privileges:true \
  --env-file .env.docker toxicity-api-migrations:local
```

Then start the API on that same network:

```sh
docker run -d --name toxicity-api --platform linux/amd64 --network toxicity \
  --publish 127.0.0.1:8000:8000 \
  --read-only --tmpfs /tmp:rw,noexec,nosuid,size=256m \
  --cap-drop ALL --security-opt no-new-privileges:true \
  --pids-limit 256 --memory 2g --cpus 2 --stop-timeout 75 \
  --env-file .env.docker toxicity-api:local
```

The image runs as UID/GID **10001**. Source, dependencies, and models are root-owned;
the application cannot modify them. Writable caches are confined to `/tmp`, which
is disposable. PostgreSQL data belongs outside the API container.

The loopback port binding is for local access. A hosting platform should route
HTTPS traffic to container port 8000. `EXPOSE` documents that port; publishing
or platform routing makes it reachable. Credentials are supplied at runtime,
never in the Dockerfile or build arguments.

## Server, health, and shutdown

One Uvicorn worker owns one model/runtime instance. JSON-array `CMD` starts Python
directly, allowing it to receive SIGTERM. Application lifecycle logs and access
logs go to standard output using `docker/logging.json`.

The Docker probe calls `/v1/health/ready` with a two-second request timeout and
checks that the JSON body contains boolean `is_ready: true`. HTTP 200 alone is
insufficient with the current application endpoint. External load balancers must
likewise understand that body or use a future HTTP-status readiness contract.
The probe allows 120 seconds for startup, checks every 30 seconds, and marks the
container unhealthy after three failed checks. Docker health status alone does
not restart containers or decide where traffic is routed.

`UVICORN_LIMIT_CONCURRENCY=16` provides a provisional server admission guard;
excess traffic receives Uvicorn HTTP 503. It does not repair the executor/runtime
queue mismatch or establish the application's intended per-category scheduling.
`OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` constrain applicable numerical libraries;
ONNX Runtime still uses its own default session threading. Tune resources and
threading from measurements on the actual hosting hardware.

`UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN=60` gives Uvicorn up to 60 seconds to drain
requests. The Docker command allows 75 seconds before forced termination. These
are provisional limits, not a guarantee that active Torch operations finish:
the current runtime does not forcibly cancel model execution and its lifespan
shutdown can wait indefinitely. Verify worst-case explanation behavior before
choosing a production shutdown budget.

Inspect and stop the container with:

```sh
docker logs toxicity-api
docker inspect --format '{{.State.Health.Status}}' toxicity-api
docker stop --timeout 75 toxicity-api
```

## Verification commands

Run the entire suite inside the image, including real classification and
explanation, without external networking:

```sh
docker run --rm --platform linux/amd64 --network none \
  --read-only --tmpfs /tmp:rw,noexec,nosuid,size=256m \
  --cap-drop ALL --security-opt no-new-privileges:true \
  --mount "type=bind,source=$PWD/tests,target=/opt/app/tests,readonly" \
  --env TOXICITY_REAL_MODEL_TESTS=1 \
  toxicity-api:local python -m unittest discover -s tests -t .
```

Against an API connected to a **disposable migrated database**, run:

```sh
python scripts/smoke-container.py http://127.0.0.1:8000
```

This creates test classification and correction rows. It checks health, model
metadata, classification, validation, persisted lookup, real explanation, and
moderator correction. It does not benchmark capacity or establish accuracy.

## Remaining deployment work

Compose will make the database network, persistent storage, and migration ordering
repeatable. Public hosting also requires HTTPS, access controls for comment history
and moderation, a persistence durability decision, HTTP admission verification,
and backup/restore and rollback procedures. Existing background writes still
log and drop failures; returning a classification ID does not guarantee durability.
Repeated text can conflict with the current database uniqueness policy.

## Verification performed

- Both runtime and migration targets built successfully for Linux amd64; Docker's
  built-in Dockerfile checks reported no warnings.
- The image audit confirmed UID/GID 10001, root-owned model files inaccessible
  for writing by the application, exactly five serving artifacts, and no secrets,
  training tools, CUDA libraries, or migration dependencies in the API image.
- All **136 tests passed**, including the real-model smoke test, with networking
  disabled, a read-only root filesystem, and restricted capabilities. No tests
  were skipped. The new helper tests cover false/malformed readiness responses,
  connection failures, missing/corrupted artifacts, and invalid artifact paths.
- The migration image applied the schema to disposable PostgreSQL 18. The API's
  real HTTP classification, persisted lookup, explanation, correction, and
  health checks passed against that database under the documented restrictions.
- Docker reported the running API healthy. SIGTERM at idle completed application
  shutdown with exit code 0 and no out-of-memory termination.

The tests used amd64 emulation on Apple Silicon. Their timings do not establish
production latency or capacity. Active-request termination, load behavior,
durable persistence under failure, and backup recovery remain unverified.

References: [Docker build practices](https://docs.docker.com/build/building/best-practices/),
[Docker runtime](https://docs.docker.com/engine/containers/run/),
and [Uvicorn settings](https://www.uvicorn.org/settings/).
