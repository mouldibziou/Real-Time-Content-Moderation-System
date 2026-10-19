# Flagged

**Flag harmful content. Understand why.**

Flagged is an end-to-end, explainable content moderation project for comments, community forums, and chat applications. A fine-tuned DistilBERT model scores text across six toxicity categories; a separate explanation worker shows which words contributed to a decision; human reviewers can save corrections for future retraining.

The project includes the model experiments, inference service, durable job processing, moderation interface, deployment configuration, and monitoring.

**[Explore the project](https://contentflagged.tech) · [Try the live demo](https://contentflagged.tech/demo) · [API metadata](https://flagged.duckdns.org/v1/model_info)**

## What makes it useful

- **Multiple signals per comment.** Six independent scores capture overlapping forms of harmful content instead of forcing a single category.
- **Recall-focused decisions.** Class-weighted training and per-label F2 threshold tuning emphasize reducing missed harmful content.
- **Explanations on demand.** Integrated Gradients produces signed text contributions without putting explanation computation on the classification request path.
- **A complete review loop.** The interface connects classification, explanation, and saved moderator corrections while retaining the original prediction.
- **Durable results.** Successful classification responses return committed, deduplicated IDs that can immediately be used for explanations and corrections.
- **CPU deployment.** FP32 ONNX inference runs through NVIDIA Triton on AWS EC2, with managed PostgreSQL/Redis and Prometheus/Grafana monitoring.

## Try the moderation journey

Open **[contentflagged.tech/demo](https://contentflagged.tech/demo)**. No account is required.

1. Choose an editable example or enter a comment, then classify it.
2. Inspect all six probabilities, threshold markers, and triggered labels. Scores are independent and **do not sum to 100%**.
3. Request an explanation for a label. Follow the real job state and inspect supporting/opposing text highlights and ranked contributors.
4. Adjust the review decisions and explicitly submit a correction. The interface displays the saved audit receipt.
5. Expand batch classification to process several comments and inspect each result in the same workspace.

The workspace uses the hosted API, not simulated predictions. Landing-page sketches are illustrative. **Submitted text, predictions, explanation results, and corrections are stored; use non-sensitive examples.** Public demo corrections carry `moderator_id: "public-demo"`. Corrections support later offline retraining; they do not update the deployed model automatically.

## Architecture

The deployed system separates the fast classification path from the more expensive explanation path.

```mermaid
flowchart TD
    Browser["User's browser"]
    Pages["Cloudflare Pages · React / TypeScript"]
    Postgres[("Managed PostgreSQL")]
    Redis[("Managed Redis")]

    Browser -->|Load frontend| Pages
    Browser -->|API requests over HTTPS| Caddy

    subgraph EC2["AWS EC2 · containerized services"]
        Caddy["Caddy · TLS and public route allowlist"]
        API["FastAPI · validation, classification, persistence"]
        Triton["NVIDIA Triton · FP32 ONNX CPU inference"]
        Worker["Explanation worker · PyTorch / Captum"]
        Prometheus["Prometheus"]
        Grafana["Grafana"]
        Migrate["Alembic migration container · deployment only"]

        Caddy --> API
        API -->|Inference| Triton
        Prometheus -.->|Scrape| API
        Prometheus -.->|Scrape| Triton
        Prometheus -.->|Scrape| Worker
        Grafana -->|Query| Prometheus
    end

    API -->|Commit predictions, create jobs, save corrections| Postgres
    API -->|Notify explanation work| Redis
    Redis -->|Wake worker| Worker
    Worker <-->|Claim jobs, renew leases, save explanations| Postgres
    Migrate -->|Versioned schema changes| Postgres
```

### Classify → explain → review

**Classification:** FastAPI validates the input, obtains six scores, and commits the prediction before responding. Repeated text for the same model version reuses its canonical stored ID. Batch responses preserve input order, including duplicate entries; a failed persistence transaction rolls back the batch.

**Explanation:** The API checks the explanation cache by classification, label, and model version. A miss creates or reuses a durable job. Redis provides notifications; **PostgreSQL owns the job state**. The worker uses leases, heartbeat renewal, ownership tokens, bounded retries, and database scanning to recover missed notifications or abandoned work. Explanation results and completion state are committed together. Attempts may repeat; this is not an exactly-once computation guarantee.

**Review:** Corrections are append-only audit events. The original prediction stays intact, and classification detail exposes the latest corrected view. Public reviews should be checked before inclusion in a training dataset.

## Model and inference

| Component | Implementation |
| --- | --- |
| Deployed release | Fine-tuned **DistilBERT v2.0.0** |
| Task | Multilabel text classification with six independent sigmoid outputs |
| Training approach | Class-weighted loss; per-label threshold selection using F2 to emphasize recall |
| Classification | **FP32 ONNX on CPU**, served by Triton's Python backend with ONNX Runtime and dynamic batching |
| Local alternative | In-process ONNX inference; PyTorch backend also available |
| Explanation | Captum **Layer Integrated Gradients** in a separate PyTorch worker; 32 steps by default |
| Input limits | 10,000 characters per text; 1–64 texts per classification batch; truncation at 512 tokens including special tokens |

Current label order and thresholds:

| Label | Threshold |
| --- | ---: |
| `toxic` | 0.74 |
| `severe_toxic` | 0.45 |
| `obscene` | 0.74 |
| `threat` | 0.59 |
| `insult` | 0.61 |
| `identity_hate` | 0.76 |

A label is predicted when its probability meets or exceeds its threshold. Query [`GET /v1/model_info`](https://flagged.duckdns.org/v1/model_info) for the serving model version, thresholds, backend, device, and input limits. No quantization is used in the deployed model.

The research path includes exploratory analysis, classical baselines, encoder comparison, fine-tuning, and threshold selection. See the [training notes](Finetuning/transformer/notebooks/README.md) and [model provenance](docs/model-provenance.md) for experiment results and their scope. Reported development scores are not presented here as verified held-out test performance.

## Measured performance

The supplied **180-second, classification-only benchmark** used an offered load of **50 classifications/s**:

| Measurement | Result |
| --- | ---: |
| Successful classification throughput | **49.99/s** |
| p95 latency | **84 ms** |
| Mean / median latency | 43.9 ms / 34 ms |
| p99 latency | 185 ms |
| Measured request errors | **0%** |
| Dropped load-generator iterations | 0 |
| Sampled average host CPU | 66.9% |

**These are historical measurements from before the commit-before-response persistence change.** They exclude explanations and do not establish the current deployment's latency or sustained capacity. Browser request round trip and live readiness are separate observations.

Recheck after deployment changes using the same hardware, network path, settings, and input distribution. The [classification load probe](scripts/frontend-classification-load.js) and [cloud load-testing guide](docs/cloud-load-testing.md) describe measurement scope; the probe's default sample mix is a substitute, not a reproduction of the original workload.

## API usage

The hosted API base URL is `https://flagged.duckdns.org`. Examples below create persisted demo records.

### Classify one comment

```sh
curl --fail-with-body https://flagged.duckdns.org/v1/classify \
  -H 'Content-Type: application/json' \
  --data '{"text":"Thanks for helping me understand this."}'
```

The response contains the text, a `classification_id`, and six `labels` entries, each with `probability` and the authoritative `predicted` boolean.

### Classify a batch

```sh
curl --fail-with-body https://flagged.duckdns.org/v1/classify_batch \
  -H 'Content-Type: application/json' \
  --data '{"texts":["Thanks for the explanation.","Your opinion is stupid."]}'
```

The `results` array follows the input order. All returned IDs refer to committed records.

### Explain a saved classification

Replace the placeholder with a `classification_id` returned above:

```sh
curl --fail-with-body https://flagged.duckdns.org/v1/explain \
  -H 'Content-Type: application/json' \
  --data '{"classification_id":"REPLACE_WITH_CLASSIFICATION_UUID","label":"toxic"}'
```

A cache hit returns **200** with explanation spans. New or reused jobs return **202** with a `job_id`. Poll:

```sh
curl --fail-with-body https://flagged.duckdns.org/v1/jobs/REPLACE_WITH_JOB_UUID
```

States are `queued`, `running`, `done`, or `failed`. Completed jobs include `result`; failed jobs include an error. Re-requesting a failed job does not promise to restart it. Span offsets refer to Unicode code points, not JavaScript UTF-16 indices.

### Endpoint reference

| Method | Path | Purpose | Public deployment |
| --- | --- | --- | --- |
| GET | `/v1/health/live` | Process liveness | Yes |
| GET | `/v1/health/ready` | Model/inference and database readiness | Yes |
| GET | `/v1/model_info` | Model, thresholds, backend, and limits | Yes |
| POST | `/v1/classify` | Single classification | Yes |
| POST | `/v1/classify_batch` | Ordered batch classification | Yes |
| POST | `/v1/explain` | Cached explanation or durable job | Yes |
| GET | `/v1/jobs/{job_id}` | Explanation state and result | Yes |
| GET | `/v1/classifications/{id}` | Original prediction, corrected view, explanations | Yes |
| POST | `/v1/classifications/{id}/corrections` | Append a full six-label correction | Yes |
| POST | `/v1/explain_batch` | Explanation jobs for multiple saved IDs | Local/private |
| GET | `/v1/classifications` | Paginated classification history | Local/private |
| GET | `/v1/status` | Runtime queues and capacity | Local/private |

Corrections accept `corrected_labels` containing **all six labels as booleans**, plus an optional `moderator_id`. They return **201** with the saved audit event. See the exact [request/response schemas](app/api/schemas.py) and [routes](app/api/routes.py); interactive OpenAPI documentation is available at `/docs` on a local API.

Typical errors: **422** invalid input, **404** unknown ID, **409** unavailable model version, **429** runtime capacity, **503** unready service/database failure or admission rejection, and **504** operation wait timeout.

## Run locally

### 1. Prepare the model artifacts

You need Git, Docker with Compose v2, and the serving artifacts. Containers and dependency locks target **Linux amd64 / Python 3.12 / CPU**; Apple Silicon runs them through amd64 emulation.

```sh
git clone https://github.com/mouldibziou/Real-Time-Content-Moderation-System.git
cd Real-Time-Content-Moderation-System
```

**Weights, exported ONNX files, and training data are excluded from Git. A clone alone cannot build the backend.** Obtain the matching release artifacts from the project author and restore:

```text
models/distilbert/
├── v2.0.0/
│   ├── model.safetensors
│   ├── config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── v2.0.0-onnx/
    └── model.onnx
```

The build verifies their sizes and hashes against [`docker/model-manifest.json`](docker/model-manifest.json). Both model formats are needed: classification uses ONNX; the explanation worker uses PyTorch. There is no automatic artifact download during startup. You can try the hosted demo without these files.

### 2. Start the backend

From the repository root:

```sh
export TOXICITY_CORS_ORIGINS=http://127.0.0.1:5173,http://localhost:5173
docker compose -f compose.yaml up -d --build --wait --wait-timeout 180
docker compose -f compose.yaml ps -a
curl --fail http://127.0.0.1:8000/v1/health/ready
```

This starts local PostgreSQL, Redis, the one-shot Alembic migration, the API, and the explanation worker. API/worker startup waits for migrations. The local stack uses in-process ONNX inference; the API is at `http://127.0.0.1:8000` and its interactive docs are at `http://127.0.0.1:8000/docs`.

`compose.yaml` contains development-only database credentials. Keep this setup local. The worker must run for uncached explanations to complete.

To use Triton locally instead, append its overlay when starting the stack:

```sh
docker compose -f compose.yaml -f compose.triton.yaml up -d --build
```

Stop with the same Compose files and `down`; named volumes retain local data. `down --volumes` deletes that data.

### 3. Start the frontend

Use **Node 24** and **pnpm 11.25.0**. In another terminal:

```sh
cd frontend
pnpm install --frozen-lockfile
```

Create `frontend/.env.local` with this value to connect to your local backend:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Then run `pnpm dev` and open the printed URL. The checked-in `.env.example` points to the hosted API, so override it for local work. All `VITE_` values are public build-time configuration. See the [frontend README](frontend/README.md) for interface details and Pages build settings.

## Configuration and deployment

| Configuration | Role |
| --- | --- |
| `compose.yaml` | Local API, worker, PostgreSQL, Redis, and migrations |
| `compose.triton.yaml` | Triton server and API inference backend override |
| `compose.monitoring.yaml` | Local Prometheus and Grafana overlay |
| `compose.cloud.yaml` | Cloud images, managed database/Redis URLs, monitoring |
| `compose.https.yaml` | Caddy HTTPS ingress and loopback-only API binding |
| `compose.ec2.yaml` | Alternative single-host topology with local PostgreSQL/Redis containers |

The public deployment uses the **managed-service topology**, combining `compose.cloud.yaml`, `compose.triton.yaml`, and `compose.https.yaml`. The frontend builds independently with root `frontend`, command `pnpm build`, and output `dist`.

For backend environments, use [`docker/runtime.env.example`](docker/runtime.env.example) as a configuration reference. Supply private database/Redis credentials outside Git. The migration role uses the synchronous PostgreSQL driver; application roles use `postgresql+asyncpg`.

- `TOXICITY_INFERENCE_BACKEND`: `onnx`, `torch`, or `triton`.
- `TOXICITY_MODEL_DIR` / `TOXICITY_ONNX_MODEL_DIR`: local serving artifact paths.
- `TOXICITY_TRITON_URL`: internal Triton address when using remote inference.
- `TOXICITY_CORS_ORIGINS`: comma-separated exact browser origins, with no paths, trailing slashes, or wildcards.
- `MODEL_VERSION`: must match the packaged release, currently `v2.0.0`.

The current public frontend origins are `https://contentflagged.tech` and `https://real-time-content-moderation-system.pages.dev`. CORS is **not authentication**. The public demo has no login, and Caddy restricts exposed routes; global history, `/docs`, runtime status, and monitoring are not publicly routed. Individual IDs are not an authorization mechanism. Production integrations handling private content need their own access policy.

Application containers run as non-root with read-only filesystems and restricted capabilities. Dependency locks and artifact manifests make build inputs explicit. Monitoring covers HTTP outcomes, latency, queues, prediction counts, explanation attempts, and worker health; structured logs carry request/job context. See [monitoring deployment](docs/monitoring-deployment.md), [metrics](docs/metrics.md), and [logging](docs/logging.md).

## Tests and verification

Backend tests use Python's `unittest`. In a **Python 3.12 Linux amd64** development environment, install the locked dependencies and run from the repository root:

```sh
python -m pip install --require-hashes --only-binary=:all: \
  -r requirements.txt \
  -r requirements/migrations.txt \
  -r requirements/triton-client.txt

DATABASE_URL=postgresql+asyncpg://toxicity:replace-me@127.0.0.1:5432/toxicity_tests \
  python -m unittest discover -s tests -t .
```

The default suite mocks external services. Database integration tests are opt-in with `TOXICITY_DB_TESTS=1` and require a migrated **disposable database whose name ends in `_tests`**; they clear test records. Real-model smoke tests use `TOXICITY_REAL_MODEL_TESTS=1` and require the artifacts. Never direct integration tests at the hosted database. Linux CPU-affinity checks are not portable to native macOS.

Frontend checks:

```sh
cd frontend
pnpm test
pnpm build
```

Coverage includes validation and backpressure, batch ordering, duplicate/concurrent persistence, rollback, explanation cache and job delivery, API errors, Unicode attribution rendering, stale-request cancellation, polling, and correction submission. Tests establish behavior; they do not establish held-out model quality or production capacity. See [frontend verification](frontend/VERIFICATION.md) for the recorded integration checks.

## Repository guide

| Path | Contents |
| --- | --- |
| [`app/api/`](app/api/) | FastAPI routes, schemas, dependencies, and error handling |
| [`app/db/`](app/db/) | SQLAlchemy persistence, corrections, explanation cache, and leased jobs |
| [`app/worker.py`](app/worker.py) | Separate explanation process and recovery loop |
| [`app/`](app/) | Classification, explanations, runtime coordination, and observability |
| [`frontend/`](frontend/) | React/TypeScript landing page and moderation workspace |
| [`Finetuning/`](Finetuning/) | Data exploration, baseline experiments, model selection, and fine-tuning |
| [`docker/`](docker/) | Artifact verification, Triton backend, health checks, and Caddy configuration |
| [`alembic/`](alembic/) | Versioned database migrations |
| [`monitoring/`](monitoring/) | Prometheus configuration and provisioned Grafana dashboard |
| [`scripts/`](scripts/) | Smoke checks, load probes, environment helpers, and dependency locking |
| [`tests/`](tests/) | Backend unit and opt-in integration tests |
| [`docs/`](docs/) | Design, model provenance, operations, and measurement notes |

This README describes the current source behavior. Some detailed design documents capture earlier development stages, including background persistence and pre-Triton runtime decisions; use the current [API code](app/api/routes.py), [job implementation](app/db/jobs.py), and [model provenance](docs/model-provenance.md) when comparing those stages.

## Contact

Want to use the backend API directly or discuss an integration?

**Mohamed Mouldi Bziou** · [Email](mailto:bzioumohamedmouldi@gmail.com) · [LinkedIn](https://www.linkedin.com/in/mohamedmouldibziou/)
