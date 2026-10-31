# Flagged

**Flag harmful content. Understand why.**

An end-to-end content moderation system: six toxicity scores, on-demand explanations, and an audit trail of moderator corrections. Built with React, FastAPI, NVIDIA Triton, PostgreSQL, and Redis; deployed on AWS EC2 with a Cloudflare Pages frontend.

**[Explore the project](https://contentflagged.tech) · [Try the demo](https://contentflagged.tech/demo) · [Model metadata](https://flagged.duckdns.org/v1/model_info)**

## Try it

Enter a comment in the demo, inspect its predictions, request an explanation, and submit a review. Batch classification is available in the same workspace. The demo uses the real API, with no account required.

- **Classify:** six independent probabilities and label-specific thresholds. Scores do not sum to 100%.
- **Explain:** Integrated Gradients highlights text that supports or opposes a selected label.
- **Review:** save corrected labels while preserving the original prediction and an append-only audit record.

**Submissions, predictions, explanations, and corrections are stored. Use non-sensitive examples.** Public reviews are marked `public-demo`; they do not automatically retrain the model.

## Architecture

```mermaid
flowchart TD
    Browser[Browser] -->|Frontend assets| Pages[Cloudflare Pages · React]
    Browser -->|HTTPS API requests| Caddy
    subgraph EC2[AWS EC2 · Docker containers]
        Caddy[Caddy · TLS / route allowlist] --> API[FastAPI]
        API -->|Classification| Triton[NVIDIA Triton · ONNX CPU]
        Worker[Explanation worker · PyTorch / Captum]
        Prometheus[Prometheus] -->|Scrape| API
        Prometheus -->|Scrape| Worker
        Prometheus -->|Scrape| Triton
        Grafana[Grafana] -->|Query| Prometheus
        Migrate[Alembic · one-shot migration]
    end
    API -->|Predictions / jobs / corrections| PG[(Managed PostgreSQL)]
    API -->|Job notification| Redis[(Managed Redis)]
    Redis -->|Wake worker| Worker
    Worker <-->|Claim / renew lease / save result| PG
    Migrate --> PG
```

### Engineering decisions

- **Fast classification, separate explanations.** Triton dynamically batches FP32 ONNX inference on CPU. Expensive gradient-based explanations run in a separate worker container.
- **Durable jobs, lightweight notifications.** PostgreSQL owns explanation job state; Redis wakes workers. Leases, heartbeats, ownership tokens, retries, and database scans recover abandoned work and missed notifications. Computation may be retried.
- **Committed results.** Classification returns only after persistence succeeds. Text/model-version deduplication produces canonical IDs; batches preserve input order and commit transactionally. Explanation results and job completion are also committed together.
- **Explicit capacity and lifecycle.** The local in-process runtime coordinates shared model access, queued work, and shutdown. The deployed Triton path uses its own admission limit and server queue; it bypasses the local inference executor.
- **Observable services.** Prometheus and Grafana cover HTTP outcomes, latency, queues, predictions, and worker behavior. Structured logs carry request/job context.

The [API implementation](app/api/), [job state machine](app/db/jobs.py), [local runtime](app/runtime.py), and [Triton runtime](app/triton_runtime.py) contain the corresponding contracts.

## Model and measurements

The packaged release is **DistilBERT v2.0.0**, with six independent sigmoid outputs: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`. Class-weighted fine-tuning and per-label F2 threshold selection emphasize recall. Classification runs with ONNX Runtime; explanations use Captum Layer Integrated Gradients.

| Measurement | Scope | Result |
| --- | --- | ---: |
| Macro F1 | Historical reported development result after threshold tuning | **0.9148** |
| Classification p95 | Historical 180-second classification-only run, offered 50 requests/s | **84 ms** |
| Successful throughput | Same historical run | **49.99/s** |
| Request errors | Same historical run | **0%** |

**The latency benchmark predates commit-before-response persistence and excludes explanations. It is not a current production guarantee.** Later production observations included CPU saturation and 503 responses; no numerical report for those runs is included here. The development score is not a verified held-out test result.

See the [model card](MODEL_CARD.md) for thresholds, artifact identity, evaluation scope, and limitations. The [classification load probe](scripts/frontend-classification-load.js) provides a repeatable measurement tool; its sample texts do not reproduce the original workload.

## API

Base URL: `https://flagged.duckdns.org`. Requests to the hosted demo persist records.

```sh
curl --fail-with-body https://flagged.duckdns.org/v1/classify \
  -H 'Content-Type: application/json' \
  --data '{"text":"Thanks for helping me understand this."}'
```

The response includes a committed `classification_id` and six `labels` entries with `probability` and the authoritative `predicted` boolean.

| Method | Public endpoint | Purpose |
| --- | --- | --- |
| POST | `/v1/classify` | Classify one text |
| POST | `/v1/classify_batch` | Classify 1–64 texts in input order |
| POST | `/v1/explain` | Explain a saved classification and selected label |
| GET | `/v1/jobs/{job_id}` | Poll explanation status and result |
| GET | `/v1/classifications/{id}` | Read prediction, explanations, and corrected view |
| POST | `/v1/classifications/{id}/corrections` | Append a full six-label review |
| GET | `/v1/model_info` | Model version, thresholds, backend, and input limits |
| GET | `/v1/health/live`, `/v1/health/ready` | Liveness and readiness |

`/v1/explain` accepts `{"classification_id":"UUID","label":"toxic"}`. A cache hit returns **200**; asynchronous jobs return **202** with a `job_id`. Poll until `done` or `failed`. Corrections require all six booleans in `corrected_labels` and return **201**.

Input is limited to 10,000 characters per text and truncated to 512 tokens. Invalid requests return **422**; capacity and dependency failures can return **429**, **503**, or **504**. See [request/response schemas](app/api/schemas.py) and [routes](app/api/routes.py) for exact contracts.

The public demo is unauthenticated. Caddy exposes an explicit route allowlist; global history, runtime status, OpenAPI docs, and monitoring stay private. CORS and record IDs do not provide authorization. Integrations handling private content need an appropriate access policy.

## Run locally

### Backend

Use Docker with Compose v2. Container images and dependency locks target **Linux amd64 / Python 3.12 / CPU**; Apple Silicon uses amd64 emulation.

```sh
git clone https://github.com/mouldibziou/Real-Time-Content-Moderation-System.git
cd Real-Time-Content-Moderation-System
```

**Model weights, ONNX exports, and datasets are excluded from Git.** Obtain the matching release artifacts from the author before building the backend:

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

The image build verifies five artifacts against [`docker/model-manifest.json`](docker/model-manifest.json). Both formats are required: classification uses ONNX and explanations use PyTorch. Startup does not download them.

```sh
export TOXICITY_CORS_ORIGINS=http://127.0.0.1:5173,http://localhost:5173
docker compose -f compose.yaml up -d --build --wait --wait-timeout 180
curl --fail http://127.0.0.1:8000/v1/health/ready
```

This starts PostgreSQL, Redis, migrations, API, and worker using local development credentials. API and worker wait for migrations. Interactive API documentation is at `http://127.0.0.1:8000/docs`.

The default local stack uses in-process ONNX. To use Triton, start with `docker compose -f compose.yaml -f compose.triton.yaml up -d --build`. Stop with the same Compose files and `down`; named volumes retain data. Adding `--volumes` deletes it.

### Frontend

Use **Node 24 / pnpm 11.25.0**:

```sh
cd frontend
pnpm install --frozen-lockfile
cp .env.example .env.local
```

Set `VITE_API_BASE_URL=http://127.0.0.1:8000` in `.env.local`, then run `pnpm dev`. The example otherwise points to the hosted API. All `VITE_` values are public build-time configuration. See the [frontend README](frontend/README.md) for Pages settings.

## Deployment

The deployed topology combines **`compose.cloud.yaml` + `compose.triton.yaml` + `compose.https.yaml`**: containerized services on EC2, managed PostgreSQL/Redis, and Caddy HTTPS ingress. React builds separately on Cloudflare Pages.

| Configuration | Purpose |
| --- | --- |
| [`compose.yaml`](compose.yaml) | Local database, Redis, API, worker, migrations |
| [`compose.triton.yaml`](compose.triton.yaml) | Triton inference and API override |
| [`compose.monitoring.yaml`](compose.monitoring.yaml) | Local Prometheus / Grafana overlay |
| [`compose.cloud.yaml`](compose.cloud.yaml) | Managed-service deployment and monitoring |
| [`compose.https.yaml`](compose.https.yaml) | HTTPS ingress |
| [`compose.ec2.yaml`](compose.ec2.yaml) | Single-host database / Redis overlay, applied after `compose.cloud.yaml` |

Use [`docker/runtime.env.example`](docker/runtime.env.example) as a configuration reference. Keep real credentials in ignored environment files. Application database URLs use `postgresql+asyncpg`; migrations use the synchronous driver. Configure exact frontend origins through `TOXICITY_CORS_ORIGINS`.

Application containers run as non-root with read-only filesystems. The build context explicitly allows only required inputs; dependency locks and artifact hashes pin build inputs. The current EC2 deployment is a single-host topology, with finite CPU capacity rather than automatic scaling.

## Development checks

Backend tests use standard-library `unittest`. In a **Python 3.12 Linux amd64** environment:

```sh
python -m pip install --require-hashes --only-binary=:all: \
  -r requirements.txt -r requirements/migrations.txt -r requirements/triton-client.txt
DATABASE_URL=postgresql+asyncpg://toxicity:replace-me@127.0.0.1:5432/toxicity_tests \
  python -m unittest discover -s tests -t .
```

The default suite mocks external services. Opt-in database tests (`TOXICITY_DB_TESTS=1`) require a migrated, disposable database whose name ends in `_tests` and clear test records. Real-model smoke tests (`TOXICITY_REAL_MODEL_TESTS=1`) require artifacts. Never run integration tests against production. Linux CPU-affinity checks are not portable to native macOS.

Frontend checks, from `frontend/`:

```sh
pnpm test
pnpm build
```

## Repository guide

| Path | Contents |
| --- | --- |
| [`app/`](app/) | API, inference coordination, explanations, persistence, and observability |
| [`frontend/`](frontend/) | React landing page and moderation workspace |
| [`Finetuning/`](Finetuning/) | EDA, baseline, encoder selection, fine-tuning, and threshold notebooks |
| [`alembic/`](alembic/) | Database migrations |
| [`docker/`](docker/) | Artifact checks, Triton backend, health checks, and ingress |
| [`monitoring/`](monitoring/) | Prometheus configuration and Grafana dashboard |
| [`requirements/`](requirements/) | Role-specific dependency inputs and locks |
| [`scripts/`](scripts/) | Smoke checks, load probes, and deployment helpers |
| [`tests/`](tests/) | Backend unit and opt-in integration tests |

## Contact

For API access, matching model artifacts, or a discussion about the project:

**Mohamed Mouldi Bziou** · [Email](mailto:bzioumohamedmouldi@gmail.com) · [LinkedIn](https://www.linkedin.com/in/mohamedmouldibziou/)
