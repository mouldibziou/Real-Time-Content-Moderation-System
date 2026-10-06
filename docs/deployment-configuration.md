# Deployment configuration

The ASGI entry point is `app.api.main:app`. It calls
`load_runtime_settings()` once per application process before constructing the
FastAPI application. Routes use that application's runtime rather than reading
environment variables per request.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `TOXICITY_DEVICE` | `cpu` | Torch device: `cpu`, `mps`, `cuda`, or `cuda:<index>` |
| `TOXICITY_INFERENCE_BACKEND` | `onnx` | Classification backend: `onnx` or `torch` |
| `TOXICITY_MODEL_DIR` | `models/distilbert/v2.0.0` | Torch weights, configuration, and fast tokenizer |
| `TOXICITY_ONNX_MODEL_DIR` | `models/distilbert/v2.0.0-onnx` | ONNX classification model |
| `DATABASE_URL` | Required, no default | Existing database configuration; use the `postgresql+asyncpg` driver |

Missing runtime variables use defaults. Explicitly empty or whitespace-only
values fail configuration validation. Backend and device values are exact and
case-sensitive. Model paths become `Path` objects; relative paths are resolved
against the process working directory when resources load. Paths containing
spaces are preserved.

Configuration loading uses only Python's standard library. It does not check
files, probe accelerators, load model weights, or connect to the database.
Resource loading during the application lifespan verifies files and device
availability; an explicitly requested unavailable accelerator fails startup
without silently falling back to CPU. ONNX classification still needs the Torch
model for explanations. Label order and thresholds remain fixed in runtime
settings.

`DATABASE_URL` remains owned by `app.db.engine`, which currently loads a local
`.env` file without overriding existing process variables and requires the URL
at import time. The settings loader itself does not load `.env`. Supply process
environment variables explicitly in deployment, and keep credentials out of
images and configuration logs.

These changes establish configuration behavior only; they do not yet establish
a container build or verify full application startup.
