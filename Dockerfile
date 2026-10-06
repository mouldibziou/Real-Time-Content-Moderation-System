# syntax=docker/dockerfile:1

# Matches the Python/OS image used to validate the Linux dependency locks.
FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258 AS base
ARG TARGETARCH
RUN test "$TARGETARCH" = amd64 || \
    (echo 'The dependency locks require --platform linux/amd64.' >&2; exit 1)

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH"

RUN groupadd --gid 10001 app && \
    useradd --uid 10001 --gid 10001 --no-create-home --no-log-init \
    --home-dir /nonexistent --shell /usr/sbin/nologin app
WORKDIR /opt/app

FROM base AS serving-dependencies
COPY requirements.txt /tmp/requirements.txt
RUN python -m venv /opt/venv
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --require-hashes --only-binary=:all: -r /tmp/requirements.txt && \
    python -m pip check

FROM base AS migration-dependencies
COPY requirements/migrations.txt /tmp/migrations.txt
RUN python -m venv /opt/venv
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --require-hashes --only-binary=:all: -r /tmp/migrations.txt && \
    python -m pip check

# Build with --target migrations; no model weights or serving dependencies.
FROM base AS migrations
COPY --from=migration-dependencies /opt/venv /opt/venv
COPY app/__init__.py ./app/
COPY app/db/__init__.py app/db/models.py ./app/db/
COPY alembic.ini ./
COPY alembic/ ./alembic/
USER 10001:10001
CMD ["alembic", "upgrade", "head"]

# Last stage is the default image produced by docker build.
FROM base AS runtime
COPY --from=serving-dependencies /opt/venv /opt/venv
COPY app/ ./app/
COPY docker/healthcheck.py docker/verify_artifacts.py docker/model-manifest.json docker/logging.json ./docker/
COPY --chmod=0644 models/distilbert/v2.0.0/model.safetensors \
    models/distilbert/v2.0.0/config.json \
    models/distilbert/v2.0.0/tokenizer.json \
    models/distilbert/v2.0.0/tokenizer_config.json ./models/distilbert/v2.0.0/
COPY --chmod=0644 models/distilbert/v2.0.0-onnx/model.onnx ./models/distilbert/v2.0.0-onnx/
# COPY's file mode can also apply to newly created parent directories.
# Directories need the search bit; files remain root-owned and non-writable to app.
RUN find models -type d -exec chmod 0755 {} +

ENV TOXICITY_DEVICE=cpu \
    TOXICITY_INFERENCE_BACKEND=onnx \
    TOXICITY_MODEL_DIR=/opt/app/models/distilbert/v2.0.0 \
    TOXICITY_ONNX_MODEL_DIR=/opt/app/models/distilbert/v2.0.0-onnx \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/cache \
    HF_HOME=/tmp/huggingface \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    UVICORN_LIMIT_CONCURRENCY=16 \
    UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN=60

USER 10001:10001
RUN python docker/verify_artifacts.py
EXPOSE 8000
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "/opt/app/docker/healthcheck.py"]
CMD ["python", "-m", "uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-config", "/opt/app/docker/logging.json"]
