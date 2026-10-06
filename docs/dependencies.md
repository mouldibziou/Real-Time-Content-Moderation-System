# Dependency environments

The first deployment target is **CPython 3.12, Linux amd64, glibc, CPU**.
`.python-version` records the Python minor version. These locks are target-specific;
they are not a claim of macOS, ARM64, Alpine/musl, or CUDA compatibility. The
container step will pin the Python base image digest and therefore its patch
version and OS contents.

## Inputs and locks

| Environment | Direct requirements (edit these) | Generated lock (install these) |
| --- | --- | --- |
| Serving | `requirements/serving.in` | `requirements.txt` |
| Migrations | `requirements/migrations.in` | `requirements/migrations.txt` |
| Training | `requirements/training.in` | `requirements/training.txt` |
| ONNX export | `requirements/export.in` | `requirements/export.txt` |
| Development tooling | `requirements/development.in` | `requirements/development.txt` |

Direct packages start at the versions already exercised by the application in
the local environment, with Torch's matching CPU build selected for Linux.
Generated locks pin all transitive versions and include SHA-256 distribution
hashes. A lock establishes dependency identity; clean installation and tests
establish compatibility.

The serving group explicitly includes Pydantic because the schemas import it,
and Uvicorn because it starts the ASGI server. `fastapi[standard]` is replaced by
FastAPI plus Uvicorn without its unrelated CLI and optional extras. Torch and
Captum remain necessary for explanations. Captum's required Matplotlib dependency
also remains; omitting a package's required dependencies is not a size optimization.

Alembic and the synchronous PostgreSQL driver are migration-only dependencies.
SQLAlchemy's asyncio extra requests Greenlet for serving. Migrations constrain
their shared packages against the serving lock so both can be installed together.
Training and export resolve independently using the same direct ML versions.
`requirements/constraints.in` holds the tested `fsspec==2026.2.0` compatibility
constraint required by datasets 4.8.4. This constrains an existing Torch/Hub
dependency without adding training packages to serving.

Pandas, datasets, scikit-learn, joblib, and Accelerate are isolated to training.
Optimum and its ONNX dependencies are isolated to export. `math` is part of the
standard library and is not installed. The test suite uses standard-library
`unittest`; development tooling only adds the pinned lock generator.

## CPU wheel source

`torch==2.11.0+cpu` selects the CPU distribution explicitly. The generator uses
uv's scoped CPU backend to resolve Torch and obtain its wheel hashes, then records
a `--find-links` URL for PyTorch's official CPU Torch wheel listing in the
generated serving, training, and export locks. Other packages use PyPI. This avoids treating the entire PyTorch
index as an alternate source for unrelated dependencies, and lets standard pip
install the generated locks without requiring uv in the runtime image.

## Installation

In a fresh Python 3.12 Linux amd64 environment, install serving with:

```sh
python -m pip install --require-hashes --only-binary=:all: -r requirements.txt
python -m pip check
python -m unittest discover -s tests -t .
```

For a deployment environment that also runs migrations:

```sh
python -m pip install --require-hashes --only-binary=:all: \
  -r requirements.txt -r requirements/migrations.txt
```

Training and export locks are standalone environments; they do not include the
API or database stack. Do not install them in the serving image. The training
lock targets CPU; GPU training needs a separately resolved and tested lock.

The real-model smoke test remains opt-in and requires both serving model formats
to be present. The ordinary test suite does not establish database availability,
model quality, or production capacity.

## Regeneration and updates

Use uv **0.12.23** and an installed Python 3.12 interpreter:

```sh
sh scripts/lock-dependencies.sh
```

The script resolves serving first and then the other environments. It refuses a
different uv version and automatic Python downloads. Commit changed inputs and
generated locks together. To update a direct package, change its input pin,
regenerate, and repeat clean installation and tests. Existing transitive pins
are retained when compatible; deliberate transitive upgrades require passing
`--upgrade-package` or `--upgrade` to the relevant compile command and validating
the resulting lock.

Never hand-edit hashes or copy a full freeze from the existing Mac environment.
The original `venv/` is intentionally left untouched.

## Verification performed

The locks were verified using Python **3.12.15** on **Linux x86_64**, in isolated
virtual environments inside `python:3.12-slim-bookworm` at image digest
`sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258`.

- Serving installed from the lock with hashes required and source builds disabled.
- `pip check` passed and the API imported successfully with explicit test database
  configuration, without loading weights or contacting PostgreSQL.
- Torch identified itself as `2.11.0+cpu`; CUDA, training, export, migration, and
  development packages were absent from the serving environment.
- The complete test suite ran 125 tests: 124 passed and the opt-in real-model
  smoke test was skipped because model artifacts were not mounted.
- The separate migration environment installed successfully, passed `pip check`,
  and generated the Alembic upgrade SQL offline. No live database migration ran.

Training and export locks were resolved with binary wheels and package hashes;
training and model export were not executed. Full model startup and capacity
measurements remain for container verification.

References: [uv locking requirements](https://docs.astral.sh/uv/pip/compile/),
[uv cross-platform resolution](https://docs.astral.sh/uv/concepts/resolution/),
and [PyTorch installation](https://pytorch.org/get-started/locally/).
