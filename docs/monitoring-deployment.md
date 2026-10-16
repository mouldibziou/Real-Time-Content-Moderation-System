# Build and deploy the five-service stack

For the current AWS deployment, use this file alone with managed RDS PostgreSQL
and node-based ElastiCache Redis OSS (cluster mode disabled). `compose.ec2.yaml`
is an alternative for self-hosted databases and is not part of that deployment.

## AWS managed connections

On EC2, copy the downloaded RDS CA bundle to `certs/postgres-ca.pem`. Make the
directory mode 755 and certificate mode 644: certificates are public data, and
the containers run as UID 10001, not the SSH user. The certificate directory is
mounted read-only. `POSTGRES_CA_PATH=/opt/app/certs/postgres-ca.pem` becomes
`PGSSLROOTCERT` for both the async application driver and sync migration driver.
Use `sslmode=verify-full` in `DATABASE_URL`; the application's URL helper converts
it to the equivalent asyncpg option. Do not add `sslrootcert` to the URL: SQLAlchemy
passes URL query arguments as connection keywords, which asyncpg does not accept
for this option.

Run `scripts/configure-managed-env.py` from the deployment directory with
`--database-host` and `--redis-host`. It asks for the application-role password
and Redis AUTH password privately, URL-encodes them, and writes `.env.cloud` with
mode 600 without printing secrets. Existing output files are never overwritten.
It preserves the previously generated Grafana password from `.env.ec2` when
available. That previous file's Docker database credentials are not used.

The Redis URL uses `rediss`, required certificate and hostname verification, and
the certifi CA bundle in the existing Python 3.12 serving images. Those settings
match the container-based Redis connection check. No image rebuild is needed.
Keep env files private; resolved `docker compose config` output contains secrets,
so use `config --quiet` for validation.

```sh
docker compose --env-file .env.cloud -f compose.cloud.yaml config --quiet
docker compose --env-file .env.cloud -f compose.cloud.yaml up -d --no-build --pull never
```

All five services rotate their Docker stdout/stderr logs into three 10 MB files
per container. Logs remain local to EC2 unless a separate log collector is set up.
`API_LIMIT_CONCURRENCY` defaults to 64 and is mapped by Compose to Uvicorn's
`UVICORN_LIMIT_CONCURRENCY`. This limits concurrent admitted connections/tasks,
not inference throughput; model inference remains on the single executor.
The optional certificate directory and empty default `POSTGRES_CA_PATH` also let
non-TLS development connections continue to use this configuration.

`compose.cloud.yaml` now defines these images, all built for Linux amd64:

| Service | Default image | Purpose |
| --- | --- | --- |
| api | `toxicity-api:latest` | HTTP API and `/metrics/` |
| worker | `toxicity-worker:latest` | Background explanations and private port 9101 metrics |
| migrate | `toxicity-migrations:latest` | One-shot database migration before API/worker startup |
| prometheus | `toxicity-prometheus:latest` | Scrapes API and worker; stores time series |
| grafana | `toxicity-grafana:latest` | Provisioned datasource and application dashboard |

The three application images use `Dockerfile`. The two monitoring images use
`Dockerfile.monitoring`, with official Prometheus and Grafana bases. Monitoring
configuration is copied into the images; deployment needs no source bind mounts.
All images share `IMAGE_PREFIX` and `IMAGE_TAG`. For example, a registry prefix
`123456789012.dkr.ecr.REGION.amazonaws.com/toxicity` names five separate repositories.
Building images does not upload them to GitHub or an image registry.

## Build

Copy `docker/runtime.env.example` to `.env.docker` and set PostgreSQL, Redis, and
Grafana credentials. The cloud Compose file expects reachable PostgreSQL and Redis
services; it does not provision either managed service. Keep this env file private.
The model artifacts described in `docs/container.md` must exist before building.

```sh
docker compose --env-file .env.docker -f compose.cloud.yaml config --quiet
docker compose --env-file .env.docker -f compose.cloud.yaml build
```

After making these images available on the EC2 host, start with:

```sh
docker compose --env-file .env.docker -f compose.cloud.yaml up -d
```

If building directly on EC2, use `up -d --build` instead. This configuration targets
an x86-64 EC2 instance. An ARM/Graviton deployment needs a separate application
architecture build and dependency validation.

## Access and persistence

The API listens on localhost port 8000 by default. Prometheus listens on localhost
9090 and Grafana on localhost 3000. Use an SSH tunnel from your laptop to view
monitoring on EC2:

```sh
ssh -L 3000:127.0.0.1:3000 -L 9090:127.0.0.1:9090 YOUR_EC2_SSH_ALIAS
```

Open http://localhost:3000/d/toxicity-local and sign in as `admin` with your configured
password. Unlike the local demo, cloud Grafana disables anonymous viewing. Its
password is initialized on first startup; changing the env file does not reset an
existing Grafana volume's password. Prometheus targets are at
http://localhost:9090/targets. The worker metrics port stays on the Docker network.

Prometheus scrapes every five seconds and retains history for 15 days or 1 GB,
whichever limit is reached first. Override `PROMETHEUS_RETENTION_TIME` and
`PROMETHEUS_RETENTION_SIZE` as needed. Retention applies to data blocks; current
blocks and the write-ahead log require additional disk space. Named Docker volumes
persist Prometheus history and Grafana state across container recreation, on the
EC2 host's disk. They do not survive loss of that disk automatically.

`docker compose --env-file .env.docker -f compose.cloud.yaml down` preserves these
volumes. Adding `--volumes` permanently deletes the monitoring history and Grafana
state. Migrations exit after completion; they are not a continuously running sixth
server. PostgreSQL and Redis remain external dependencies in this cloud setup.

## Local development

The existing local demo continues to use `compose.yaml` plus
`compose.monitoring.yaml`, with PostgreSQL and Redis containers and anonymous
Grafana viewing. See [local-monitoring.md](local-monitoring.md). Local application
builds now use the same three default image names as the cloud configuration.
