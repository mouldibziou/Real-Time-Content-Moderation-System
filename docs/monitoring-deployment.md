# Build and deploy the five-service stack

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
