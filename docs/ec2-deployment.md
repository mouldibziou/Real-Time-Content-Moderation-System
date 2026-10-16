# Single EC2 host deployment

Use `compose.cloud.yaml` followed by `compose.ec2.yaml`. The overlay adds private
PostgreSQL and Redis containers, persistent volumes, health-gated startup, bounded
Docker log files (three 10 MB files per container), monitoring memory limits, and
one inference thread per API/worker on the initial two-vCPU host. Docker volumes
live on the host's EBS disk. Independent backups are still required before public
launch; volumes alone are not backups.

On a fresh host, copy both Compose files and `scripts/init-ec2-env.py` to a private
deployment directory, then run:

```sh
python3 init-ec2-env.py
docker compose --env-file .env.ec2 -f compose.cloud.yaml -f compose.ec2.yaml config --quiet
```

The initializer writes new random credentials with owner-only access and refuses
to replace an existing file. Never copy the development `.env.docker` to EC2, and
never print the resolved Compose configuration in shared output: it contains
credentials. Do not change PostgreSQL credentials by regenerating the env file
after the database volume is initialized. Existing database users must be changed
with SQL and corresponding application configuration as a deliberate operation.

Transfer the five existing Linux amd64 images to this host, or publish them to a
registry and update `IMAGE_PREFIX`/`IMAGE_TAG`. The model artifacts are already in
the application images. No source build is required on EC2.

After the five images are loaded, pull only the two new dependency images:

```sh
docker compose --env-file .env.ec2 -f compose.cloud.yaml -f compose.ec2.yaml pull db redis
docker compose --env-file .env.ec2 -f compose.cloud.yaml -f compose.ec2.yaml up -d --no-build --pull never
docker compose --env-file .env.ec2 -f compose.cloud.yaml -f compose.ec2.yaml ps -a
```

The migration runs against a fresh database before API/worker startup and exits
successfully. This deployment starts with an empty database; existing local
classifications are not transferred automatically. Use a reviewed backup/restore
operation if historical local data is needed.

API (8000), Grafana (3000), and Prometheus (9090) bind to loopback. PostgreSQL,
Redis, and scrape ports have no published host ports. To test from a Mac with
local ports available, open an SSH tunnel (adjust key path and current server IP):

```sh
ssh -i /path/to/toxicity-server-key.pem -N \
  -L 18000:127.0.0.1:8000 \
  -L 13000:127.0.0.1:3000 \
  -L 19090:127.0.0.1:9090 ec2-user@SERVER_IP
```

Then access the API at `http://localhost:18000`, Grafana at
`http://localhost:13000`, and Prometheus at `http://localhost:19090`.
Public HTTP/HTTPS routing is not configured by this overlay; configure and verify
TLS, access controls, and backups before opening the API to clients.

The default tag is `latest` for the initial transfer of existing images. Before
future releases, use explicit version tags and record the deployed image IDs.
`down` preserves volumes; `down --volumes` permanently deletes all database,
queue, and monitoring data. A normal restart does not reinitialize PostgreSQL.
