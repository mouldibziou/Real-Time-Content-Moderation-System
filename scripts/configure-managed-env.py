#!/usr/bin/env python3
"""Create a private cloud env file for verified RDS and ElastiCache connections."""

import argparse
import getpass
import os
from pathlib import Path
import secrets
from urllib.parse import quote


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-host", required=True)
    parser.add_argument("--redis-host", required=True)
    parser.add_argument("--output", type=Path, default=Path(".env.cloud"))
    args = parser.parse_args()
    for hostname in (args.database_host, args.redis_host):
        if not hostname or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-" for c in hostname):
            parser.error("Supply hostnames only, without a URL scheme or port.")
    if args.output.exists():
        parser.exit(1, f"Refusing to overwrite {args.output}; edit existing credentials deliberately.\n")
    certificate = Path("certs/postgres-ca.pem")
    if not certificate.is_file() or not os.access(certificate, os.R_OK):
        parser.exit(1, "Copy the RDS certificate bundle to certs/postgres-ca.pem first.\n")

    database_password = getpass.getpass("RDS toxicity_app password: ")
    redis_password = getpass.getpass("Redis AUTH password: ")
    if not database_password or not redis_password:
        parser.error("Both passwords must be nonempty.")
    # Preserve the Grafana password previously generated on this host.
    grafana_password = None
    previous = Path(".env.ec2")
    if previous.exists():
        for line in previous.read_text().splitlines():
            if line.startswith("GRAFANA_ADMIN_PASSWORD="):
                value = line.split("=", 1)[1]
                if value and value.isascii() and value.isalnum():
                    grafana_password = value
                    break

    values = {
        "DATABASE_URL": f"postgresql://toxicity_app:{quote(database_password, safe='')}@{args.database_host}:5432/toxicity?sslmode=verify-full",
        "REDIS_URL": f"rediss://:{quote(redis_password, safe='')}@{args.redis_host}:6379/0?ssl_cert_reqs=required&ssl_check_hostname=true&ssl_ca_certs=/opt/venv/lib/python3.12/site-packages/certifi/cacert.pem",
        "POSTGRES_CA_DIRECTORY": "./certs",
        "POSTGRES_CA_PATH": "/opt/app/certs/postgres-ca.pem",
        "GRAFANA_ADMIN_PASSWORD": grafana_password or secrets.token_hex(24),
        "IMAGE_PREFIX": "toxicity",
        "IMAGE_TAG": "latest",
        "API_BIND_ADDRESS": "127.0.0.1",
        "API_MEMORY": "2G",
        "WORKER_MEMORY": "3G",
        "API_CPU_THREADS": "1",
        "API_LIMIT_CONCURRENCY": "64",
        "WORKER_CPU_THREADS": "1",
        "PROMETHEUS_RETENTION_TIME": "15d",
        "PROMETHEUS_RETENTION_SIZE": "1GB",
    }
    try:
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        parser.exit(1, f"Refusing to overwrite {args.output}.\n")
    with os.fdopen(descriptor, "w") as env_file:
        env_file.write("# Private managed-service credentials. Do not commit or share.\n")
        for key, value in values.items():
            # Single quotes prevent Compose from interpolating value contents.
            env_file.write(f"{key}='{value}'\n")
    print(f"Created {args.output}. Passwords were not printed. No services were started.")


if __name__ == "__main__":
    main()
