#!/usr/bin/env python3
"""Create private credentials on a fresh EC2 host; never overwrite existing ones."""

import argparse
import os
from pathlib import Path
import secrets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".env.ec2"))
    args = parser.parse_args()
    # Hex passwords need no URL or Compose interpolation escaping.
    postgres_password = secrets.token_hex(32)
    redis_password = secrets.token_hex(32)
    values = {
        "POSTGRES_PASSWORD": postgres_password,
        "REDIS_PASSWORD": redis_password,
        "DATABASE_URL": f"postgresql://toxicity:{postgres_password}@db:5432/toxicity",
        "REDIS_URL": f"redis://:{redis_password}@redis:6379/0",
        "GRAFANA_ADMIN_PASSWORD": secrets.token_hex(24),
        "IMAGE_PREFIX": "toxicity",
        "IMAGE_TAG": "latest",
        "API_BIND_ADDRESS": "127.0.0.1",
        "API_MEMORY": "2G",
        "WORKER_MEMORY": "3G",
        "PROMETHEUS_RETENTION_TIME": "15d",
        "PROMETHEUS_RETENTION_SIZE": "1GB",
    }
    try:
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        parser.exit(1, f"Refusing to overwrite {args.output}; retain the existing credentials.\n")
    with os.fdopen(descriptor, "w") as env_file:
        env_file.write("# Private EC2 deployment credentials. Do not commit or share.\n")
        for key, value in values.items():
            env_file.write(f"{key}={value}\n")
    print(f"Created {args.output} with private credentials. Passwords were not printed.")


if __name__ == "__main__":
    main()
