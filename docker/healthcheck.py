"""Read runtime readiness without importing the application or ML stack."""
import json
import sys
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener


def check_readiness(opener=None) -> bool:
    # Internal loopback probes must not go through a deployment's HTTP proxy.
    if opener is None:
        opener = build_opener(ProxyHandler({}))
    try:
        with opener.open("http://127.0.0.1:8000/v1/health/ready", timeout=2) as response:
            if response.status != 200:
                return False
            payload = json.load(response)
    except (OSError, ValueError, URLError):
        return False
    return isinstance(payload, dict) and payload.get("is_ready") is True


if __name__ == "__main__":
    sys.exit(0 if check_readiness() else 1)
