from urllib.request import urlopen

with urlopen("http://127.0.0.1:8000/v2/models/toxicity/versions/1/ready", timeout=3) as response:
    if response.status != 200:
        raise SystemExit(1)
