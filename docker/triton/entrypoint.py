"""Render deployment settings at container startup, then replace PID 1 with Triton."""
import json
import os
from pathlib import Path
import shutil


def positive(name, default, *, allow_zero=False):
    value = int(os.environ.get(name, default))
    if value < (0 if allow_zero else 1):
        raise ValueError(f"{name} has an invalid value")
    return value


def render_config(root=Path("/opt/artifacts")):
    config = json.loads((root / "models/distilbert/v2.0.0/config.json").read_text())
    labels = [config["id2label"][str(i)] for i in range(len(config["id2label"]))]
    batch = positive("TRITON_MAX_BATCH_SIZE", 8)
    # Keep the public maximum as the absolute internal ceiling too.
    if batch > 64:
        raise ValueError("TRITON_MAX_BATCH_SIZE must not exceed 64")
    delay = positive("TRITON_QUEUE_DELAY_US", 0, allow_zero=True)
    capacity = positive("TRITON_QUEUE_CAPACITY", 64)
    timeout = positive("TRITON_QUEUE_TIMEOUT_US", 1000000)
    threads = positive("TRITON_CPU_THREADS", 1)
    parameters = {"artifact_root": str(root), "model_version": "v2.0.0",
                  "model_type": config["model_type"], "labels": json.dumps(labels),
                  "max_length": str(config["max_position_embeddings"]), "cpu_threads": str(threads)}
    value = f'''name: "toxicity"
backend: "python"
max_batch_size: {batch}
input [ {{ name: "TEXT" data_type: TYPE_STRING dims: [1] }} ]
output [ {{ name: "LOGITS" data_type: TYPE_FP32 dims: [{len(labels)}] }} ]
instance_group [ {{ count: 1 kind: KIND_CPU }} ]
dynamic_batching {{
  max_queue_delay_microseconds: {delay}
  default_queue_policy {{
    max_queue_size: {capacity}
    timeout_action: REJECT
    default_timeout_microseconds: {timeout}
  }}
}}
'''
    for key, item in parameters.items():
        value += f'parameters {{ key: {json.dumps(key)} value {{ string_value: {json.dumps(item)} }} }}\n'
    return value


if __name__ == "__main__":
    repository = Path("/tmp/triton-models/toxicity")
    (repository / "1").mkdir(parents=True, exist_ok=True)
    shutil.copyfile("/opt/triton-model/model.py", repository / "1/model.py")
    (repository / "config.pbtxt").write_text(render_config())
    os.execvp("tritonserver", ["tritonserver", "--model-repository=/tmp/triton-models",
                             "--backend-directory=" + os.environ["TRITON_BACKEND_DIRECTORY"],
                             "--allow-grpc=false", "--allow-gpu-metrics=false",
                             "--pinned-memory-pool-byte-size=0",
                             "--disable-auto-complete-config", "--exit-timeout-secs=60"])
