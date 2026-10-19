"""Fetch small pinned official metadata files; never launch training jobs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = "e889973c84bd10c80c1a70f57ec5b426221f0264"
BASE = f"https://raw.githubusercontent.com/NVIDIA-TAO/tao-skill-bank/{REVISION}/"
FILES = {
    "versions": "versions.yaml",
    "training": "skills/models/tao-train-rtdetr/references/skill_info.yaml",
    "deploy": "skills/models/tao-train-rtdetr/references/tao-deploy-rtdetr.skill_info.yaml",
    "schema": "skills/models/tao-train-rtdetr/schemas/train.schema.json",
}


def run():
    sources, content = {}, {}
    for name, path in FILES.items():
        with urllib.request.urlopen(BASE + path, timeout=30) as response:
            payload = response.read(2_000_000)
        content[name] = payload.decode()
        sources[name] = {"url": BASE+path, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    # Check literal action contracts; this is not a general YAML parser.
    assert "command: rtdetr train -e {config_path}" in content["training"]
    assert "command: rtdetr gen_trt_engine -e {config_path}" in content["deploy"]
    assert "command: rtdetr gen_trt_engine" not in content["training"]
    assert "nvcr.io/nvidia/tao/tao-toolkit:7.0.1-pyt" in content["versions"]
    schema = json.loads(content["schema"])
    defaults = schema["default"]
    assert isinstance(defaults["dataset"]["train_data_sources"], list)
    assert isinstance(defaults["dataset"]["val_data_sources"], dict)
    return {"checked_utc_date": datetime.now(timezone.utc).date().isoformat(), "release": "7.0.1", "revision": REVISION,
            "sources": sources,
            "observations": {"train_backbone": defaults["model"]["backbone"],
                             "train_images": defaults["dataset"]["augmentation"]["train_spatial_size"],
                             "eval_images": defaults["dataset"]["augmentation"]["eval_spatial_size"],
                             "train_source_type": "list", "validation_source_type": "mapping",
                             "training_and_deploy_commands_separate": True},
            "not_executed": ["container pull", "TAO training", "TAO export", "TensorRT build", "Jetson inference"]}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=run()
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
