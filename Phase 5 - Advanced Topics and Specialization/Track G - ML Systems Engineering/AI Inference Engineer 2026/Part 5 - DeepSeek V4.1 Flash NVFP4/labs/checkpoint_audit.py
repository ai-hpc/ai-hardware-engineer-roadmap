#!/usr/bin/env python3
"""Inspect local checkpoint metadata; never load weights or claim engine support."""
import argparse
import hashlib
import json
from pathlib import Path


def audit(config, index):
    if not isinstance(config, dict) or not isinstance(index, dict):
        raise ValueError("config and index must be JSON objects")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("index needs a nonempty weight_map")
    if not all(isinstance(k, str) and isinstance(v, str) and v for k, v in weight_map.items()):
        raise ValueError("weight_map must map tensor names to nonempty shard names")
    metadata = index.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    size = metadata.get("total_size")
    if size is not None and (type(size) is not int or size <= 0):
        raise ValueError("metadata.total_size must be a positive integer byte count")
    text_config = config.get("text_config", config)
    if not isinstance(text_config, dict):
        raise ValueError("text_config must be an object")
    return {
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "text_model_type": text_config.get("model_type"),
        "layers": text_config.get("num_hidden_layers"),
        "routed_experts": text_config.get("n_routed_experts"),
        "experts_per_token": text_config.get("num_experts_per_tok"),
        "quantization_config": config.get("quantization_config"),
        "tensor_entries": len(weight_map),
        "unique_shards": len(set(weight_map.values())),
        "declared_payload_bytes": size,
        "declared_payload_gib": size / 2**30 if size is not None else None,
        "runtime_compatibility": "not established",
        "memory_note": "Index payload only; excludes runtime allocations and does not prove GPU fit.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.demo:
        if args.config or args.index:
            parser.error("--demo cannot be combined with input paths")
        config = {"model_type": "synthetic_moe", "text_config": {
            "num_hidden_layers": 2, "n_routed_experts": 4, "num_experts_per_tok": 2},
            "quantization_config": {"expert_dtype": "nvfp4"}}
        index = {"metadata": {"total_size": 3 * 2**30},
                 "weight_map": {"expert.0": "a.safetensors", "expert.1": "a.safetensors",
                                "dense": "b.safetensors"}}
        provenance = {"execution_status": "synthetic fixture"}
    else:
        if not args.config or not args.index:
            parser.error("provide --config and --index, or --demo")
        try:
            raw_config, raw_index = args.config.read_bytes(), args.index.read_bytes()
            config, index = json.loads(raw_config), json.loads(raw_index)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        provenance = {"execution_status": "metadata inspection only",
                      "config_sha256": hashlib.sha256(raw_config).hexdigest(),
                      "index_sha256": hashlib.sha256(raw_index).hexdigest()}
    try:
        result = audit(config, index)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({**provenance, **result}, indent=2))


if __name__ == "__main__":
    main()
