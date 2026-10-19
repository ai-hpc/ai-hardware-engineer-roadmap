"""Metadata and hypothetical cache allocations; not a Gemma runtime allocator."""
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class Allocation:
    name: str
    heads: int
    dimension: int
    window: int | None = None


def cache_bytes(allocations, context, batch=1, element_bytes=2):
    """Count each explicitly named physical allocation once.

    A name repeated by two consumers is one shared allocation; names do not
    imply sharing unless the complete descriptor agrees. Payload only.
    """
    for value in (context, batch, element_bytes):
        if type(value) is not int or value <= 0:
            raise ValueError("positive integer context/batch/element size required")
    unique = {}
    for item in allocations:
        if (not item.name or type(item.heads) is not int or item.heads <= 0
                or type(item.dimension) is not int or item.dimension <= 0
                or item.window is not None and (type(item.window) is not int or item.window <= 0)):
            raise ValueError("invalid allocation descriptor")
        if item.name in unique and unique[item.name] != item:
            raise ValueError("shared allocation has conflicting contracts")
        unique[item.name] = item
    return sum(batch * a.heads * a.dimension * element_bytes
               * (min(context, a.window) if a.window is not None else context)
               for a in unique.values())


def lesson_allocations():
    """Lecture 1's hypothetical 16-layer model, NOT Gemma E2B dimensions."""
    return [Allocation(f"layer{i}.{kind}", 4, 128, 512 if i < 12 else None)
            for i in range(16) for kind in ("k", "v")]


def summarize(config):
    text = config["text_config"]
    if len(text["layer_types"]) != text["num_hidden_layers"]:
        raise ValueError("layer list length disagrees with configuration")
    return {key: text[key] for key in
            ("hidden_size", "num_hidden_layers", "num_key_value_heads", "head_dim",
             "global_head_dim", "num_kv_shared_layers", "attention_k_eq_v")}


def report():
    here = Path(__file__).parent
    result = {"scope": "config snapshots and a hypothetical cache model; no runtime cache measurement"}
    for name in ("target", "assistant"):
        raw = (here / f"{name}-config.json").read_bytes()
        config = json.loads(raw)
        result[name] = {"architecture": config["architectures"], "config_sha256": hashlib.sha256(raw).hexdigest(),
                        "text": summarize(config), "layer_types": dict(Counter(config["text_config"]["layer_types"]))}
    result["hypothetical_cache_MiB"] = {str(context): cache_bytes(lesson_allocations(), context)/2**20
                                        for context in (256, 8192, 16384)}
    a = Allocation("shared.global.k", 4, 128)
    result["one_allocation_two_consumers_bytes"] = cache_bytes([a, a], 8192)
    result["assistant_backbone_hidden_size"] = json.loads((here / "assistant-config.json").read_text())["backbone_hidden_size"]
    return result


if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
