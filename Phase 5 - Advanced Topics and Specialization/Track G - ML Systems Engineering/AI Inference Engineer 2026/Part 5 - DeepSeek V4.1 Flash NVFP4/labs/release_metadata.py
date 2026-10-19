"""Compare pinned checkpoint metadata, without downloading or executing weights.

The network step is explicit (--fetch). Unit tests and --compare are offline.
Hugging Face LFS SHA256 values are publisher-declared hashes, not locally
verified weight hashes. Equal configs or index files do not imply equal weights.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.request


RELEASES = {
    "original": ("deepseek-ai/DeepSeek-V4.1-Flash", "314113f8a7dadb388215396ba5fbee2c610a04d6"),
    "1017": ("deepseek-ai/DeepSeek-V4.1-Flash-1017", "1bae0e1088993b9b82de08f59de5be5a34e42170"),
    "community_nvfp4": ("s-zaizen/DeepSeek-V4.1-Flash-NVFP4", "179b7cda25486efbaaf8637d696759d9a791d8bd"),
}
FILES = ("config.json", "model.safetensors.index.json", "tokenizer_config.json")
MAX_BYTES = 24 * 1024 * 1024


def read_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "course-metadata-audit/1"})
    with urllib.request.urlopen(request, timeout=45) as response:
        data = response.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("metadata response exceeds 24 MiB limit")
    return json.loads(data), {"url": url, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def shard_manifest(info):
    manifest = {}
    for item in info["siblings"]:
        name = item["rfilename"]
        if not name.endswith(".safetensors"):
            continue
        digest = item.get("lfs", {}).get("sha256", "")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"missing declared LFS SHA256 for {name}")
        if name in manifest:
            raise ValueError(f"duplicate shard {name}")
        manifest[name] = {"declared_sha256": digest, "bytes": item["size"]}
    if not manifest:
        raise ValueError("no safetensors shards found")
    return manifest


def fetch_release(repo, revision):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("use a full immutable 40-character revision")
    info, info_receipt = read_json(f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true")
    if info["sha"] != revision:
        raise ValueError("server returned a different revision")
    shards = shard_manifest(info)
    documents, receipts = {}, {}
    for filename in FILES:
        documents[filename], receipts[filename] = read_json(f"https://huggingface.co/{repo}/resolve/{revision}/{filename}")
    config = documents["config.json"]
    index = documents["model.safetensors.index.json"]
    referenced = set(index["weight_map"].values())
    if referenced != set(shards):
        raise ValueError("index and model repository shard lists disagree")
    text = config.get("text_config", config)
    return {
        "repo": repo, "revision": revision, "api_receipt": info_receipt,
        "metadata_receipts": receipts, "shards": shards,
        "architecture": config.get("architectures"),
        "layers": text.get("num_hidden_layers"),
        "routed_experts": text.get("n_routed_experts"),
        "max_position_embeddings": text.get("max_position_embeddings"),
        "quantization_config": config.get("quantization_config"),
        "tensor_entries": len(index["weight_map"]),
        "declared_tensor_bytes": index["metadata"]["total_size"],
        "declared_shard_file_bytes": sum(s["bytes"] for s in shards.values()),
        "weights_downloaded": False, "runtime_tested": False,
    }


def compare(left, right):
    a, b = left["shards"], right["shards"]
    common = set(a) & set(b)
    changed = sorted(name for name in common if a[name]["declared_sha256"] != b[name]["declared_sha256"])
    return {
        "left_revision": left["revision"], "right_revision": right["revision"],
        "identical_metadata_files": [name for name in FILES if left["metadata_receipts"][name]["sha256"] == right["metadata_receipts"][name]["sha256"]],
        "common_shards": len(common), "changed_declared_shard_hashes": len(changed),
        "changed_shards": changed, "only_left": sorted(set(a) - set(b)), "only_right": sorted(set(b) - set(a)),
        "runtime_compatibility": "not established by metadata",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fetch", type=Path, metavar="OUTPUT_JSON")
    mode.add_argument("--compare", type=Path, metavar="SAVED_JSON")
    args = parser.parse_args()
    if args.fetch:
        report = {"checked_utc": datetime.now(timezone.utc).isoformat(), "evidence_kind": "downloaded metadata; declared weight hashes; no weight execution",
                  "releases": {name: fetch_release(*pin) for name, pin in RELEASES.items()}}
        args.fetch.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    else:
        report = json.loads(args.compare.read_text())
    releases = report["releases"]
    print(json.dumps({"original_vs_1017": compare(releases["original"], releases["1017"]),
                      "original_vs_community_nvfp4": compare(releases["original"], releases["community_nvfp4"])}, indent=2))


if __name__ == "__main__":
    main()
