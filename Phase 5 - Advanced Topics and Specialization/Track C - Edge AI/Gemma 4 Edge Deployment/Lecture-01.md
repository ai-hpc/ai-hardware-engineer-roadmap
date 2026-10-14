# Lecture 01 — Gemma 4 identity, architecture, and memory budgets

**Input:** a checkpoint ID and sufficient host storage for its metadata. **Output:** a configuration audit and a falsifiable memory estimate. [Course map](README.md) · [Next: quantization](Lecture-02.md).

## 1. Model names are not a hardware specification

Start with `google/gemma-4-E2B-it` as a metadata case study. The official card distinguishes effective parameters from total storage; E2B uses per-layer embedding tables, and the family includes both dense and MoE designs. Do not substitute effective/active parameters for total resident weights when deciding whether a model fits. The [model card](https://ai.google.dev/gemma/docs/core/model_card_4) is the family overview; the downloaded checkpoint configuration and tensor files are the deployment evidence.

A configuration can change under `main`. Resolve and save an immutable revision before downloading weights. Record processor/tokenizer files too: a correct decoder with the wrong image processing or chat template is a different experiment.

## 2. Audit the actual configuration

Save `config.json` from your selected revision locally, then run this metadata-only exercise:

```python
import collections
import hashlib
import json
from pathlib import Path

path = Path("config.json")
raw = path.read_bytes()
config = json.loads(raw)
text = config.get("text_config", config)
keys = ("model_type", "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "head_dim", "global_head_dim",
        "sliding_window", "num_kv_shared_layers", "max_position_embeddings")
print("config_sha256", hashlib.sha256(raw).hexdigest())
print(json.dumps({k: text.get(k) for k in keys}, indent=2))
print("layer_types", dict(collections.Counter(text.get("layer_types", []))))
```

For the inspected [E2B configuration](https://huggingface.co/google/gemma-4-E2B-it/blob/main/config.json), local and full-attention layers have different head dimensions; cache-sharing metadata also exists. This invalidates a universal `2 × layers × heads × head_dim × context` estimate applied without checking model implementation and runtime allocation. Compare metadata with actual KV tensors or runtime cache logs.

## 3. Work through cache growth

For a conventional layer that stores separate K and V tensors, a useful payload estimate is:

\[
M_l = B H_{KV,l} T_l (D_{K,l}+D_{V,l})s_l.
\]

Here `s` is bytes per stored element and `T_l = min(T, window_l)` only when the runtime truly bounds that layer's retained cache. A static allocator may reserve more. Shared K/V or cross-layer cache reuse must be accounted for once per physical allocation, not once per conceptual consumer. Quantization scales, alignment, and allocator slack add overhead.

**Teaching example, not Gemma specifications:** 12 local layers with a 512-token window and four global layers, all with 4 KV heads, head dimension 128, batch one and 2-byte elements. At 8,192 tokens:

- Each retained token-layer costs `2 × 4 × 128 × 2 = 2,048` bytes.
- Local cache: `12 × 512 × 2,048 = 12 MiB`.
- Global cache: `4 × 8,192 × 2,048 = 64 MiB`.
- Total payload: **76 MiB**. At 16,384 tokens it becomes **140 MiB**.

Local cache saturates; global cache keeps growing. The mixed model is not constant-memory with respect to context.

## 4. Fit the whole service

Build a ledger for weights, embedding/vision/audio components, KV/state, activations, temporary workspace, runtime, camera buffers, Linux, and reserve. On Jetson, CPU and GPU use the same physical DRAM budget. A 4-bit file may need temporary unpacking or higher-precision activations, and conversion may need more RAM than steady-state inference.

For decode, `bytes touched per token / sustained bandwidth` is a lower bound under stated traffic assumptions. Do not divide floating-point work by sparse INT8 TOPS. For MoE or embedding lookups, bytes touched and total resident bytes are different quantities.

## Lab and acceptance

Submit the config hash, layer-type counts, a cache estimate at three contexts, and measured cache allocation if a runtime is available. Explain every discrepancy larger than a predeclared tolerance using reservation, sharing, dtype, or workspace evidence. Without hardware, submit the metadata/math result and leave measurement cells **not run**. Pass the accepted memory budget into Lecture 02.
