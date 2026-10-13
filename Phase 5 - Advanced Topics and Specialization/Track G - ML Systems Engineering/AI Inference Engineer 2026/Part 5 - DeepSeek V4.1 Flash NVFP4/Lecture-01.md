# Lecture 01 — Checkpoint identity and compatibility

**Prerequisites:** model configs, tensor shapes, and the precision stack. **Time:** 2–3 hours, CPU. **Outcome:** identify a checkpoint without allocating its weights.

## Why it matters

A repository name is a label. A deployable artifact is a repository revision, tensor serialization, tokenizer/protocol, runtime loader, kernel backend, and hardware combination. Failure in any of these can appear as an out-of-memory error, a loader error, or fluent but incorrect output.

Separate three claims: files are internally consistent; the engine understands the architecture and format; the deployed model preserves task quality. None implies the next.

## Inspect before downloading weights

Use the [course checkpoint table](README.md#checkpoint-identity-and-evidence) to choose a source. Record the resolved 40-character commit from the model repository. In a shell where the Hugging Face CLI is installed, this is an **upstream CLI workflow**, not a validated serving recipe:

```bash
MODEL_REPO='s-zaizen/DeepSeek-V4.1-Flash-NVFP4'
# Set MODEL_REVISION to the resolved commit you intend to study.
: "${MODEL_REVISION:?Set an immutable model repository commit}"
hf download "$MODEL_REPO" --revision "$MODEL_REVISION" \
  --include 'config.json' 'model.safetensors.index.json' 'README.md' \
  --local-dir checkpoint-metadata
python3 labs/checkpoint_audit.py \
  --config checkpoint-metadata/config.json \
  --index checkpoint-metadata/model.safetensors.index.json
```

Run from this part's directory. The audit uses local JSON and does not import checkpoint code, load tensors, or contact a server. Its byte count is the index's declared tensor payload size, not disk allocation or peak HBM. If the index has no total size, the result stays unknown.

Read `text_config` as well as the outer config. The [official config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/config.json) distinguishes the outer architecture, text model, vision model, and quantization fields. A top-level dtype cannot describe every tensor in a mixed-format checkpoint.

### Worked config comparison

These fields were observed in the published configs during this review; resolve a revision before using them in an experiment.

| Config | Fields that distinguish the expert format |
|---|---|
| Upstream | `quant_method: fp8`, `expert_dtype: fp4`, `weight_block_size: [32, 32]` |
| [s-zaizen conversion](https://huggingface.co/s-zaizen/DeepSeek-V4.1-Flash-NVFP4/raw/main/config.json) | Also retains `expert_dtype: fp4`, but adds `quant_algo: MIXED_PRECISION`, `moe_quant_algo: NVFP4`, `group_size: 16`, and per-layer quantization entries |
| [LibertAI conversion](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-NVFP4/raw/main/config.json) | Uses `expert_dtype: nvfp4`, explicit expert scale/block fields, and separate Engram format fields |

This is a concrete reason to inspect the whole quantization contract: checking only `quant_method` or `expert_dtype` would misclassify at least one artifact. Also avoid applying a dense-weight block size to routed experts without reading the loader.

## Build a compatibility matrix

| Layer | Evidence to record | Failure probe |
|---|---|---|
| Identity | Repository, resolved revision, hashes of local metadata | Change the index and detect the hash change |
| Architecture | Registered class and source commit | Is the class the V4.1 implementation? |
| Loader | Tensor names, packed dtypes, scale shapes, excluded modules | Audit missing/unexpected keys |
| Kernels | Selected attention, MoE, activation and scale paths | Inspect dispatch logs and a profile |
| Protocol | Tokenizer and upstream prompt encoder revision | Compare encoded IDs on a fixed request |
| Quality | Held-out tasks and reference results | A deliberate scale or prompt mutation must fail |

An architecture appearing in a support list does not prove a loader accepts custom quantization metadata. Likewise, a generic “Use this model” web snippet may be automatically generated. Prefer an actual checkpoint-specific integration test.

## Guided lab

1. Run the demo audit; identify the fields marked synthetic.
2. Inspect real metadata for upstream and one conversion.
3. Compare quantization fields without guessing that `quant_method` describes all components.
4. Count unique shard names separately from tensor entries.
5. Find the corresponding loader in a pinned runtime checkout. Record support as verified, unsupported, or unknown, with a code location or error.

**Expected output:** two JSON reports, a metadata diff, and a support matrix. No GPU is needed and no inference claim is justified yet.

**Independent exercise:** review a second conversion. Identify a changed component that could alter quality even if the routed experts reconstruct identically.

## Self-check and answer guidance

- Why can packed tensor element counts mislead parameter counts? A stored byte can contain multiple low-precision values; auxiliary scales also consume storage.
- Is a successful index parse evidence of a working engine? No; it checks metadata, not loader/kernel semantics.
- Which revision belongs in the experiment record? Both checkpoint and runtime commits, plus tokenizer/encoder and conversion provenance.

**Exit criteria:** another learner can identify your exact artifact, reproduce the audit, and explain the unresolved support checks.

[Overview](README.md) · [Next: execution and memory](Lecture-02.md)
