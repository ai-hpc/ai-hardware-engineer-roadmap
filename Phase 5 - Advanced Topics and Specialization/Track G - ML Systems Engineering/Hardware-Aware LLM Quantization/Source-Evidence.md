# Quantization case study: source evidence and revision boundaries

**Checked 2026-10-18 UTC.** This page separates official architecture metadata,
community export metadata, publisher benchmarks and locally executed exercises.
No model weights were downloaded or executed. Full file URLs, SHA-256 hashes and
byte counts are in [model-evidence.json](labs/model-evidence.json).

## Immutable identities

| Role | Repository | Revision |
|---|---|---|
| Official higher-precision model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0) | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Community target export | [NVFP4 RTX5090](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090/tree/5b7a687fc8211a5d631c8ca6a593dd37eb26ce33) | `5b7a687fc8211a5d631c8ca6a593dd37eb26ce33` |
| Community external drafter | [DSpark NVFP4](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4/tree/eba1ac5a66c74902eaa95a4000a7c5eda96d8e95) | `eba1ac5a66c74902eaa95a4000a7c5eda96d8e95` |

The official and target configs specify 64 language layers, hidden size 5120,
vocabulary 248320, 4 KV heads and head dimension 256. Their explicit `layer_types`
contain 16 full-attention and 48 linear-attention entries. The target sets
`mtp_num_hidden_layers=0`; the official config has 1. The target's shard index
reports 17,915,815,528 tensor bytes in two shards, output-head scale keys and no
MTP-named tensors. DSpark has 5 full-attention layers, head dimension 128 and 8 KV
heads; it is not the target's MTP layer.

These are checks of config/index metadata, not verified tensor shapes, loaded
memory, kernel dispatch, numerical equivalence or quality. Model-card usage
widgets and the presence of a quantization config do not prove runtime support.
The quantization config declares 4-bit weights and input activations; a serialized
weight size alone cannot establish actual runtime activation precision.

## Historical results need historical identities

The community target card describes earlier output-head/MTP variants. On the check
date the publisher's branches resolved as follows:

| Branch label | Resolved revision | Intended distinction in the card |
|---|---|---|
| `pre-lmhead4` | `2ed84505a87c4b65bf959e1601bedbc4b7bbad37` | earlier BF16 output head |
| `pre-final` | `35fd99fb8434ef70ee61372df3face5174bed2c8` | quantized output head, MTP still present |

[Resolution evidence](labs/historical-branches.json) records the API responses'
hashes. Branch identity alone does not recover the exact benchmark runtime,
prompts or thermal conditions. The historical 18.80 GiB/81.6 tok/s claim must not
be assigned to the current smaller artifact.

The [pinned DSpark card](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4/blob/eba1ac5a66c74902eaa95a4000a7c5eda96d8e95/README.md)
reports 147.87→150.73 tok/s and 2.792→2.546 acceptance length when additionally
quantizing **the drafter's** Q/K/V. A separate tuned result reports 155.75 tok/s
at 2.886. None is a course measurement. Their interpretation belongs to proposal
cost/agreement, not an inferred target-model quality regression or target-byte
reduction. The lesson calculations deliberately keep these experiments separate.

## Hardware and algorithm sources

- [NVIDIA RTX 50 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/): advertised bandwidth/capacity, not measured workload speed.
- [CUTLASS 4.8 warp-level MMA guide](https://docs.nvidia.com/cutlass/4.8.0/media/docs/pythonDSL/guides/mma/wmma_programming.html): an explicit SM120 operator path; no compilation/execution claimed here.
- [TensorRT quantization schemes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html): currently 11.4 graph semantics; graph scale dtype and packed kernel metadata are distinct.
- [JetPack boundary](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-jetson.html): TensorRT 11.4 is not the JetPack-supported runtime.
- [Speculative decoding paper](https://arxiv.org/abs/2211.17192): exact rejection correction and its assumptions.
- [AWQ](https://arxiv.org/abs/2306.00978), [GPTQ](https://arxiv.org/abs/2210.17323), [SmoothQuant](https://arxiv.org/abs/2211.10438): original method motivation; exporter-specific NVFP4 support needs separate evidence.

## Learner evidence task

Run the offline tests, then optionally rerun `verify_model_metadata.py --online`
from `labs`. It fetches only pinned small metadata/model-card files, rejects a hash
change, and never executes repository code. Explain why matching metadata hashes
is stronger than a moving model name but weaker than executing a model correctly.
Answer: identities are stable, while parser semantics, actual tensor data, runtime
compatibility and numerical behavior remain separate contracts.

Local CPU results and limitations are recorded in [validation.json](labs/validation.json)
and the [CPU lab](CPU-Quantization-Lab.md).
