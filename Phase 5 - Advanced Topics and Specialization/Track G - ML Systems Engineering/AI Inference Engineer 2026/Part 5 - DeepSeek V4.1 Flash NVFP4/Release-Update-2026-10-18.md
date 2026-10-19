# Release check: DeepSeek V4.1 Flash 1017

**Checked:** 2026-10-18. **Time:** 35–50 minutes. **Needs:** Python 3.10+, no GPU. **Outcome:** distinguish model identity, storage layout, and runtime compatibility using evidence you can reproduce.

DeepSeek [announced V4.1 Flash 1017 on October 17](https://api-docs.deepseek.com/news/news261017/). It is a separate official checkpoint. Part 5 retains the earlier V4.1/NVFP4 case because its conversions, measurements, and compatibility questions belong to that source revision. A newer model name does not update an existing conversion.

## What we actually checked

The [metadata script](labs/release_metadata.py) fetched three JSON files and repository file metadata at immutable revisions. It did not download weights, run model code, launch a server, or measure quality or speed. The [saved evidence](labs/release-evidence-2026-10-18.json) contains URLs, local SHA256 hashes for downloaded JSON, and the hosting service's declared SHA256 hashes for weight shards. Those declared weight hashes were **not independently checked against downloaded weight bytes**.

| Artifact | Pinned repository revision | Tensor entries | Declared tensor payload |
|---|---|---:|---:|
| [Official original](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/314113f8a7dadb388215396ba5fbee2c610a04d6) | `314113f8a7dadb388215396ba5fbee2c610a04d6` | 96,085 | 510,286,023,000 bytes |
| [Official 1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170) | `1bae0e1088993b9b82de08f59de5be5a34e42170` | 96,085 | 510,286,023,000 bytes |
| [Earlier community NVFP4 conversion](https://huggingface.co/s-zaizen/DeepSeek-V4.1-Flash-NVFP4/tree/179b7cda25486efbaaf8637d696759d9a791d8bd) | `179b7cda25486efbaaf8637d696759d9a791d8bd` | 188,245 | 527,273,322,840 bytes |

All have 48 safetensors shards. Payload excludes file headers and other repository files; resident GPU memory additionally depends on loading, repacking, replication, workspace, and runtime state.

**The useful surprise:** original and 1017 have identical downloaded `config.json`, `model.safetensors.index.json`, and `tokenizer_config.json`, but **all 48 matching shard names have different declared weight hashes**. An index maps tensor names to files; it does not identify the contents of those files. A configuration describes the computation's dimensions; it does not specify its learned parameter values.

The community conversion differs from the original on 40 shard hashes and retains eight matching ones. This is evidence about these files, not proof of a complete conversion lineage or compatibility with 1017. Its payload is larger than the official mixed-precision checkpoint. “NVFP4” is not a promise of a smaller checkpoint than a source that already contains FP4 expert tensors.

## What follows, and what still requires a runtime

Both official configs declare `DeepseekV41ForCausalLM`, 40 backbone layers, 384 routed experts, and `max_position_embeddings=1048576`. Their quantization metadata combines FP8 settings with `expert_dtype: fp4`; do not describe either as uniformly BF16 or uniformly FP8. The community config adds its own mixed-precision/NVFP4 group and layer metadata. A single `quant_method` string is insufficient to reconstruct the complete serialization contract.

The [1017 model card at the pinned revision](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/README.md) supplies prompt encoding guidance and reference inference material. It does not supply a generic Jinja chat template. Check its `encoding/` protocol and runtime adaptation before comparing outputs. Matching `tokenizer_config.json` alone does not establish identical tokenizer vocabulary, prompt formatting, or generated behavior.

Runtime support for an architecture is only one requirement. For 1017, separately establish loader support for the exact checkpoint, packing/scales, prompt encoding, state handling, and supported hardware. Then compare logits and held-out task behavior against a matched 1017 reference. The older NVFP4 conversion's parity evidence cannot substitute for that test. GPU loading, 1M-context operation, conversion quality, and serving benchmarks remain **not run in this update**; publisher-reported improvements are not local benchmark results.

## Exercise: reject a tempting shortcut

From this Part 5 directory:

```bash
# Offline: reproduce the comparison using saved evidence.
python3 labs/release_metadata.py --compare labs/release-evidence-2026-10-18.json
python3 -m unittest discover -s labs -p 'test_release_metadata.py' -v

# Optional network step: fetch about 32 MB of pinned metadata, no weight files.
python3 labs/release_metadata.py --fetch /tmp/deepseek-release-check.json
```

Before running, predict whether equal config and index hashes imply unchanged weights. Then answer:

1. Which equality tests pass, and which identity test fails?
2. A teammate proposes renaming the old converted directory to `1017-NVFP4`. What evidence would that label require?
3. Which fields describe storage, and which observations would be needed to claim faster serving?

**Answers:** the three metadata byte comparisons pass, while 48/48 declared shard hashes differ. A genuine 1017 conversion needs pinned source identity, conversion recipe/version, resulting artifact hashes, and correctness evidence against that source. Renaming supplies none of them. Payload and shard sizes describe stored bytes; they cannot predict resident memory or latency alone. Measure a supported runtime under matched quality, request mix, batch/context lengths, and hardware before claiming speed.

**Independent transfer:** edit a copy of the saved JSON to change one shard hash while leaving every config receipt unchanged. The comparator must report one additional or removed mismatch as appropriate. Next remove a shard and explain why equal hashes for the remaining intersection no longer establishes complete artifact equality. Submit the original comparison, your modified comparison, and a four-sentence compatibility verdict. Pass when you distinguish downloaded evidence, declared evidence, and untested execution without inventing a speedup.
