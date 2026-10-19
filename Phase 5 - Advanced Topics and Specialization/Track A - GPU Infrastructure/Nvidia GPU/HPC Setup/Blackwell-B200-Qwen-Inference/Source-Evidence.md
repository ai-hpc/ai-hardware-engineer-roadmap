# B200/Qwen: source evidence and execution boundary

**Checked 2026-10-18.** Hardware inventory, format semantics, model identity and
runtime compatibility are separate evidence. This course review used CPU checks
and source inspection; it did not run a B200, NVL72 or a 72B model.

## Model snapshot

The official [Qwen2.5-72B-Instruct config](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/495f39366efef23836d0cfae4fbe635880d2be31/config.json)
is pinned at `495f39366efef23836d0cfae4fbe635880d2be31`. The fetched config, index
and model-card hashes are in [qwen-metadata.json](evidence/qwen-metadata.json).
The index reports 145,412,407,296 tensor bytes across 37 shards. The metadata
identifies the Qwen license, not Apache-2.0. The model's 32K default configuration
and the card's context-extension recipe must not be conflated.

This is an explicit historical full-attention teaching model. The newer hybrid
case in the [quantization course](../../../../Track%20G%20-%20ML%20Systems%20Engineering/Hardware-Aware%20LLM%20Quantization/Source-Evidence.md)
has a different vocabulary, hidden size and state architecture; its memory
formula and community quantization export cannot simply replace this model ID.

## Platform and software sources

| Source | What was checked | Limit |
|---|---|---|
| [DGX B200 guide](https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html) | eight GPUs, 1440 GB aggregate HBM | actual available memory needs device query |
| [GB200 specification](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) | two GPUs/Grace Superchip, 372 GB GPU HBM, NVL72 composition | a different SKU from DGX B200; aggregate links are not one CUDA device |
| [CUTLASS 4.8 SM100](https://docs.nvidia.com/cutlass/4.8.0/media/docs/cpp/blackwell_functionality.html) | operand/scale/accumulator contracts | no GPU compilation or kernel result here |
| [vLLM serve documentation](https://docs.vllm.ai/en/stable/cli/serve/) | current CLI concepts | moving docs, backed by pinned source check below |
| [Transformer Engine NVFP4](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html) | 2.20.2 training recipe documentation | not proof of a serving checkpoint loader |
| [TensorRT schemes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html) | 11.4 graph semantics | distinct from TensorRT-LLM backend and JetPack-supported versions |

[Software source hashes](evidence/software-source-snapshot.json) pin vLLM 0.32.0
at `5df63a7203cdb9328d824a27b682e77c7b9928be` and CUTLASS 4.8.0 at
`098de2a652cf8f00fd70b2df54051c7eccbb855a`. Serving flags were found in the selected
parser sources; a full runtime parse/startup or B200 dispatch was **not executed**.
Versioned vLLM documentation URLs returned HTTP 403 during the check, so the
course links the accessible stable docs plus immutable source.

## Locally executed evidence

[The CPU budget record](evidence/cpu-budget-result.json) uses existing quantization
and collective arithmetic: 10 GiB of BF16 KV at 32K; a hypothetical free-memory
budget fits one such sequence but not two; TP=72 fails the simple equal-head
partition; normalized AllReduce bandwidth is not its algorithm bandwidth.
All calculations passed their stated assertions. They are not hardware fit tests.

The linked [FP4 lab](../../../../Track%20G%20-%20ML%20Systems%20Engineering/Hardware-Aware%20LLM%20Quantization/CPU-Quantization-Lab.md),
[attention lab](../../../../../Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/02%20-%20Kernel%20Engineering/FlashAttention%20Course/CPU-Attention-Lab.md)
and [Gloo collectives lab](../NCCL-Deep-Dive/CPU-Collectives-Lab/Guide.md) contain
actual framework/test evidence and their limitations. This pass reuses those
fixtures instead of inventing another set of model or GPU results.

[Validation](evidence/validation.json) and [source URL checks](evidence/source-check-2026-10-18.json)
record the final checks. Remaining device work includes loading the selected
checkpoint, output/quality evaluation, dispatch validation, per-rank memory,
NCCL/topology testing, latency/goodput and recovery.
