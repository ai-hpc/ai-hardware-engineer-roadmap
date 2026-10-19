# 05 — Measure an L40S serving system

**Outcome:** compare feasible deployments under the same quality and latency constraints. Source review: 2026-10-18. No L40S hardware or LLM serving run was executed for this update. Untraceable throughput/perplexity tables have been removed; they are not measurements.

## First reject impossible configurations

Each L40S has 48 GB GDDR6, 864 GB/s advertised local memory bandwidth, PCIe Gen4 x16, and **no NVLink**. Twelve cards provide 576 GB aggregate capacity, distributed across devices. Raw BF16 weights for a 70B model require about 140 GB; TP=2 gives about 70 GB per rank before overhead, so **70B BF16 does not fit on two 48 GB L40S cards** without a separate offloading or other memory strategy. Quantized weights also need scales, metadata, unquantized tensors, KV cache and workspace. [L40S specifications](https://www.nvidia.com/en-us/data-center/l40s/).

The official current table lists FP8 at 733 dense / 1,466 structured-sparse TFLOP/s, TF32 at 183 / 366, and BF16 roughly 362–366 dense / 733 sparse. Do not compare a dense BF16 product with the old incorrect 183 BF16 number. A peak ceiling is not an expected application score.

## Reuse a correct small benchmark

Run the bounded correctness/GEMM/copy script in [H200 validation](../8x-H200-Training-Inference/06-Benchmarks-and-Validation.md#small-correctness-test-then-a-bounded-gpu-microbenchmark) on the L40S and record the actual device. The script does not assume HBM; label local memory as **GDDR6** for L40S. Do not allocate two 40 GiB buffers on a 48 GB card. The script caps each buffer at 256 MiB and uses a fraction of free memory.

For a GPU0→GPU1 copy, synchronize both source and destination before/after the timed interval, verify the destination, and calculate `payload_bytes × repeats / seconds`. Do not multiply payload by two unless deliberately reporting a separately defined bidirectional aggregate. Decimal GB and GiB must not be mixed. Check peer-access support and topology; a successful copy may use host staging. Compare same-switch, host-bridge and cross-NUMA paths on the actual server.

## Compare a feasible deployment matrix

| Candidate | What must be established before benchmarking |
|---|---|
| One GPU per small-model replica | Full per-rank working set fits; model quality satisfies task |
| Two-GPU quantized replica | Format/kernels supported, actual weights+KV fit, quality retained, topology acceptable |
| Four-GPU larger-model replica | Most-loaded rank fits, divisibility constraints hold, TP/PP communication worthwhile |
| Twelve independent replicas | Shared CPU, NIC, tokenizer and storage capacity can feed them; load balancing meets tail latency |

The rows are experiments, not rankings. Twelve replicas need not deliver exactly twelve times throughput when shared resources saturate. A supposed “Llama-3 13B” checkpoint is not a valid baseline model identity; use a real official model ID and immutable revision.

## Run the same workload on each candidate

Follow the [current vLLM benchmark workflow](../8x-H200-Training-Inference/06-Benchmarks-and-Validation.md#inference-throughput-and-latency-need-a-workload). Save tokenizer/model/runtime revisions, prompt/output distributions, concurrency, precision and parallel topology. Report successes, errors, TTFT/TPOT/completion tails, output tokens/s and goodput. Include the entire measurement interval, including request draining; counting completed requests while dividing only by submission time can overstate throughput.

Do not use a custom streaming client that equates one JSON event with one token. Test chunk parsing and completion/usage handling or use the runtime's supported benchmark client. For a robotics fleet, include the task's deadline and stale-response handling; peak offline tokens/s does not establish timely action guidance.

## Quantization is an accuracy experiment

Compare the same source checkpoint and evaluation corpus with a baseline and a supported quantized artifact. Record format, group size/scales, calibration provenance, runtime and kernels. Evaluate held-out task quality as well as perplexity. Perplexity is `exp(total negative log likelihood / predicted token count)` for token normalization; word-normalized perplexity and different tokenizers are not directly comparable.

LM Evaluation Harness returns metric keys with aggregation suffixes that depend on task/version. Inspect the actual result rather than assuming a `word_perplexity` key or interpreting it as token perplexity. Omit a quantization argument for the baseline instead of passing an unsupported string `none`. A 4× raw-weight compression does not guarantee 4× total-memory reduction, speedup or less than 2% quality loss. [Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness), [AWQ paper](https://arxiv.org/abs/2306.00978), [GPTQ paper](https://arxiv.org/abs/2210.17323).

**Worked memory check:** 35 GB packed weights + 2 GB metadata/unquantized weights + 6 GB KV cache + 3 GB workspace/reserve = 46 GB decimal. That is only a planning estimate, close to a nominal 48 GB capacity; runtime-reported available memory and worst-rank peak decide feasibility. Doubling concurrency can invalidate it through the KV term.

**Submission:** one measured device baseline, one feasible deployment comparison and one quality slice where quantization fails or passes the stated tolerance. Cost/token needs dated actual prices, utilization and energy measurements; it cannot be inferred from a vendor-name comparison.
