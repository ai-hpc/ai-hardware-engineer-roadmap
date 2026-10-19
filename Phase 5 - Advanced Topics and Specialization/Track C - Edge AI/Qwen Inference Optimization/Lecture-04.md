# Lecture 4: Qwen2.5-72B memory, tensor parallelism, and communication

**Reviewed 18 October 2026.** This is a fixed BF16 checkpoint case study. The portable lab demonstrates partitioning arithmetic; no 72B checkpoint, NCCL job, or multi-GPU inference was executed for this lesson. Use [Lab-Guide.md](Lab-Guide.md) for the runnable work.

## 1. Fit the entire service, not just the parameter count

The metadata-verified unique parameter count is 72,706,203,648. Two bytes per parameter gives **145,412,407,296 bytes**, about **145.41 GB / 135.42 GiB**. BF16 and FP16 both use two bytes but have different exponent/precision ranges. The checkpoint ships BF16; casting to FP16 changes the numerical experiment and is not a memory reduction.

A single accelerator can have enough physical memory for this payload; for example, [AMD specifies 192 GB HBM3 for MI300X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html). Whether the full service fits still depends on supported runtime, workspace, cache, reserve, and context/concurrency. Multi-GPU execution is a deployment choice driven by capacity and performance, not a mathematical requirement for every 145 GB model.

The unsharded FP16/BF16 KV payload per token is:

```text
2 × 80 layers × 8 KV heads × 128 dimensions × 2 bytes = 327,680 bytes
At 32,768 tokens and batch 1: 10 GiB
At 32,768 tokens and batch 8: 80 GiB
```

Paging can reduce wasted allocation but does not turn those 80 GiB into 10 GiB. TP may shard KV heads; replicated heads, static reservation, quantization, and allocator metadata alter actual per-rank allocation. Measure it.

## 2. Derive tensor parallelism with small matrices

For a linear map `y=W x`, split **output rows** into `W0` and `W1`. Each rank computes a disjoint piece of `y`; concatenation reconstructs the full output. This is the output-partitioned or column-parallel linear layer in common frameworks whose mathematical weight orientation is input-by-output.

For an input/reduction split, write `W=[W0 W1]` and `x=[x0;x1]`:

\[
y=W_0x_0+W_1x_1.
\]

Ranks compute partial outputs and **sum** them. Averaging gives the wrong result. The portable `row_parallel()` experiment returns both partial vectors and the combined result, then compares to the unsplit computation. Its tests deliberately distinguish summation from averaging.

In a conventional transformer TP layout, Q/K/V and gate/up split their output channels; O and down combine corresponding partial outputs. This often creates two reduction points per decoder layer. Actual runtimes may fuse, rearrange, or combine communication, so “two all-reduces” describes this algorithm rather than every possible implementation. See the [Megatron tensor-parallel design](https://arxiv.org/abs/1909.08053).

## 3. Map the 72B case to four ranks

With TP=4, 64 Q heads and 8 KV heads divide evenly:

| Tensor | Per-rank stored shape `[output,input]` |
|---|---:|
| Q | `[2048,8192]` |
| K and V, each | `[256,8192]` |
| Fused QKV | `[2560,8192]` |
| O partial | `[8192,2048]` |
| Gate/up, each | `[7392,8192]` |
| Down partial | `[8192,7392]` |

Bias terms must be handled at the correct stage; adding a full output bias separately on every rank and then summing multiplies that bias. Normalization and small tensors may be replicated. Vocabulary-parallel embedding/head handling is another implementation detail, so “total parameters divided by four” is a useful first estimate, not an exact allocator prediction.

At TP degrees beyond eight, KV-head partitioning may require replication or a different supported arrangement. Validate attention-head divisibility and runtime rules rather than declaring one degree universally optimal.

## 4. Count communication with units

For batch-one decode, a hidden activation of width 8,192 at two bytes/value contains **16,384 bytes**. A ring all-reduce transfers approximately `2*(P−1)/P * payload` sent bytes per rank over its two phases, excluding protocol overhead. For `P=4`, that is 24,576 bytes per reduction per rank; 160 reductions would send 3,932,160 bytes per token per rank.

This small payload does not imply negligible time. Startup latency across many sequential reductions can dominate. Prefill/batched workloads use larger activation matrices and change the latency/bandwidth balance. Network topology, NUMA placement, peer access, congestion, algorithm choice and overlap all matter. Summing GPU peak HBM bandwidth does not turn it into one fully usable pool with zero communication.

Use the [NCCL course](../../Track%20A%20-%20GPU%20Infrastructure/Nvidia%20GPU/HPC%20Setup/NCCL-Deep-Dive/README.md) for actual collective measurements and correctness checks. `NCCL_P2P_DISABLE=1` disables NCCL's P2P transport path; it does not physically break NVLink. Such an experiment must be isolated and interpreted through the selected transport logs.

## 5. TP, PP, and replicas solve different problems

**Tensor parallelism** splits work inside layers and can reduce each layer's local arithmetic/weight traffic at the cost of communication. **Pipeline parallelism** places different layers on stages; microbatches can overlap, while a single request still traverses the stages and pays boundaries. **Data-parallel replicas** each serve requests independently and multiply model residency.

Choose a layout using the actual memory budget, topology and workload. A single-stream latency target can prefer a different layout from aggregate throughput. More GPUs can make a small or communication-heavy workload slower. Consult the runtime's [parallelism documentation](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/) for supported combinations.

## 6. Source-checked optional serving procedure

For an already installed, compatible vLLM environment and sufficient GPUs, first record package version and `vllm serve --help`. A local, bounded experiment could use:

```bash
vllm --version
vllm serve Qwen/Qwen2.5-72B-Instruct \
  --revision 495f39366efef23836d0cfae4fbe635880d2be31 \
  --dtype bfloat16 --tensor-parallel-size 4 \
  --max-model-len 4096 --max-num-seqs 4 \
  --host 127.0.0.1 --port 8000
```

This is an optional command outline checked against the [server interface](https://docs.vllm.ai/en/latest/cli/serve/), not an executed or fully pinned container deployment. It downloads large weights if absent. Select an official runtime build that matches the actual accelerator/driver; record its exact version or image digest. Avoid `latest` in a reproduction manifest. A desktop CUDA wheel is not automatic evidence of Jetson support.

Start with a short known request, then vary concurrency and context while measuring completed requests, failures, memory, TTFT and inter-token latency. Query the served metrics endpoint and compare it to that version's documented names. Low KV occupancy can simply mean low demand; it is not by itself a misconfiguration.

**Exercise:** for batch 8 and 32k context, derive weight+KV capacity at TP=4 before workspace. Then state which terms remain replicated. Explain why average token throughput alone cannot demonstrate a p95 latency target. Next: [speculation, routing and capacity](Lecture-05.md).
