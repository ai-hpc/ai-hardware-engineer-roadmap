# Edge LLM inference: projections, quantized weights, and a memory budget

**Reviewed 18 October 2026.** This lesson uses Qwen3-4B-Instruct-2507 as a fixed architecture case study. It teaches estimates you can check on any CPU, then shows how to test those estimates on a supported inference runtime. Allow 60–90 minutes. Prerequisites: matrix multiplication, bytes versus bits, and the transformer forward pass.

By the end, produce a projection-shape table, a weight/KV memory budget, and a clear statement of what a timing result actually measures. Batch-one dense-model decoding often has low arithmetic intensity, but memory bandwidth is a hypothesis to test alongside launch overhead, dequantization, attention, CPU work, and synchronization.

## 1. One linear layer, two workloads

Let stored weights have shape `W[M,K]`, with `M` outputs and `K` inputs. For one token, `y=W x` is a matrix-vector operation. For `B` independent token vectors, `Y=X Wᵀ` is a matrix-matrix operation. Prefill supplies many prompt tokens; decode supplies one new token **per active sequence**. Several simultaneous sequences can therefore make decoding a matrix-matrix workload too.

Count one multiply plus one addition as two FLOPs. Assuming one read of each weight and input and one write of each output:

\[
F=2BMK,\quad D=MKs_w+B(K+M)s_a,\quad I=F/D.
\]

`s_w` and `s_a` are bytes per weight and activation. If weights dominate, `I≈2B/s_w`: approximately **1 FLOP/byte for FP16 batch one**, 4 for ideal INT4, and 32 for FP16 batch 32. Quantization metadata, kernel rereads, cache reuse, and dequantization instructions modify this model.

A roofline estimate is `achievable FLOP/s ≤ min(compute ceiling, useful bandwidth × I)`. The compute ceiling must match the numerical path; sparse INT8 TOPS and dense FP16 FLOPS cannot be substituted. A large prefill does not automatically make every layer compute-bound.

## 2. Derive dimensions from the checkpoint

The supplied [config snapshot](labs/qwen3-config.json) comes from [Qwen3-4B-Instruct-2507 revision cdbee75](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f958dc650907174af0554/config.json). It has hidden size 2,560, 32 query heads, 8 KV heads, head dimension 128, 36 layers, intermediate size 9,728, and vocabulary size 151,936.

Notice that `32 × 128 = 4096`, **not 2560**. Query width need not equal hidden width. Assuming equality silently gives the wrong parameter count.

| Projection | Stored shape `[output,input]` | Purpose |
|---|---:|---|
| Q | `[4096,2560]` | 32 query heads |
| K and V, each | `[1024,2560]` | 8 shared KV heads |
| O | `[2560,4096]` | return attention output to hidden width |
| Gate and up, each | `[9728,2560]` | two SwiGLU inputs |
| Down | `[2560,9728]` | return FFN output to hidden width |
| LM head | `[151936,2560]` | vocabulary logits |

Grouped-query attention assigns four query heads to each KV head here. It saves KV storage and projection work relative to 32 KV heads; it also changes model architecture and quality. It is more than a runtime bandwidth setting. Qwen3 additionally normalizes Q and K within each head before RoPE; a correct implementation must preserve that operation.

Packing Q, K, and V along the **output** dimension gives `[6144,2560]`. Splitting its result into 4,096, 1,024, and 1,024 values recovers the three projections. It can reduce launches and input rereads. It does not remove weight reads, and fusion support depends on the runtime, format, and kernel. The CPU test verifies a tiny numerical example of this equivalence.

A dimension-only trace is ambiguous: two projections can share a shape. Record tensor name, layer, operation, datatype, strides, and runtime revision. The synthetic string `type=12 M=2560 K=2560` is not a universal log format or proof of a particular model.

## 3. Tied embeddings save storage, not the output projection

For input token `i`, embedding lookup reads row `E[i]`. At the output, tied embeddings compute **all vocabulary logits** as `E h`. The same table is reused, but the LM-head matrix-vector multiplication still happens.

In this checkpoint, `151936 × 2560 = 388,956,160` parameters are shared. Exact counting of projection matrices, this tied table, and Qwen3's normalization weights gives **4,022,468,096 unique parameters**. Untying the head would add another 388,956,160 parameters. It would not add a previously absent output computation.

`7 × layers + 1` describes the number of conceptual linear projections in this particular decoder. It is not a guarantee of that many GPU kernels: fusion, tiling, quantization, and attention implementations change the kernel count.

## 4. Quantization formats contain more than packed integers

The following tensor-format IDs and block layouts were checked against the [ggml enum](https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml.h) and [block definitions](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h). The [source record](labs/source-review.json) stores fetched-file hashes because these branch links can change.

| Tensor format | Enum ID | Weights/block | Bytes/block | Effective bits/weight |
|---|---:|---:|---:|---:|
| Q4_0 | 2 | 32 | 18 | 4.5 |
| Q8_0 | 8 | 32 | 34 | 8.5 |
| Q4_K | 12 | 256 | 144 | 4.5 |
| Q5_K | 13 | 256 | 176 | 5.5 |
| Q6_K | 14 | 256 | 210 | 6.5625 |

For example, Q4_K's 256 weights occupy 128 payload bytes plus 16 bytes of scales/minimum metadata. Q6_K uses 210 bytes, not 208. A matrix's rows must satisfy the format's block layout; the lab rejects unaligned dimensions instead of silently padding them.

**Q4_K_M is a model quantization recipe, not one tensor datatype.** Its artifact can contain several tensor formats. Measure the actual file and tensor inventory; do not estimate it as exactly `parameter_count/2`. File headers, alignment, unquantized norms, workspace, and KV allocation also affect the process footprint.

Many kernels consume packed weights directly and dequantize tiles on chip. Others have different unpacking or repacking behavior. Neither the format name nor its byte ratio predicts an exact speedup. Lower precision also needs task-quality evaluation, such as the [tool-dispatch lesson](../Agent%20Tool-Dispatch%20Evaluation%20with%20BFCL/Lecture-01.md).

## 5. Budget the KV cache separately

For dense full-context attention, an unsharded, uncompressed KV cache has payload

\[
D_{KV}=2L H_{KV}d_h T B s_{KV}.
\]

For this model at FP16, batch one and 4,096 tokens:

```text
2 × 36 × 8 × 128 × 4096 × 1 × 2 = 603,979,776 bytes = 576 MiB
```

At the configured maximum context of 262,144 tokens it is **36 GiB**, before weights and workspace. A model's advertised context limit is not a statement that it fits on an 8 GB device at that length.

Paged allocation reduces fragmentation and enables flexible sharing/allocation; it does **not** by itself reduce bytes per stored key/value or the mathematical number of keys attended. Quantized KV changes element storage plus metadata; sliding-window attention changes the attention contract. Use only supported combinations and measure quality after changing precision or context behavior.

## 6. Run the budget and falsify an assumption

From the repository root:

```bash
cd 'Phase 5 - Advanced Topics and Specialization/Track C - Edge AI/Edge LLM Inference Internals'
python3 labs/budget.py
python3 -m unittest discover -s labs -p 'test_*.py' -v
```

Read [budget.py](labs/budget.py), [eight tests](labs/test_budget.py), and the [recorded output](labs/budget-reference.json). No model weights or accelerator are needed.

The script defines an **illustrative mixture**: all seven layer projection matrices use Q4_K, the tied embedding/head table uses Q6_K, and norms use FP32. This is an explicit storage example, not a promise about a Q4_K_M exporter. Its weight payload is 2,363,590,144 bytes.

At an assumed useful bandwidth of **50 GB/s**, reading that payload once per generated token gives 21.15 token/s. Adding one complete 4,096-token KV read per step gives 16.85 token/s. These are optimistic traffic estimates under the stated assumptions, **not Jetson measurements or universal hard ceilings**. They omit cache writes, other activations, intermediate traffic, launches, computation, and scheduler work. Small models cached on chip, batching, sparse expert activation, and speculative decoding have different traffic/reuse patterns.

**Exercise:** change context to 8,192 and batch to 2. Cache payload becomes 2.25 GiB: four times the original. Weight storage does not become four times larger, because both requests can share the same weights. Explain why batch throughput and latency per request are different metrics.

**Exercise:** compare FP16 and ideal INT4 arithmetic intensity at batch one. Then include Q4_K's 4.5 effective bits per weight. Explain why “decode has two FLOPs per byte” is incomplete without stating precision.

## 7. Move from arithmetic to measurements

For a machine with an existing compatible llama.cpp checkout and a local GGUF file, inspect its revision and `llama-bench --help`. The current upstream [build instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) use CMake; CUDA builds use `-DGGML_CUDA=ON`. Old `LLAMA_CUDA=1` recipes are not the current build interface.

The [benchmark documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) distinguishes prompt processing (`pp`), generation (`tg`), and combined processing (`pg`). An example inside that checkout is:

```bash
git rev-parse HEAD
./build/bin/llama-bench --help
# Replace the model path with a downloaded, hash-recorded compatible GGUF.
./build/bin/llama-bench -m /path/to/model.gguf -p 512 -n 128 -r 5 -o json
```

This is an optional measurement procedure, not an executed model benchmark in this lesson. Record model hash, runtime revision, backend/offload, cache precision, context depth, batch/concurrency, warmup, repetitions, memory use, and power/cooling configuration. In current `llama-bench`, `-d` can set context depth; check the chosen build's help. A synthetic `tg` result is not an application time-to-first-token measurement. Measure actual prompt/template/tokenization and request queueing for application latency.

For Jetson, use the maintained [platform compatibility guide](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/Compatibility.md), [memory guide](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Memory-Architecture/Guide.md), and [real-time inference course](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Real-Time-Inference/Guide.md). Power-mode IDs depend on module, BSP, and flash configuration; query the actual device before selecting a mode. Clock locking is an experimental condition, not a prerequisite for all valid inference.

A `tegrastats` sample with idle GPU does not prove that an entire inference ran on CPU: sampling can miss bursts, or the pipeline may be waiting elsewhere. Check backend logs and a CUDA trace. Shared physical DRAM does not make zero-copy access free: caching, coherence, access patterns, and synchronization still cost time. Host-to-device copy bandwidth is also not automatically the GPU's useful weight-read bandwidth.

## Check your understanding

1. Why can Q have 4,096 outputs when hidden size is 2,560? **Query width is query-head count times head dimension; this checkpoint sets those independently.**
2. What exactly does tying the embedding save? **A second stored parameter table; it preserves the full vocabulary output operation.**
3. Does a 4-bit artifact always use half a byte per parameter? **No; scales, mixed tensor types and other storage must be counted.**
4. Does paged attention halve KV traffic? **No; paging changes allocation/access organization, not precision by itself.**
5. Is a 21 token/s estimate a performance result? **Only a result of the stated arithmetic model. Measure the actual runtime separately.**

Continue with [Qwen architecture and optimization](../Qwen%20Inference%20Optimization/README.md) and [Gemma 4 deployment](../Gemma%204%20Edge%20Deployment/README.md) to compare architectures and deployment contracts.
