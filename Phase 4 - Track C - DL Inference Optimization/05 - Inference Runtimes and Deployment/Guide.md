# 05 — Inference Runtimes & Deployment Targets

<div class="course-identity auto-course" style="--course-accent: #7c3aed; --course-accent-rgb: 124, 58, 237;" markdown="1">
<div class="course-identity__icon">IRDT</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Compiler Track</p>
<p class="course-identity__title">Specialized course identity for 05 — Inference Runtimes & Deployment Targets.</p>
<p class="course-identity__meta">Artifact: compiler/inference optimization · Measure: op count, memory, latency</p>
</div>
</div>


**Order:** Fifth. After graph, kernels, compiler, and quantization (01–04), you deploy and measure in production-like settings.

**Role target:** DL Inference Optimization Engineer · **MTS Kernels** (deployment, production reliability, measurable outcomes).

---

## Why this comes fifth

Your kernels and optimizations only matter if they run correctly in a runtime and meet latency/throughput goals. This unit covers the main inference runtimes and how to measure and compare them.

---

## 1. Runtimes

* **TensorRT** — Engine build, plugins, and dynamic shapes. Hardware features such as DLA are release/platform specific; do not infer support from a generic TensorRT label.
* **ONNX Runtime** — Execution providers (CUDA, TensorRT, OpenVINO). Graph optimizations and provider selection.
* **Triton Inference Server** — Batching, model concurrency, metrics. Serving multiple models and dynamic batching.

---

## 2. LLM runtimes: choose a release and a workload

[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) and
[vLLM](https://github.com/vllm-project/vllm) provide scheduling, cache management,
and model execution for autoregressive workloads. Neither has a universal
performance advantage. Compare the same checkpoint revision, tokenizer,
precision/quality gate, input/output length distribution, arrival trace,
hardware, and latency objective.

### TensorRT product and platform versions

As checked 2026-10-18, the TensorRT `latest` guide is **11.4**, with strongly typed networks by default. It does **not** support JetPack; Jetson requires a supported 10.x stack and matching archived documentation. DLA availability is a separate release/platform question. Do not copy 10.x precision flags/build APIs into 11.x examples or use 11.x support tables to certify a JetPack deployment. See [official Jetson migration guidance](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-jetson.html) and [10.x Python documentation](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/python-api-docs.html).

### Current and historical TensorRT-LLM paths

**Currency review: 2026-10-18.** NVIDIA's current `latest` documentation,
[updated 2026-10-09 at source commit `cc63e66`](https://nvidia.github.io/TensorRT-LLM/latest/legacy/tensorrt-backend-removal.html),
documents removal of its legacy TensorRT engine backend and directs users to the
PyTorch-backed LLM API. It also removes the old per-model conversion and
`trtllm-build` workflow. This is a claim about that documentation snapshot;
check the release/container you actually install. An older Jetson or archived
server recipe may deliberately target an earlier supported stack.

TensorRT itself remains a separate inference product. Do not infer the
TensorRT-LLM execution backend from the product name, or mix old engine-build
instructions with a newer checkpoint-loading runtime.

**Upstream-documented shape, not locally executed GPU code:**

```python
from tensorrt_llm import LLM, SamplingParams

# Replace with a locally downloaded, revision-pinned supported checkpoint.
llm = LLM(model="/absolute/path/to/pinned-checkpoint")
outputs = llm.generate(["Explain a matrix multiply."], SamplingParams(max_tokens=32))
for result in outputs:
    print(result.outputs[0].text)
```

Use the release's [quick start](https://nvidia.github.io/TensorRT-LLM/latest/quick-start-guide.html)
and hardware/model support documentation to prepare the environment. Record
container digest, checkpoint hash/revision, runtime version, and command. The
[CPU compiler lab](../00%20-%20CPU%20Compiler%20Lab/Guide.md) is the executed local
route; it does not validate either GPU LLM runtime.

### What the runtime must manage

| Mechanism | What it changes | What to measure |
|---|---|---|
| Continuous batching | Adds/removes sequences while other requests run | Useful throughput, queue time, TTFT, inter-token latency |
| Paged KV storage | Allocates state in blocks and can reduce wasted capacity | Allocated/used bytes, fragmentation, cache reuse rules |
| Quantized execution | Changes representations and kernel choices | Quality, actual arithmetic path, scale/metadata overhead |
| Tensor/pipeline/expert parallelism | Distributes tensors, layers, or experts | Communication critical path, memory, per-device useful throughput |
| Speculative decoding | Drafts candidates and verifies them with the target | Acceptance rate, decode latency, draft cost, sampling correctness |
| CUDA Graph replay | Reuses captured launch structures on compatible paths | Host overhead, capture limits, static-shape/memory constraints |

Paging does not eliminate all fragmentation. Speculation does not promise
better time-to-first-token, and capture does not eliminate all execution or
host overhead. Measure the mechanism you enabled. vLLM also supports multiple
parallelism strategies; consult its [parallelism documentation](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/)
rather than treating it as tensor-parallel-only.

### Worked KV-cache estimate

For a conventional full-attention decoder with cached keys and values:

```text
KV payload bytes = 2 × layers × cached_tokens × KV_heads × head_dim × bytes_per_element
```

With 32 layers, 4,096 cached tokens, 8 KV heads, head dimension 128, and FP16
(2 bytes), this is `536,870,912 bytes = 512 MiB` per sequence. Changing only the
KV-head count to 32 produces `2 GiB`. Parameter count alone cannot distinguish
these cases. Multi-query/grouped-query attention, sliding windows, latent
attention, cache quantization, and runtime sharding require their actual state
layout rather than blindly applying this formula.

This is payload arithmetic, not peak device allocation. Subtract model weights,
activations/workspaces, runtime buffers, graph reservations, and fragmentation
from device capacity before estimating concurrency, then measure actual use.
A nominal 80 GB device does not imply a fixed number of concurrent “7B” requests.

### Prefill and decode are hypotheses to profile

Long-prompt prefill often offers greater arithmetic intensity; small-batch
decode often exposes weight/cache bandwidth and launch overhead. Batch size,
attention structure, sequence lengths, precision, and device can change the
bottleneck. Record the measured roofline/trace evidence instead of labeling
all prefill compute-bound and all decode memory-bound.

---

## 3. Measurable outcomes

* **Latency** — p50, p99; what to measure (single request, batch). For LLMs: time-to-first-token (TTFT) and inter-token latency (ITL).
* **Throughput** — QPS, tokens/s; how batch size and concurrency affect it. For LLMs: output tokens/s across all concurrent requests.
* **Memory footprint** — Peak GPU/system memory; impact of batching and precision. For LLMs: model weights + KV-cache + activation memory.
* **Methodology** — Reproducible benchmarks; A/B comparison (e.g. before/after kernel change, or TensorRT-LLM vs vLLM).

---

## Resources

* [TensorRT Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html)
* [TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM) — Source, examples, model support matrix.
* [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/) — Build, deploy, and optimize guides.
* [vLLM](https://github.com/vllm-project/vllm) — Alternative LLM serving engine for comparison.
* [Triton Inference Server](https://github.com/triton-inference-server/server) — Production serving with TRT-LLM backend.
* [MLPerf Inference](https://mlcommons.org/benchmarks/inference/) — Reference benchmarks and methodology.

---

## Projects

1. **Runtime comparison** — Deploy the same model with ONNX Runtime and TensorRT (same hardware). Compare latency and throughput; document configuration and measurement method.
2. **Triton server** — Set up a minimal Triton server with dynamic batching. Measure QPS vs batch size and document how batching affects latency and throughput.
3. **Benchmark report** — For one model and one runtime, produce a one-page benchmark report: latency (p50/p99), throughput, memory, and exact environment (GPU, driver, runtime version).
4. **LLM runtime comparison** — Load one supported, revision-pinned checkpoint in the selected TensorRT-LLM release and vLLM. Match precision and quality first; measure useful tokens/s, TTFT, inter-token latency, memory, and errors under the same arrival trace. Use engine builds only for an explicitly archived release that supports that workflow.
5. **Multi-GPU LLM serving** — Measure a model that fits in both compared configurations, keeping model and workload fixed while changing GPU count. A comparison against a smaller model is a different workload, not a scaling-efficiency measurement.
6. **Serving integration** — Use a server integration supported by the exact selected runtime release. Load test concurrent clients, cancellation, and overload; do not assume an archived Triton engine backend accepts a newer runtime artifact.

---

## Next

→ **[06 — tinygrad Deep Dive](../06%20-%20tinygrad%20Deep%20Dive/Guide.md)** (optional) — Hands-on compiler/kernel interface: IR, scheduler, backends, and adding a simple optimization.


## Integration lab: At what load does a fast engine become a slow service?

Replay a fixed arrival trace with bounded admission and compare concurrency levels. Measure queue time, processing time, errors, and per-request latency while keeping quality and token lengths fixed.

**Acceptance evidence:** Report request throughput separately from per-request latency and reject overload explicitly. Include cold start, cancellation, and deployment compatibility in the runtime manifest.

**Continue:** take this artifact to [06 - tinygrad Deep Dive](../06%20-%20tinygrad%20Deep%20Dive/Guide.md). Explain which assumption from this course the next engineer must preserve.
