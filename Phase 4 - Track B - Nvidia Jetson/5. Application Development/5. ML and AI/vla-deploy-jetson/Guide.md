# VLA Deployment on Edge GPUs — Stack Selection for Vision-Language-Action Models

**Parent:** [ML and AI](../Guide.md)

> **Pick the right inference stack for VLA workloads, then prove correctness with strict parity, then optimize what your profiler says is slow.** Vision-Language-Action models are not LLMs — applying LLM serving wisdom to a robot's control loop wastes both time and silicon.

This guide distills a real review of an open-source VLA deploy layer ([reflex-vla](https://github.com/rylinjames/reflex-vla)) into a reusable framework: how to pick between ORT + TensorRT EP, TensorRT-LLM, and raw CUDA for a VLA workload, what parity gate to insist on, and what to profile before claiming you are optimal.

---

## Why this exists

VLA models — SmolVLA, π0, π0.5, GR00T N1.6, OpenVLA — combine a vision encoder, a language-model backbone, and an action head into a single inference call that drives a robot. Most public inference-stack writing is about LLM serving. **Almost none of that advice applies cleanly to VLA inference**, and applying it anyway is the most common mistake in this space.

| Workload axis | LLM serving (chatbot) | VLA inference (robot) |
|---------------|----------------------|----------------------|
| Batch size | 1 to 256, often dynamic | 1 (one robot per process) |
| Decode length | 256 - 8192 tokens | 50-step action chunk, often unrolled into ~10 fused passes |
| Sequence pattern | Long autoregressive | Heterogeneous: encoder + prefill + diffusion / flow-matching |
| KV cache pressure | Dominant | Negligible |
| Optimization target | Throughput per GPU | p50 / p95 latency per request |
| Hardware | Datacenter (H100, A100, L40S) | Edge (Jetson Orin, Thor) + cloud |

If you read an inference-engineering blog and the words "paged attention," "in-flight batching," "speculative decoding," or "FP8 attention plugin" appear, that blog is about LLM serving, not VLA inference. None of those features pay for themselves on batch=1, 50-step action chunks.

---

## The deploy-stack tradeoff space

Three real options for the inference path. They are not interchangeable.

### Option A — ONNX Runtime + TensorRT Execution Provider + CUDA Graphs

```
Trained VLA (PyTorch)
   │
   ▼  torch.onnx.export
ONNX graph (vision encoder, LM, action head, sampler loop unrolled)
   │
   ▼  ORT session.create()
ORT session
   ├──▶ TensorRT EP   ── compiles eligible subgraph into a TRT engine
   ├──▶ CUDA EP       ── fallback for unsupported ops
   └──▶ CPU EP        ── last-resort fallback
   │
   ▼  cudaGraphCapture / cudaGraphLaunch
Replayable CUDA graph for batch=1 fixed-shape inference
```

**Strengths for VLA:**

- **Heterogeneity-friendly.** Whatever ONNX subgraph you export, TRT compiles. Vision encoder + LM + action head all coexist as subgraphs with no architectural assumption that the workload is "transformer decoder."
- **CUDA Graphs collapse launch overhead.** For batch=1 fixed-shape, replaying a captured graph eliminates per-kernel host costs that dominate small-batch inference.
- **Cross-architecture deploy.** ORT runs on Ampere, Ada, Hopper, Jetson Orin, Thor — same code path, different EP backends.
- **Parity is feasible.** ONNX is a deterministic IR; comparing TRT-EP outputs to PyTorch reference at machine epsilon is a real correctness gate.

**Weaknesses:**

- **No LLM-specialized kernels.** TRT EP gets you generic fused attention; it does not get you the hand-tuned attention/RMSNorm/RoPE plugins TRT-LLM carries.
- **Dynamic shapes are awkward.** CUDA Graphs require fixed shapes. Variable image resolution or variable action-chunk length forces a graph rebuild or an op-by-op fallback.
- **FP8 is immature** in this path versus TRT-LLM, which has FP8 attention plugins as a first-class feature.

### Option B — TensorRT-LLM

TRT-LLM is built around the LLM-as-decoder workload: paged KV cache, in-flight continuous batching, speculative decoding, FP8 attention. The model definition surface is "transformer decoder."

**When it pays for itself:** batch ≥ 8, decode length ≥ 512 tokens, datacenter serving.

**Why it is wrong as the primary path for VLA:**

- Its wins all scale with batch × decode length. VLA inference is batch=1, ~10 decode passes.
- Bending Eagle 2.5 + Qwen3 + DiT into TRT-LLM's transformer-decoder shape costs more in glue code than you recover in kernel speed.
- Two model-definition surfaces (ONNX for non-LLM components, TRT-LLM for LLM components) is a permanent maintenance tax.

**When TRT-LLM is correctly used in VLA deployment:** narrow architecture-specific fallback when the primary ORT/TRT-EP path cannot reach a target (e.g., a Blackwell sm_100 path during a transitional ORT packaging gap). It should not be the primary runtime.

### Option C — Raw CUDA / hand-written kernels

Maximum control, maximum maintenance burden.

**The deploy-layer rule:** do not write hand kernels for the velocity field, attention, RMSNorm, or RoPE. TRT does these well, the kernels evolve faster than you can keep up, and every touch breaks your parity gate.

**The two places raw CUDA legitimately wins:**

1. A specific op that ONNX cannot represent and TRT cannot fuse efficiently. Wrap it as a TRT plugin, not a separate runtime.
2. Stream / event orchestration — overlapping vision-encoder forward, LM prefill, and action-head decode across multiple CUDA streams. This is "raw CUDA thinking" without writing kernels.

---

## The recommendation, summarized

| Workload | Right primary stack | When to escalate |
|----------|---------------------|------------------|
| VLA inference, batch=1, edge or cloud GPU | **ORT + TRT-EP + CUDA Graphs** | Add TRT plugin if a specific op profiles hot |
| LLM serving, batch ≥ 8, datacenter | TRT-LLM or vLLM | n/a |
| Mixed VLA + LLM serving on shared infra | ORT/TRT-EP for VLA, TRT-LLM for LLM | Keep them as separate processes |

If you take one thing from this guide: **TRT-LLM is the wrong primary architecture for VLA inference, and raw CUDA wholesale is a maintenance trap. The boring answer (ORT + TRT-EP + CUDA Graphs) is the correct one.**

---

## Case study: reflex-vla

[reflex-vla](https://github.com/rylinjames/reflex-vla) is an open-source deploy layer for VLAs (SmolVLA, π0, π0.5, GR00T N1.6, OpenVLA) targeting x86 NVIDIA + Jetson Orin / Thor. Reading the project as a reference implementation:

### What the stack does well

| Decision | Why it matters |
|----------|---------------|
| **TRT EP as default**, CUDA EP as fallback | Correct primary/fallback ordering for batch=1 latency. Measured 5.55× on A10G/SmolVLA (19.49 ms vs 108.11 ms p50). |
| **`reflex validate` as a first-class CLI** | Parity is treated as a regression gate, not an afterthought. CI scaffolder lands a GitHub Actions workflow. |
| **Strict-equality parity** at FP32 reference | Per-model first-action `max_abs` ≈ 1e-7 to 1e-5, all at machine epsilon for FP32. A "close enough" gate would forfeit safe refactoring later. |
| **Abandoning the decomposed-ONNX path** | KV-state plumbing across ORT session boundaries is painful and the throughput claim was not worth the complexity. |
| **`reflex doctor` operational health check** | Verifies `libnvinfer.so.10` loadable, cuBLAS / cuDNN present, TRT EP active. Real ops primitive. |

### Reproducing the parity gate pattern

Independent of any specific tool, a serious VLA deploy layer should:

```python
# 1. Run the reference forward pass in PyTorch FP32 with seeded inputs.
ref = run_pytorch_fp32(model, fixture, seed=0)

# 2. Run the deployed engine on the same seeded inputs.
got = run_deployed(engine, fixture)

# 3. Strict comparison against machine epsilon.
cos = cosine_similarity(ref.flatten(), got.flatten())
max_abs = (ref - got).abs().max()

assert cos >= 1.0 - 1e-7,  f"cos parity failed: {cos}"
assert max_abs < 1e-4,      f"max_abs parity failed: {max_abs}"
```

The point of this pattern is not the exact tolerances. The point is that you can **refactor the engine, change EPs, change export pipeline, and the gate either holds or breaks immediately**. The moment you accept "1e-2 close enough," you have lost the ability to change anything safely.

### What can still leave perf on the table

Even on a well-architected stack, four common gaps. These are derived from reading reflex-vla's public design; the same gaps appear in nearly every VLA deploy layer.

**1. External diffusion / sampler loops.** If the velocity-field unroll lives inside the ONNX (as in SmolVLA / π0 / π0.5: 10-step Euler baked in), TRT can fuse across the loop. If the sampler lives outside the ONNX (as in GR00T N1.6: 4-step DDIM in the serve loop), each step is its own CUDA Graph replay with a Python boundary in between. **Bake the loop into the ONNX wherever possible** — you trade flexibility (changing step count requires re-export) for one fat captured graph instead of N small ones.

**2. Multi-engine handoffs without IO binding.** When the inference pipeline is two engines chained (e.g., Eagle 2.5 vision-language → DiT action head), the boundary between them is a memcpy candidate unless you are explicitly chaining device pointers. ORT's IO binding lets you pass `OrtValue` device tensors directly between sessions on the same stream.

**3. Precision conservatism.** Default FP16 leaves BF16 unexplored. On Ada / Hopper, BF16 is the same throughput as FP16 with much better numerical headroom — flow-matching velocity fields are FP16-sensitive at trajectory start where dx/dt is largest. Add BF16 as a config knob and validate against the existing parity harness.

**4. CPU-side preprocessing.** If image resize / normalize / CHW-swap happens in NumPy on CPU before the inference call, the deploy layer pays PCIe twice — once for the raw image, once for the preprocessed tensor. On Orin Nano this regularly dominates latency. Move to NPP, kornia-on-GPU, or a graph-able preprocess subgraph.

---

## Profile-driven optimization

Before claiming any deploy layer is optimal, the answer to "what should I optimize next?" comes from a profiler, not from intuition.

```bash
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=vla-trace.qdrep \
    python -m your_serve_module --model your-model
# fire one inference call, then SIGINT
```

Open in Nsight Systems and check, in order:

| Question | What it tells you |
|----------|-------------------|
| Is each `/act` one CUDA Graph launch or many? | Many → external sampler loop or graph capture is broken |
| Gap between `Run()` returning and next CUDA op? | Non-trivial → ORT session overhead or buffer allocation; fix with IO binding |
| Any host-device transfers during inference? | Yes → fallback op outside the TRT subgraph, or CPU-side preprocessing |
| Stream concurrency on multi-engine pipelines | Serial → switch encoder and decoder to different streams with event sync |
| Memory-bandwidth utilization (Orin specifically) | High → you are memory-bound, not compute-bound; quantization beats kernel fusion |

The Orin-specific point matters. **Edge VLA inference is much more often memory-bound than compute-bound.** The optimization order on a Jetson is approximately:

1. Reduce weight bandwidth: FP16 / BF16 / INT8 / INT4 quantization
2. Eliminate host-device transfers: GPU-resident preprocessing, IO binding
3. Capture larger CUDA Graphs: bake sampler loops into the ONNX
4. Improve kernel fusion: TRT plugins for hot ops the compiler cannot fuse
5. Multi-stream concurrency: overlap encode / decode

Doing step 4 before step 1 on a Jetson is wasted work.

---

## Hardware-aware decisions

| Target | Compute | Memory | What changes |
|--------|---------|--------|--------------|
| **Orin Nano 8 GB** (sm_8.7) | ~40 TOPS INT8 | 8 GB unified, ~50 GB/s | SmolVLA-class only; aggressive quantization mandatory; preprocessing on GPU non-negotiable |
| **Orin AGX 64 GB** (sm_8.7) | ~275 TOPS INT8 | 64 GB unified, ~204 GB/s | π0 / π0.5 / GR00T fit; FP16 is the comfortable default; BF16 if you need numerical headroom |
| **Jetson Thor 128 GB** (sm_10) | ~2 PFLOPS FP4 | 128 GB unified | FP8 first-class; can run multiple VLAs concurrently |
| **A10G** (sm_8.6) | ~31 TFLOPS FP32 | 24 GB GDDR6, ~600 GB/s | Cloud reference target; not memory-bound; CUDA Graph + TRT fusion dominates |
| **H100** (sm_9.0) | ~989 TFLOPS BF16 | 80 GB HBM3, ~3 TB/s | FP8 attention shines; overkill for batch=1 VLA but useful for fleet emulation |
| **Blackwell** (sm_10.0) | ~20 PFLOPS FP4 | 192 GB HBM3e | Currently blocked by transitional ORT packaging gap; pin newer TRT + rebundle cuBLAS / cuDNN before reaching for TRT-LLM |

The Orin Nano column is the design constraint that drives most of the architectural decisions. If your VLA deploy layer cannot fit SmolVLA-class on 8 GB unified memory, it will not deploy onto the most numerous robot platform in the world.

---

## Anti-patterns to avoid

**1. Quantizing before having a parity gate.** FP16 / BF16 / INT8 / FP8 each break strict-equality parity by construction. Without an FP32-reference parity gate first, you cannot tell whether a quantized regression is an acceptable precision loss or a bug.

**2. Running TRT-LLM as the primary VLA runtime.** Wrong workload axis. Use it as a narrow fallback for a specific architecture only.

**3. Hand-rolled kernels in the deploy layer.** TRT evolves faster than you can keep up. Wrap one specific bottleneck as a TRT plugin if profiling demands it; do not maintain a parallel kernel library.

**4. CPU-side preprocessing on Jetson.** PCIe-equivalent traffic on a unified-memory device is still wasted bandwidth, and bandwidth is the constraint.

**5. Optimistic dynamic shapes.** CUDA Graphs require fixed shapes. Either commit to a fixed image resolution and action-chunk length per deployment, or accept the cost of graph rebuilds on shape change. Pretending you can have both is how you ship a deploy layer that benchmarks well in isolation but jitters in production.

**6. "Close enough" parity tolerances.** Once `1e-2` is acceptable, every refactor leaks numerical drift. Pin to FP32 machine epsilon and let lower precision break the gate explicitly.

---

## Build it

If you are working on a VLA deploy layer of your own, the artifacts that prove you have done it correctly:

- **`bench A10G`:** p50 / p95 latency for one model, batch=1, with and without TRT EP. The ratio should be in the 3-6× range for a flow-matching VLA. If it is below 2× something is wrong with the EP wiring.
- **`validate <model>`:** strict-equality parity report. Per-fixture `cos`, `max_abs`, `mean_abs`. Wired to CI on every push.
- **`doctor`:** operational health check. Verifies the EP loaded, libraries are reachable, and the test fixture passes parity within tolerance.
- **`nsys` trace of one inference call:** annotated to show one CUDA Graph launch per `/act` (or, if there are many, an open issue tracking why).
- **Memory budget report:** for the smallest target (typically Orin Nano 8 GB), a token-by-token memory accounting that demonstrates the model fits with headroom for the action history buffer.

---

## What good outcomes look like

| Area | Weak outcome | Strong outcome |
|------|--------------|----------------|
| Stack selection | "Let's use vLLM for the LM and figure out the rest later" | ORT + TRT-EP + CUDA Graphs as primary; TRT plugins escalation path documented |
| Parity | "Outputs look right" | `cos = 1 - 1e-7`, `max_abs < 1e-4` against FP32 reference, in CI |
| Hardware support | "Works on my A100" | A10G + Orin AGX + Orin Nano benchmarks, with target-specific optimization order |
| Optimization order | "Fuse more kernels" | "Quantize first; we are memory-bound on Orin" |
| Maintenance | "I'll update kernels when needed" | "We do not maintain a kernel library; everything is upstream TRT or one documented plugin" |

---

## References

- reflex-vla — open-source VLA deploy layer (the case study used in this guide): [https://github.com/rylinjames/reflex-vla](https://github.com/rylinjames/reflex-vla)
- LeRobot — SmolVLA, π0, π0.5: [https://github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)
- NVIDIA Isaac GR00T: [https://developer.nvidia.com/isaac/gr00t](https://developer.nvidia.com/isaac/gr00t)
- OpenVLA: [https://openvla.github.io/](https://openvla.github.io/)
- ONNX Runtime — TensorRT Execution Provider: [https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)
- TensorRT-LLM (for context, not as VLA primary): [https://github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- CUDA Graphs API: [https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
- Nsight Systems: [https://developer.nvidia.com/nsight-systems](https://developer.nvidia.com/nsight-systems)
- Sibling module: [Jetson LLM Runtime](../jetson-llm-runtime/Guide.md) — same memory-first discipline applied to LLM-only inference

---

## Related Pages

- [ML and AI on Jetson — overview](../Guide.md)
- [Jetson LLM Runtime](../jetson-llm-runtime/Guide.md)
- [LLM Optimization on Jetson](../llm-optimization-jetson/Guide.md)
- [Robotics — Advanced Perception and AI](../../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/4.%20Robotics/Advanced%20Perception%20and%20AI%20for%20Robotics/Lecture-01.md)
