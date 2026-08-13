# VLA Deployment on Edge GPUs — Stack Selection, Compression, and Profile-Driven Optimization

<div class="course-identity auto-course" style="--course-accent: #be123c; --course-accent-rgb: 190, 18, 60;" markdown="1">
<div class="course-identity__icon">VDOE</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Jetson Track</p>
<p class="course-identity__title">Specialized course identity for VLA Deployment on Edge GPUs — Stack Selection, Compression, and Profile-Driven Optimization.</p>
<p class="course-identity__meta">Artifact: Jetson integration demo · Measure: latency, memory, power, logs</p>
</div>
</div>


**Parent:** [ML and AI](../Guide.md)

> **Vision-Language-Action models are not LLMs.** Treating them as if they were — paged KV, in-flight batching, FP8 attention plugins — wastes silicon and engineering time. This guide is a working survey of how the field is actually deploying VLAs to edge GPUs, NPUs, and 100-gram robots, with cited numbers and concrete tradeoffs.

Sources used in this guide:

- **EdgeVLA / EVLA** (Budzianowski et al., 2025) — [arXiv 2507.14049](https://arxiv.org/abs/2507.14049)
- **AsyncVLA** (Hirose et al., 2025) — [asyncvla.github.io](https://asyncvla.github.io/)
- **LiteVLA-Edge** (Williams et al., 2026)
- **NVlabs/vla-perf** — [github.com/NVlabs/vla-perf](https://github.com/NVlabs/vla-perf)
- **NXP i.MX 95 + SmolVLA** (HuggingFace + NXP, Mar 2026) — [HF blog post](https://huggingface.co/blog/nxp/bringing-robotics-ai-to-embedded-platforms)
- **reflex-vla** — [github.com/rylinjames/reflex-vla](https://github.com/rylinjames/reflex-vla)
- **RoboECC** (multi-factor edge-cloud collaborative deployment, 2026)
- **OmniVLA / OmniVLA-edge** (Hirose et al., 2025)

---

## 1. Why VLA edge inference is its own discipline

A VLA model takes pixels + state + a language instruction and emits an action — a continuous joint command, an end-effector pose delta, a chunk of future actions, or a gripper toggle. The architecture typically composes:

```
camera frames ─┐
                ├─▶ vision encoder ─┐
state vector  ─┘                    │
                                    ▼
                            language-model backbone ──▶ action head ──▶ action(s)
                                    ▲                     │
                language token  ────┘                     │
                                                          ▼
                                              robot at real-time control rate
```

Compared to a chatbot serving a 4 K-token reply at batch 64, a robot inference call has a completely different operating point:

| Workload axis | LLM serving (chatbot) | VLA inference (robot) |
|---|---|---|
| Batch size | 1–256, often dynamic | **1** (one robot per process) |
| Decode length | 256–8192 tokens | **50-step action chunk**, often unrolled into ~10 fused passes |
| Sequence pattern | Long autoregressive | **Heterogeneous:** encoder + prefill + diffusion / flow-matching |
| KV cache pressure | Dominant | **Negligible** |
| Optimization target | Throughput per GPU | **p50 / p95 latency per request** |
| Hardware | Datacenter (H100, L40S) | **Edge (Jetson Orin / Thor, NXP i.MX 95) + cloud** |
| Failure mode | Slow chat | **Robot crashes into a wall** |

That last row is not a joke. A 100 ms tail latency on a chatbot is a UX wart. A 100 ms tail latency on a 30 Hz visuomotor controller is a missed control deadline.

The deeper principle behind every other lesson in this guide:

> **Almost no LLM-serving optimization pays for itself on batch=1, fixed-decode VLA workloads.** The wins come from somewhere else.

---

## 2. The latency budget — how fast does inference actually have to be?

Two numbers set the whole design.

**Control-loop frequency.** Different robot tasks demand different rates:

| Task class | Typical control rate | Tolerable inference latency |
|---|---|---|
| Slow tabletop manipulation | 5–10 Hz | < 100 ms |
| Standard manipulation (LIBERO-class) | 15–30 Hz | < 33 ms |
| Reactive grasping, dual-arm | 30–50 Hz | < 20 ms |
| Bipedal / legged whole-body | 100–500 Hz | < 5 ms (often classical control under VLA target poses) |
| Drone / vehicle | 50–200 Hz | < 10 ms |

**Action-chunk length.** A VLA can amortize this. If the model emits *k* future actions per inference call, the effective inference rate becomes `control_rate / k`. SmolVLA, π0, π0.5 emit 50-step chunks; OpenVLA emits one action per call. This is the single biggest inference-rate lever the field has, and it costs nothing at run time — only training-time changes.

```
single-step VLA at 30 Hz:           inference must finish in 33 ms,    every step
50-action-chunk VLA at 30 Hz:       inference must finish in 1.66 s,   every 50 steps
                                    + adapter / smoother runs at 30 Hz onboard
```

A 1.66 s budget is enormous on a Jetson. **Action chunking is what makes edge VLA inference feasible at all** for anything more than a 100 M-parameter policy.

---

## 3. The seven techniques the field is actually using

Every published VLA edge-deployment system reaches for some subset of these. Order matters: the earlier ones in the list usually pay off first.

### 3.1 Quantization

The first lever, the cheapest, and the one with the biggest single multiplier on edge.

**Bit widths in practice:**

- **FP16 / BF16:** the comfortable default on TRT EP, no accuracy loss for most VLA backbones. BF16 is preferable on Ada / Hopper / Thor where it has the same throughput as FP16 with much better numerical headroom — flow-matching velocity fields are FP16-sensitive at trajectory start where dx/dt is largest.
- **INT8:** post-training quant on the vision encoder and LM backbone; usually leaves the action head at higher precision because diffusion / flow-matching denoising loops accumulate error per step.
- **INT4 / 4-bit (AWQ, GPTQ, NF4):** aggressive but standard now for the LM backbone. The HuggingFace + NXP team report taking SmolVLA's vision encoder and LLM prefill from "8-bit mixed precision to 4-bit" while keeping the action expert at FP32 to preserve the iterative denoising path.
- **FP8:** Hopper / Blackwell / Thor first-class. Less mature in the ORT path; first-class in TRT-LLM but rarely the right primary VLA stack.

**A real number from the field.** SmolVLA on the NXP i.MX 95 (6× Cortex-A55 + Mali GPU + eIQ Neutron NPU):

```
ONNX FP32 baseline             29.1 s  per inference
Selective quantization +        6.15 s per inference         (≈ 4.7× speedup)
decomposition (vision / LLM
to 4-bit, action expert FP32)
```

For comparison on the same blog, an ACT model (different architecture, simpler action head) went from 2.86 s FP32 to 0.32 s optimized — an 88.8 % latency reduction with a ~7 % drop in global accuracy. The ACT result shows the cost: aggressive whole-model quant on a non-VLA architecture lost real accuracy. The SmolVLA result shows the discipline: keep the iterative denoiser at high precision, quantize only the components that tolerate it.

**Rule of thumb derived from these results:** quantize the encoder and the LM prefill; do not quantize the iterative diffusion / flow-matching action head until you have measured the accuracy delta against an FP32 reference and accepted it on a real task suite.

### 3.2 Distillation and small-language-model backbones

The second lever, and the one that re-architects the model. Two flavors in the public literature:

**Architectural shrink — EdgeVLA (EVLA), Budzianowski et al. 2025.** Keeps OpenVLA's vision pipeline but makes two structural changes:

1. **Eliminates autoregressive end-effector position prediction** — instead of generating action tokens one at a time, predicts the 7-DoF pose jointly. This single change is reported to give a **7× inference speedup** by itself.
2. **Replaces the 7B-class LLM backbone with a Small Language Model** — substantially reduces both compute and memory.

The authors report comparable training characteristics to OpenVLA at significantly reduced inference cost. The lesson: the autoregressive-action assumption was inherited from "VLA = LLM with action tokens," and removing it costs nothing in capability while collapsing inference time.

**Lightweight specialized policies — OmniVLA-edge.** Hirose et al. 2025 ship a 108 M-parameter edge variant of their larger OmniVLA system specifically for navigation tasks where the full 7B-class backbone is overkill. The point: pick the model class for the task, not "the biggest available."

**Practical takeaway.** If you are deploying a VLA from a research paper without considering EdgeVLA-style architectural changes, you are probably leaving a 5–10× speedup on the table that quantization alone cannot give you.

### 3.3 Action chunking

Already introduced above; deserves its own section because it interacts with everything else.

**The trade.** Predicting `k` future actions per call divides the required inference rate by `k`, but the model now has to handle the longer-horizon prediction and the rollout becomes brittle to environmental change between chunks. In practice:

- `k = 1` (OpenVLA): inference must hit control rate. Hard on edge.
- `k = 8–16` (mid-range): standard for many manipulation tasks.
- `k = 50` (SmolVLA, π0, π0.5): a full second of actions at 30 Hz. Tractable on Jetson.
- `k = 100` (HF/NXP SmolVLA on i.MX 95 with chunk-size threshold 0.2 + weighted-average aggregation).

**The complication.** With a long action chunk, the world drifts during execution. Two correct responses, both used in published systems:

1. **Re-plan opportunistically** — start the next inference call before the chunk completes, so a fresh plan is ready when needed. This is the asynchronous-inference idea (next section).
2. **Aggregate / smooth** — the HF/NXP deployment mixes overlapping chunks with a `weighted_average` aggregator, weighted by how recent each chunk's context was. The robot never executes a stale chunk verbatim.

### 3.4 Asynchronous inference and the edge-adapter pattern

The third major lever, and the one that makes large remote VLAs usable on real robots.

**Problem.** A 7B-class VLA backbone may take 200 ms–6 s on a remote workstation depending on network and load. A robot at 30 Hz cannot wait.

**AsyncVLA (Hirose et al., 2025) solution.** Decouple semantic reasoning from reactive execution:

```
robot                                     remote workstation
─────                                     ──────────────────
camera frames ──┬────────────────────────▶ base VLA
                │                          (OmniVLA: SigLIP + DINOv2 + LLaMA2-7B)
                │                                  │
                │                                  ▼
                │                          coarse action tokens
                │                                  │
                │                                  ▼
                │                          token projector
                │                          (8×4×4096  →  1×1024)
                │                                  │
                │                                  ▼  ◄── high-latency network
                │                          ─────────────
                │                          edge adapter
                │                          (2 MLP-ResNet blocks, onboard)
                │                                  │
                ▼                                  ▼
            latest observation     ───▶    action refinement (fast)
                                                   │
                                                   ▼
                                               robot at real-time rate
```

The edge adapter is small enough to run on the robot's onboard controller, sees the latest observation, and refines whatever stale-but-rich action token came in from the remote VLA. AsyncVLA reports a **40 % higher success rate than state-of-the-art baselines** under artificial delays of 0.2 s, 2.0 s, and 5.0 s, and tolerates real-world delays up to 6 s.

**Edge-cloud collaborative inference at large scale: RoboECC** (2026) reports a **3.28× speedup** on VLA task execution by routing the right pieces of the pipeline to the right tier (edge for latency-sensitive, cloud for compute-heavy). Same general pattern, different splitting policy.

**The structural insight:** if a VLA must be large to be capable, and a robot must be reactive to be safe, you cannot have both on one device. The async / edge-adapter pattern is the way out.

### 3.5 Edge-cloud collaborative inference

A spectrum, not a binary. Five operating points, in increasing latency-criticality:

| Mode | Where the VLA runs | Where the robot's action-time loop runs | When to use |
|---|---|---|---|
| **Cloud-only** | Datacenter | Datacenter | Demos, simulation, no real robot in loop |
| **Cloud + edge adapter** | Datacenter | Onboard (AsyncVLA) | Large VLA, tolerable network, safe failures |
| **Edge GPU + onboard** | Edge GPU box (Jetson AGX nearby) | Onboard MCU | Mobile robots with workstation in radio range |
| **Onboard only** | Onboard Jetson / NPU | Onboard | Drones, tetherless robots, safety-critical |
| **Onboard tiny + cloud assist** | Tiny model onboard, cloud for hard cases | Onboard | Battery-constrained, intermittent connectivity |

The right mode is a function of three independent things: model size, network reliability, and failure consequence. There is no universally correct choice.

### 3.6 Runtime stack selection

The original framing from this guide's earlier draft, refined.

**Three real options for the inference path.** They are not interchangeable.

#### Option A — ONNX Runtime + TensorRT Execution Provider + CUDA Graphs

```
Trained VLA (PyTorch)
   │
   ▼  torch.onnx.export
ONNX graph (vision encoder, LM, action head, sampler loop unrolled)
   │
   ▼  ORT session
ORT session
   ├──▶ TensorRT EP   ── compiles eligible subgraph into a TRT engine
   ├──▶ CUDA EP       ── fallback for unsupported ops
   └──▶ CPU EP        ── last-resort fallback
   │
   ▼  cudaGraphCapture / cudaGraphLaunch
Replayable CUDA graph for batch=1 fixed-shape inference
```

**Why this is the right primary architecture for VLA:**

- Heterogeneity-friendly. Vision encoder + LM + action head all coexist as ONNX subgraphs.
- CUDA Graphs collapse launch overhead — critical for batch=1 fixed-shape.
- Cross-architecture: same code path on Ampere / Ada / Hopper / Jetson Orin / Thor.
- Strict parity is feasible. ONNX is deterministic; FP32 reference comparison at machine epsilon is a real correctness gate.

Empirical anchor: the [reflex-vla](https://github.com/rylinjames/reflex-vla) project reports 19.49 ms p50 on A10G/SmolVLA with TRT EP vs 108.11 ms p50 with CUDA EP — a measured **5.55× speedup** on this stack, batch=1, no algorithmic change. That ratio is consistent with what TRT-EP fusion should give on a flow-matching velocity-field unroll where CUDA EP dispatches kernel-by-kernel.

#### Option B — TensorRT-LLM

Built around the LLM-as-decoder workload: paged KV cache, in-flight continuous batching, speculative decoding, FP8 attention.

**Where it pays for itself:** batch ≥ 8, decode length ≥ 512 tokens, datacenter serving.

**Why it is wrong as the primary VLA runtime:**

- Its wins all scale with batch × decode length. VLA inference is batch=1, ~10 decode passes.
- The model-definition surface is "transformer decoder." Bending Eagle 2.5 + Qwen3 + DiT into that shape costs more in glue than you recover in kernels.
- Two model-definition surfaces (ONNX for non-LLM, TRT-LLM for LLM) is a permanent maintenance tax.

**When TRT-LLM is correctly used in VLA deployment:** narrow architecture-specific fallback (e.g., a Blackwell sm_100 path during a transitional ORT packaging gap). It should not be the primary runtime.

#### Option C — Raw CUDA / hand-written kernels

Maximum control, maximum maintenance burden.

**Deploy-layer rule:** do not write hand kernels for the velocity field, attention, RMSNorm, or RoPE. TRT does these well, the kernels evolve faster than you can keep up, every touch breaks parity.

**The two places raw CUDA legitimately wins in a VLA deploy layer:**

1. A specific op that ONNX cannot represent and TRT cannot fuse efficiently. Wrap as a TRT plugin, not a separate runtime.
2. Stream / event orchestration — overlapping vision encoder, LM prefill, and action head across multiple CUDA streams. This is "raw CUDA thinking" without writing kernels.

#### NPU-specific runtimes

Outside the NVIDIA stack, the runtime story is per-vendor and changes more often. As of writing:

- **NXP eIQ + ONNX Runtime / TFLite** — used in the i.MX 95 SmolVLA deployment.
- **Hailo-8 / Hailo-15** — proprietary compiler; works for ViT-based encoders; LM backbones over ~1B parameters typically don't fit.
- **Qualcomm SNPE / QNN** — first-class on Snapdragon; ONNX import is pragmatic.
- **Apple ANE / Core ML** — feasible for small VLAs; mostly used for prototyping.

The ORT + EP architecture generalizes across all of these — the "execution provider" abstraction is doing real work here.

### 3.7 Profile-driven optimization

The seventh and last technique: **measure before changing anything**.

The canonical entry point is Nsight Systems for NVIDIA targets and the vendor profiler for NPUs:

```bash
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=vla-trace.qdrep \
    python -m your_serve_module --model your-model
# fire one inference call, then SIGINT
```

Open the trace and answer, in order:

| Question | What it tells you | If "yes," do this |
|---|---|---|
| Is each `/act` one CUDA Graph launch or many? | Many → external sampler loop or graph capture is broken | Bake the loop into the ONNX |
| Gap between `Run()` returning and next CUDA op? | Non-trivial → ORT session overhead or buffer allocation | Use IO binding |
| Any host-device transfers during inference? | Yes → fallback op outside TRT subgraph or CPU preprocessing | Move op to GPU or graph-able subgraph |
| Stream concurrency on multi-engine pipelines | Serial → switch encoder and decoder to different streams | Add CUDA stream + event sync |
| Memory-bandwidth utilization (Orin specifically) | High → memory-bound, not compute-bound | Quantize first, fuse second |

**Optimization order on Jetson** (almost always in this sequence):

1. Reduce weight bandwidth: FP16 → BF16 → INT8 → INT4 quantization
2. Eliminate host-device transfers: GPU-resident preprocessing, IO binding
3. Capture larger CUDA Graphs: bake sampler loops into the ONNX
4. Improve kernel fusion: TRT plugins for hot ops the compiler cannot fuse
5. Multi-stream concurrency: overlap encode / decode

Doing step 4 before step 1 on a Jetson is wasted work.

---

## 4. The model zoo

Concrete VLAs you will encounter, with the inference characteristics that matter for deployment.

| Model | Params | Vision encoder | LM backbone | Action head | Action chunk | Notes |
|---|---|---|---|---|---|---|
| **OpenVLA** | 7.5 B | dual SigLIP + DINOv2 | Llama-2 7B | autoregressive token | 1 | The "VLA = LLM with action tokens" baseline. Hard to deploy on edge as-is. |
| **SmolVLA** (LeRobot) | 450 M | small ViT | small LM | flow-matching, 10-step Euler | 50 | Fits Orin Nano 8 GB. The reference small VLA. |
| **π0** (LeRobot) | 3.5 B | SigLIP | PaliGemma | flow-matching, 10-step Euler | 50 | Needs Orin 16 GB+. |
| **π0.5** (LeRobot) | 3.62 B | SigLIP | PaliGemma | flow-matching, 10-step Euler | 50 | Decomposed-export-friendly. |
| **GR00T N1.6** (NVIDIA) | 3.29 B | SigLIP | Qwen3 (Eagle 2.5 VLM) | DiT, 4-step DDPM | varies | Two-ONNX chain in deploy stacks. |
| **EdgeVLA / EVLA** | smaller | OpenVLA-style | SLM | non-AR joint pose | 1 | 7× inference speedup vs OpenVLA via non-AR head + SLM. |
| **OmniVLA** | 7B+ | SigLIP + DINOv2 | LLaMA-2 7B | tokens + adapter | varies | Used as the remote VLA in AsyncVLA. |
| **OmniVLA-edge** | 108 M | smaller | smaller | task-specific | varies | Lightweight navigation policy. |
| **CogACT** | varies | ViT | varies | action chunking | varies | MulticoreWare deployment writeups available. |

Two patterns to notice:

- The 7B-class single-step autoregressive (OpenVLA, OmniVLA) family wants async / edge-adapter for edge deployment.
- The 0.5–4 B-class chunked flow-matching / diffusion (SmolVLA, π0/.5, GR00T, EVLA) family fits on Jetson directly.

---

## 5. The hardware zoo

Per-target constraints that change the optimization order.

| Target | Compute | Memory | What changes |
|---|---|---|---|
| **NXP i.MX 95** | Mali GPU + eIQ Neutron NPU | shared with CPU, varies | NPU-quant-first; selective per-component quant; CPU fallback for action expert. SmolVLA achievable at ~6 s per inference with 4-bit encoder/LLM + FP32 action expert. |
| **Jetson Orin Nano 8 GB** (sm_8.7) | ~40 TOPS INT8 | 8 GB unified, ~50 GB/s | SmolVLA-class only; aggressive quant mandatory; preprocessing on GPU non-negotiable. |
| **Jetson Orin AGX 64 GB** (sm_8.7) | ~275 TOPS INT8 | 64 GB unified, ~204 GB/s | π0 / π0.5 / GR00T fit; FP16 default; BF16 if numerical headroom needed. |
| **Jetson Thor 128 GB** (sm_10) | ~2 PFLOPS FP4 | 128 GB unified | FP8 first-class; multi-VLA per device feasible. |
| **A10G** (sm_8.6) | ~31 TFLOPS FP32 | 24 GB GDDR6, ~600 GB/s | Cloud reference target; not memory-bound; CUDA Graph + TRT fusion dominates. |
| **RTX 4090** (sm_8.9) | ~83 TFLOPS FP32 | 24 GB GDDR6X, ~1 TB/s | Workstation reference; commonly used in vla-perf studies. |
| **H100** (sm_9.0) | ~989 TFLOPS BF16 | 80 GB HBM3, ~3 TB/s | FP8 attention shines; useful for fleet emulation. |
| **B100 / Blackwell** (sm_10.0) | ~20 PFLOPS FP4 | 192 GB HBM3e | Modeled by vla-perf; runtime stack support is currently transitional in some toolchains. |

The Orin Nano column is the design constraint that drives most architectural decisions for "real" robot deployment. **If your VLA cannot fit on 8 GB unified memory, it cannot deploy onto the most numerous robot platform in the world.**

---

## 6. Reference deploy stacks

Three publicly visible VLA deployment efforts, side by side. These are not endorsements — they are useful design reference points.

| Project | Primary stack | Targets | Key claim | Strength | Notable choice |
|---|---|---|---|---|---|
| **reflex-vla** | ORT + TRT EP + CUDA Graphs | x86 NVIDIA + Jetson Orin / Thor | 5.55× CUDA EP → TRT EP on A10G/SmolVLA; cos=+1 / max_abs ≈ 6e-7 strict parity | Cross-arch, CI-gated parity, mature ops primitives (`reflex doctor`) | Abandoned decomposed ONNX in favor of monolithic + baked sampler loop. |
| **HF + NXP i.MX 95** | ONNX Runtime + eIQ NPU | i.MX 95 (no GPU) | 29.1 s → 6.15 s SmolVLA | First serious VLA-on-NPU writeup; per-component quant policy | Action expert kept FP32 deliberately; vision + LLM dropped to 4-bit. |
| **AsyncVLA / OmniVLA** | Async edge adapter; base VLA on workstation | Robot onboard MCU + remote GPU | 40 % nav success-rate improvement under 0.2–6 s delays | Concrete asynchronous architecture with token compression | Edge adapter is just 2 MLP-ResNet blocks. Compression: 8×4×4096 → 1×1024. |

The three are not in competition — they target different operating points (cloud-anchor + small edge, fully on-device GPU, fully on-device NPU). A serious deployment effort eventually borrows from all three.

---

## 7. The reproducible parity-gate pattern

Independent of any specific tool, a serious VLA deploy layer should:

```python
# 1. Run the reference forward pass in PyTorch FP32 with seeded inputs.
ref = run_pytorch_fp32(model, fixture, seed=0)

# 2. Run the deployed engine on the same seeded inputs.
got = run_deployed(engine, fixture)

# 3. Strict comparison against machine epsilon (or reported tolerance).
cos = cosine_similarity(ref.flatten(), got.flatten())
max_abs = (ref - got).abs().max()

assert cos >= 1.0 - 1e-7,  f"cos parity failed: {cos}"
assert max_abs < 1e-4,      f"max_abs parity failed: {max_abs}"
```

The point of this pattern is not the exact tolerances. The point is that **you can refactor the engine, change EPs, change the export pipeline, and the gate either holds or breaks immediately**. Once you accept "1e-2 is close enough," you have lost the ability to change anything safely.

For **quantized** paths, strict equality breaks by construction. The replacement gate is **task-success-rate parity** on a held-out fixture suite, with a bounded acceptance delta:

```python
ref_tasks  = simulate(reference_model, task_suite, n=100)
quant_tasks = simulate(quantized_model, task_suite, n=100)

assert quant_tasks.success_rate >= ref_tasks.success_rate * 0.95
```

The HF + NXP team's ACT result (FP32 0.96 global → optimized 0.89, a –7 % delta) is roughly the boundary of what is generally tolerable. Beyond ~10 % task-success drop, you have over-quantized.

---

## 8. Benchmarking with vla-perf

[NVlabs/vla-perf](https://github.com/NVlabs/vla-perf) is the closest thing the field has to a standardized analytical benchmark. Built on top of the GenZ LLM Analyzer, it estimates VLA inference latency given:

- architecture (Pi0, OpenVLA, plus extension hooks for new VLAs),
- target hardware (A100, H100, B100, RTX 4090, the full Jetson family — Thor, AGX Orin, Orin NX, Orin Nano, AGX Xavier, Xavier NX),
- precision (FP32 / FP16 / BF16 / FP8 / INT8 / INT4),
- parallelism strategy (tensor / pipeline parallel).

It is **analytical**, not measured. The output is a roofline-style estimate per pipeline stage (vision encoder prefill, VLM backbone, action expert / DiT decode), with CSVs, plots, and LaTeX tables.

**How to use it well:**

1. Use it for sizing — "will π0 fit at acceptable latency on Orin AGX 64 with FP16?" — before exporting an ONNX.
2. Use it to compare hardware tiers — Thor vs Orin AGX vs A10G — before buying / spec'ing.
3. Do not use it as a substitute for measured Nsight traces. Analytical models miss launch overhead, EP fallbacks, host-device transfers, and CUDA Graph capture wins.

**Workflow:**

```
research / spec phase    →   vla-perf estimates       (will it fit at all?)
implementation phase     →   nsys measured traces     (what is the actual bottleneck?)
optimization phase       →   parity gate + nsys diff  (did the change work?)
release phase            →   bench harness in CI      (does it still work?)
```

---

## 9. Anti-patterns to avoid

**1. Quantizing before having a parity gate.** FP16 / BF16 / INT8 / FP8 each break strict equality by construction. Without an FP32 gate first, you cannot tell whether a quantized regression is acceptable precision loss or a bug.

**2. Running TRT-LLM as the primary VLA runtime.** Wrong workload axis. Use it as a narrow fallback for a specific architecture only.

**3. Quantizing the iterative diffusion / flow-matching action head aggressively.** Per-step error compounds. The HF + NXP discipline (encoder + LLM 4-bit, action expert FP32) is the right shape; reverse it and watch the rollouts diverge.

**4. Hand-rolled kernels in the deploy layer.** TRT evolves faster than you can keep up. Wrap one specific bottleneck as a TRT plugin if profiling demands it; do not maintain a parallel kernel library.

**5. CPU-side preprocessing on Jetson.** Even on a unified-memory device, bandwidth is the constraint. Move resize / normalize / CHW-swap onto the GPU.

**6. Optimistic dynamic shapes.** CUDA Graphs require fixed shapes. Either commit to a fixed image resolution and action-chunk length per deployment, or pay the cost of graph rebuilds. You cannot have both.

**7. "Close enough" parity tolerances.** Once `1e-2` is acceptable, every refactor leaks numerical drift. Pin to FP32 machine epsilon and let lower precision break the gate explicitly.

**8. Synchronous inference on a 7B-class remote VLA.** If the model lives on a workstation and the network round-trip is > 100 ms, the robot is open-loop during inference. Adopt the AsyncVLA edge-adapter pattern or move the model onboard (with all the associated quantization / distillation work).

**9. Treating the action head and the LM the same way.** They have different precision sensitivities. Apply different optimization strategies.

**10. No task-suite gate after quantization.** Strict equality cannot survive precision change. A held-out task-success-rate gate must replace it.

---

## 10. Build it — artifacts that prove correctness

If you are working on a VLA deploy layer of your own, the artifacts that prove you have done it correctly:

- **`bench`:** p50 / p95 latency on each target (cloud GPU, Jetson AGX, Jetson Nano, NPU if relevant), batch=1, with and without each EP/runtime variant. The TRT-EP-vs-CUDA-EP ratio for a flow-matching VLA on A10G should be in the 3–6× range; below 2× indicates an EP wiring problem.
- **`validate`:** strict-equality parity report for FP32 export, plus task-success-rate report for each lower-precision variant. CI-gated on every push.
- **`doctor`:** operational health check verifying EP loaded, libraries reachable, fixture passes parity within tolerance.
- **`nsys` trace** of one inference call per target, annotated to show one CUDA Graph launch per `/act` (or, if there are many, an open ticket explaining why).
- **Memory-budget report** for the smallest target (typically Orin Nano 8 GB or i.MX 95), with token-by-token memory accounting demonstrating the model fits with headroom for the action history buffer and KV cache.
- **Quantization recipe** documenting which components are quantized to what precision, with the task-suite delta vs FP32.

---

## 11. What good outcomes look like

| Area | Weak outcome | Strong outcome |
|---|---|---|
| Stack selection | "We use vLLM for the LM and figure out the rest" | ORT + TRT-EP + CUDA Graphs primary; TRT plugin escalation path documented; NPU runtime pluggable |
| Parity | "Outputs look right" | `cos = 1 - 1e-7`, `max_abs < 1e-4` against FP32 reference, in CI, plus task-success-rate gate for quant |
| Hardware support | "Works on my A100" | A10G + Orin AGX + Orin Nano + i.MX 95 benchmarks, with target-specific optimization order |
| Optimization order | "Fuse more kernels" | "Quantize first; we are memory-bound on Orin" |
| Precision policy | "FP16 everywhere" | "Encoder + LLM 4-bit; action expert FP32; documented and validated" |
| Async | "Robot waits for inference" | "Edge adapter at 30 Hz; remote VLA at 5 Hz; AsyncVLA-style refinement" |
| Maintenance | "Update kernels when needed" | "We do not maintain a kernel library; everything is upstream TRT or one documented plugin" |

---

## 12. Open research directions worth tracking

These are unsettled as of mid-2026; expect the answers to move in the next 12 months.

1. **Native FP8 attention for VLAs on Hopper / Thor.** TRT-LLM has it; the ORT path is catching up. The first VLA deploy layer with first-class FP8 + parity gate will set a new bar.
2. **NPU-native VLA compilers.** NXP, Hailo, Qualcomm, and Apple are all racing here. The runtime story below "ONNX + EP" is rapidly fragmenting and will probably re-converge around a small number of compilers.
3. **Speculative action decoding.** Borrowing speculative decoding from LLM serving for the autoregressive subset of VLAs (OpenVLA-style). Limited public work as of writing.
4. **Edge-cloud collaborative training** (not just inference). RoboECC-style splits for online fine-tuning would unlock fleet-scale robot learning without per-robot upload of full demonstrations.
5. **Action chunk length as a function of task confidence.** Current systems pick `k` statically. Confidence-conditioned `k` would be a clean inference-time win for variable-difficulty tasks.
6. **Standardized benchmarks beyond LIBERO.** vla-perf is a step toward analytical standardization; measured benchmarks across hardware tiers are still missing.

---

## References

### Primary papers and projects

- **EdgeVLA / EVLA** — Budzianowski et al., 2025. *EdgeVLA: Efficient Vision-Language-Action Models.* arXiv: [2507.14049](https://arxiv.org/abs/2507.14049).
- **AsyncVLA** — Hirose et al., 2025. *An Asynchronous VLA for Fast and Robust Navigation on the Edge.* [asyncvla.github.io](https://asyncvla.github.io/).
- **LiteVLA-Edge** — Williams et al., 2026. *LiteVLA-Edge: Quantized On-Device Multimodal Control for Jetson Orin-class Hardware.*
- **OmniVLA / OmniVLA-edge** — Hirose et al., 2025.
- **RoboECC** (2026) — *Multi-Factor-Aware Edge-Cloud Collaborative Deployment for VLA Models.*
- **NVlabs/vla-perf** — analytical performance modeling tool: [github.com/NVlabs/vla-perf](https://github.com/NVlabs/vla-perf).
- **CogACT** — referenced in MulticoreWare deployment writeup.

### Reference deployments

- **reflex-vla** — open-source VLA deploy layer: [github.com/rylinjames/reflex-vla](https://github.com/rylinjames/reflex-vla).
- **HuggingFace + NXP — Bringing Robotics AI to Embedded Platforms** (Mar 2026): [HF blog](https://huggingface.co/blog/nxp/bringing-robotics-ai-to-embedded-platforms).
- **MulticoreWare — Deploying VLA AI Models on Edge** (Jul 2025).
- **deepsense.ai — Embodied AI on a 100 g Device** (Aug 2025).

### Model implementations

- **LeRobot (SmolVLA, π0, π0.5)**: [github.com/huggingface/lerobot](https://github.com/huggingface/lerobot).
- **NVIDIA Isaac GR00T N1 / N1.6**: [developer.nvidia.com/isaac/gr00t](https://developer.nvidia.com/isaac/gr00t).
- **OpenVLA**: [openvla.github.io](https://openvla.github.io/).

### Runtime / tooling

- **ONNX Runtime — TensorRT Execution Provider**: [onnxruntime.ai/docs](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html).
- **TensorRT-LLM** (for context, not as VLA primary): [github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM).
- **CUDA Graphs API**: [docs.nvidia.com/cuda](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs).
- **Nsight Systems**: [developer.nvidia.com/nsight-systems](https://developer.nvidia.com/nsight-systems).
- **NXP eIQ / i.MX 95**: NXP developer documentation.

### Sibling roadmap modules

- [ML and AI on Jetson — overview](../Guide.md)
- [Jetson LLM Runtime](../jetson-llm-runtime/Guide.md)
- [LLM Optimization on Jetson](../llm-optimization-jetson/Guide.md)
- [Robotics — Advanced Perception and AI](../../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20D%20-%20Robotics/Advanced%20Perception%20and%20AI%20for%20Robotics/Lecture-01.md)
