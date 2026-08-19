# Hardware-Aware LLM Quantization & Inference Optimization

<div class="course-identity mlsys" markdown="1">
<div class="course-identity__icon">NVFP4</div>
<div markdown="1">
<p class="course-identity__eyebrow">Phase 5 · ML Systems Engineering · Research-Engineering Course</p>
<p class="course-identity__title">Which bits should I remove, from which tensors, using which method, to gain real tok/s without changing model behavior?</p>
<p class="course-identity__meta">Artifact: a hardware-aware quantization policy engine + measured ablation report · Measure: tok/s, bytes/token, KL vs reference, speculative acceptance length</p>
</div>
</div>

> *A smaller checkpoint is not a faster model. A faster model is not a preserved model. This course is about the difference.*

Most quantization material answers "how do I make the file smaller?" That is the wrong question for a decode-bound LLM on modern silicon. The right question has four parts fused into one, and it is the sentence this entire course exists to answer:

```text
   Which bits should I remove   ──▶  precision allocation, not uniform bit-width
   from which tensors           ──▶  runtime traffic, not parameter count
   using which method           ──▶  calibration matched to the error mode
   to gain real tok/s           ──▶  hardware-native formats, not nominal bits
   without changing behavior?   ──▶  KL and acceptance length, not file size
```

Miss any one clause and you ship a regression that looks like a win. A 3-bit model that is *smaller and slower* than a 4-bit one. A vision tower you quantized for zero decode benefit. A Q/K projection that bought 2 tok/s and cost you a quarter of your speculative acceptance rate.

**Target platform:** NVIDIA Blackwell, GeForce RTX 5090 (GB202, `sm_120`), 32 GB GDDR7 @ 1792 GB/s.
**Primary case study:** a ~27 B dense multimodal model in NVFP4 with an MTP speculation head — reconstructed throughout from published measurements of [`Qwen3.8-27B-NVFP4-RTX5090`](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090) and [`Qwen3.8-27B-DSpark-NVFP4`](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4).
**Level:** senior inference engineer / research engineer. This course starts *above* introductory PyTorch quantization.

**Layer mapping:** L4–L6 — the numerical-format layer where model math, kernel dispatch, and memory bandwidth meet. It sits below the serving runtime and above the CUDA kernel.

**Role targets:** Inference Systems Engineer · Model-Compression Engineer · GPU Runtime Engineer · LLM Runtime Optimization Engineer · Research Engineer (efficiency)

---

## Prerequisites

| Prerequisite | Why you need it |
|---|---|
| [Logprobs, Perplexity & KL Divergence](../Logprobs,%20Perplexity%20and%20KL%20Divergence/README.md) | Module 8 assumes you can already read `H(p,q) = H(p) + D_KL(p‖q)` and grade a quant with mean KLD and top-token agreement. This course *uses* those instruments; that course *derives* them. |
| [AI Inference Engineer 2026 — Part 1](../AI%20Inference%20Engineer%202026/Part%201%20-%20Fundamentals/README.md) | Roofline, the precision stack, the runtime landscape. Module 1 here goes deeper on one specific roofline; Part 1 gives you the general one. |
| [Phase 4 — Quantization & Low-Precision Inference](../../../Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/04%20-%20Quantization/Guide.md) | PTQ/QAT vocabulary, per-tensor vs per-channel scales, TensorRT/ONNX tooling. That page is the *general* introduction; this course is the LLM-and-Blackwell-specific research treatment. |
| CUDA fluency | You must be able to read a Nsight Compute report and know what a memory-bound kernel looks like. |

**Pairs with:** [MLSys Deep Dives](../MLSys%20Deep%20Dives/README.md) (speculative decoding and the kernel-language layer as systems) and [AI Inference Engineer 2026 — Part 4](../AI%20Inference%20Engineer%202026/Part%204%20-%20Optimizing%20a%20Real%20Engine/README.md) (the same measure-first discipline applied to an 8× H200 engine).

---

## The equation the whole course hangs from

For batch-1 decode on a bandwidth-bound GPU:

```text
                 BW_effective                    ← what the memory system actually delivers
   tok/s  ≈  ──────────────────
                   B_token                       ← bytes that MUST be fetched per generated token
```

Every module is a different way of attacking one of those two terms, or of proving you did not break the model while doing it:

| Module | Attacks | How |
|---|---|---|
| 01, 04 | `B_token` | find which bytes are actually on the per-token critical path |
| 02, 03 | `B_token` + `BW_eff` | pick a format that is both smaller *and* natively executable |
| 05, 06, 07 | behavior | remove bits where the model can afford it |
| 08 | behavior | prove you did not break it |
| 09 | `B_token` | the KV term that grows with context |
| 10 | the equation itself | speculation emits multiple tokens per weight-read |
| 11, 12 | all of it | allocate precision optimally and prove the result |

---

## Course Map (12 modules + capstone)

<div class="lecture-map" markdown>

| # | Module | The thread |
|---|--------|-----------|
| [01](Lecture-01.md) | **Inference Physics** — arithmetic intensity, the bandwidth ceiling, why checkpoint size ≠ bytes/token, and the `Traffic × Compressibility × HardwareSpeedup × BehaviorTolerance` opportunity framework | why compression becomes speed |
| [02](Lecture-02.md) | **Quantization Mathematics** — BF16 → FP8 E4M3 → NVFP4 E2M1, block scaling, the 16-element group, and one activation walked through the NVFP4 grid by hand | the numerical machinery |
| [03](Lecture-03.md) | **Blackwell Hardware** — what `sm_120` actually accelerates, block-scaled `mma.sync` vs `tcgen05`, ridge points per precision, and why 3-bit can be slower than 4-bit | the hardware boundary |
| [04](Lecture-04.md) | **Model Anatomy** — reconstructing a 27 B checkpoint from its bytes, the per-token traffic ledger, and the tensors that cost VRAM but not bandwidth | find the bytes that matter |
| [05](Lecture-05.md) | **Calibration** — absmax vs percentile vs MSE-optimal clipping, AWQ, GPTQ/Hessian methods, SmoothQuant, and which one matches which error mode | choosing the scales |
| [06](Lecture-06.md) | **Activation Outliers** — systematic channel outliers vs rare token spikes, attention sinks, massive activations, and why W4A4 is the hard one | why activations fight back |
| [07](Lecture-07.md) | **Layer Sensitivity** — the softmax amplification argument for why Q/K break while O/MLP survive, RoPE phase error, and why sensitivity ≠ outlier magnitude | which layers can pay |
| [08](Lecture-08.md) | **Behavior Preservation** — the metric ladder from MSE to benchmarks, and speculative acceptance length as the sharpest cheap behavioral probe you have | proving you didn't break it |
| [09](Lecture-09.md) | **KV Cache & Long Context** — GQA KV math, FP8/FP4 KV, the weight-traffic/KV-traffic crossover, and 262 K-context economics on a 32 GB card | the term that grows |
| [10](Lecture-10.md) | **Speculative Decoding** — the rejection-sampling acceptance rule, the `τ/(1+K·c)` model, MTP heads, and how quantizing the target silently taxes acceptance | multiplying the ceiling |
| [11](Lecture-11.md) | **Hardware-Aware AutoQuant** — precision allocation as constrained optimization, the sensitivity/traffic ratio, and a greedy solver with hardware-native guard rails | the allocation algorithm |
| [12](Lecture-12.md) | **Research Methodology** — controlled ablations, clock locking, the paired-comparison discipline, and the invalid comparisons that produce most published quantization claims | proving something |
| [13](Lecture-13.md) | **Capstone: TurboQuant** — build the policy engine, run the ablation grid, and beat 155.75 tok/s @ 2.886 acceptance without losing behavior | the artifact |

</div>

---

## Course Outcomes

By the end you should be able to:

* Decide, from measurements alone, whether a decode workload is **bandwidth-bound, compute-bound, or launch-bound** — and refuse to quantize anything until you know.
* Build a **per-token byte ledger** for any checkpoint that separates resident bytes from per-token-read bytes, and explain why the two differ by 20 % or more on a typical multimodal model.
* Encode and decode an **NVFP4 block by hand**, state its effective bits-per-weight (4.5, not 4), and explain why the 16-element group with an E4M3 scale beats MXFP4's 32-element E8M0 group.
* Name what `sm_120` executes **natively** and what it executes by **unpacking to a wider type**, and predict which "smaller" formats will be slower.
* Choose a calibration method by **error mode** rather than by popularity — clipping vs rounding, weight outliers vs activation outliers.
* Explain the **softmax-amplification mechanism** that makes Q/K quantization disproportionately damaging, and why that damage is not predicted by outlier magnitude.
* Grade a quantization with **KL divergence and speculative acceptance length**, and defend acceptance length as a more sensitive instrument than perplexity.
* Compute the **KV/weight traffic crossover context length** for a given config and say when long-context work changes the optimization target.
* Allocate precision across tensors as a **constrained optimization** — maximize tok/s subject to a behavior budget, a VRAM budget, and a hardware-native-format constraint.
* Run an **ablation another engineer would believe**.

---

## What this course is not

* Not an introduction to quantization. Start at [Phase 4 — Quantization](../../../Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/04%20-%20Quantization/Guide.md) if you need per-tensor vs per-channel scales explained.
* Not a survey of every published method. GPTQ, AWQ, SmoothQuant and friends appear as **tools selected by error mode**, not as a literature tour.
* Not a claim that lower bit-width is better. Several modules exist specifically to show it is not.

---

## Currency / Refresh Discipline

* **Timeless:** the roofline argument, arithmetic intensity, the softmax amplification mechanism, the rejection-sampling acceptance rule, block-scaling mathematics.
* **Moves with hardware:** what `sm_120` and `sm_100` accelerate natively; ridge points; whether a given bit-width has a tensor-core path. Module 03 is the refresh surface.
* **Moves with tooling:** TensorRT Model Optimizer, llm-compressor, vLLM/TensorRT-LLM quantized kernel coverage. Modules 05 and 11 are the refresh surface.
* Every module closes with a **`## Current as of`** note separating settled math from 2026 tooling.

---

## Exit Criteria

You are done when you can take an unfamiliar checkpoint on unfamiliar silicon and, without looking anything up:

* Produce its per-token byte ledger and predict its batch-1 decode ceiling within ~10 %.
* Say which three tensors to quantize first and why — citing traffic, not size.
* Name the format you would use and prove it has a native tensor-core path on that silicon.
* State the behavior budget in KL and acceptance-length terms *before* running the experiment.
* Run the ablation, report it honestly, and correctly identify the case where your speedup came from a bug.

If you can recite quantization algorithms but cannot say which bytes a token actually reads, you have vocabulary. The point of this course is the allocation decision.

---

*Related: [Logprobs, Perplexity & KL Divergence](../Logprobs,%20Perplexity%20and%20KL%20Divergence/README.md) · [AI Inference Engineer 2026](../AI%20Inference%20Engineer%202026/README.md) · [MLSys Deep Dives](../MLSys%20Deep%20Dives/README.md) · [Phase 5 — ML Systems Engineering Guide](../Guide.md)*
