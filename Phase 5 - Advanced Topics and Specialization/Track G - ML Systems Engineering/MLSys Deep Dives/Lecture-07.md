# Lecture 07 - The Edge and Physical-AI Frontier: 1000-TOPS Hardware, On-Device Models, and the Capstone

**Collection:** [MLSys Deep Dives](README.md) | **Previous:** [← Lecture 06](Lecture-06.md) | **Next:** [MLSys Deep Dives index](README.md)

---

The course has lived in the datacenter. This final lecture takes the *same* MLSys discipline to the other budget — the **edge**, where the constraint is not dollars-per-GPU-hour but **watts**, and where AI meets the physical world in robots, cars, and embedded devices. Then it closes the loop: the architecture, kernel, compiler, and decode-algorithm layers of Lectures 1–6 are **one co-designed system**, and the capstone is to prove it on a real model with a number.

We start by untangling a confusion the field invites — the two different "1000s" — because keeping them straight is itself a senior-engineer signal.

---

## Learning objectives

By the end of this lecture, you should be able to:

1. Distinguish **1000 TOPS** (edge hardware *capacity*) from **1000 tokens/s** (datacenter serving *throughput*) — and read TOPS numbers with the right precision/sparsity skepticism.
2. Describe **1000-TOPS-class edge hardware** (Jetson AGX Thor, DRIVE Thor) and its real constraint: **memory bandwidth and power**, not peak TOPS.
3. Explain why edge inference is the *same* MLSys discipline at a **power-capped** budget, and why **perf/watt** is the binding metric.
4. Choose on-device models (small/hybrid) and runtimes (TensorRT, MLC-LLM, llama.cpp) for an edge target.
5. Draw the **co-design loop** connecting all seven lectures to tokens/s and tokens/s/watt.
6. Execute the **capstone**: an optimization-ladder report tying each rung to a roofline bound and a cost number.

---

## 1. The two "1000s"

You will hear "1000" attached to AI systems in two completely different ways, and conflating them marks a novice. Untangle them:

```text
   "1000 TOPS"   = edge HARDWARE compute capacity   (Jetson Thor, DRIVE Thor)
                   operations/second the chip CAN do — a SUPPLY-side spec
                   ⚠ almost always quoted at the LOWEST precision WITH sparsity (FP4 sparse);
                     the honest dense number is roughly half (FP8)

   "1000 tok/s"  = datacenter SERVING throughput     (MiMo + TileRT, Lecture 6)
                   tokens/second a STACK actually DELIVERS — a DEMAND-side result
                   produced by architecture + quantization + spec-decode + runtime, stacked
```

One is what the silicon *could* do; the other is what a software stack *actually* achieved. The first is a number on a datasheet; the second is the output of this entire course. The lesson embedded here is **TOPS skepticism**: whenever you see a TOPS figure, ask *at what precision, dense or sparse?* — because vendors quote the most flattering combination (FP4 + structured sparsity), and the number you can actually sustain on a real workload is often a fraction of it. This is the same discipline as the "printed throughput is a dated anchor" rule from Lecture 1, applied to the supply side.

---

## 2. 1000-TOPS-class edge hardware

The concrete device behind "1000 TOPS" in 2025–2026 is the **NVIDIA Jetson AGX Thor** class (and its automotive sibling **DRIVE Thor**), built for **physical AI** — robotics, autonomous machines, embodied agents.

| Spec | Jetson AGX Thor |
|---|---|
| GPU | **Blackwell**, 2560 CUDA cores, 96 5th-gen Tensor Cores |
| AI perf | **~2070 FP4 TFLOPS (sparse)** / **~1035 FP8 TFLOPS (dense)** |
| CPU | 14× Arm Neoverse-V3AE |
| Memory | **128 GB LPDDR5X, 273 GB/s** |
| Power | **75–130 W** |
| Dev kit | $3,499 (Nov 2025); ~7.5× AI perf, ~3.5× efficiency vs AGX Orin |

Note where the headline number lands and where the *real* constraint is:

* The "2070 TOPS" is **FP4 with sparsity**; the honest sustained figure is **~1035 FP8 TFLOPS dense**. (TOPS skepticism, §1.)
* The binding constraint for LLM decode is **not** the TOPS — it's the **273 GB/s memory bandwidth**. Compare that to a datacenter GPU's **multiple TB/s of HBM**: the edge has ~10–30× less bandwidth. Since decode is memory-bound (Lecture 1, 6), **the edge is even more bandwidth-starved than the datacenter** — which makes quantization, hybrid architectures, and speculative decoding *more* important here, not less.
* **128 GB of unified memory** is generous (it's a robot's whole brain), but **75–130 W** is the hard wall. You are power-capped, full stop.

Run Lecture 1's **bandwidth-ceiling check** on this box and the whole edge story falls out of three lines of arithmetic:

```text
   batch-1 decode ceiling on Thor (273 GB/s):
       7B  @ FP16  ≈ ~14  GB weights   →   273 / 14    ≈  ~19 tok/s
       7B  @ INT4  ≈ ~3.5 GB weights   →   273 / 3.5   ≈  ~78 tok/s
       70B @ INT4  ≈ ~35  GB weights   →   273 / 35    ≈   ~8 tok/s
   add speculative decoding at τ ≈ 3 (Lec 6) → the 7B-INT4 box clears ~200 tok/s in theory
```

Notice what *didn't* appear in that math: the 2070-TOPS headline. Bandwidth and bytes decided everything — which is the concrete proof of the bullet above, and of why on the edge **quantization is not a nice-to-have, it is the difference between a usable robot and a 19-tok/s one.**

The competitive set (DRIVE Thor, Qualcomm Snapdragon Ride Flex) lives in the same **1000–2000 TOPS, sub-130 W** band, targeting centralized automotive/robotics compute. This is the hardware where a reasoning model has to run *inside a power budget*, in real time, next to sensors.

---

## 3. The edge is the same discipline, power-capped

Here is the unifying claim of the whole course. The cost equation from Lecture 1 —

```text
   value delivered  =        tokens per second
                      ───────────────────────────────
                       budget you are capped against

   datacenter:  budget = $/GPU-hour   →  optimize TOK/$  (TCO/Mtok)
   edge:        budget = WATTS         →  optimize tokens/s/WATT  (and fit in memory)
```

— is *the same equation*; only the denominator's units change. And the **levers are identical**: quantize to fewer bits (FP4/INT4), pick a hybrid architecture to shrink the KV cache so the model fits the bandwidth, use speculative decoding to get more tokens per memory pass, use good kernels/compilers and megakernel runtimes to cut waste. Everything in Lectures 2–6 applies unchanged at the edge — it just gets graded on **perf/watt** instead of **perf/dollar**.

So an MLSys engineer does not switch disciplines moving from cloud to robot. They re-point the *same* discipline at a power budget. A 7B reasoning model that runs at acceptable tokens/s inside 40 W on a Jetson is the same kind of win as a 1T MoE at 1000 tok/s in a datacenter — both are tokens-per-(capped resource), both built from the same stack.

---

## 4. On-device models and runtimes

What runs on a 1000-TOPS, 273-GB/s, 100-W box? The model choices map directly onto Lectures 4–5:

* **Small dense reasoning models** — **MiMo-7B**, Nemotron Nano, Phi-class. 7B at INT4 fits comfortably in memory and bandwidth; MiMo's MTP heads (Lec 5) give on-device speculative decoding for free.
* **Hybrid / SSM models** — **Falcon-H1-1.5B/3B**, small Mamba hybrids. Their **flat-in-context memory** (Lec 4) is doubly valuable when bandwidth is the wall — a constant-state model doesn't thrash the 273 GB/s bus as context grows.
* **Aggressive quantization is mandatory**, not optional: FP4/INT4 weights to fit memory *and* to multiply effective bandwidth (fewer bytes per parameter streamed per token). The edge is where the precision floor gets pushed hardest.

The runtimes:

| Runtime | Edge fit |
|---|---|
| **TensorRT / TensorRT-LLM** | best on NVIDIA Jetson/DRIVE; closed but peak |
| **MLC-LLM** (TVM Unity) | cross-platform — the same model to Jetson, phone GPU, browser; quantized, dlight schedules (see [TVM Deep Dives](../TVM%20Deep%20Dives/README.md)) |
| **llama.cpp** | ubiquitous CPU/edge GGUF runtime, great for the smallest targets |

The choice is the Lecture-3 decision (closed-vendor-peak vs portable-compiler) re-asked under a power budget. On a Jetson you'll often run TensorRT-LLM for peak; for a model that must *also* hit a phone and a browser, MLC-LLM's one-model-many-targets path wins.

---

## 5. The co-design loop closes

Step back and see the whole course as one diagram. Every lecture was a different lever on the same denominator, and they **compound**:

```text
   THE CO-DESIGN LOOP  (the whole course, as one system)
   ┌──────────────────────────────────────────────────────────────────┐
   │  ARCHITECTURE   (Lec 4–5)  MoE · MLA · SSM/hybrid                  │  ← less work per token
   │       │                                                            │
   │  KERNELS+COMPILERS (Lec 2–3)  tiles · fusion · megakernels          │  ← each op faster, gaps gone
   │       │                                                            │
   │  INFERENCE ALGOS (Lec 6)  spec decode (EAGLE-3/DFlash) · Flash      │  ← more tokens / memory pass
   │       │                                                            │
   │  HARDWARE+PRECISION (Lec 7)  FP4 · right chip · right batch         │  ← right bits, right silicon
   └───────────────────────────────┬──────────────────────────────────┘
                                   ▼
            tokens/s ↑   AND   tokens/s/watt ↑   →   $/Mtok ↓   (Lecture 1)
```

The MiMo + TileRT result (Lec 6) was this loop, fully stacked: sparse MoE (architecture) + MXFP4 (precision) + DFlash (algorithm) + TileRT megakernel (runtime). No single layer produced 1000 tok/s; the *product* did. That is the senior-MLSys worldview: **you do not optimize one layer, you co-design the stack**, and you measure the compound at the top in tokens/s and dollars (or watts).

---

## 6. Capstone: the optimization ladder

The course artifact. Pick **one model and one target** (a datacenter GPU *or* an edge board — the discipline is the same) and walk it down an optimization ladder, **measuring every rung**:

```text
   THE OPTIMIZATION LADDER  — one model, one target, a number at every rung
   ─────────────────────────────────────────────────────────────────────────────
   rung 0  baseline (eager, FP16)                          tokens/s · $/Mtok or tok/s/W · TTFT/TPOT
   rung 1  + quantize (FP8 / FP4 / INT4)                   Δ + which roofline bound moved
   rung 2  + better kernels / compiler (Triton/TVM/TRT)    Δ + GFLOP/s vs roofline
   rung 3  + speculative decoding (EAGLE-3 / DFlash)        Δ + acceptance length τ + outputs-identical?
   rung 4  + (if applicable) hybrid arch / megakernel       Δ + KV-cache or gap reduction
   ─────────────────────────────────────────────────────────────────────────────
   for EACH rung: name the roofline bound it moved, and the $/Mtok (or tok/s/W) delta.
```

Rules that make it real engineering, not a recipe log:

* **Measure, don't estimate.** Every rung gets a measured tokens/s and a recomputed cost (Lecture 1's model). A rung you can't measure didn't happen.
* **Name the bound.** For each rung, state which roofline regime it moved — memory-bound → quantization/hybrid; compute-bound → better kernels; launch/gap-bound → megakernel; serial-decode-bound → speculation. If you can't name the bound, you don't yet understand why the rung helped.
* **Parity at every rung.** Quantization and (especially) speculative decoding must preserve outputs within budget — spec decode *losslessly*. A faster wrong model is a regression.
* **End with the compound.** The headline is the full-ladder result: baseline `$/Mtok` (or tok/s/W) → final, and the multiplier. That single number is your portfolio.

This is a **Level-5 artifact**: another engineer clones your repo, runs the ladder, and reproduces your numbers on the same hardware class. It is also the most honest possible demonstration that you understand MLSys — because it forces every layer of the course to show up as a measured, defended, cost-connected step.

---

## 7. Mini-lab (and course wrap)

If the full capstone is too large, do a three-rung version: **baseline → quantize → speculative decode**, on any model+target you can run, measuring tokens/s and `$/Mtok` (or tok/s/W) at each, with the roofline bound named and parity confirmed.

Then answer the course's closing question in writing: *Why is a hybrid 7B at 40 W on a Jetson and a 1T MoE at 1000 tok/s in a datacenter the same MLSys discipline?* If your answer is "because both maximize tokens-per-(capped resource) by co-designing architecture, kernels, compilers, and decode algorithms, and both are measured against the cost equation" — you have the worldview this course exists to build.

---

## Key takeaways

- **Two "1000s"**: 1000 **TOPS** = edge hardware *capacity* (supply-side, quoted FP4-sparse — be skeptical); 1000 **tok/s** = datacenter serving *throughput* (demand-side, the output of the whole stack).
- **1000-TOPS-class edge** (Jetson/DRIVE Thor) is bounded by **memory bandwidth (~273 GB/s) and power (75–130 W)**, not peak TOPS. The edge is *more* bandwidth-starved than the datacenter, so Lec 2–6's levers matter *more*.
- The **edge is the same discipline, power-capped**: `value = tokens/s ÷ budget`, with budget = watts instead of dollars. **perf/watt** is the binding metric; the levers (quantize, hybrid arch, spec decode, good kernels) are unchanged.
- On-device: small dense (MiMo-7B) and hybrid (Falcon-H1) models, **aggressive FP4/INT4** quantization, runtimes TensorRT-LLM (peak NVIDIA) / MLC-LLM (cross-platform) / llama.cpp (smallest).
- The **co-design loop** ties all seven lectures together: architecture × kernels/compilers × inference algorithms × hardware/precision **compound** into tokens/s and tokens/s/watt. You co-design the stack; you don't optimize one layer.
- The **capstone** is an optimization ladder — baseline → quantize → kernels/compiler → spec decode → (hybrid/megakernel) — with a measured cost number and a named roofline bound at every rung. That is the proof you understand MLSys.

---

## References

- NVIDIA Jetson AGX Thor (physical AI platform): [https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/](https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/)
- Jetson AGX Thor dev kit specs (2070 TOPS FP4 / 1035 TFLOPS FP8, 128 GB, 273 GB/s): [https://www.cnx-software.com/2025/08/19/3499-nvidia-jetson-agx-thor-developer-kit-2070-tops-jetson-t5000-som-for-robotics-and-edge-ai/](https://www.cnx-software.com/2025/08/19/3499-nvidia-jetson-agx-thor-developer-kit-2070-tops-jetson-t5000-som-for-robotics-and-edge-ai/)
- MLC-LLM (cross-platform on-device LLM, TVM Unity): [https://github.com/mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm)
- llama.cpp: [https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- SemiAnalysis InferenceMAX / InferenceX (perf/$, perf/watt): [https://newsletter.semianalysis.com/p/inferencemax-open-source-inference](https://newsletter.semianalysis.com/p/inferencemax-open-source-inference)
- *TVM Deep Dives* — [Lecture 05 — Shipping it: runtime, microTVM, MLC-LLM](../TVM%20Deep%20Dives/Lecture-05.md), for the on-device deployment path.
- *MLSys Deep Dives* — [Lecture 01 — the cost equation](Lecture-01.md), which this capstone measures against.

---

## Current as of

2026-06. Pins: Jetson AGX Thor (Blackwell, ~2070 FP4 / ~1035 FP8 TFLOPS, 128 GB LPDDR5X @ 273 GB/s, 75–130 W, $3,499 dev kit Nov 2025), DRIVE Thor / Snapdragon Ride in the 1000–2000 TOPS band. TOPS figures are FP4-sparse marketing numbers — sustained dense FP8 is roughly half; always re-check precision/sparsity and re-benchmark on the actual workload.

---

*Back to: [MLSys Deep Dives index](README.md)*
