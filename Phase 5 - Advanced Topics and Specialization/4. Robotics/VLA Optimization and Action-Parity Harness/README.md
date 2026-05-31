# VLA Optimization and Action-Parity Harness — Special Course

<div class="course-identity robotics" markdown="1">
<div class="course-identity__icon">VLA</div>
<div markdown="1">
<p class="course-identity__eyebrow">Phase 5 · Robotics · Special Course</p>
<p class="course-identity__title">Shrink and accelerate Vision-Language-Action policies until they run on a real robot, then prove the optimized policy still does the task.</p>
<p class="course-identity__meta">Artifact: optimized VLA + action-parity harness · Measure: tokens/s, control-loop latency, action MSE, task success-rate parity</p>
</div>
</div>

> *Optimization is only credible when the robot still picks up the cup.*

This special course pairs two halves that almost always have to ship together:

1. **VLA optimization** — quantization, KV/action-chunk caching, distillation, speculative action decoding, and runtime tricks that take a 3-7B-parameter Vision-Language-Action model from "runs in a notebook on a H100" to "runs at usable rate on a Jetson AGX Orin or a single workstation GPU on a real robot."
2. **Action-parity harness** — the measurement framework that decides whether each optimization is safe to deploy: per-step action error, trajectory divergence, closed-loop sim success-rate parity, and on-robot regression gating against a reference policy.

The two halves are inseparable: an optimization that you cannot measure against the reference policy is not an engineering result, it is a vibe.

**Scope:** inference and deployment of pretrained VLAs (OpenVLA, RT-2-X / OpenX-style policies, π₀ / π0.5, NVIDIA GR00T N-series, RDT, Octo). Training and finetuning are explicit non-goals — when you need adaptation, use the [Robot Learning section of Lecture 3](../Advanced%20Perception%20and%20AI%20for%20Robotics/Lecture-01.md#c1-reinforcement-learning) of the Perception lecture.

**Layer mapping:** L3-L8. Touches model architecture, runtime / kernels, edge accelerators, ROS 2 integration, and the closed-loop sim / real evaluation harness.

**Role targets:** Robot Learning Engineer · Robotics Foundation-Model Engineer · Edge AI / Embodied AI Runtime Engineer · Applied Research Engineer (VLA deployment) · Eval Infra Engineer for Robot Foundation Models.

**Prerequisites:**

* Phase 5 — Robotics — [Advanced Perception and AI for Robotics, Part C (Robot Learning)](../Advanced%20Perception%20and%20AI%20for%20Robotics/Lecture-01.md) — you need to already understand what a VLA *is* and how it sits above ROS 2 skills.
* Phase 5 — Edge AI — [Qwen Inference Optimization](../../3.%20Edge%20AI/Qwen%20Inference%20Optimization/README.md) — the LLM half of a VLA is a transformer; the optimization toolkit transfers.
* Phase 4 Track B — [Jetson Real-Time Inference](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Real-Time-Inference/Guide.md) — Jetson Orin is the default reference edge target.
* Phase 4 Track C — [Quantization](../../../Phase%204%20-%20Track%20C%20-%20ML%20Compiler%20and%20Graph%20Optimization/DL%20Inference%20Optimization/04%20-%20Quantization/Guide.md) — INT8 / FP8 / AWQ-style weight-only quant is reused verbatim on the LLM backbone of the VLA.

**What comes after:** a reproducible repo containing an optimized policy checkpoint, a runtime config, a harness CLI, a CSV of parity metrics across optimization levels, and an on-robot or closed-loop sim recording of the optimized policy hitting the parity bar.

---

## Lecture map

| # | Title | Focus |
|---|-------|-------|
| 01 | [VLA Optimization for Real-Time Control](Lecture-01.md) | What a VLA actually executes per control tick · quantization of the LLM backbone · vision-tower fusion · action-chunk and KV caching · speculative action decoding · Jetson / single-GPU deployment paths |
| 02 | [The Action-Parity Harness](Lecture-02.md) | Reference vs candidate rollouts · per-step action error · trajectory divergence · closed-loop success-rate parity (LIBERO / RoboCasa / Isaac Lab) · tolerance budgets · CI gating · on-robot canary protocol |

Each lecture follows the standard *Why it matters → Mental model → Build it → Measure it → Ship it* shape from the [Curriculum Authoring Guide](../../../Curriculum-Authoring-Guide.md), with a self-check and a runnable lab.

---

## Why this is a hardware-first course

A VLA is, mechanically:

```text
images (N cameras, ~224x224 or 384x384)
   └─► vision encoder (SigLIP / DINOv2 / EVA-style, ~300M-1B params)
        └─► projector / Perceiver resampler
              └─► LLM backbone (Llama-7B / Gemma-2B / Qwen2-VL-class, 2-8B params)
                    └─► action head (discretized tokens, MLP regression, or diffusion head)
                          └─► 7-DoF (or 14-DoF bimanual) action @ 10-50 Hz
```

The hardware reality every embodied-AI engineer hits:

* the **control loop deadline** is 20-100 ms; a vanilla 7B VLA at fp16 takes 200-600 ms per action on consumer / edge hardware
* **the vision tower is not free** — at 4 cameras × 384×384 it can rival the LLM backbone in FLOPs
* **memory budget on Jetson AGX Orin (64 GB)** is shared between the robot stack, ROS 2 nodes, perception, and the VLA — you rarely have more than ~16-24 GB for the policy
* **deterministic latency matters more than peak throughput** — a P99 spike that misses a control tick can crash a real robot
* the **action head is a different runtime problem** from text generation: action-token VLAs do short autoregressive bursts (4-16 tokens per chunk), diffusion heads do 5-20 denoising steps

Every optimization in Lecture 01 is justified by one of those bullets. Every metric in Lecture 02 is the thing the optimization is allowed to trade away.

---

## What you ship

By the end of the course you should have, in one repo:

* a **baseline reference** — the unmodified policy at fp16/bf16 on a desktop GPU, with logged actions on a fixed task suite
* at least **three candidate variants** at increasing aggressiveness — e.g. AWQ-INT4 backbone, INT4 + FP8 KV cache + action-chunk caching, distilled student with 1.5-2B-param backbone
* a **parity report** (CSV + a short markdown writeup) covering per-step action MSE, per-axis error, trajectory divergence under closed-loop sim, and success-rate parity vs the reference on a fixed seed list
* a **runtime config** for the deployment target (Jetson AGX Orin or a single L4 / RTX 6000 Ada workstation) with measured P50 / P95 / P99 control-loop latency
* a **CI-style gate script** that another engineer can run on a new checkpoint and get a pass/fail against the parity tolerances you defined

That bundle — optimization recipe + parity harness + reproducible numbers — is the differentiating artifact. It is what a robotics foundation-model team actually hires for.

---

## Exit criteria

You are done with this special course when you can:

* draw the per-tick compute and memory diagram for one specific VLA (OpenVLA, π₀, or GR00T-N) including vision tower, projector, LLM backbone, action head
* name which optimization saves which milliseconds, and which milliseconds are unrecoverable
* defend a tolerance budget for action MSE and success-rate parity to a roboticist who does not trust your optimization
* explain why a passing sim-parity check is necessary but not sufficient for on-robot deployment, and what your canary protocol looks like
* point to one body of work (the repo above) that another team could fork

If you cannot do these things, you have built an optimization demo, not a deployment artifact. Re-run the harness.
