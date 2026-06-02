# Lecture 1: VLA Optimization for Real-Time Control

## Overview

A Vision-Language-Action policy is, from the runtime's point of view, an unusually inconvenient transformer:

* it eats **N camera streams** every control tick (not a text prompt typed once)
* it runs **inference end-to-end inside a hard real-time loop** (not interactively)
* it emits **a small burst of action tokens or denoising steps**, not a long generation
* it has to **share a Jetson or workstation GPU** with perception, ROS 2, costmaps, and a teleop UI

The job of this lecture is to convert a published **VLA checkpoint** — OpenVLA-7B, π₀ (3.3B), GR00T N1.5 (2-3B class), or a Qwen2-VL-class custom policy — into something that **closes the control loop on real hardware** without changing what the robot does. The companion measurement framework lives in [Lecture 2](Lecture-02.md); we will assume it exists and validates every step.

By the end you should be able to:

* draw the per-tick FLOP and bandwidth budget for a specific VLA on a specific target (e.g. OpenVLA-7B at 224×224 × 1 camera on Jetson AGX Orin 64 GB)
* pick the right quantization recipe for the LLM backbone, the vision tower, and the action head — these are usually three different decisions
* explain why the vision tower is often the worst offender once you have already quantized the LLM
* implement action-chunk caching and KV-cache reuse for VLAs that emit short autoregressive bursts
* know when to walk away from quantization and distill to a smaller backbone instead

---

## 1. What runs every control tick

Strip the marketing away and a VLA control tick is:

```text
                  ┌──────────────────────────────────────────┐
   sensors  ──►   │  preprocess: resize, normalize, layout    │  ~1-5 ms
                  └──────────────┬───────────────────────────┘
                                 ▼
                  ┌──────────────────────────────────────────┐
   vision tower   │  SigLIP / DINOv2 / EVA — frozen ViT       │  10-80 ms
                  │  N_cameras × 1-2 image tokens stream      │
                  └──────────────┬───────────────────────────┘
                                 ▼
                  ┌──────────────────────────────────────────┐
   projector      │  MLP or Perceiver resampler               │  <1 ms
                  └──────────────┬───────────────────────────┘
                                 ▼
                  ┌──────────────────────────────────────────┐
   LLM backbone   │  prefill (256-2048 tokens) + decode       │  20-300 ms
                  │  Llama-7B / Gemma-2B / Qwen2-VL class     │
                  └──────────────┬───────────────────────────┘
                                 ▼
                  ┌──────────────────────────────────────────┐
   action head    │  token detok / MLP regression / diffusion │  1-30 ms
                  └──────────────┬───────────────────────────┘
                                 ▼
                            7-DoF action  ──►  ROS 2 controller
```

Three things to internalize:

1. **Prefill dominates if the prompt is large.** Some OpenX-style policies serialize task description + N camera views + proprioceptive state into >1500 tokens of prefix every tick. Prefill becomes a **compute-bound matmul**; decode is a different, **bandwidth-bound** problem.
2. **Decode is short.** OpenVLA emits 7 discretized action tokens. π₀ runs a 10-step flow-matching head over a 50-step action chunk. RT-2-style emits a few action tokens per arm. You almost never decode more than ~32 tokens, so per-token decode tricks that help long-form LLMs (continuous batching, paged KV) help less here.
3. **Vision tower is constant per tick and not text-shaped.** It does not benefit from KV cache. It is the most common **surprise bottleneck** after the LLM is quantized.

### 1.1 Concrete budget for OpenVLA-7B on Jetson AGX Orin (64 GB, MAXN)

Reference numbers, rounded, single 224×224 camera, prompt ~280 tokens, 7 action tokens, fp16:

| Stage | ms | Memory traffic | Notes |
|-------|----|----------------|-------|
| Preprocess (CPU) | 2 | — | resize + normalize, single camera |
| Vision tower (SigLIP-So400m, ~400M) | 25 | weights ~800 MB | dominated by attention matmuls, batch=1 |
| Projector | <1 | small | negligible |
| LLM prefill (Llama-2-7B, 280 tok) | 110 | weights ~13 GB | compute-bound at this prompt length |
| LLM decode (7 tokens) | 60 | KV per tick ~0 (reset) | bandwidth-bound, ~8.5 ms/token |
| Action detokenize | <1 | — | 256-bin lookup per axis |
| **Total** | **~200** | | misses 50 ms control budget by 4× |

This is the table you write *before* you optimize anything. Every later number gets compared to this row.

---

## 2. The optimization ladder

Cheapest to most invasive, with the action-parity harness as the gate at every rung:

| Rung | Technique | Typical speedup on Jetson AGX | Parity risk |
|------|-----------|-------------------------------|-------------|
| 0 | bf16 → fp16, fuse RMSNorm, kernel selection | 1.1-1.3× | none if numerics validated |
| 1 | Static KV-cache + CUDA Graphs for decode | 1.2-1.5× on decode only | none, deterministic |
| 2 | Vision tower → TensorRT FP16 / FP8 | 2-4× on vision stage | low; ViT is robust to FP8 |
| 3 | LLM backbone weight-only INT8 / INT4 (AWQ / GPTQ) | 2-3× on prefill + decode | medium; needs action-MSE check |
| 4 | FP8 KV cache, FP8 attention | 1.2-1.5× on decode | medium-high; affects long action chunks more |
| 5 | Action-chunk caching / temporal reuse | up to 5-10× wall-clock effective | task-dependent |
| 6 | Speculative action decoding (small draft model) | 1.5-2× on decode | low if draft is well-distilled |
| 7 | Distill to a smaller backbone (e.g. 7B → 2-3B class) | 2-3× across the board | high; full re-evaluation required |
| 8 | Architecture surgery (drop a camera, smaller image, fewer action steps) | 1.5-3× | high; changes the policy class |

The rule: take the **cheapest rung** that still meets the parity bar from [Lecture 2](Lecture-02.md), then stop. Most teams overshoot and ship a policy that **drifts at the third rollout** in a row.

---

## 3. Rung 0-1: free wins (numerics + scheduling)

### 3.1 Precision selection

* The vision tower is almost always shipped in bf16 / fp16. Stay in fp16 on Jetson — bf16 lacks tensor-core acceleration on Ampere / older Orin SKUs.
* The LLM backbone in published VLAs is usually bf16. Convert weights once, offline. Validate with the parity harness on at least one episode before going further.
* The action head, if it is a regression MLP or a diffusion head, is **the place you keep highest precision**. Errors here are not absorbed by softmax; they **hit the actuator directly**. Default fp16; only drop further if Lecture 2's per-axis error budget allows it.

### 3.2 Static KV cache + CUDA Graphs

Because decode bursts are short and the action-token count is fixed (OpenVLA always emits 7), you can:

* preallocate the KV buffer for `max_prefix + max_action_tokens`
* capture the decode step as a CUDA Graph after the first warmup tick
* avoid the per-token Python overhead that dominates a 7-token decode on Jetson

This is pure scheduling. The output bits are unchanged. Run a bit-exactness check against the un-graphed version on the same prompt before moving on.

### 3.3 Pinned memory + zero-copy from the camera node

In ROS 2 land, the slowest path is often `sensor_msgs/Image` → host buffer → CUDA copy → ViT input. Use `image_transport` with a CUDA-mapped buffer pool, or NITROS / Isaac ROS GXF if you are on a Jetson with NITROS support. Saves 3-8 ms per tick at zero numerical cost.

---

## 4. Rung 2: TensorRT-ify the vision tower

Once the LLM is small or quantized, the **ViT becomes the visible cost**. SigLIP-So400m at 384×384 × 2 cameras can be 60-90 ms on Jetson AGX Orin before this work.

Recipe:

1. Export the vision encoder to ONNX with fixed input shape (batch, N_cameras, 3, H, W). Dynamic shapes hurt TRT.
2. Build a TensorRT engine in FP16; on Orin with FP8 support, try FP8 with per-tensor scales calibrated on ~200 task-typical images.
3. Replace the PyTorch encoder with a thin TRT wrapper that returns the projector input on the same CUDA stream.
4. Validate the projector input embedding cosine similarity vs the PyTorch reference: target ≥ 0.999 mean, ≥ 0.99 worst-case.

If similarity drops below those thresholds, do not proceed — quantizing the LLM on top of a degraded vision embedding silently destroys task success without any one stage looking wrong.

---

## 5. Rung 3: backbone weight-only quantization

Same recipe family as the [Qwen3-4B Q4 lecture](../../3.%20Edge%20AI/Qwen%20Inference%20Optimization/Lecture-02.md), applied to the LLM half of the VLA. Differences worth knowing:

* **Calibration data must be VLA-shaped, not text-shaped.** A WikiText calibration set will produce a backbone that hallucinates plausible English completions and fails on action tokens. Use 256-1024 trajectory prompts from your task suite as calibration input.
* **Action-token logits are concentrated on a small subvocab** (typically 256 bins × 7 axes for OpenVLA). Per-channel weight quantization can over-suppress these tokens. Validate with the harness on action-token logit KL divergence in addition to weight error.
* **Do not quantize the LM head if action tokens come from it.** The LM head matmul is small, so the throughput cost of keeping it FP16 is negligible, and the parity payoff is large.

AWQ and GPTQ both work; AWQ tends to preserve action-token logits better in practice because it is activation-aware. INT4 group-128 is the usual sweet spot. INT3 is rarely worth the parity hit on 7B-class backbones.

---

## 6. Rung 4: FP8 KV cache and FP8 attention

Worthwhile only if your action chunk is long enough that decode-side bandwidth matters — π₀-style flow-matching policies that re-attend over a 50-step chunk benefit; OpenVLA's 7-token burst does not noticeably.

If you do enable it:

* keep the KV scales **per-head** rather than per-tensor — VLAs have heads that specialize in the visual-token region of the prefix, and per-tensor scales overflow them
* re-run the per-step action MSE check; FP8 KV most often shows up as a drift on the *later* tokens of a chunk, not the first

---

## 7. Rung 5: action-chunk caching and temporal reuse

This is the rung that is unique to VLAs and gets neglected because it looks like cheating.

The observation: a robot's scene **does not change at the rate of the camera frame**. If the policy emits an N-step action chunk (π₀'s 50-step horizon, ALOHA-style chunked imitation, RT-2 with action chunking), you can:

* run the full VLA every K-th tick (e.g. every 5th, at 10 Hz instead of 50 Hz)
* execute the cached action chunk in between
* re-run the policy early if a closed-loop monitor (proprio mismatch, force spike, vision delta) triggers

Effective control rate is unchanged; effective compute drops by K. This is the difference between "runs on a Jetson at 4 Hz" and "runs on a Jetson at 30 Hz effective."

The hazard: the parity harness in Lecture 2 must include a **chunk-cache rollout mode**, not just per-tick parity. A policy that is bit-exact per tick can still fail because the cached actions become stale during a fast-moving sub-trajectory (e.g. final grasp closure). The harness's success-rate-parity metric is what tells you the safe K per task.

---

## 8. Rung 6: speculative action decoding

Standard speculative decoding adapted to action tokens:

* a tiny draft model (the action head from a distilled smaller VLA, or a 0.5B-class LLM) proposes the next k action tokens
* the full VLA verifies in parallel and accepts the longest matching prefix

For 7-token OpenVLA bursts, the wins are modest (~1.3-1.7× decode). For policies that emit longer chunks, or for bimanual 14-DoF action sequences, the wins are real (~2×).

The harness must validate that the verified outputs are bit-exact to the un-speculated decoded outputs — this is the one optimization in this lecture where you can prove parity at the *logit* level, not just the action level.

---

## 9. Rung 7: distillation when quantization runs out

If you have exhausted rungs 0-6 and still miss the control budget, the next step is to **swap the backbone**. The pattern:

* keep the vision tower frozen (it is doing the heavy perception work)
* swap the 7B LLM for a 2-3B class one (Gemma-2-2B, Qwen2.5-3B, Llama-3.2-3B)
* distill on the original policy's action distribution using your trajectory dataset
* re-train the action head from scratch on the new backbone's hidden states

This is a different project — it is now policy training, not policy deployment. But the *deployment side* still uses every rung above on the new student. The parity harness in Lecture 2 is the same harness, with the un-distilled teacher as the reference.

---

## 10. Rung 8: architecture surgery (last resort)

Sometimes you have to change the policy class to fit the robot:

* drop a camera (3 → 2 or 2 → 1) — measure success-rate impact per task category
* downscale images (384 → 224 or 224 → 160) — usually safe for short-range manipulation, fatal for fine grasping
* shorten the action chunk (50 → 20 steps) — interacts with rung 5; the harness will tell you when this breaks
* drop a modality (proprioception text, force tokens) — almost never safe; revisit only if you have replaced it with a faster encoding

Every one of these requires the harness to re-baseline and re-publish parity numbers against the un-surgery'd policy. There is no other way to know.

---

## 11. Reference deployment paths

Two targets covered explicitly because most readers will choose one of these:

### 11.1 Jetson AGX Orin 64 GB (edge)

* Backbone: AWQ-INT4 (group-128), LM head FP16
* Vision tower: TensorRT FP16 (FP8 if 6.x JetPack with FP8 path is available)
* Action head: FP16
* Runtime: vLLM build for ARM64 + TRT for vision, or llama.cpp CUDA backend if you want a single binary
* Control loop wrapper: ROS 2 node that publishes `JointTrajectory` at 30-50 Hz; policy itself runs at chunk-cache K=5
* Memory budget: ~14 GB policy, ~6 GB perception, ~4 GB ROS 2 + costmap + nav

### 11.2 Single-GPU workstation (RTX 6000 Ada / L4 / 4090)

* Backbone: FP8 (if Ada / Hopper) or AWQ-INT4
* Vision tower: TRT FP8 or torch.compile + flash-attn
* Action head: FP16 / BF16
* Runtime: vLLM, SGLang, or TensorRT-LLM
* Use case: lab robot, teleop assist, sim-in-the-loop development; latency budget is looser (~30 ms) but determinism still matters

---

## 12. Lab — Build a baseline + 3 candidates

You will need the lab data and the parity harness from [Lecture 2](Lecture-02.md) to grade your work. Set up the harness first, then do this lab.

1. **Reproduce the published checkpoint.** Pick OpenVLA-7B or π₀-base. Run 50 episodes on LIBERO-Spatial or LIBERO-Object at the published evaluation protocol. Record actions, success/failure, and wall-clock per tick. This is your reference row.
2. **Candidate A — free wins.** fp16 + static KV + CUDA Graphs + pinned camera input. No quantization. Re-run the same 50 episodes with a fixed seed list. Diff against reference.
3. **Candidate B — quantized backbone.** AWQ-INT4 backbone, TRT FP16 vision tower, fp16 head. Same 50 episodes.
4. **Candidate C — quantized + chunk caching.** Same as B, plus K=5 chunk cache (if the policy supports chunked actions) or speculative decoding (if it does not). Same 50 episodes.
5. **Produce the parity table.** Per-step action MSE, per-axis max error, success-rate parity, P50/P95/P99 control-loop latency. This goes in your repo as `parity_report.md`.

Pass criterion for the lab: at least one candidate beats the control-loop deadline on your target hardware **and** stays inside the parity tolerances you set in Lecture 2. If none do, the answer is either to distill or to revisit the budget — not to ship.

---

## Self-check

1. You quantized the LLM backbone and the vision tower and decode latency on a 7-token VLA dropped from 60 ms to 22 ms, but task success-rate fell from 78% to 41%. Which two stages do you investigate first, and which metric in Lecture 2's harness would have caught this before deployment?
2. A teammate proposes FP8 KV cache "because it worked for our 7B chat model." Why is the parity risk different for a π₀ flow-matching policy than for OpenVLA-7B, and what specifically does the parity harness need to add to evaluate it fairly?
3. You have a 200 ms-per-tick policy and a 50 ms control budget. Action-chunk caching with K=5 gets you to 40 ms *effective*. Why is "effective" doing a lot of work in that sentence, and what is the harness experiment that decides whether K=5 is actually safe for this task?
4. Why is it usually safe to TRT the vision tower in FP8 but dangerous to quantize the action head to INT8?

---

## References

* OpenVLA: An Open-Source Vision-Language-Action Model — [project page](https://openvla.github.io/), [paper](https://arxiv.org/abs/2406.09246)
* π₀ / π0.5 (Physical Intelligence) — [tech report](https://www.physicalintelligence.company/blog/pi0)
* NVIDIA GR00T N1 / N1.5 — [model card](https://huggingface.co/nvidia/GR00T-N1-2B), [Isaac Lab integration](https://docs.omniverse.nvidia.com/isaacsim/latest/isaac_lab_tutorials/index.html)
* Open X-Embodiment dataset — [project](https://robotics-transformer-x.github.io/)
* AWQ: Activation-aware Weight Quantization — [paper](https://arxiv.org/abs/2306.00978)
* RT-2 / RT-2-X — [paper](https://arxiv.org/abs/2307.15818)
* LIBERO benchmark — [paper](https://arxiv.org/abs/2306.03310), [code](https://github.com/Lifelong-Robot-Learning/LIBERO)
* TensorRT for ViTs — [NVIDIA technical blog](https://developer.nvidia.com/blog/tag/tensorrt/)
* CUDA Graphs for decode — covered in [Phase 5 — CUDA Advanced Optimization, Lecture 01](../../1.%20GPU%20Infrastructure/Nvidia%20GPU/HPC%20Setup/CUDA-Advanced-Optimization/01-CUDA-Graphs.md)

---

## Next in this special course

* Next: [Lecture 2 — The Action-Parity Harness](Lecture-02.md)
* Back: [VLA Optimization and Action-Parity Harness — Overview](README.md)
