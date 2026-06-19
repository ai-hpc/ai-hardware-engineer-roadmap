# Phase 6 — Interview Preparation

<div class="course-identity" markdown="1">
<div class="course-identity__icon">INT</div>
<div markdown="1">
<p class="course-identity__eyebrow">Phase 6 · Career Progression</p>
<p class="course-identity__title">Role-specific interview preparation for AI hardware, inference, and systems engineers — real Q&A at senior level, not trivia.</p>
<p class="course-identity__meta">Artifact: confident, specific answers grounded in hands-on experience · Measure: can you defend the tradeoffs, not just recall the definition</p>
</div>
</div>

> *The difference between a pass and a hire at senior level is not knowing what FlashAttention is — it's explaining which bottleneck it moves, which tradeoff it makes, and what you hit when you actually implemented it.*

This phase is not a flashcard bank. Each answer here is written at the level of someone who has shipped the system: specific numbers, real failure modes, tradeoffs defended, and the insight you only get by doing the work. That is exactly what a staff/senior interview expects.

---

## Roles Covered

| # | Role | Focus | Status |
|---|------|-------|--------|
| [1](1.%20MLSys%20Engineer/README.md) | **MLSys Engineer** | Inference systems, attention kernels, KV cache, speculative decode, TRT-LLM architecture integration | ✅ Active |

More roles coming: Edge AI Engineer, CUDA Kernel Engineer, AI Chip Architect, Robotics AI Engineer.

---

## How to use this material

**Don't memorize — reconstruct.** Read an answer once, close it, and say it back in your own words. The goal is to be able to derive the answer from first principles during the interview, not recite it.

**Calibrate level.** Each Q&A is written at senior/staff level. If you're interviewing at L4/E4, you don't need every detail — but you do need the first-principles reasoning and one concrete tradeoff per answer.

**Extend with your own numbers.** The answers reference specific measurements (e.g., 152→620 tok/s prefill, 1289ms→348ms with double-buffer). Swap those for your own project numbers. Interviewers can tell generic answers from ones grounded in real work.

**Breadth first, then depth.** Read all questions to calibrate what topics an interviewer cares about, then go deep on the two or three you know best. Be honest about the others.

---

## Interview format by company type

| Company type | Typical format | What they probe |
|---|---|---|
| Inference startup (Groq, Cerebras, Together, Fireworks) | Deep system design, 1 or 2 coding, no LC grind | Bottleneck analysis, kernel writing, latency math |
| NVIDIA (NIM, TRT-LLM, cuBLAS) | Mix of design + coding + HW questions | PTX, occupancy, plugin APIs, TRT engine lifecycle |
| Google (XLA, TPU, Google AI) | Design heavy, some coding, ML breadth | Distributed systems, compiler IR, quantization theory |
| Meta (PyTorch, AITER, infra) | Coding + design + infra | Python extension APIs, CUDA/Triton, distributed training |
| Edge / Jetson (NVIDIA EGX, Qualcomm, Apple) | End-to-end system design | Memory budget, latency budget, runtime portability |
| Big tech ML infra | LC coding + broad system design | Less HW-specific, more distributed, scheduling, cost |

---

*Start here: [MLSys Engineer Interview Prep](1.%20MLSys%20Engineer/README.md)*
