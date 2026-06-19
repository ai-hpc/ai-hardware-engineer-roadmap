# MLSys Engineer — Interview Preparation

<div class="course-identity" markdown="1">
<div class="course-identity__icon">ML</div>
<div markdown="1">
<p class="course-identity__eyebrow">Phase 6 · MLSys Engineer</p>
<p class="course-identity__title">Senior-level interview preparation for ML Systems engineers — inference, kernels, KV cache, speculative decoding, architecture integration.</p>
<p class="course-identity__meta">Target roles: Inference Engineer · ML Systems Engineer · CUDA Kernel Engineer · LLM Serving Engineer · Staff ML Infra Engineer</p>
</div>
</div>

---

## What companies actually test

A senior MLSys interview probes three things:

1. **Systems reasoning** — can you identify the bottleneck (compute vs memory vs latency), back it with numbers, and design around it? Not "what is roofline" but "what does the roofline tell you this specific kernel should do differently."

2. **Implementation depth** — have you actually written a fused kernel, debugged a quantization regression, ported a new architecture? The tell is specificity: real implementations have failure modes, gotchas, and measured numbers. Generic descriptions do not.

3. **Tradeoff ownership** — can you defend a decision? "We used INT8 KV because X but we left V in FP16 for Gemma because Y" is a senior answer. "INT8 saves memory" is not.

---

## Topic map

| Topic area | Questions | File |
|------------|-----------|------|
| KV cache & memory management | PagedAttention, KV optimization, prefix caching | [01 — Inference Systems](01-Inference-Systems-QA.md) |
| Attention kernels | FlashAttention tiling, HBM traffic analysis, fused kernel design | [01 — Inference Systems](01-Inference-Systems-QA.md) |
| Speculative decoding | EAGLE-3 in TRT-LLM, edge scheduling, accept-length economics | [01 — Inference Systems](01-Inference-Systems-QA.md) |
| Latency optimization | TTFT reduction, prefill throughput, prefix caching | [01 — Inference Systems](01-Inference-Systems-QA.md) |
| Architecture integration | Porting a new model to TRT-LLM, divergence checklist | [01 — Inference Systems](01-Inference-Systems-QA.md) |

---

## Self-assessment rubric

Before the interview, score yourself 1–3 on each topic:

```text
1 = I know the concept and can explain it
2 = I've implemented or debugged something in this area
3 = I have shipped this in production and can defend tradeoffs with numbers
```

Target: 3 on your two strongest topics, 2 on the rest, 1 on at most one. Interviewers at senior level expect depth on at least two areas.

---

## How to prepare

**Week 1 (breadth):** Read all 8 Q&As once. Identify which two feel most natural to you — those are your anchor topics. Identify which two feel weakest — those need work.

**Week 2 (depth on weak areas):** For each weak topic, build something: write a toy fused attention kernel, implement a PagedAttention block table in Python, run speculative decoding benchmarks. Talking from code beats talking from notes.

**Week 3 (delivery):** Practice saying answers out loud at interview pace (3–4 minutes per question). Record yourself once. The most common failure at senior level is being technically correct but too slow to hit the key insight — interviewers interrupt before you get there.

**Day before:** Review your own project numbers. The interviewer will ask "and what speedup did you actually measure?" Know your numbers.

---

## Files in this section

| File | Content |
|------|---------|
| [01-Inference-Systems-QA.md](01-Inference-Systems-QA.md) | 8 deep-dive Q&As: PagedAttention, EAGLE-3, KV optimization, fused attention kernel, FlashAttention HBM analysis, speculative decode on Jetson, TTFT on Orin Nano, new architecture integration in TRT-LLM |

---

*Up: [Interview Preparation](../README.md)*
