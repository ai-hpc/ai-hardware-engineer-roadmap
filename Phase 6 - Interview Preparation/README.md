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

Use this phase to defend the engineering decisions made in the roadmap. The examples are practice material; do not present them as your own production experience. A clear distinction between estimates, published results, and your measurements is part of a strong technical answer.

---

## Roles Covered

| # | Role | Focus | Status |
|---|------|-------|--------|
| [1](1.%20MLSys%20Engineer/README.md) | **MLSys Engineer** | Inference systems, attention kernels, KV cache, speculative decode, TRT-LLM architecture integration | ✅ Active |
| [2](2.%20Robotics%20AI%20Engineer/README.md) | **Robotics AI Engineer** | Frames, control timing, learning from demonstrations, evaluation, planning, VLA deployment | Worked scenarios and answer guidance |
| [3](3.%20Embedded%20and%20Edge%20AI%20Engineer/README.md) | **Embedded and Edge AI Engineer** | Sensor bandwidth, queues, ownership, quantization, updates, sustained operation | Worked scenarios and answer guidance |
| [4](4.%20Compiler%20and%20Accelerator%20Engineer/README.md) | **Compiler, Kernel, and Accelerator Engineer** | Roofline reasoning, legal fusion, SRAM planning, tile boundaries, runtime interfaces | Worked scenarios and answer guidance |

**Start at your current level.** The three scenario workbooks begin with small
calculations and add follow-up design changes. Use them before the advanced MLSys
questions if you are still building your first project. These exercises are
practice material, not a prediction of a particular company's interview process.

---

## How to use this material

**Don't memorize — reconstruct.** Read an answer once, close it, and say it back in your own words. The goal is to be able to derive the answer from first principles during the interview, not recite it.

**Calibrate level.** Each Q&A is written at senior/staff level. If you're interviewing at L4/E4, you don't need every detail — but you do need the first-principles reasoning and one concrete tradeoff per answer.

**Bring your own evidence.** For each performance claim, bring the workload, hardware/software versions, metric definition, sample count, correctness check, and raw result. The Q&A calculations are illustrative exercises, not measurements to put on a résumé.

**Breadth first, then depth.** Read all questions to calibrate what topics an interviewer cares about, then go deep on the two or three you know best. Be honest about the others.

---

## Practice by engineering responsibility

Interview formats vary by team and role; use the job description and recruiter guidance for the actual process.

| Responsibility | Practice task | Evidence to bring |
|---|---|---|
| Inference/runtime | Explain a latency or memory regression | Trace, workload manifest, quality comparison |
| Kernel/compiler | Derive a legal transformation and its limit | Reference tensors, generated code, profiler counters |
| Embedded/robotics | Follow an observation through driver, runtime, and controller | Timestamped trace, stale-data and restart tests |
| Accelerator/SoC | Defend compute, SRAM, DRAM, and interface choices | Operator inventory, traffic model, simulation/implementation boundary |

---

*Start here: [MLSys Engineer Interview Prep](1.%20MLSys%20Engineer/README.md)*

## Defend the complete model-to-machine chain

Prepare a ten-minute review of one design-studio artifact: task/quality requirement, model/operator inventory, compiler lowering, accelerator traffic, Linux/driver ownership, wireless behavior, and action validity. Mark any unimplemented component as a model or proposal.

Ask a reviewer to double the image rate or halve the memory budget. Revise the design using your measurements and state which conclusions need a new experiment. A complete answer can identify an infeasible requirement instead of hiding it behind peak TOPS.

**Carry forward:** A traceable design defense grounded in the learner’s own evidence. Continue with [the next course](1.%20MLSys%20Engineer/README.md) and reuse that evidence.
