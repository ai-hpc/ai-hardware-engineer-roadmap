# Agentic Chip Design 2026

<div class="course-identity ai-chip-design" markdown="1">
<div class="course-identity__icon">ACD</div>
<div markdown="1">
<p class="course-identity__eyebrow">Special Course · AI Chip Design × AI Agents</p>
<p class="course-identity__title">Using LLMs and agents across the RTL-to-silicon flow — generation, verification, and agentic EDA.</p>
<p class="course-identity__meta">Artifact: an RTL-generation + self-verification agent · Measure: pass@1, functional coverage, PPA, $ / verified-module</p>
</div>
</div>

**Parent:** [AI Chip Design](../Guide.md) · **Bridges:** [AI Agent Development 2026](../../../Phase%203%20-%20Artificial%20Intelligence/Track%20B%20-%20Agentic%20AI%20and%20ML%20Engineering/3.%20Agentic%20AI%20and%20GenAI/Guide.md) · [AI Inference Engineer 2026](../../Track%20G%20-%20ML%20Systems%20Engineering/AI%20Inference%20Engineer%202026/README.md)

> *Apply the agent harness you built in AI Agent Development to the hardest verification problem there is: designing the chip itself.*

This course applies an agent harness to design tasks with machine-checkable outcomes: RTL behavior, synthesis constraints, and traceable tool results. The purpose is to accelerate a bounded design experiment while preserving an independent specification and checker. It does not depend on a product launch or a claim that a language model can sign off silicon.

**Prerequisites:** comfort with Verilog/RTL, simulation, and basic ASIC flow ([AI Chip Design](../Guide.md)); its lectures 01–05 cover LLVM/MLIR, so use the Phase 1 HDL material for RTL prerequisites, and the agent fundamentals ([AI Agent Development 2026](../../../Phase%203%20-%20Artificial%20Intelligence/Track%20B%20-%20Agentic%20AI%20and%20ML%20Engineering/3.%20Agentic%20AI%20and%20GenAI/Guide.md) Modules 1–3: harness, tools, guardrails, evaluation).

**Role targets:** Design-automation / AI-for-EDA Engineer · RTL Engineer (agent-augmented) · ML-for-hardware researcher.

---

## Why this course, and why now

* **A design-productivity gap.** Transistor budgets and product cadence are outrunning human RTL+verification throughput. **Verification** — not generation — is the dominant cost of a tapeout, and it is exactly the kind of judgment-heavy, unstructured-data work that agents are suited to (see [AI Agent Development · "when to build an agent"](../../../Phase%203%20-%20Artificial%20Intelligence/Track%20B%20-%20Agentic%20AI%20and%20ML%20Engineering/3.%20Agentic%20AI%20and%20GenAI/Lectures/Lecture-03.md)).
* **Precedent exists.** NVIDIA's **ChipNeMo** (2023) showed domain-adapted LLMs doing real chip-design work — engineering Q&A, EDA-script generation, and bug-report summarization — inside a production design org. The field has since exploded into RTL generation, testbench synthesis, and multi-agent EDA flows.
* **A living leaderboard.** The **[Chip-Design-LLM-Zoo](https://iprc-dip.github.io/Chip-Design-LLM-Zoo/)** tracks RTL-generation models against shared benchmarks (VerilogEval, RTLLM, RealBench) by **pass@1 / correct-rate**, distinguishing fine-tuned vs base and open vs closed weights. Use it for discovery, then inspect each paper and the exact benchmark revision. Scores from different task sets, sampling budgets, or checking protocols are not directly comparable.

> **Honesty / currency note.** Model rankings, benchmark scores, and "best RTL LLM" change every few months — always check the live [Zoo leaderboard](https://iprc-dip.github.io/Chip-Design-LLM-Zoo/) and primary papers before quoting a number. This course teaches the *stable* layer: the flow, where agents fit, how to evaluate them, and the silicon-cost discipline that makes verification and human-in-the-loop non-negotiable.

---

## Curriculum

### Module 1 · Foundations

<div class="lecture-map" markdown>

| # | Lecture |
|---|---------|
| [01](Lectures/Lecture-01.md) | Why agents for chip design — the 2026 landscape *(start here)* |
| 02 | The RTL-to-silicon flow and where agents fit (spec → RTL → verify → synth → PPA → physical → signoff) |

</div>

### Module 2 · The core task — RTL generation and its evaluation

<div class="lecture-map" markdown>

| # | Lecture |
|---|---------|
| 03 | LLMs for RTL / Verilog generation — models, fine-tuning, and the Chip-Design-LLM-Zoo |
| 04 | Evaluation discipline — VerilogEval, RTLLM, RealBench; pass@1, correct-rate, syntax vs functionality |

</div>

### Module 3 · Beyond generation

<div class="lecture-map" markdown>

| # | Lecture |
|---|---------|
| 05 | Verification & testbench generation with agents — the real bottleneck, coverage closure |
| 06 | Agentic EDA flows — multi-agent spec→RTL→verify→debug loops, tool-calling into EDA tools (the harness, applied) |
| 07 | PPA optimization and debugging loops with agents |

</div>

### Module 4 · Systems & practice

<div class="lecture-map" markdown>

| # | Lecture |
|---|---------|
| 08 | Serving design agents — latency, token/tool budgets, and cost per successful task |
| 09 | **Capstone** — build an RTL-generation + self-verification agent loop, evaluated on VerilogEval-style tasks |

</div>

> Lectures 02–09 are the curriculum to be built out; Lecture 01 (the opener) is written. The capstone reuses the **genie-claw** harness pattern from AI Agent Development, retargeted: generate RTL → run a simulator/linter as a tool → check against a testbench → iterate within guardrails.

---

## What you ship

A small **agentic RTL pipeline**: an agent that, given a module spec, generates Verilog, drives a simulator/linter as tools, self-checks against a testbench, and iterates — with a measured report on pass@1, functional coverage, and cost per verified module, plus an honest account of where it failed and needed a human.

---

## Ready-to-run study design: a bounded RTL repair experiment

Lecture 01 is available; lectures 02–09 remain planned, so the table above is not a claim of nine completed lectures. Use this workshop to turn the opener into a concrete investigation:

1. Specify a one-entry ready/valid buffer: transfer occurs only on `valid && ready`, accepted input is neither lost nor duplicated, output stays stable while stalled, and reset clears validity. Keep the specification and reference scoreboard outside the agent's editable files.
2. Prepare directed tests for empty/full transitions, consecutive transfers, random backpressure, and reset while full. Add a deliberately incorrect candidate that drops an item under stall; confirm the checker rejects it before evaluating generated RTL.
3. Compare a single proposal with an agent allowed at most three repair rounds. Freeze model revision, prompts, tool versions, timeout, and token budget. Give both arms the same visible tests; use separate held-out tests for scoring.
4. Report first-proposal success separately from final success after repair. Include tool calls, wall time, cost, and failures. An agent editing its own tests until they pass has changed the target rather than verified a design.
5. Synthesize only functionally accepted candidates under identical constraints. Compare area and timing with a hand-written baseline; power needs a stated activity source and is still an estimate unless measured on hardware.

**Handoff to the SoC project:** apply the same protocol to a DMA command FIFO or completion counter. A passing small-module testbench does not establish SoC correctness, clock-domain crossing correctness, or tapeout readiness.

**Review date: 2026-10-14.** This course uses primary papers and reproducible protocols rather than an unverified “latest best model” ranking. [VerilogEval's maintained harness](https://github.com/NVlabs/verilog-eval) distinguishes its original code-completion task from specification-to-RTL evaluation; save the exact task and harness commit with each result.

---

## References

* **Chip-Design-LLM-Zoo** (live RTL-generation leaderboard) — [iprc-dip.github.io/Chip-Design-LLM-Zoo](https://iprc-dip.github.io/Chip-Design-LLM-Zoo/)
* **ChipNeMo: Domain-Adapted LLMs for Chip Design** (NVIDIA) — [arXiv:2311.00176](https://arxiv.org/abs/2311.00176)
* **VerilogEval** — [arXiv:2309.07544](https://arxiv.org/abs/2309.07544) · **RTLLM** — [arXiv:2308.05345](https://arxiv.org/abs/2308.05345)
* Bridge courses: [AI Agent Development 2026](../../../Phase%203%20-%20Artificial%20Intelligence/Track%20B%20-%20Agentic%20AI%20and%20ML%20Engineering/3.%20Agentic%20AI%20and%20GenAI/Guide.md) · [AI Inference Engineer 2026](../../Track%20G%20-%20ML%20Systems%20Engineering/AI%20Inference%20Engineer%202026/README.md)

## Practice and assessment

**Entry diagnostic (prerequisite check):** Explain which design decisions can be proposed automatically and which need objective verification. If this is unfamiliar, use the prerequisite material above or the [roadmap diagnostics](../../../Learning-Guide.md#diagnose-before-reading) before starting the lab.

1. **Baseline lab:** Use a small RTL task with an independent testbench and bounded design constraints.
2. **Guided variation:** Propose a change and test counterexamples before considering PPA improvement.
3. **Independent transfer:** repeat the method on a new input or configuration and explain where your original assumptions stop applying.

**Evidence to submit:** Functional verification, constraint satisfaction, synthesis evidence where available, and reproducible design history. Keep raw results and reproduction steps; distinguish measured results from estimates or work not run.

**Self-check:** Which observation would disprove your explanation of the result? Identify a competing explanation and the smallest experiment that separates them. A passing answer names a concrete predicted difference, not only another tool to try.

**Exit criteria:** reproduce the baseline, demonstrate the negative/variation check, and meet the [shared completion rubric](../../../Learning-Guide.md#completion-rubric). Use the [lesson cycle](../../../Learning-Guide.md#a-repeatable-lesson-cycle) for retrieval practice and spaced review.
