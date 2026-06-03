# Agentic Chip Design 2026

<div class="course-identity ai-chip-design" markdown="1">
<div class="course-identity__icon">ACD</div>
<div markdown="1">
<p class="course-identity__eyebrow">Special Course · AI Chip Design × AI Agents</p>
<p class="course-identity__title">Using LLMs and agents across the RTL-to-silicon flow — generation, verification, and agentic EDA.</p>
<p class="course-identity__meta">Artifact: an RTL-generation + self-verification agent · Measure: pass@1, functional coverage, PPA, $ / verified-module</p>
</div>
</div>

**Parent:** [AI Chip Design](../Guide.md) · **Bridges:** [AI Agent Development 2026](../../../Phase%203%20-%20Artificial%20Intelligence/Track%20B%20-%20Agentic%20AI%20and%20ML%20Engineering/3.%20Agentic%20AI%20and%20GenAI/Guide.md) · [AI Inference Engineer 2026](../../7.%20ML%20Systems%20Engineering/AI%20Inference%20Engineer%202026/README.md)

> *Apply the agent harness you built in AI Agent Development to the hardest verification problem there is: designing the chip itself.*

This course sits at the intersection of two tracks. From **AI Agent Development** it takes the harness — run loops, tools, guardrails, evaluation. From **AI Chip Design** it takes the target — RTL, verification, synthesis, PPA, and the unforgiving economics of silicon. The thesis: as the hardware cadence accelerates (NVIDIA put **Vera Rubin into full production** at Computex / GTC Taipei 2026, with a Grace-Blackwell rack now assembled in ~5 minutes), the bottleneck moves upstream to **design and verification throughput** — exactly where LLM agents are starting to help.

**Prerequisites:** comfort with Verilog/RTL and the basic ASIC flow ([AI Chip Design](../Guide.md) lectures 01–05), and the agent fundamentals ([AI Agent Development 2026](../../../Phase%203%20-%20Artificial%20Intelligence/Track%20B%20-%20Agentic%20AI%20and%20ML%20Engineering/3.%20Agentic%20AI%20and%20GenAI/Guide.md) Modules 1–3: harness, tools, guardrails, evaluation).

**Role targets:** Design-automation / AI-for-EDA Engineer · RTL Engineer (agent-augmented) · ML-for-hardware researcher.

---

## Why this course, and why now

* **A design-productivity gap.** Transistor budgets and product cadence are outrunning human RTL+verification throughput. **Verification** — not generation — is the dominant cost of a tapeout, and it is exactly the kind of judgment-heavy, unstructured-data work that agents are suited to (see [AI Agent Development · "when to build an agent"](../../../Phase%203%20-%20Artificial%20Intelligence/Track%20B%20-%20Agentic%20AI%20and%20ML%20Engineering/3.%20Agentic%20AI%20and%20GenAI/Lectures/Lecture-03.md)).
* **Precedent exists.** NVIDIA's **ChipNeMo** (2023) showed domain-adapted LLMs doing real chip-design work — engineering Q&A, EDA-script generation, and bug-report summarization — inside a production design org. The field has since exploded into RTL generation, testbench synthesis, and multi-agent EDA flows.
* **A living leaderboard.** The **[Chip-Design-LLM-Zoo](https://iprc-dip.github.io/Chip-Design-LLM-Zoo/)** tracks RTL-generation models against shared benchmarks (VerilogEval, RTLLM, RealBench) by **pass@1 / correct-rate**, distinguishing fine-tuned vs base and open vs closed weights. We use it as the *truth-at-time-of-reading* benchmark, the way the inference course uses live benchmark dashboards.
* **The 2026 hardware backdrop.** Computex / GTC Taipei 2026 (June 1–5) underscored the cadence: **Vera Rubin in full production**, **Cosmos 3** (a fully-open omnimodel mapping text/image/video/audio → action), and a broad push of AI into every layer of the stack. Faster silicon ⇒ more pressure on the *design* loop.

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
| 08 | Serving design agents — inference cost, Blackwell/Vera Rubin context (bridge to AI Inference Engineer 2026) |
| 09 | **Capstone** — build an RTL-generation + self-verification agent loop, evaluated on VerilogEval-style tasks |

</div>

> Lectures 02–09 are the curriculum to be built out; Lecture 01 (the opener) is written. The capstone reuses the **genie-claw** harness pattern from AI Agent Development, retargeted: generate RTL → run a simulator/linter as a tool → check against a testbench → iterate within guardrails.

---

## What you ship

A small **agentic RTL pipeline**: an agent that, given a module spec, generates Verilog, drives a simulator/linter as tools, self-checks against a testbench, and iterates — with a measured report on pass@1, functional coverage, and cost per verified module, plus an honest account of where it failed and needed a human.

---

## Current as of 2026-06

Anchored on the Chip-Design-LLM-Zoo leaderboard (VerilogEval / RTLLM / RealBench) and the Computex / GTC Taipei 2026 hardware backdrop (Vera Rubin in production, Cosmos 3). RTL-LLM rankings move fast — refresh against the live Zoo and primary papers. Refresh when a new RTL-generation SOTA or agentic-EDA system materially shifts the leaderboard.

---

## References

* **Chip-Design-LLM-Zoo** (live RTL-generation leaderboard) — [iprc-dip.github.io/Chip-Design-LLM-Zoo](https://iprc-dip.github.io/Chip-Design-LLM-Zoo/)
* **ChipNeMo: Domain-Adapted LLMs for Chip Design** (NVIDIA) — [arXiv:2311.00176](https://arxiv.org/abs/2311.00176)
* **VerilogEval** — [arXiv:2309.07544](https://arxiv.org/abs/2309.07544) · **RTLLM** — [arXiv:2308.05345](https://arxiv.org/abs/2308.05345)
* NVIDIA Computex / GTC Taipei 2026 (Vera Rubin production, Cosmos 3) — [ServeTheHome live coverage](https://www.servethehome.com/nvidia-computex-2026-keynote-live-coverage/) · [NVIDIA GTC Taipei](https://www.nvidia.com/en-tw/gtc/taipei/computex/)
* Bridge courses: [AI Agent Development 2026](../../../Phase%203%20-%20Artificial%20Intelligence/Track%20B%20-%20Agentic%20AI%20and%20ML%20Engineering/3.%20Agentic%20AI%20and%20GenAI/Guide.md) · [AI Inference Engineer 2026](../../7.%20ML%20Systems%20Engineering/AI%20Inference%20Engineer%202026/README.md)
