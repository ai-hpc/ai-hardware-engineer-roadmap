# Lecture 01 - Why Agents for Chip Design: The 2026 Landscape

**Course:** [Agentic Chip Design 2026](../Guide.md) | **Previous:** [Course Guide](../Guide.md) | **Next:** Lecture 02 *(curriculum)*

---

This opener frames the course: the design-productivity problem agents are meant to attack, where LLMs genuinely help versus where silicon economics forbid it, the precedent that proved it works, and how to judge any claim in this fast-moving field.

This lecture covers:

1. The design-productivity gap.
2. Where LLMs/agents fit in the flow — and where they must not.
3. The precedent: ChipNeMo.
4. The 2026 task landscape (and the live leaderboard).
5. How to measure — because in hardware, evaluation is everything.
6. Transfer from benchmark tasks to a real accelerator module.
7. The harness connection.

---

## 1. The design-productivity gap

Transistor budgets and product cadence keep climbing; the number of engineers who can write and **verify** RTL does not. The expensive, slow part of building a chip is not typing Verilog — it is **verification**: testbenches, coverage closure, debugging, and signoff routinely consume the majority of a tapeout's engineering effort.

That profile — judgment-heavy, exception-laden, drowning in unstructured logs and specs — is precisely the "when to build an agent" signal from [AI Agent Development · Building Agents I](../../../../Phase%203%20-%20Artificial%20Intelligence/Track%20B%20-%20Agentic%20AI%20and%20ML%20Engineering/3.%20Agentic%20AI%20and%20GenAI/Lectures/Lecture-03.md): complex decisions, brittle hand-maintained rules, heavy unstructured data. Chip design is an unusually good fit *and* an unusually unforgiving one.

---

## 2. Where agents fit — and where they must not

The RTL-to-silicon flow, with the honest agent opportunity at each stage:

<div class="lecture-map" markdown>

| Stage | Agent opportunity | Risk |
|-------|-------------------|------|
| **Spec → RTL** | Draft modules, refactor, translate intent to Verilog | Hallucinated/plausible-but-wrong logic |
| **Verification / testbench** | Generate tests, chase coverage, triage failures | Tests that pass while masking real bugs |
| **Synthesis / lint** | Auto-fix lint, suggest timing fixes | Fixes that change function |
| **PPA optimization** | Propose area/power/timing trade-offs | Local wins, global regressions |
| **Debug** | Summarize logs, localize bugs, propose patches | Confident wrong root-cause |
| **Signoff** | Assist review, never decide | — |

</div>

> **The silicon-cost asymmetry.** A wrong token in a chat reply is free to fix; a wrong token that reaches a **mask set** costs millions and months. This is why every stage above is **human-in-the-loop and evaluation-gated** — the agent proposes, a deterministic checker (simulator, formal tool, linter) and a human dispose. The agent is a throughput multiplier on the design loop, not an autonomous tapeout button.

---

## 3. The precedent: ChipNeMo

NVIDIA's **ChipNeMo** ([arXiv:2311.00176](https://arxiv.org/abs/2311.00176), 2023) is the proof-of-concept the field grew from. By **domain-adapting** general LLMs to a chip-design org's internal data (designs, docs, bug databases) it delivered three concrete applications inside a real design flow:

* an **engineering assistant** chatbot (architecture/design Q&A),
* **EDA-script generation** (drive the tools in their own languages),
* **bug-report summarization and triage.**

The lesson that carries: **domain adaptation + tool access + tight scope** beat a bigger general model used naively. That is the same harness thesis as the agent course, applied to silicon.

---

## 4. The 2026 task landscape

The field now spans generation, verification, and full agentic flows. The most-measured task is **RTL/Verilog generation**, tracked publicly on the **[Chip-Design-LLM-Zoo](https://iprc-dip.github.io/Chip-Design-LLM-Zoo/)** — a live leaderboard ranking models by **pass@1 / correct-rate** across shared benchmarks, distinguishing **fine-tuned vs base** and **open vs closed** weights.

<div class="lecture-map" markdown>

| Task family | What it covers |
|-------------|----------------|
| **RTL generation** | Spec/NL → Verilog modules (the Zoo's focus) |
| **Verification** | Testbench synthesis, coverage closure, assertion generation |
| **Agentic EDA** | Multi-agent spec→RTL→verify→debug loops, tool-calling into EDA tools |
| **Debug / PPA** | Failure triage, lint/timing fixes, area/power/timing trade-offs |

</div>

Use the Zoo to find primary papers, then check task definitions, benchmark revisions, sample counts, and repair budgets. A leaderboard is an index of reported results, not an independent verification of your design.

---

## 5. How to measure — evaluation is everything

In hardware you cannot judge an agent by a nice transcript. The benchmarks that matter:

* **VerilogEval** ([arXiv:2309.07544](https://arxiv.org/abs/2309.07544)) — spec-to-RTL with machine and human-written problem sets; reports **pass@k**.
* **RTLLM** ([arXiv:2308.05345](https://arxiv.org/abs/2308.05345)) — RTL generation scored on **syntax** *and* **functional** correctness.
* **RealBench** — module- and system-level tasks, separating syntax from functionality.

Two hard truths these encode:

1. **Syntax ≠ function.** Code that compiles and lints can still be functionally wrong — and only a testbench or formal check catches it. A high syntax-pass rate with a low functional-pass rate is the field's recurring failure mode.
2. **Match the metric to the protocol.** pass@1 measures a single sampled candidate; pass@k estimates whether at least one of k candidates passes the reference checker. A bounded repair agent additionally needs final success rate, tool calls, latency, and cost under a fixed budget. Report first-attempt and repaired success separately. The checker and selection procedure are part of the result. [VerilogEval harness](https://github.com/NVlabs/verilog-eval).

---

## 6. Transfer from a benchmark to the accelerator

A benchmark test passing means the candidate satisfied that checker on that task. It does not prove freedom from every RTL defect, nor timing closure after physical implementation. Transfer the method to one accelerator interface first: a command FIFO, DMA length counter, or interrupt-status register.

For a command FIFO, enumerate reset, simultaneous push/pop, backpressure, overflow, and counter wrap. Keep a reference queue and held-out stimulus independent of the generator. If you use formal verification, document clock/reset assumptions and the properties proved; an overconstrained environment can exclude the bug you intended to find. Run the bounded experiment in the [course workshop](../Guide.md#ready-to-run-study-design-a-bounded-rtl-repair-experiment) before proposing a larger agentic flow.

---

## 7. The harness connection

Everything you built in **AI Agent Development 2026** applies here, retargeted to silicon:

* the **run loop** → generate RTL, call a tool, check, iterate to an exit condition;
* **tools** → a simulator, a linter, a formal checker, a synthesis report parser;
* **guardrails** → never let unverified RTL advance; gate high-cost actions on a human;
* **evaluation** → pass@1 / functional coverage, not vibes.

The course capstone is exactly this: a **genie-claw**-style harness that generates a Verilog module, drives a simulator/testbench as tools, and iterates within guardrails — scored on VerilogEval-style tasks.

---

## Key takeaways

* The chip-design bottleneck is **verification throughput**, and that judgment-heavy, unstructured work is a strong agent fit — under a brutal silicon-cost asymmetry that mandates **human-in-the-loop + deterministic checkers**.
* **ChipNeMo** proved domain-adapted LLMs + tools + scope work inside a real design org.
* Judge everything by **pass@1 and functional correctness** (syntax is not function); use the live **Chip-Design-LLM-Zoo** rather than stale rankings.
* Small verified interfaces provide a useful first target; system integration and physical signoff require additional evidence.

---

## Self-check

1. Why is verification — not RTL generation — the part of the flow where agents add the most value?
2. Explain the silicon-cost asymmetry and what it implies for autonomy in this domain.
3. A model scores 90% syntax-pass but 45% functional-pass on RTLLM. What does that tell you, and what catches the gap?
4. What is the difference between pass@1, pass@5, and success after three repair rounds? What budget and checker details make the comparison fair?
5. Name the three ChipNeMo applications and the general lesson they teach about deploying LLMs in a specialized domain.

---

## References

* **Chip-Design-LLM-Zoo** (live leaderboard) — [iprc-dip.github.io/Chip-Design-LLM-Zoo](https://iprc-dip.github.io/Chip-Design-LLM-Zoo/)
* **ChipNeMo** — [arXiv:2311.00176](https://arxiv.org/abs/2311.00176)
* **VerilogEval** — [arXiv:2309.07544](https://arxiv.org/abs/2309.07544) · **RTLLM** — [arXiv:2308.05345](https://arxiv.org/abs/2308.05345)
* Bridge: [AI Agent Development 2026 — when to build an agent](../../../../Phase%203%20-%20Artificial%20Intelligence/Track%20B%20-%20Agentic%20AI%20and%20ML%20Engineering/3.%20Agentic%20AI%20and%20GenAI/Lectures/Lecture-03.md)

---

*Next: Lecture 02 — The RTL-to-silicon flow and where agents fit (curriculum).*
