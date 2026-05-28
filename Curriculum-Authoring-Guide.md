# Curriculum Authoring Guide

> *How to add or expand roadmap content without diluting the project.*

This repository is strongest when every new page does three things well:

- teaches a real systems or hardware concept
- produces a visible engineering artifact
- connects cleanly to the 8-layer stack and downstream roles

Use this guide when adding modules, deep dives, labs, or project pages.

---

## Core Rules

### 1. Stay hardware-first

This is not a generic AI course.

New content should answer at least one of these:

- What workload does the hardware need to run?
- How does software map onto hardware resources?
- How is performance, power, memory, or deployment affected?
- What role in the AI chip stack does this teach?

If a topic is interesting but does not sharpen hardware intuition, it probably does not belong here.

### 2. Prefer build-first content

Avoid pages that are only reading lists or concept summaries.

Every substantial guide should push the learner toward:

- code
- profiling
- debugging
- measurement
- deployment
- architecture reasoning

### 3. Produce artifacts, not just notes

Each module should suggest one or more artifacts that another engineer could inspect:

- benchmark table
- profiling trace
- board bring-up checklist
- device-tree patch
- CUDA kernel
- MLIR pass demo
- FPGA timing report
- system diagram

### 4. Keep the role mapping explicit

Every guide should make it easy to answer:

- Which layer does this belong to?
- What role does this help with?
- What phase or track should come next?

### 5. Define what "done" means

Every substantial course page should make completion observable. A learner should not finish a module by saying "I read it." They should finish with evidence:

- code that runs
- a build log
- a profiler trace
- a benchmark table
- a debug capture
- a design report
- a deployment checklist
- a system diagram tied to measurements

If a page cannot name the artifact, the page is probably still only a topic list.

---

## Course Quality Standard

Use this standard when reviewing old modules or adding new ones.

| Standard | Weak page | Strong page |
|----------|-----------|-------------|
| Purpose | Lists topics | States the engineering problem and why it matters |
| Structure | Long resource dump | Guides the learner through build, use, measure, ship |
| Hardware relevance | Implied | Explicitly connects workload, runtime, memory, power, timing, or deployment to hardware |
| Practice | "Try X" | Defines a concrete lab with commands, inputs, and expected outputs |
| Measurement | Optional | Names the metrics that prove progress |
| Completion | Vague | Defines an artifact another engineer can inspect |
| Navigation | Isolated | Links prerequisites and next modules |
| Role mapping | Generic | Names roles and daily work this prepares for |

### Minimum viable course page

Every substantial `Guide.md` should include:

- one-sentence purpose
- layer mapping
- prerequisites
- what comes after
- role targets
- why the module exists
- course outcomes
- unit map or learning path
- build/lab work
- metrics to collect
- final artifact
- exit criteria

Short reference pages can be lighter, but they should still state why the page exists and what the learner should do with it.

### Artifact ladder

Learners should move from simple artifacts to portfolio-grade artifacts:

| Level | Artifact quality |
|-------|------------------|
| 1 | Notes, diagrams, and command logs |
| 2 | Working code, RTL, or configuration |
| 3 | Reproducible build/test/benchmark script |
| 4 | Measurement report with raw data and interpretation |
| 5 | Reusable subsystem, lab, or case study another engineer can run |

Most modules should target Level 3 or Level 4. Capstones should target Level 5.

---

## Standard Lesson Template

For new guides, use this structure unless there is a strong reason not to:

## Why it matters

State the practical problem and the hardware relevance.

## Mental model

Explain the core concept in plain language before implementation details.

## Build it

Implement the concept from scratch or close to the metal.

## Use it in the real stack

Show the production version with real tools, frameworks, drivers, or boards.

## Measure it

Ask for concrete metrics: latency, throughput, occupancy, bandwidth, power, accuracy, utilization, memory, boot time, thermal behavior.

## Ship it

Define the final artifact that proves the learner completed the unit.

## Exit criteria

List what the learner must be able to explain, build, measure, or debug before moving on.

---

## What Good Module Outcomes Look Like

| Area | Weak outcome | Strong outcome |
|------|--------------|----------------|
| CUDA | "Read about warps" | Nsight report comparing coalesced vs uncoalesced kernels |
| Embedded Linux | "Learn Yocto" | Reproducible image build + boot log + package diff |
| Jetson | "Try TensorRT" | FP16 vs INT8 benchmark with latency, RAM, power |
| FPGA | "Study HLS" | Synthesized kernel with utilization and timing results |
| ML Compiler | "Understand MLIR" | Minimal pass or backend lowering demo |

---

## Page Header Checklist

At the top of a substantial guide, include:

- one-sentence purpose
- layer mapping
- prerequisites
- what comes after
- role targets if the page is specialized

This keeps the roadmap navigable for learners entering from different backgrounds.

---

## Writing Style

- Write like an engineer teaching another engineer.
- Prefer concrete claims over inspirational language.
- Use tables for comparison, not for decoration.
- Explain tradeoffs, not just definitions.
- Show where things fail in practice.
- Keep resource lists curated; do not dump large unsorted link lists.

---

## Depth Guidelines

Use this rough split:

- Overview guides: orientation, role mapping, module structure, project ideas
- Sub-guides: implementation detail, performance mechanics, debugging workflow
- Labs/projects: step-by-step execution and measurable outputs

Do not overload overview guides with every implementation detail. Link downward instead.

---

## Contribution Checklist

Before merging new curriculum content, verify:

- The page clearly belongs in the roadmap
- Hardware relevance is explicit
- The learner is asked to build something
- Measurement is part of completion
- A final artifact is defined
- Links and prerequisites are correct
- The content does not duplicate an existing page unnecessarily

---

## Recommended Companion Files

When a module becomes substantial, consider adding:

- a lab page
- an artifact tracker
- a benchmark template
- a worked project or case study

---

## Related Pages

- [Home / Roadmap Overview](README.md)
