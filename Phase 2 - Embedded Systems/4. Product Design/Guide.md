# Product Design for Embedded Systems

<div class="course-identity product-design" markdown="1">
<div class="course-identity__icon">PRD</div>
<div markdown="1">
<p class="course-identity__eyebrow">Module 4 · Product Design</p>
<p class="course-identity__title">Turn working hardware into a device with credible setup, controls, trust, and deployment boundaries.</p>
<p class="course-identity__meta">Artifact: V1 product brief · Measure: setup friction, reliability, trust cues</p>
</div>
</div>


A structured mini-course for engineers who can already reason about boards, firmware, Linux, and connectivity, but want to learn how those technical choices become a **real product**.

This course sits under **Phase 2 - Embedded Systems** because product design here does not mean marketing language or app mockups. It means:

- what the device is for
- where it lives
- what tradeoffs it makes
- how it signals trust
- how hardware, acoustics, controls, software, and setup work together

The running example is an **AI smart speaker / home AI appliance**.

---

## Why this course exists

A lot of embedded engineers can build a working system and still struggle with product decisions such as:

- should this device be voice-first or screen-first?
- should it feel like a tool, an appliance, or a room object?
- where should microphones, speakers, vents, and controls go?
- when is a physical mute switch mandatory?
- how much setup friction is acceptable?
- when does a technically interesting feature actually weaken the product?

Those are product-design questions, but they are also embedded-system questions because they change:

- enclosure layout
- thermal design
- acoustic architecture
- board constraints
- software boundaries
- privacy behavior
- user trust

---

## What you will learn

- Why product design belongs inside embedded engineering, not outside it.
- How to separate documented product features, design hypotheses, and user evidence.
- How to use the HomePod 2018 launch as a dated primary-source case study.
- What later HomePod / 2023-era platform lessons taught about room role, setup, and household workflows.
- How to translate those lessons into a local-first AI smart-speaker design.
- How to think about trust, microphones, speakers, compute placement, setup, and ecosystem boundaries as one product system.

---

## Step-by-step lectures

Each lecture is a separate file under **[Lecture/](Lecture/README.md)**. Work in order.

| # | Topic | Lecture |
|---|-------|---------|
| 1 | Why product design belongs in embedded systems | [Lecture-01.md](Lecture/Lecture-01.md) |
| 2 | HomePod 2018: specifications and testable product hypotheses | [Lecture-02.md](Lecture/Lecture-02.md) |
| 3 | Later HomePod / 2023-era platform lessons | [Lecture-03.md](Lecture/Lecture-03.md) |
| 4 | AI smart speaker design studio | [Lecture-04.md](Lecture/Lecture-04.md) |

---

## Execute one product contract

After Lecture 04, run the [CPU product-state lab](CPU-Product-Contract-Lab.md). Its local timer, software mute, delayed result, cancellation and prior-boot cases turn the brief into an executable interaction. Eight standard-library tests and a recorded transcript are provided. Virtual time tests event rules; they do not measure physical response latency.

## Recommended study pattern

For each lecture:

1. identify the product decision being discussed
2. map it to a real embedded consequence
3. ask what gets better and what gets worse
4. write one short product rule you would keep for your own design

Do not study this as abstract brand analysis. Study it as system design.

---

## What this course is not

This course is not:

- UI design theory
- generic startup advice
- consumer-electronics history for its own sake

It is a practical product-thinking course for embedded engineers using a smart-speaker case study.

---

## Suggested output

By the end of this mini-course, you should be able to write:

- a one-page product intent note
- a V1 hardware/product tradeoff list
- a trust and privacy design checklist
- a simple AI smart-speaker design brief

That is the right kind of artifact because it proves you can connect engineering choices to product behavior.

---

**Next:** [Lecture 01 - Why product design belongs in embedded systems](Lecture/Lecture-01.md)

## Turn the product brief into system requirements

Keep the speaker case study, then transfer its method to a simulated mobile robot or drone mission. Specify one user task, an operating envelope, time-to-ready, observation-age limit, power/thermal budget, wireless-loss behavior, and a visible recovery/status flow. Classify each value as a product requirement, engineering estimate, or measured result.

The final brief must connect those requirements to compute, SRAM/DRAM, boot/service policy, and control responsibilities. A VLA model does not replace the controller’s validity checks. Have a second person walk through startup, task execution, reconnect, and a stale-result scenario and record where the intended behavior was unclear.

**Carry forward:** A measurable product brief feeding SoC requirements and system tests. Continue with [the next course](Lecture/README.md) and reuse that evidence.
