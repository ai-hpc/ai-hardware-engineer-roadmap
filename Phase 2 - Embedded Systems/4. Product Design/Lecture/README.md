# Product Design for Embedded Systems — lectures

Work through these in order. This mini-course uses a **local-first AI smart speaker** as the running example.

**Hub:** [Product Design for Embedded Systems guide](../Guide.md)

| Lecture | Topic | Main product question |
|---------|-------|-----------------------|
| [Lecture-01](Lecture-01.md) | Why product design belongs in embedded systems | What does "product design" actually change in hardware and software? |
| [Lecture-02](Lecture-02.md) | HomePod 2018: specifications and testable product hypotheses | Why can strong hardware still produce a weak product? |
| [Lecture-03](Lecture-03.md) | Later HomePod / 2023-era platform lessons | How does a device become a trusted room object instead of just a gadget? |
| [Lecture-04](Lecture-04.md) | AI smart speaker design studio | How do you turn those lessons into a concrete V1 product? |

## Carry each product decision into the final design brief

L1 defines the user’s task and environment. Treat L2–3’s HomePod judgments as design hypotheses, then name the observation or user study that would test each one. L4 turns the selected hypotheses into a brief with measurable requirements and a startup/fault walkthrough.

For the physical AI extension, add a robot mission and trace one decision—such as operating during wireless loss—through local model capacity, storage, control policy, and user feedback. Submit requirements and evidence separately; attractive hardware and plausible narratives do not establish task success.

**Carry forward:** Case-study hypotheses translated into testable product and hardware requirements. Continue with [the next course](../../../Physical-AI-Design-Studio.md) and reuse that evidence.

**Practical continuation:** [CPU product-state lab](../CPU-Product-Contract-Lab.md), with a worked transcript, eight executed tests and an independent failure/recovery extension.
