# Part 3 — MoE inference: route the work and account for the state

Use DeepSeek-V3.1 and Qwen3-235B-A22B as pinned model-configuration case studies. The key distinction is between total resident parameters, per-token active work, and the union of experts touched by a batch. Attention layout, routing policy, topology, and state ownership then determine what the runtime must do.

The [CPU lab](../Part%201%20-%20Fundamentals/labs/README.md) runs the beginner route: inspect the two public config snapshots, calculate GQA/MLA cache payload, draw a dispatch matrix, combine toy expert outputs, and prove a one-step speculation probability identity. No large weights or Blackwell hardware are required.

| Lesson | Question |
|---|---|
| [01 — MoE anatomy](Lecture-01.md) | Which experts and cache components exist in these configurations? |
| [02 — Blackwell products](Lecture-02.md) | Which capacity, precision, and data-path figures belong to this exact SKU? |
| [03 — expert parallelism](Lecture-03.md) | Where do tokens travel, what can be deduplicated, and which rank is hot? |
| [04 — prefill/decode separation](Lecture-04.md) | Do scheduling savings justify transfer and ownership costs? |
| [05 — speculation and constraints](Lecture-05.md) | What policy is preserved, what work is added, and which outputs are useful? |

DeepSeek's first three layers use dense MLPs; “256 routed experts per layer” requires that qualification. Qwen3's attention head dimension is explicitly 128 despite `hidden_size/query_heads = 64`. The ideal cache calculations are 70272 bytes/token for the stated DeepSeek compressed layout and 192512 bytes/token for Qwen3 BF16 GQA, before allocator/runtime overhead.

Hardware extensions require a supported model/format/runtime combination and actual product topology. HGX B200, GB200, and B300 have different capacities. A large NVLink domain does not create one uniformly fast allocation, and TP/EP group sizes are not universally multiplied to obtain GPU count.

Deliver a source-backed compatibility table, per-rank resource/traffic budget, and quality/performance experiment. Mark hardware results unmeasured until executed. A full deployment adds traces, failure/cancellation handling, useful throughput under latency targets, and cost from real allocations/prices. There is no prescribed FP4 winner or unmeasured expected tokens/s figure.

Continue to [Part 4 — Optimizing a Real Engine](../Part%204%20-%20Optimizing%20a%20Real%20Engine/README.md) or the [course overview](../README.md). Carry the routing/cache contracts into the [Physical AI Design Studio](../../../../Physical-AI-Design-Studio.md) when building a physical system.
