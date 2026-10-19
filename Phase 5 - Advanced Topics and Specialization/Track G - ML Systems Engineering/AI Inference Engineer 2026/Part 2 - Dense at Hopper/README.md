# Part 2 — Dense models on Hopper: shapes, memory, and serving

Llama 3.3 70B and Qwen2.5 72B are concrete historical architecture case studies. Their similar widths make it easier to see how MLP width, vocabulary, bias, position configuration, and normalization affect a deployment. They are not presented as the newest available models.

Start with the [Part 1 CPU lab](../Part%201%20-%20Fundamentals/labs/README.md). It includes a pinned public Qwen2.5 configuration and tested KV arithmetic; the gated Llama configuration was not downloaded. Qwen's saved limit is 32768 positions, with a separate documented YaRN recipe for extension. Do not describe both checkpoints as identically configured for 128K.

| Lesson | Worked result |
|---|---|
| [01 — derive model shapes](Lecture-01.md) | Projection/MLP dimensions, RMSNorm, parameter and cache accounting |
| [02 — Hopper hardware](Lecture-02.md) | SKU capacities, cluster limits, TMA/WGMMA synchronization contracts |
| [03 — quantization experiment](Lecture-03.md) | A supported format/runtime recipe with paired quality evaluation |
| [04 — tensor parallelism](Lecture-04.md) | Column/row partition algebra and collective traffic assumptions |
| [05 — serving and scheduling](Lecture-05.md) | Chunked prefill, attention backends, cache identity, capture boundaries |
| [06 — long context](Lecture-06.md) | Position recipe, 40 GiB cache example, allocation and quality limits |
| [07 — collective implementation](Lecture-07.md) | Current NCCL APIs, trace interpretation, versioned capability checks |

The CPU route produces a configuration audit, tensor-shape table, memory budget, partition proof, and benchmark plan. These are complete learning artifacts without a GPU cluster. A full deployment milestone additionally requires a supported licensed checkpoint, appropriate hardware, real quality tests, per-rank memory, traces, and raw service measurements. The course does not claim those full 70B hardware runs occurred.

For a hardware extension, compare supported cells in a model × format × runtime × topology table. Keep quality, arrival process, prompt/output lengths, cache state, and total reserved resources aligned. Record unsupported cells instead of forcing every combination to run. Include startup and failure behavior, not only a favorable steady-state number.

You are ready for the next part when you can explain a resource budget and a collective's semantics, identify which measurements would test your bottleneck hypothesis, and distinguish current TensorRT-LLM's documented PyTorch path from the archived engine-build workflow.

Continue to [MoE at Blackwell](../Part%203%20-%20MoE%20at%20Blackwell/README.md), or carry the tested contracts into the [Physical AI Design Studio](../../../../Physical-AI-Design-Studio.md).
