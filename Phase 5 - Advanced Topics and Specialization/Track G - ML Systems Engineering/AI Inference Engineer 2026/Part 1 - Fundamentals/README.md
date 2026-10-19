# Part 1 — Fundamentals of AI inference

Learn to trace a token, count its memory, and define a useful service measurement. The complete beginner route runs on a CPU and requires no model weights. Large-model GPU serving is an optional later experiment with separate evidence.

| Lesson | What you will be able to explain |
|---|---|
| [01 — workload and service metrics](Lecture-01.md) | TTFT, average TPOT versus individual ITL, useful throughput, tails, and cost |
| [02 — transformer execution](Lecture-02.md) | Prefill, incremental decoding, GQA, cache growth, and absolute-position masks |
| [03 — roofline bounds](Lecture-03.md) | FLOPs/bytes, compatible ceilings, units, and why a bound is not a benchmark |
| [04 — precision contracts](Lecture-04.md) | Weights/activations/KV, scales and tails, calibration and task quality |
| [05 — runtime selection](Lecture-05.md) | Source-backed compatibility and a fair matched comparison |
| [CPU lab](labs/README.md) | Run attention/cache, expert-routing, storage, and speculation checks |

Read one lesson, predict its worked example, then run the relevant lab case. The lab has 22 standard-library tests and two optional PyTorch CPU comparisons, all executed in the recorded validation environment. Three pinned public model configurations support the memory exercises; no pretrained checkpoint is loaded.

Your first artifact is a benchmark contract and a small tested reference, not an invented cluster throughput result. Explain the 40 GiB dense-cache example, the non-square cached-mask counterexample, and Qwen3's explicit head dimension. Keep precision, cache layout, and allocator overhead separate.

Before progressing, distinguish storage capacity from transfer traffic; a per-token active parameter count from batch work; and a kernel timing from client-observed latency. State which conditions make a stage compute-, memory-, communication-, or scheduling-limited instead of memorizing that all decode is bandwidth-bound.

Continue to [dense-model deployment reasoning](../Part%202%20-%20Dense%20at%20Hopper/README.md), then [MoE routing](../Part%203%20-%20MoE%20at%20Blackwell/README.md). Carry the benchmark contract into the [Physical AI Design Studio](../../../../Physical-AI-Design-Studio.md) with the hardware measurements clearly separated from CPU reference evidence.
