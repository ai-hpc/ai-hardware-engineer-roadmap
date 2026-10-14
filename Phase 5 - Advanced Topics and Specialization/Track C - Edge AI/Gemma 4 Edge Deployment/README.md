# Gemma 4 at the edge — verified deployment and physical-AI integration

<div class="course-identity edge-ai" markdown="1">
<div class="course-identity__icon">G4E</div>
<div markdown="1">
<p class="course-identity__eyebrow">Edge AI · Six connected lectures</p>
<p class="course-identity__title">Turn a verified multimodal checkpoint into a measured edge service and an explicit robot interface.</p>
<p class="course-identity__meta">Artifact: deployment dossier · Measure: quality, memory, TTFT, token latency, observation age</p>
</div>
</div>

**Evidence reviewed: 2026-10-14.** Gemma 4 launched on April 2, 2026 under Apache 2.0. Its family includes E2B, E4B, 12B Unified, 26B A4B, and 31B; these are not interchangeable configurations. Use the official [launch announcement](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/) and [model card](https://ai.google.dev/gemma/docs/core/model_card_4), then pin the exact checkpoint revision used in your experiment.

The course goal is an evidence-backed deployment decision, not a promised token rate. An E2B effective parameter count is not its complete weight storage; local attention does not eliminate global-layer KV growth; a VLM producing text is not yet a VLA producing calibrated actions. Those distinctions shape the accelerator, memory system, compiler, and control interface you eventually design.

**Before:** [Edge LLM internals](../Edge%20LLM%20Inference%20Internals/Lecture-01.md), [quantization](../../../Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/04%20-%20Quantization/Guide.md), and basic Jetson/Linux operation. A host with enough RAM can complete metadata and numerical exercises; label on-device measurements not run until you have a compatible target.

| Lecture | Engineering question | Artifact passed onward |
|---|---|---|
| [01 — Identity and memory](Lecture-01.md) | Which checkpoint, tensor shapes, and cache rules are real? | Hashed configuration and memory model |
| [02 — Quantization](Lecture-02.md) | Which representation preserves task quality and maps to a kernel? | Format/quality/compatibility record |
| [03 — Runtime](Lecture-03.md) | Can the selected runtime execute the exact model and modality? | Reproducible service and benchmark protocol |
| [04 — Speculation](Lecture-04.md) | Does drafting reduce cost per committed token? | Accepted-prefix and timing analysis |
| [05 — Multimodal robot interface](Lecture-05.md) | How old is the evidence behind a proposed action? | Timestamped observation/result contract |
| [06 — Official MTP assistant](Lecture-06.md) | Is the assistant paired and supported by this runtime? | Compatibility audit and capstone decision |

## What completion means

Keep one `deployment-manifest.json` containing checkpoint ID/revision, file hashes, processor/tokenizer, runtime commit, build flags, BSP, module SKU, power mode, dtype/quantization, prompt or image shapes, and measurement definitions. Each lecture extends this manifest instead of restarting with an unrelated demo.

Submit the float reference where feasible, one supported optimized deployment, held-out quality, memory at several context lengths, cold/warm TTFT, decode inter-token latency, and failures. A comparison must preserve input/output lengths and sampling policy. Peak advertised TOPS and unsourced “expected tok/s” are not results.

**Continue:** use the latency, live-memory, and operator-shape evidence in [VLA deployment on Jetson](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/5.%20Application%20Development/5.%20ML%20and%20AI/vla-deploy-jetson/Guide.md), then the physical-AI SoC proposal. The original six lecture URLs remain stable; unverified commands and invented benchmark tables have been replaced with explicit experiments.
