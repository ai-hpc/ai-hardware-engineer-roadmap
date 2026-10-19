# Gemma 4 at the edge — verified deployment and physical-AI integration

<div class="course-identity edge-ai" markdown="1">
<div class="course-identity__icon">G4E</div>
<div markdown="1">
<p class="course-identity__eyebrow">Edge AI · Six connected lectures</p>
<p class="course-identity__title">Turn a verified multimodal checkpoint into a measured edge service and an explicit robot interface.</p>
<p class="course-identity__meta">Artifact: deployment dossier · Measure: quality, memory, TTFT, token latency, observation age</p>
</div>
</div>

**Evidence reviewed: 2026-10-18.** Gemma 4 launched on April 2, 2026 under Apache 2.0. Its family includes E2B, E4B, 12B Unified, 26B A4B, and 31B; these are not interchangeable configurations. Use the official [launch announcement](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/) and [model card](https://ai.google.dev/gemma/docs/core/model_card_4), then pin the exact checkpoint revision used in your experiment.

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

## Start with the supplied CPU exercises

From the repository root:

```bash
cd 'Phase 5 - Advanced Topics and Specialization/Track C - Edge AI/Gemma 4 Edge Deployment'
python3 labs/cache_plan.py
python3 -m unittest discover -s labs -p 'test_*.py' -v
```

Seven tests verify the worked 76/140 MiB cache example, local-window saturation, global growth, shared-allocation accounting, invalid metadata, and the different target/assistant configurations. [Code](labs/cache_plan.py), [tests](labs/test_cache_plan.py), and [recorded output](labs/cache-reference.json) are included. The 16-layer cache example is deliberately hypothetical; it is not a Gemma E2B allocator model.

The supplied [target config](labs/target-config.json) is pinned to `google/gemma-4-E2B-it` revision `3e22461f65e89153144f8adb70e3b8c2cc9845a7`; [assistant config](labs/assistant-config.json) is pinned to revision `2d874ef7d29f9a30599a1e4b3c1cbc9595f005df`. Hashes and metadata appear in the recorded output. These small configuration files require no weight download.

An optional [Transformers meta-device inspection](labs/transformers_meta.py) was actually executed with Transformers 5.20.1 and PyTorch 2.14.0+cpu. Reuse the optional environment from the [Qwen CPU lab](../Qwen%20Inference%20Optimization/Lab-Guide.md), then run this script using that environment's Python. It creates shapes without allocating model data and verifies **5,104,297,504 unique parameters**, including modality components. Its [recorded result](labs/meta-reference.json) identifies 15 KV-owner layers and the two state-provider layers in this implementation. It does not allocate/measure a runtime cache or run Gemma inference.

**Practice:** compare the target's 35 layers and hidden size 1,536 with the assistant's four layers, hidden size 256 and backbone hidden size 1,536. Explain why a separate assistant config is an execution-interface contract, not proof that an arbitrary speculative runtime supports the pair. Continue through the six lessons with these checked facts.
