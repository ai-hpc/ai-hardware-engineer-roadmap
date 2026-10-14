# Part 5 — DeepSeek V4.1 Flash NVFP4: inference optimization engineering

**Outcome:** build an auditable deployment decision for a named checkpoint, then optimize a supported execution path while preserving model quality.

**Prerequisites:** Part 1 (execution, rooflines, precision), Part 3 (MoE and parallelism), and Part 4 Lecture 02 (measurement). You should be able to distinguish checkpoint size from peak GPU memory and TTFT from decode throughput.

**Effort:** ten lectures; approximately 25–40 hours for the analysis route, plus installation and hardware time for full serving experiments. Stop a deployment experiment at its compatibility gate if the exact format is unsupported.

**Access:** Lectures 01–03 and the supplied fixtures run on a CPU with Python 3.10+. Full-checkpoint work requires a capacity plan for hundreds of GiB of weights, runtime memory, and the chosen GPU topology. An FP4-capable GPU alone does not establish model support.

## Checkpoint identity and evidence

Checked **2026-10-13**. These are distinct artifacts:

| Artifact | Role in this course | Evidence boundary |
|---|---|---|
| [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | Official architecture and source-format reference | Its evaluations apply to the upstream model |
| [s-zaizen/DeepSeek-V4.1-Flash-NVFP4](https://huggingface.co/s-zaizen/DeepSeek-V4.1-Flash-NVFP4) | Default community conversion for the checkpoint audit | Backbone routed experts converted; other components retain source formats. Publisher reports no quality/throughput benchmark or validated launch recipe |
| [LibertAIDAI/DeepSeek-V4.1-Flash-NVFP4](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-NVFP4) | Optional comparison of format and quality tradeoffs | Also reduces Engram precision; publisher explicitly leaves end-to-end quality unestablished |

Neither community conversion is an official NVIDIA checkpoint. Upstream experts already use FP4; “convert to NVFP4” does not imply halving the model. Read each actual config, index, and loader. The s-zaizen card identifies its upstream source revision; resolve and record a separate immutable revision for the conversion you inspect.

The current [vLLM architecture documentation](https://docs.vllm.ai/en/latest/api/vllm/models/deepseek_v41/) contains a DeepSeek V4.1 implementation. That is evidence of architecture work, **not** proof that either community serialization works on a particular released NVIDIA backend. Do not repeat the older claim that no engine supports the architecture; verify the exact runtime commit and checkpoint together.

## Lecture map

| Lecture | Engineering question | Artifact |
|---|---|---|
| [01 — Checkpoint identity and compatibility](Lecture-01.md) | What did you actually download, and can a loader read it? | Metadata manifest and support matrix |
| [02 — Execution and memory budgets](Lecture-02.md) | What is resident, active, cached, or gathered? | Per-rank memory and traffic budget |
| [03 — NVFP4 representations and numerical checks](Lecture-03.md) | Which values and scales did the conversion change? | Reconstruction checks and error cases |
| [04 — Establishing a serving baseline](Lecture-04.md) | Is a successful launch also correct inference? | Pinned smoke-test report |
| [05 — Prefill, decode, and kernel profiling](Lecture-05.md) | Which stage limits this workload? | Trace-backed bottleneck hypothesis |
| [06 — MoE placement and communication](Lecture-06.md) | When does sharding save memory but lose time? | Topology and EP/TP sweep |
| [07 — Long context and cache behavior](Lecture-07.md) | What does cache reuse save in your runtime? | Context, cache, and memory curve |
| [08 — Scheduling, load, and goodput](Lecture-08.md) | How much useful work meets the latency target? | Reproducible load experiment |
| [09 — Quality and speculative decoding](Lecture-09.md) | Does the optimized system still perform the task? | Quality gate and speculation ablation |
| [10 — Capstone deployment decision](Lecture-10.md) | Should this exact deployment ship? | Reproduction package and decision |

## Start on a laptop

From this directory:

```bash
python3 labs/checkpoint_audit.py --demo
python3 labs/inference_metrics.py --demo
python3 labs/fp4_layout.py
python3 -m unittest discover -s labs -p 'test_*.py'
```

These standard-library exercises use **synthetic fixtures**, not DeepSeek performance measurements. See the [lab guide](labs/README.md) for real inputs and expected outputs.

## Completion levels

**Analysis route:** submit inspected metadata, the memory derivation, format checks, a compatibility decision, and a benchmark design. Mark full-model execution “not run.” Unsupported deployment is an acceptable evidence-based decision.

**Deployment route:** additionally submit a working pinned launch, reference-quality comparisons, at least three repeated load runs, per-rank memory, and one successful or rejected optimization backed by traces. Apply the [shared rubric](../../../../Learning-Guide.md); an integrity check alone cannot satisfy model-quality assessment.

[Back to the inference course](../README.md) · [Start Lecture 01](Lecture-01.md)


## Carry this course into the physical-AI system

**Start from:** Exact DeepSeek V4.1 Flash checkpoint identity and loader compatibility.

**Build and challenge:** Audit payload/scales, total versus active parameters, hybrid state, and parallel placement. A text-output multimodal model is an inference case study or optional planner; it is not a native robot action policy.

**Pass forward:** A checkpoint/memory/kernel dossier; transfer supported numerical and memory lessons to the selected VLA. Continue with [the connected course](../../Hardware-Aware%20LLM%20Quantization/README.md) and keep the artifact in the [Physical AI Design Studio](../../../../Physical-AI-Design-Studio.md).
