# Lecture 04 — Establishing a serving baseline

**Prerequisites:** Lectures 01–03. **Time:** 2 hours for a compatibility dossier; deployment time depends on the node. **Outcome:** a correct, pinned starting point or a precise unsupported-path report.

## Define the deployment tuple

Record checkpoint revision, runtime commit or image digest, driver, CUDA toolchain, GPU model/count, topology, attention and MoE backends, parallel groups, and memory settings. Record the prompt-encoding implementation separately.

A branch name such as `main` or an image tag such as `latest` is discovery information. Resolve it to an immutable identifier before measuring.

## A launch is a hypothesis

The [vLLM V4.1 module](https://docs.vllm.ai/en/latest/api/vllm/models/deepseek_v41/) demonstrates an architecture implementation. It does not by itself validate the s-zaizen or LibertAI loader contract. Inspect the chosen commit for:

- accepted architecture and nested config layout;
- expert payload and scale names, including global scales;
- mixed formats in components outside the converted experts;
- Engram and vision handling;
- compatible attention/cache and speculative paths.

Use the runtime's documented recipe for the **exact artifact**, then save that command verbatim. This lecture deliberately does not supply an unverified `serve MODEL --quantization ...` incantation. A wrong generic quantization override can replace a checkpoint's intended mixed-format loader.

## Baseline ladder

| Step | Input | Pass condition |
|---|---|---|
| Load | Minimal supported runtime settings | No unexplained missing/unexpected tensors; all ranks healthy |
| Encode | Fixed text and one structured conversation | Token IDs match the pinned upstream encoder |
| Generate | Short deterministic smoke test | Finite outputs, sensible termination, reference comparison |
| Exercise | Multiple prompt/output lengths | Correct outputs without memory errors |
| Concur | Small bounded concurrency | Request accounting and timings remain valid |
| Extend | Images, cache reuse, speculation | Each added feature has its own correctness gate |

DeepSeek's [prompt-encoding section](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash#prompt-encoding) supplies its protocol implementation; the release does not include a generic Jinja chat template. Retain reasoning settings, tool framing, and image preprocessing when comparing engines.

## Lab procedure

1. Complete the support matrix before downloading hundreds of GiB.
2. Choose a runtime/checkpoint pair with evidence of matching loader semantics.
3. Save launch logs and peak memory per rank. Keep warmup separate.
4. Run a fixed ten-request smoke set: short text, multi-turn text, explicit stop behavior, and an image case if supported.
5. Compare encoded prompts and reference outputs or task scores.
6. Freeze the successful configuration before tuning a single parameter.

If the loader fails, reduce the problem to a tensor/config mismatch and file it in your local experiment record. Do not modify metadata merely to bypass an assertion. An assertion may be the only thing preventing incorrect execution.

**Reduced exercise:** use a small supported model to rehearse request logging and benchmark analysis. Label it as that model; it does not establish DeepSeek support.

## Self-check

- Why is plausible text insufficient? Severe loader or protocol errors can leave output superficially fluent.
- Why begin with short contexts? They isolate loading and basic execution from cache/capacity limits.
- Can you substitute the upstream reference inference script for a converted-checkpoint loader? Only after verifying its tensor and scale contract.

**Exit artifact:** either a reproducible baseline with the evidence above, or an unsupported-path report identifying the failing interface and the work needed to enable it.

[Previous](Lecture-03.md) · [Overview](README.md) · [Next: profiling](Lecture-05.md)
