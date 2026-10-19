# Source and Execution Evidence — Agent Lectures

Checked **2026-10-18 UTC**. This note separates package/source observations from execution evidence. Moving documentation and model names are not immutable release pins.

## Executed locally

The [provider contract lab](Provider-Contract-Lab.md) runs a strict local arithmetic dispatcher and real SDK clients with fixture HTTP transports:

- **8 standard-library tests:** exact argument types/keys/bounds, duplicate JSON keys, unknown tools and whole-batch call budgets.
- **20 optional SDK tests:** actual request serialization/response parsing, multi-call correlation, preserved reasoning items, refusals/incomplete output, turn budgets, explicit continuation signals, duplicate IDs, unsupported server-tool scope and structured-output/local validation.
- **Two fixture conversations:** actual tool sums and captured fictional request bodies in [provider-demo.json](labs/evidence/provider-demo.json).
- **134 fenced Python/shell blocks:** parsed with `ast.parse` or `bash -n`; syntax checks are not execution or dependency/API validation.

The [validation record](labs/evidence/validation.json) records actual commands, return codes, package versions, lock hash and timestamps. The [20-package hashed wheel lock](labs/requirements-provider.lock) was installed in an isolated CPython 3.12 Linux x86-64 environment. Anthropic 1.15.0 uses `httpx2`; OpenAI 3.35.1 uses `httpx`. Passing `httpx.Client` to the current Anthropic SDK, and passing its removed `temperature` argument, were both tested to raise `TypeError`.

**Not executed:** hosted model inference, current model quality/price/latency, Chroma deployment, OpenClaw Gateway/CLI, browser-use benchmark, local Qwen/Granite inference, live multi-agent orchestration, confidential hardware attestation or zero-knowledge proving. Fixture responses are intentionally authored data, not a benchmark dataset. The tool arithmetic is real; the model response text is fictional.

## Immutable source observations

The [machine-readable source manifest](labs/evidence/source-evidence.json) records 33 downloaded files with exact revision, URL, byte count and SHA-256.

| Project | Revision | What it supports |
|---|---|---|
| [OpenClaw](https://github.com/openclaw/openclaw/tree/24aac3c15a01feb1946dad5b212282e78dc3941e) | `24aac3c15a01feb1946dad5b212282e78dc3941e` | Package version 2026.10.4; Node engine; current loop/writer fencing, automation/storage, SDK and Gateway schemas |
| [OpenAI Agents Python](https://github.com/openai/openai-agents-python/blob/9abc9182d06dbd28330af888306031269d57a175/docs/guardrails.md) | `9abc9182d06dbd28330af888306031269d57a175` | Blocking versus parallel input guardrails and tool-side timing distinction |
| [Reflex benchmark](https://github.com/reflex-dev/agent-benchmark/tree/3528c2cc11d561599fa92df0316bbe222b50269f) | `3528c2cc11d561599fa92df0316bbe222b50269f` | Historical author's small-sample browser/API results, operation units and cache accounting |
| [Agent Skills Eval](https://github.com/darkrishabh/agent-skills-eval/tree/f10ae2b3c93055cadd82c02a3b7fe4cfbd9a4645) | `f10ae2b3c93055cadd82c02a3b7fe4cfbd9a4645` | Source package version 0.1.1; optional CLI case study, not executed evaluation |
| [LLM workshop](https://github.com/angelos-p/llm-from-scratch/tree/63069f99af8ce9a6ab6e261000f4f9f1daaa1ee9) | `63069f99af8ce9a6ab6e261000f4f9f1daaa1ee9` | Workshop configurations and author's M3 Pro training-time estimates |

OpenClaw's **package version, wire protocol 4 and device-signature payload v2/v3 are distinct**. The private `@openclaw/sdk` manifest is `0.0.0-private`; this does not contradict separately published gateway protocol/client packages. The current SDK implements scoped artifact methods, environment methods (with `delete` explicitly unsupported) and `tools.invoke`. Exported method existence does not establish remote authorization or compatible Gateway configuration.

## Model metadata and worked arithmetic

[Qwen3.5-4B revision 851bf6e](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a) and [Granite embedding revision 835ad140](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2/tree/835ad14087e140460703cf0fae09f97d469d65c2) were checked through config and pipeline metadata only. Both repositories declare Apache-2.0; the bundled JSON retains its upstream identity through manifest URLs/hashes. No weights or training data were downloaded.

For Qwen, the actual text config has 8 full-attention and 24 linear-attention layers. The manifest records the evaluated BF16 full-attention KV arithmetic: 32 KiB/token/sequence and 128/256/512 MiB for 4K/8K/16K. It separately records an illustrative 48 MiB recurrent-matrix term. These are calculated components, not measured process allocations. Convolution state, padding, rollback, weights, workspaces and other model/application components are excluded explicitly.

Granite's metadata specifies 384-dimensional CLS pooling followed by normalization. Changing to mean pooling would change the embedding pipeline. Encoder maximum positions do not imply that equally long RAG chunks are useful or fast.

## Moving primary documentation checked

- [Claude model lifecycle and removed sampling parameters](https://platform.claude.com/docs/en/about-claude/model-deprecations), [structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs), [tool-result handling](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls).
- [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling), [structured output](https://developers.openai.com/api/docs/guides/structured-outputs), [reasoning](https://developers.openai.com/api/docs/guides/reasoning), [agent architecture choices](https://developers.openai.com/api/docs/guides/agents).
- [Chroma clients](https://docs.trychroma.com/reference/python/client) and [collection metrics/configuration](https://cookbook.chromadb.dev/core/collections/).
- [RISC Zero receipt types](https://dev.risczero.com/proof-system/), [Groth16 trusted setup](https://dev.risczero.com/api/trusted-setup-ceremony), and [EZKL](https://github.com/zkonduit/ezkl).

These pages support the stated API/protocol distinctions, not a claim that every integration example was run. HTTP source checks, including blocked requests, are preserved in [source-links.json](labs/evidence/source-links.json). A 403 from a raw HTTP probe is recorded separately from a successful browser-tool read; it is not relabeled as a successful download.

## Revalidation exercise

Choose one pinned source. Fetch it at its recorded revision, recompute SHA-256 and compare the exact field used in the lecture. Then inspect a newer release and write a three-line migration note: changed contract, affected example, test that would detect the incompatibility. Do not update a date or model name without checking the behavior it changes.
