# Part 1 · Lecture 01 — The 2026 Inference Engineer's Mental Model

## Overview

An AI inference engineer in 2026 is not a model engineer, not a prompt engineer, and not a generalist MLOps practitioner. The role is narrower and more specific than any of those:

> **Given a model, a workload, and a hardware target, ship the lowest-cost serving configuration that meets the latency and accuracy SLO — and *prove* it with measurements another engineer can reproduce.**

Everything in this course follows from that sentence. The model is fixed (or chosen at the top of the project). The workload — chat, agent, batch, embedding — sets the metric that matters. The hardware target — single Jetson Orin Nano up to GB200 NVL72 — sets the physical ceiling. The job is the configuration in between, defended by reproducible numbers.

This lecture builds that mental model in five passes:

1. The **four inference shapes** and what each one optimizes.
2. The **metrics** that decide whether an inference deployment is good.
3. The **diagnostic flow** — how a good engineer figures out *why* a system is slow before changing anything.
4. **Reading a model card** to predict cost before you spin a GPU.
5. The **role contract** — what a senior inference engineer is expected to ship.

By the end you should be able to look at a new model + workload + GPU triple and, on a whiteboard, predict (a) which metric will be the bottleneck, (b) which precision floor you would try first, (c) which runtime you would prototype on. Part 2 and Part 3 then deepen this with two concrete model families each.

---

## 1. The four inference shapes

Every production inference workload reduces to one of four shapes. They have different bottlenecks and want different optimizations. Mis-identifying the shape is the most common reason expensive optimization work produces zero metric movement.

### 1.1 Chat

```text
user message → prefill (1–4K tokens) → decode (50–500 tokens) → user
                                       ↑
                                 repeat in a turn loop with growing KV cache
```

Characteristics:

* Long-lived sessions, growing KV cache across turns.
* TTFT (time to first token) matters because the human is waiting.
* Decode dominates wall-clock per turn (lots of small steps).
* **Prefix cache is the single biggest win** — the system prompt and prior turns are reused.
* Batch size is low to medium (1–32 concurrent users on a single replica).

What you optimize: **TTFT, p99 inter-token latency, prefix-cache hit rate.**

### 1.2 Agent loop

```text
prompt + tool catalog → short prefill → short decode (tool call) → tool run → result → repeat
                                        ↑
                                 1–16 token bursts, JSON-structured
```

Characteristics:

* Many short turns, each with a fresh-ish prefix (system + tools + history).
* **Decode is dominated by the structural format constraint** (JSON, function calling).
* Tool calls expand the conversation between LLM steps with non-LLM latency.
* **Tool-call accuracy** (see the [BFCL evaluation lecture](../../../3.%20Edge%20AI/Agent%20Tool-Dispatch%20Evaluation%20with%20BFCL/Lecture-01.md)) matters as much as latency.

What you optimize: **TTFT per turn, structured-output throughput, tool-call accuracy at the chosen precision.**

### 1.3 Batch (offline)

```text
N prompts (10K–10M) → maximize tokens/sec/GPU → write results
```

Characteristics:

* Latency does not matter; throughput does.
* Large batch sizes, often disaggregated across many GPUs.
* Prefix cache wins enormously if prompts share structure (RAG re-encoding, classification with shared instructions).
* This is where **disaggregated prefill / decode** earns its keep (see Part 3).

What you optimize: **tokens/sec/GPU, $/MTok, scheduler utilization.**

### 1.4 Embedding / retrieval

```text
N input documents → encoder-only forward → vectors out
```

Characteristics:

* No autoregression; **prefill is the whole job**.
* No KV cache.
* Pure compute-bound matmul, ideal for tensor cores at low precision.
* Batch size limited by HBM (large activations).

What you optimize: **tokens/sec, peak HBM, latency per batch.**

### 1.5 Why the shape matters

The same model on the same GPU will look different in each shape:

| Shape | Dominant cost | Wins from |
|-------|---------------|-----------|
| Chat | Decode bandwidth + KV cache | Prefix cache, paged KV, speculation |
| Agent | TTFT + structured-output enforcement | Prefix cache, grammar-constrained decode, fast tool dispatch |
| Batch | Throughput | Continuous batching, P/D disaggregation, large batch sizes |
| Embedding | Tensor-core matmul throughput | Low precision (FP8/INT8), packed batches |

If you optimize batch-style (continuous batching, big batch sizes) for a chat product you will tank TTFT. If you optimize chat-style (low batch, prefix cache) for a batch job you will burn 3–5× the cost. **Pick the shape first.**

---

## 2. The metrics that decide whether the work was good

A senior engineer is paid to optimize *the right metric* and to *defend the choice in numbers*. The five that matter:

### 2.1 TTFT — Time to First Token

The wall-clock from request acceptance to the first decoded output token reaching the client.

* For chat: this is what the human perceives as "responsiveness."
* For agent: this gates the entire tool-call loop's clock.
* Dominated by prefill cost + queue wait time + (for streamed responses) the first decode step.
* Targets you see in production: <300 ms for chat with short prompts on H200; <50 ms for fast agent loops at <512-token prompt.

### 2.2 TPOT — Time Per Output Token

Wall-clock per decoded token after the first. Sometimes called inter-token latency (ITL).

* For chat: this is what makes streamed output feel "fast" or "stuttery."
* Targets: 10–50 ms per token for human-readable chat (matches reading speed); 5–15 ms for agent loops where a single response is short.
* Dominated by **HBM bandwidth** (decode is bandwidth-bound). This is why Hopper H200 (4.8 TB/s HBM3e) beats H100 (3.35 TB/s HBM3) at decode by roughly the bandwidth ratio.

### 2.3 Throughput — tokens / sec / GPU

The denominator of cost. Total output tokens emitted across all concurrent requests per second per GPU.

* For batch: this *is* the optimization target.
* For chat + agent: this is what determines $/MTok and how many users one GPU serves.
* Improves with batch size (up to a memory ceiling) and with speculation (effective tokens-per-step > 1).

### 2.4 p50 / p95 / p99 latency

The tail. The mean is a lie; the tail is the SLO.

* Most production SLOs are stated as "p95 TTFT < X ms" or "p99 TPOT < Y ms".
* The gap between p50 and p99 tells you whether the scheduler is well-balanced or whether one stuck request poisons the rest.
* When p99 explodes while p50 stays flat, look at: prefill-bursts starving decode, KV-cache eviction storms, NCCL all-reduce stragglers.

### 2.5 $/MTok — cost per million output tokens

The metric that decides whether the product is viable.

```text
$/MTok = (replica_$/hour) / (output_tokens/sec/replica) × (10^6 / 3600)
```

* The number a CFO understands.
* Determines whether your optimization actually moved the business needle.
* A 30% throughput improvement that requires switching from 1× H200 to 2× H100 is *not* a $/MTok win — re-check the math.

### 2.6 The metric you cannot fake — accuracy parity

Every optimization in this course (quantization, speculation, prefix cache, KV compression) has a parity question hidden in it: *did the model still produce the same answers?*

* Use a fixed eval set per workload class. For chat, MMLU subset + a domain set. For agent, [BFCL](../../../3.%20Edge%20AI/Agent%20Tool-Dispatch%20Evaluation%20with%20BFCL/Lecture-01.md). For code, HumanEval / MBPP. For long context, RULER or needle-in-haystack.
* A 50% throughput win that costs 3 pp on BFCL is not a win — it ships incidents.
* The pattern is the same as the [VLA Action-Parity Harness](../../../4.%20Robotics/VLA%20Optimization%20and%20Action-Parity%20Harness/Lecture-02.md), applied to LLMs: every optimization needs a parity gate.

---

## 3. The diagnostic flow

Bad inference engineering changes things first and measures after. Good inference engineering looks like this loop:

```text
observe ──► hypothesize ──► isolate ──► benchmark ──► profile ──► explain ──► change ──► verify
   ▲                                                                                       │
   └───────────────────────────────────────────────────────────────────────────────────────┘
```

Concretely:

1. **Observe the metric that is bad.** "Decode is slow" is not observed; "p99 TPOT is 47 ms, target is 25 ms" is observed.
2. **Hypothesize the regime.** Is this compute-bound, memory-bound, scheduler-bound, or comm-bound? Lecture 02 builds the test.
3. **Isolate.** Reduce the workload until only the suspect remains. Run at batch=1, no other traffic, fixed seed, fixed prompt.
4. **Benchmark.** Get a stable baseline with at least 50 iterations and warmup. Report p50/p95/p99, not mean.
5. **Profile.** Nsight Systems for the timeline, Nsight Compute for kernel-level. PyTorch profiler if you cannot get below Python.
6. **Explain.** Write down in one sentence what you believe is happening. "Decode kernel A is bandwidth-bound at 78% of HBM3e peak; the rest is launch overhead."
7. **Change one thing.** Not three.
8. **Verify.** Re-bench. If the metric did not move predictably from the hypothesis, the explanation was wrong — back to step 2.

The number-one mistake is skipping step 6. If you cannot say *why* something is slow, you cannot say *what* to change.

---

## 4. Reading a model card to predict cost

Before spinning up a GPU, you can extract the inference cost shape of any modern model from its `config.json` (or model card) alone. Five fields do almost all of the work.

| Field | What it tells you |
|-------|-------------------|
| `num_hidden_layers` (L) | Total transformer blocks; multiplies almost every cost |
| `hidden_size` (d) | Width of activations; sets FFN GEMM size and KV bytes per token |
| `intermediate_size` (d_ff) | FFN expansion; usually 2.5–4× hidden, dominates FLOPs |
| `num_attention_heads` (h_q) and `num_key_value_heads` (h_kv) | GQA ratio; sets KV cache size (smaller h_kv = smaller KV) |
| `head_dim` | Per-head dimension; usually 128 in modern models |

From these alone you can compute:

**KV cache bytes per token (per request):**

```text
kv_bytes_per_token = 2 (K and V) × L × h_kv × head_dim × bytes_per_element
```

For Llama 3.3 70B at FP16 KV: 2 × 80 × 8 × 128 × 2 = **327 KB/token**.
At 128K context: 327 KB × 128 × 1024 ≈ **42 GB**. Per request. This is why long-context serving needs FP8 KV (cuts it in half) or INT4 KV (cuts it to ~10 GB).

**Approximate FLOPs per token (decode):**

```text
flops_per_token ≈ 2 × P
```

where P is the active parameter count (= total params for dense, = total/expert × experts_active for MoE). The factor of 2 captures multiply + accumulate.

For a 70B dense model: ~140 GFLOPs/token at decode. On H200 (~990 BF16 TFLOPs peak), if you could keep the kernel at peak you would do ~7000 tokens/sec — but you cannot, because decode is bandwidth-bound, not compute-bound.

**Decode bandwidth ceiling (the real ceiling at batch=1):**

```text
decode_tps_ceiling ≈ HBM_bandwidth / (model_size_in_bytes)
```

For Llama 3.3 70B at FP16 (140 GB) on H200 (4.8 TB/s): 4800 / 140 ≈ **34 tokens/sec**. Real vLLM numbers on H200 are around 30 tok/s at batch=1, which matches.

At INT4 (35 GB): 4800 / 35 ≈ **137 tokens/sec ceiling**. Real numbers: ~110 tok/s.

The takeaway: you can predict the bandwidth-bound ceiling from the model card and the HBM spec alone. Anything you do above that ceiling came from batching (sharing the weight read across more tokens), speculation (more accepted tokens per weight read), or precision drop.

---

## 5. The role contract — what a senior AI inference engineer ships

A senior in this role ships, recurrently:

* **Defended configurations.** Not "we use vLLM with batch=64" but "vLLM 0.7 with TP=4, batch=64, prefix cache on, AWQ-INT4, FP8 KV, parity verified at MMLU=82.1 vs FP16 reference 82.4 (Δ=0.3 pp within budget)."
* **Reproducible benchmarks.** A harness another engineer clones and gets the same numbers within ±5% on the same hardware class.
* **Bottleneck explanations.** Profile traces with annotated dominant cost. "Decode is 78% of step time; of that, 91% is HBM weight read."
* **Cost models.** $/MTok at the chosen configuration, with what would change at 2× scale and at 10× scale.
* **Parity gates.** A regression test that fails the build if any future optimization regresses the workload's metric eval set beyond budget.

A junior in this role ships individual fixes. The structural difference is the reproducibility layer.

What a senior is *not* paid to do: pick the model. Pick the product. Choose the SLO. These come from above; the engineer defends what is achievable inside them.

---

## Lab — set up your benchmark template

Goal: a repo you will use throughout Parts 1–3. Cap of one day.

1. **Pick one small model** (Qwen3-4B Instruct AWQ-INT4 is a good default — fits on a 16 GB GPU).
2. **Pick one runtime** (vLLM 0.7+ recommended).
3. **Build a benchmark CLI** that takes `--prompt-length`, `--output-length`, `--concurrency`, `--iters`, `--warmup` and emits a JSON line per run with: hardware (GPU model + driver + CUDA), software (runtime version), git commit, p50/p95/p99 TTFT, p50/p95/p99 TPOT, throughput, peak HBM.
4. **Add a parity check** — for the workload class you care about (chat / agent / batch / embedding), pick one fixed eval set and emit a single accuracy number.
5. **Wire to your `$/MTok` formula** in a `cost.py` that takes a $/hour and reads throughput from the benchmark output.

Pass criterion: you can run `bench --model qwen3-4b --shape chat --concurrency 4 --iters 200` and produce a single JSON file that another engineer on the same GPU could reproduce within ±5% by cloning the repo.

You will run this harness against every model + runtime + hardware combination in Parts 2 and 3.

---

## Self-check

1. You are deploying a customer-support chat product on H100 80GB and the product team wants "average response time < 1 second." What two TTFT-and-TPOT-shaped metrics do you actually need to commit to, and what is the right percentile?
2. A teammate proposes switching from continuous batching to disaggregated prefill/decode for a chat workload at concurrency=8. Without running it: predict whether $/MTok will improve. Why?
3. Given a model with `num_hidden_layers=80`, `num_key_value_heads=8`, `head_dim=128`, and FP16 KV, what is the KV cache cost (in MB) for one request at 32K tokens? At 256K?
4. You change FP16 → FP8 weights on a 70B model and TPOT improves from 38 ms to 22 ms. The team wants to ship. What single number do you require before agreeing?
5. Decode TPOT on an L40S is 60 ms at batch=1 with INT4 weights. Predicted bandwidth ceiling from the model card was 18 ms. What three things would you check, in order, to explain the gap?

---

## References

* vLLM documentation — [docs.vllm.ai](https://docs.vllm.ai/)
* SGLang documentation — [sgl-project.github.io](https://sgl-project.github.io/)
* NVIDIA Inference Tuning Guide — [docs.nvidia.com/deeplearning/tensorrt/](https://docs.nvidia.com/deeplearning/tensorrt-llm/)
* Reading list (companion to this lecture):
  * "Efficient Memory Management for Large Language Model Serving with PagedAttention" — [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
  * "SGLang: Efficient Execution of Structured Language Model Programs" — [arXiv:2312.07104](https://arxiv.org/abs/2312.07104)
  * "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving" — [arXiv:2401.09670](https://arxiv.org/abs/2401.09670)

Cross-references in this roadmap:

* [Phase 5 → Edge AI → Agent Tool-Dispatch Evaluation with BFCL](../../../3.%20Edge%20AI/Agent%20Tool-Dispatch%20Evaluation%20with%20BFCL/Lecture-01.md) — agent-shape parity discipline
* [Phase 5 → Robotics → VLA Action-Parity Harness](../../../4.%20Robotics/VLA%20Optimization%20and%20Action-Parity%20Harness/Lecture-02.md) — the same gating discipline applied to embodied policies
* [Phase 5 → ML Systems Engineering → Stage 0 Measurement Discipline](../../Guide.md#stage-0-measurement-discipline) — the benchmark harness pattern

---

## Current as of 2026-06

Pinned: vLLM 0.7.x (V1 engine), SGLang 0.4.x, TensorRT-LLM 0.18.x, llama.cpp post-2026-04, H100 / H200 / B200 hardware. Update when V1 stabilizes or a runtime ships a breaking API change.

---

## Next

* Next: [Lecture 02 — Transformer execution, from tokens to bits](Lecture-02.md)
* Up: [Part 1 — Fundamentals](README.md)
