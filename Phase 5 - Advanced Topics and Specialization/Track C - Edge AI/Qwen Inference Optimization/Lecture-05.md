# Lecture 5: Cross-Model Strategies and Production Serving — Tying Qwen3-4B and Qwen2.5-72B Together

## Overview

Lectures 3 and 4 treated the edge model (Qwen3-4B-Q4) and the datacenter model (Qwen2.5-72B-FP16) as separate worlds. They aren't. In production they coexist — usually because the same product wants edge latency *and* server-class quality. This lecture covers the architectures that span both:

* **Speculative decoding with Qwen3-4B drafting for Qwen2.5-72B** — the most direct cross-model use.
* **Edge↔cloud routing** — when to escalate which queries, observability needed for that decision.
* **Cascaded inference** — small model handles 80% of traffic, large model handles the rest.
* **Production observability and capacity planning** — what to measure, how to size, where workloads break.

This is the system-engineering lecture. Less kernel math, more dashboards and on-call decisions.

By the end you should be able to:

* Compute the expected wall-clock win from speculative decoding given a draft/target pair.
* Design an edge↔cloud routing policy that meets a p95 latency budget under cost constraint.
* Pick the production observability metrics that catch the most common failure modes.
* Size a Qwen2.5-72B serving fleet for a given user load.

---

## 1. Speculative Decoding: 4B Draft for 72B Target

The basic idea was sketched in Lecture 3. Here we make it concrete for the Qwen pair.

### 1.1 The math

Let `t_draft` = wall time to decode one token on the draft, `t_target_K` = wall time for the target to verify a window of K tokens, `α` = average acceptance rate (fraction of drafted tokens the target would have produced).

Effective tok/s:

```
tok_per_step = 1 + α + α² + … + α^(K-1) + α^K       (geometric + one free)
             = (1 - α^(K+1)) / (1 - α)              for K finite
             ≈ 1 / (1 - α)                           for K → ∞

wall_per_step = t_draft × K + t_target_K

tok_per_sec  = tok_per_step / wall_per_step
```

For Qwen3-4B-Q4_K_M as draft and Qwen2.5-72B-FP16 as target on 4×H100:

```
t_draft (one Qwen3-4B token on a single H100 hosting draft) ≈ 5 ms
t_target_K (Qwen2.5-72B forward pass with seq_len=K)         ≈ 30 ms + 0.5 ms × K
α (Qwen3 → Qwen2.5 same-family acceptance)                   ≈ 0.55
```

With K=5:

```
wall = 5 × 5 ms + 30 + 2.5 ms = 57.5 ms
tok_per_step = (1 - 0.55^6) / (1 - 0.55) = 2.18
tok/s = 2.18 / 0.0575 ≈ 38 tok/s
```

vs. baseline target alone:

```
30 ms per token → 33 tok/s
```

**Speculative dec wins ~15%** at this α and K. To get bigger wins you need higher α (better draft–target alignment) or lower draft cost.

### 1.2 Why the Qwen3-Qwen2.5 pair is awkward

The cross-family acceptance rate is moderate (~50–60%) because:
- Different post-training (Qwen3 has thinking mode embedded).
- Different vocabularies (151 936 vs 152 064) — the **tokenizer mismatch alone** disqualifies vanilla speculative decoding. You need a vocab projection or to share tokenizers.

**Practical recipe:** use **Qwen2.5-0.5B-Instruct** or **Qwen2.5-1.5B-Instruct** as the draft for Qwen2.5-72B. Same family, same tokenizer, ~0.7 acceptance rate. This is the standard production pairing.

The Qwen3-4B for Qwen2.5-72B pair is interesting **only** if you don't care about cross-family quality — i.e. you treat the 4B as a "small distilled version of approximately the same intent." Not a clean win.

### 1.3 EAGLE and Medusa — inline alternatives

Two-model speculative decoding has **memory overhead** (two models loaded). **Inline-speculative-decoding** variants avoid this:

* **Medusa** adds extra decoding heads to the target model. Each head predicts a different future token position; you verify all simultaneously.
* **EAGLE / EAGLE-2** trains a small autoregressive draft head on top of the target's frozen activations. Less memory than a full draft model, higher α than naïve heads.

For Qwen2.5-72B in production, EAGLE-2 has shipped widely and gives 1.6–2.0× over baseline at α ≈ 0.8 — substantially better than two-model spec dec.

---

## 2. Edge↔Cloud Routing

The product question: a user query arrives, do we answer **on-device** with Qwen3-4B or **escalate to cloud** Qwen2.5-72B?

### 2.1 Routing signals

| Signal | Send to 72B if | Reasoning |
|---|---|---|
| Prompt token count | > 8 k | Edge model context handling degrades faster |
| Detected language | Not in top-3 of edge model's training | Edge models drop multilingual quality first |
| Task category (classifier) | Code generation, math reasoning, multi-step planning | These are where 4B → 72B gap is largest |
| Conversation history depth | > N turns | Coherence over long sessions favors larger model |
| User tier | Premium | Cost-based |
| Explicit user request | "be thorough" / "think carefully" | Direct signal |
| Network availability | Cloud reachable + latency budget allows | Hybrid systems must handle offline |

### 2.2 A two-stage decision

```
                     User query
                          │
              ┌───────────┴──────────┐
              ▼                       ▼
    fast classifier (50 ms)     simple length/lang check
              │                       │
              └───────────┬──────────┘
                          ▼
                  Route decision
                          │
              ┌───────────┴──────────┐
              ▼                       ▼
     Run on Qwen3-4B (Edge)   Run on Qwen2.5-72B (Cloud)
              │                       │
              ▼                       ▼
     If confidence < τ →   ──┐    Return answer
     escalate to 72B         │
                             ▼
                        Cloud call
```

The "self-evaluate then escalate" path is powerful but adds latency. Use it sparingly — for a "verify high-stakes outputs only" policy.

### 2.3 Cost vs latency curve

A simple production data point (numbers from typical mid-2026 deployments):

```
Pure 72B:   $0.50/M tokens   |  p50 = 600 ms TTFT, p95 = 2.5 s
Pure  4B:   $0.02/M tokens   |  p50 = 100 ms TTFT, p95 = 800 ms (on-device)
70/30 mix:  $0.16/M tokens   |  p50 = 250 ms TTFT, p95 = 2.0 s
```

The 70/30 mix is usually the right starting point. Tune toward 90/10 or 50/50 based on quality complaints in dogfood.

---

## 3. Production Observability

What to put on a dashboard for Qwen serving, in priority order:

### 3.1 Latency

* **TTFT (time-to-first-token)** — p50, p95, p99. The most user-visible metric.
* **ITL (inter-token latency)** — p50, p95. After first token, how long between subsequent tokens.
* **Total request time** — for completeness.

### 3.2 Throughput

* **Decoded tokens per second per GPU** — saturation indicator.
* **Generated tokens per second cluster-wide** — capacity.
* **Prefill tokens per second per GPU** — separate metric because the path is different.

### 3.3 Utilization

* **GPU compute utilization** (`nvidia-smi`) — should be ≥ 70% under load.
* **GPU memory utilization** (% of HBM in use) — should be 85–92% on weights + KV combined.
* **KV cache occupancy** (vLLM exposes this) — higher = better continuous-batching efficiency.
* **NCCL collective time** (% of step time spent in NCCL) — > 20% means rebalance TP or check fabric.

### 3.4 Quality

* **Sampling rejection rate** (if using guided decoding) — too high = model fighting your schema.
* **EOS rate per response length** — sudden drops suggest the model is failing to terminate.
* **Refusal rate** — for safety-tuned models, drift in this metric indicates a bug or unintended retraining.

### 3.5 Cost

* **$ per 1k input tokens** and **$ per 1k output tokens** — your real economics.
* **GPU-hour idle time** — capacity overprovisioning.

### 3.6 What "good" looks like for Qwen2.5-72B on 4×H100 at moderate load

| Metric | Healthy |
|---|---|
| TTFT p50 | 200–400 ms |
| TTFT p95 | < 2 s |
| ITL p50 | 25–40 ms |
| ITL p95 | < 80 ms |
| GPU util | 70–85% |
| HBM util | 85–92% |
| KV occupancy | 75–95% |
| NCCL fraction | 5–10% |

---

## 4. Common Production Pathologies

A short field guide of "things you'll see, things they mean":

### 4.1 Edge

| Symptom | Cause | Fix |
|---|---|---|
| Decode at 0.2 tok/s, GPU @ 0 MHz | DVFS parked | `nvpmodel -m 0; jetson_clocks` |
| Tok/s drops 50% after device wakes from suspend | Clocks unlocked by suspend | Re-run `jetson_clocks` in resume hook |
| First-decode token slow, rest fast | CUDA Graph not captured yet | Warm up explicitly |
| OOM at long prompts on 8GB Orin Nano | KV cache hitting limit | Lower `n_ctx`, INT8 KV, smaller batch |
| Gibberish past token 10 | RoPE flavor wrong (vanilla vs NeoX) | Verify rope_kernel layout |
| Refuses on benign prompts | Wrong chat template | Verify against tokenizer.apply_chat_template() |

### 4.2 Datacenter

| Symptom | Cause | Fix |
|---|---|---|
| 5× slower than published vLLM benchmarks | NCCL on PCIe instead of NVLink | `NCCL_P2P_LEVEL=NVL` + check `nvidia-smi topo -m` |
| TTFT spike at random intervals | Long prompts not chunked | `--enable-chunked-prefill` |
| Throughput collapses under high concurrency | KV cache fragmentation | Newer vLLM with prefix-caching; raise `--gpu-memory-utilization` |
| 50% GPU util at saturation | Static batching, not continuous | Confirm runtime version supports it |
| Inconsistent answers across replicas | Different RoPE or YaRN config | Pin runtime container version and config |
| OOM at 32 k context with batch=16 | KV blocks exhausted | Lower batch or shorten `--max-model-len` |
| Sudden p99 latency spike | One sequence ran to `--max-model-len` | Cap output length, kill runaway generations |

---

## 5. Capacity Planning Example

Goal: serve a chat product with these characteristics:

* 1000 daily active users.
* Average 50 turns/user/day.
* Average input 300 tokens, output 400 tokens.
* 70% of queries on Qwen3-4B edge (free), 30% on Qwen2.5-72B cloud.

Cloud load:

```
Queries per day: 1000 × 50 × 0.30           = 15 000
Tokens per query in/out: 300 / 400
Output tokens per day: 15 000 × 400         = 6 000 000
Output tok/s averaged over 24h:             ≈ 70 tok/s
```

Peak factor (typical chat traffic ~5× average during peak hour):

```
Peak output tok/s: 70 × 5 ≈ 350 tok/s
```

Per Lecture 4: Qwen2.5-72B on 4×H100 sustains ~250 tok/s single-stream-saturated and ~560 tok/s at batch=32. With continuous batching at moderate concurrency, ~400 tok/s is a safe planning number for one 4-H100 box.

**You need one 4-H100 box at peak**, with a second as failover/burst capacity. Total capex: ~$200k for 8 × H100 SXM + servers (2026 pricing), or ~$25/hour × 24 × 365 × 2 = ~$440k/year cloud rental.

Per-1k-tokens cost at amortized cost: ~$0.20-0.40 — competitive with the OpenAI API at the time of this writing.

---

## 6. Hybrid Failure Modes

The edge↔cloud split has its own pathologies:

1. **Edge fails open** — when the cloud is unreachable, do you serve the (worse) edge response, or fail explicitly? Both are valid, but you must decide and instrument.

2. **Cloud is consistently better — users learn to skip the edge.** If your "be more thorough" button always produces better output, users tap it for everything, and your cost model breaks. Solution: make the edge model good enough that the difference is small for most queries.

3. **Quality drift between edge updates.** Your edge model ships with the app. Updating it is slow. Your cloud model can move daily. Eventually they drift; users on stale app builds get worse experiences. Add server-side analytics that tag responses by which model produced them, and watch the quality delta.

4. **Privacy boundary mismatch.** Edge queries don't leave the device; cloud queries do. Users will assume the boundary is at "the obvious one" — usually they assume the edge model handles "private" queries. Be explicit about which queries go where.

---

## 7. Looking Forward — What Changes by Late 2026

Selected developments visible in the literature as of May 2026:

* **MoE-class Qwen** — A "Qwen3-30B-A3B" mixture-of-experts release. 30 B total params, ~3 B active per token. Inference economics radically different — much closer to 4B compute, much closer to 30B quality. Worth tracking.

* **Speculative decoding gets easier** — EAGLE-3 reduces draft-head training cost and pushes acceptance to ~0.85 across families.

* **KV cache compression beyond INT4** — methods like H2O, StreamingLLM-style sinks, and KIVI have moved from research to production. Expect long-context decode bandwidth to halve again.

* **Edge silicon catches up** — Apple's M5-series and Qualcomm Snapdragon X75/X80 successors push edge throughput past 100 tok/s for 4B-class models, eroding the cloud-only quality advantage.

* **Standardized agent harnesses** — model serving stacks growing in to expose tool-calling, streaming partial responses, and structured outputs as first-class primitives (vLLM has `--tool-call-parser`, SGLang has compile-time structured generation).

The systems engineering doesn't fundamentally change. The constants in the roofline math get better, the models get smarter at the same parameter count, and the production observability we've talked about stays the right things to measure.

---

## Hands-On Exercises

1. **Build a 4B↔72B router.** Use a small embedding model + logistic regression on prompts to predict whether a query "needs" 72B (label by comparing outputs by quality on a held-out set). Measure routing accuracy and total cost saved at various confidence thresholds.

2. **Speculative-decoding wall-clock measurement.** Set up Qwen2.5-1.5B drafting Qwen2.5-72B in vLLM. Measure aggregate tok/s and acceptance rate on three workloads: chat, code, math. Discuss why α differs by workload.

3. **Production observability deployment.** Run vLLM with `--engine-metrics-enable` and scrape the Prometheus endpoint. Build a Grafana dashboard with the §3 metrics. Generate synthetic load (e.g., `vllm-benchmark`). Identify which metrics move first under load — that's your saturation signal.

4. **Failure injection.** While serving from 4×H100 vLLM, deliberately break NVLink (`NCCL_P2P_DISABLE=1`). Measure the throughput drop. Compare to expected based on collective bandwidth — confirm the diagnosis from Lecture 4.

5. **Capacity sizing.** Given the §5 scenario but with **2× higher peak factor** and **40% routed to 72B**, redo the capacity math. How many 4-H100 boxes do you need? What's your per-token cost?

6. **Edge-update drift study.** Pick a query set. Run on Qwen3-4B (edge) and Qwen2.5-72B (cloud) every week for 4 weeks (simulate with model snapshots). Track BLEU/ROUGE/semantic similarity between edge and cloud responses over time. This is the kind of telemetry that catches drift early.

---

## Key Takeaways

| Takeaway | Why it matters |
|---|---|
| Cross-family speculative decoding is fragile | Tokenizer + post-training mismatch caps α |
| Inline spec dec (EAGLE-2/3) beats two-model | Lower memory + higher α in practice |
| Routing 70/30 to 4B/72B is a strong default | Tune with quality complaints, not aesthetics |
| TTFT, ITL, KV occupancy are the right top-level dashboards | They catch the most common pathologies first |
| Pathology lists are short and repeatable | Most production "weird performance" bugs are on the same list |
| Hybrid edge/cloud has its own failure modes | Privacy boundary, update drift, fail-open behavior |
| MoE-Qwen will change the math again in late 2026 | Track active params, not total params |

---

## Resources

* **[Speculative Decoding paper](https://arxiv.org/abs/2211.17192):** Two-model spec dec formal analysis.
* **[Medusa paper](https://arxiv.org/abs/2401.10774):** Inline spec dec via extra heads.
* **[EAGLE paper](https://arxiv.org/abs/2401.15077) and [EAGLE-2](https://arxiv.org/abs/2406.16858):** Better inline spec dec; the de-facto production technique.
* **[vLLM operator metrics](https://docs.vllm.ai/en/latest/serving/metrics.html):** Production observability surface.
* **[SGLang structured generation](https://github.com/sgl-project/sglang):** When constrained outputs matter.
* **[H2O — heavy-hitter KV compression](https://arxiv.org/abs/2306.14048):** Long-context decode bandwidth reduction.
* **[StreamingLLM — attention sinks](https://arxiv.org/abs/2309.17453):** Long-running session KV management.
* **[Phase 5 — Qwen Inference Optimization series](README.md):** Other lectures in this series.
* **[Phase 5 — Edge LLM Inference Internals](../Edge%20LLM%20Inference%20Internals/Lecture-01.md):** The foundational roofline lecture this series builds on.
* **[Phase 5 — NCCL Deep Dive](../../Track%20A%20-%20GPU%20Infrastructure/Nvidia%20GPU/HPC%20Setup/NCCL-Deep-Dive/README.md):** Multi-GPU collective layer for the 72B path.
