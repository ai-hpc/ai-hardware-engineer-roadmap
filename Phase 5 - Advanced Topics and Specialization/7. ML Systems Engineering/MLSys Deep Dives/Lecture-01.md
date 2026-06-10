# Lecture 01 - MLSys as the Economic-Value Layer

**Collection:** [MLSys Deep Dives](README.md) | **Previous:** [← MLSys Deep Dives index](README.md) | **Next:** [Lecture 02](Lecture-02.md)

---

Before any kernel, compiler, or architecture, one number governs this entire course: **the cost of a token**. Everything an MLSys engineer does — every fused kernel, every quantization scheme, every speculative-decoding head, every hybrid-architecture layer — exists to move that number. So we start there, because if you cannot connect a technique to the cost of a token, you cannot prioritize, defend, or even recognize the work that matters.

This lecture builds the spine the other six hang from: **why systems work is now the product**, the metrics that measure it, and the one equation that turns a kernel speedup into a dollar figure.

---

## Learning objectives

By the end of this lecture, you should be able to:

1. Quantify the 2023–2026 collapse in inference cost and explain what drove it (systems, not silicon).
2. Decompose **price-per-token** into hardware and MLSys factors, and name the lever each course topic pulls.
3. Use the metric stack correctly: **tokens/s, TTFT, TPOT, TOK/$, TCO/Mtok, perf/watt** — and say which is the user's and which is the operator's.
4. Read a perf/$ benchmark (SemiAnalysis-style) and explain why a printed throughput number is a teaching anchor, not deployment truth.
5. Compute a back-of-envelope **$/Mtok** from GPU rental cost and tokens/s, and show how a 2× systems win halves it.

---

## 1. The collapse

In November 2022, GPT-3.5-class intelligence cost roughly **$20 per million tokens**. By late 2024 the same capability was available near **$0.07 per million** — about **280× cheaper in two years**. GPT-4-class input fell from **$30/Mtok** at launch (March 2023) to **$2.50** (GPT-4o, 2024) to **~$0.10** (nano-class, 2025) — over **99%**. DeepSeek-V3 arrived in December 2024 at roughly **$0.14/Mtok** for frontier-class quality — about one-hundredth of GPT-4's launch price.

Epoch AI's rigorous version, holding *capability* fixed: to keep a fixed benchmark score, price falls **between 9× and 900× per year, median ~50×/year**, and faster (median ~200×/year) for the cheapest models since 2024. The declines are **uneven** — cheap, common tasks drop fastest; the frontier of hard reasoning drops slower.

Here is the part that matters for your career: **almost none of that came from cheaper hardware.** An H100 did not get 280× cheaper. The collapse came from MLSys —

```text
   FlashAttention & better kernels    → more tokens/s per GPU
   quantization (FP16→FP8→FP4/INT4)   → more model per byte, more throughput
   continuous batching, paged KV      → higher GPU utilization
   speculative decoding               → more tokens per memory pass
   MoE + MLA + hybrid architectures   → less compute / less KV per token
   compiler fusion & scheduling       → less wasted memory traffic
```

Every one of those is a lecture in this course. The collapse *is* the field. When someone asks what an MLSys engineer does, the honest answer is: **we are why intelligence got 280× cheaper, and we are not done.**

---

## 2. The one equation

Strip inference economics to its core and you get this:

```text
                    energy  +  capital            $ per second to run the box
   price per token = ──────────────────  =  ───────────────────────────────────
                       tokens per second        tokens per second it produces
                       │                          │
                       └── set by hardware ───────┘
                              + power + utilization        ← MLSys lives here too
```

Two ways to lower the price of a token: make the box cheaper (hardware, power, financing — mostly not your job), or make the box produce **more tokens per second** (kernels, compilers, architectures, decode algorithms — *entirely* your job). The denominator is the MLSys engineer's whole world.

This is why the course is ordered the way it is. Each layer is a different way to grow the denominator:

| Layer | Course lectures | How it grows tokens/s |
|---|---|---|
| **Kernel** | 02 | a faster GEMM/attention kernel does the same work in less time |
| **Compiler / runtime** | 03 | fusion cuts memory traffic; megakernels cut launch overhead |
| **Architecture** | 04–05 | SSM/MLA shrink the KV cache; MoE cuts compute per token |
| **Inference algorithm** | 06 | speculative decoding emits multiple tokens per memory pass |
| **Hardware / deployment** | 07 | the right precision on the right silicon at the right batch |

Memorize the equation. Every time this course introduces a technique, locate it on the equation. A technique you cannot place there is a technique you do not yet understand.

---

## 3. The metric stack

"Tokens per second" is too coarse to engineer with. The working metrics split into what the **user feels** and what the **operator pays**.

```text
   USER-FACING (latency / interactivity)      OPERATOR-FACING (cost / efficiency)
   ─────────────────────────────────────      ──────────────────────────────────
   TTFT  time to first token (prefill)         $/GPU-hour   (capital + power + colo)
   TPOT  time per output token (decode)        throughput   tokens/s per GPU (or node)
   tokens/s  per-request interactivity         TOK/$   = throughput ÷ ($/s)
   p50/p99   tail latency                       TCO/Mtok = 1e6 ÷ TOK/$
                                                perf/watt   tokens/s ÷ watts
```

The two are in tension, and managing that tension is the job. You can almost always buy throughput (TOK/$) with latency (batch harder, more requests share each weight-load) — up to the point where TPOT violates the interactivity SLO. So the real target is never one number:

```text
   the SLO frontier:  maximize tokens/s (and TOK/$)
                      subject to  TTFT < X ms  and  TPOT < Y ms
```

Two facts that trip up newcomers:

* **Prefill and decode are different machines.** Prefill (processing the prompt) is compute-bound and parallel — TTFT scales with prompt length. Decode (generating tokens) is **memory-bandwidth-bound** — one token at a time streams the whole weight set from HBM. Most of this course's tricks target decode, because decode is where the time and the cost live for long generations.
* **perf/watt is becoming the binding constraint.** At datacenter scale you are power-limited before you are space-limited. tokens/s/watt is increasingly the number that decides which stack a hyperscaler deploys — which is why FP4 silicon and energy-frugal kernels matter beyond their raw speed.

---

## 4. Reading a perf/$ benchmark

The industry now has open, continuously-updated cross-stack benchmarks — most notably **SemiAnalysis InferenceMAX** (open-source) and its **InferenceX** TCO calculator. They do the thing this lecture insists on: normalize **throughput by total cost of ownership**, and express the result as **TCO per million tokens** plotted against interactivity, for different operator types (hyperscaler vs neocloud vs self-host).

What they show, and what you should internalize:

* TCO is not just the GPU. It is **server capex + power (perf/watt) + colocation/electricity + cost-of-capital**. A cheaper GPU that burns more watts can lose on TCO/Mtok.
* Generational hardware jumps are large *and* software-multiplied: a GB200 NVL72 has been reported at roughly **15× lower cost-per-million-tokens than the prior generation** on reasoning workloads — but that figure already bakes in a year of kernel and runtime improvements on top of the silicon.
* **The numbers move monthly.** A throughput figure printed in any lecture (including this course) is a *teaching anchor* — correct in shape, stale in magnitude. At deployment time, you read the live dashboard, not the textbook.

That last point is a discipline, not a disclaimer. When you quote a tokens/s number in a design review, state the date, the stack version, and the hardware — or you are quoting a number that has already rotted.

---

## 5. The "too cheap to meter" thesis — and why it makes MLSys the job

There is a thesis, riffing on the Atomic-Age promise of "electricity too cheap to meter," that the marginal cost of a token trends toward negligible. The evidence is the §1 collapse: a workload that cost **$10,000/month in 2023 can run for under $200 now**. Whether or not the slope continues, the structural conclusion holds and it is the reason this course exists:

> **Once a capability is commoditized, the only remaining differentiator is the cost of delivering it. That cost is set by MLSys.**

When every serious lab has a comparable model, the competition moves to **who serves it cheapest and fastest** — which is a kernel, compiler, architecture-co-design, and decode-algorithm contest. The economic value of a model-quality improvement decays as competitors catch up; the economic value of a systems improvement is **immediate and compounding**, because it multiplies TOK/$ across every token the company will ever serve. That is why, in 2026, the inference team is a profit center and the systems engineer is the person turning research into margin.

---

## 6. Hands-on: build the cost model

You will reuse this calculation in every later lecture's "Measure it." Build it once, properly.

Given a deployment that rents `G` GPUs at `C` dollars/GPU-hour and produces `S` aggregate tokens/second:

```python
def usd_per_mtok(gpus, usd_per_gpu_hr, agg_tokens_per_sec):
    usd_per_sec   = gpus * usd_per_gpu_hr / 3600.0
    tokens_per_sec = agg_tokens_per_sec
    tok_per_usd   = tokens_per_sec / usd_per_sec
    return 1e6 / tok_per_usd            # $ per million tokens

# Illustrative anchors (verify live $/GPU-hr and tokens/s at deployment):
node = dict(gpus=8, usd_per_gpu_hr=2.50)      # an 8-GPU node at ~$20/hr

for S in (2_000, 5_000, 10_000, 20_000):
    print(f"{S:>6} tok/s  ->  ${usd_per_mtok(agg_tokens_per_sec=S, **node):.2f}/Mtok")
```

The output is the entire thesis in four lines:

```text
  2000 tok/s  ->  $2.78/Mtok
  5000 tok/s  ->  $1.11/Mtok
 10000 tok/s  ->  $0.56/Mtok
 20000 tok/s  ->  $0.28/Mtok
```

The hardware cost (`$20/hr`) never changed. Every drop came from the **denominator** — from tokens/s, which is to say from MLSys. A kernel engineer who doubles attention throughput, an architecture that halves the KV cache so you can batch twice as deep, a speculative decoder that emits two tokens per memory pass: each one walks you *down that column*, and the column is dollars.

> **One caveat to keep you honest:** aggregate tokens/s depends on batch size, and batching trades against TPOT. The cost model above is only valid *at a fixed interactivity SLO*. Always quote `$/Mtok` together with the TPOT it was measured at — a cheap token nobody will wait for is not cheap, it is unsold.

---

## 7. Mini-lab: place the field on the equation

A two-part exercise that sets up the whole course.

1. **The cost model.** Take a real model + runtime you can run (or a published benchmark). Measure or read its aggregate tokens/s at a fixed TPOT, and compute `$/Mtok` with §6. Record TTFT, TPOT, tokens/s, TOK/$, and `$/Mtok` as your baseline row — you will add rungs to this table in every later lecture.
2. **The map.** Write the §2 equation at the top of a page. Under the denominator, list the seven lecture topics of this course and, for each, one sentence on *how* it grows tokens/s. If you cannot write the sentence yet, that lecture is where you'll learn it — but the slot on the equation should be obvious even now.

Deliverable: one baseline cost-model row, and the annotated equation. Keep both; the course is, in a sense, the exercise of filling in that page.

---

## Key takeaways

- Inference cost fell ~**280×** (GPT-3.5-class) and **>99%** (GPT-4-class) from 2023–2026, a **median ~50×/year** at fixed capability — driven by **MLSys, not cheaper silicon**.
- The governing equation: **price/token = (energy + capital) / tokens-per-second**. Hardware sets the numerator; **MLSys grows the denominator**, and that is the whole job.
- Metrics split into **user-facing** (TTFT, TPOT, tokens/s, p99) and **operator-facing** (TOK/$, TCO/Mtok, perf/watt). The target is the **SLO frontier**: max throughput subject to latency bounds — never a single number.
- **Prefill is compute-bound; decode is memory-bound.** Most acceleration targets decode, because that is where long-generation time and cost live.
- Read perf/$ benchmarks (SemiAnalysis InferenceMAX/InferenceX) by **TCO/Mtok**, and treat any printed throughput as a dated teaching anchor, not deployment truth.
- Once capability commoditizes, **cost-to-serve is the differentiator** — so every systems win is immediate, compounding economic value. That is why MLSys is the product.

---

## References

- Epoch AI, "LLM inference price trends" (median ~50×/year): [https://epoch.ai/data-insights/llm-inference-price-trends](https://epoch.ai/data-insights/llm-inference-price-trends)
- Token cost / AI price index (GPT-3.5 ~280×, GPT-4 >99%): [https://tokencost.app/blog/ai-price-index](https://tokencost.app/blog/ai-price-index)
- SemiAnalysis, "InferenceMAX — open-source inference benchmarking": [https://newsletter.semianalysis.com/p/inferencemax-open-source-inference](https://newsletter.semianalysis.com/p/inferencemax-open-source-inference)
- SemiAnalysis InferenceX TCO calculator: [https://inferencex.semianalysis.com/calculator](https://inferencex.semianalysis.com/calculator)
- Introl, "Inference unit economics — true cost per million tokens": [https://introl.com/blog/inference-unit-economics-true-cost-per-million-tokens-guide](https://introl.com/blog/inference-unit-economics-true-cost-per-million-tokens-guide)
- *AI Inference Engineer 2026* — the production serving-stack companion to this course.

---

## Current as of

2026-06. Cost figures: GPT-3.5-class ~280× (Nov 2022→Oct 2024), GPT-4-class >99% (2023→2025), DeepSeek-V3 ~$0.14/Mtok (Dec 2024), Epoch median ~50×/yr. `$/Mtok` worked example uses an illustrative $2.50/GPU-hr — **verify live GPU rental rates and tokens/s at deployment** via InferenceMAX/InferenceX; the numbers move monthly.

---

*Next: [Lecture 02 — The kernel-language explosion](Lecture-02.md)*
