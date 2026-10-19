# Part 4 · Lecture 01 — The Workload, the Baseline, and the Ladder

**Practice:** [CPU source lab](CPU-Source-Lab.md). Historical GPU results are source reports; use the **Answers** after attempting the self-check.


## Overview

Before any optimization, three questions have to be answered concretely: **what is the workload**, **what are you being compared against**, and **where are you now**. Get any of them wrong and every number you produce afterwards is decoration.

This lecture answers all three for the case study, then lays out the full measured ladder — every frontier advance, in order, as recorded in the repository's own pinned baseline file.

By the end you should be able to read a model card and a node spec and derive the *shape* of the inference problem before writing a line of code: how much HBM the weights need, how many collectives a token costs, which phase will dominate, and what a credible baseline even is when the mainstream engines cannot load your model.

---

## 1. The model — Kimi K3 read as a systems artifact

Part 3 Lecture 01 taught how to read an MoE model card. Kimi K3 is that exercise on hard mode.

| | |
|---|---|
| Parameters | **2.8T** total, **104B activated** in the official model card; legacy `2.8T.A50B` implementation label is not authoritative |
| Layers | **93** — **24 MLA** (full attention) + **69 KDA** (linear, recurrent) |
| Hidden / vocab | 7168 / 163,840 |
| Context | 1,048,576 |
| MoE | **896 routed experts**, top-16, **2 shared** · latent MoE at **3584**, expert FFN 3072 |
| Attention (full) | **MLA** — `q_lora 1536`, `kv_lora 512`, **NoPE-only**, sigmoid output gate |
| Attention (linear) | **KDA** — 96 heads × 128, conv kernel 4, full-rank gate, `gate_lower_bound −5.0` |
| Activation | SiTU nonlinearity in the gated FFN (`β 4.0`, linear `β 25.0`) |
| Extras | cross-layer residual attention, `block_size 12` |
| Vision | MoonViT-V2 in the official card; the pinned implementation describes its encoder as MoonViT-3d |

Four things in that table change the systems problem qualitatively, and each is a *different* lesson from the dense-70B and 671B-MoE cases in Parts 2 and 3.

**896 experts exceeded the selected historical baseline's limit.** The case-study fork raises a 512-expert cap to 1024. This describes that pinned engine history, not a current limit of all mainstream runtimes. The model does not load slowly — it does not load. §3 is about what that does to the idea of a baseline.

**The hybrid attention stack has two cost curves, not one.** 24 MLA layers grow a KV cache with context; 69 KDA layers carry fixed-size recurrent state. Optimizing one does nothing for the other, and a profile that averages over layers hides which is which. This is the Mamba/hybrid architecture family from [MLSys Deep Dives Lecture 04](../../MLSys%20Deep%20Dives/Lecture-04.md), but as an engineering constraint rather than a survey entry.

**The routed experts live in a down-projected space.** `expert_latent_length` is **3584**, not `hidden_size` 7168. Use hidden width for an expert input/output axis and that axis is wrong by 2×; the separate FFN intermediate width remains 3072. It is also why the tensor-parallel all-reduce is 3584-wide — [Lecture 07](Lecture-07.md).

**Top-16 of 896 is aggressive sparsity.** 104B active from 2.8T is about a 26.9:1 total/active ratio. Routed expert traffic depends on the selected experts and quantized tensor layout; top-16 alone does not imply uniformly random access. Measure routing correlation and per-rank load.

### 1.1 Three traps that produce fluent garbage

The repository documents these as configuration traps, and all three share one property worth naming immediately, because it is the theme of [Lecture 10](Lecture-10.md): **they do not crash. They produce plausible text.**

```text
1.  full_attn_layers is 1-INDEXED.
    The converter tests  (il + 1) in full_attn_layers.
    Off by one  ->  you run KDA where MLA belongs, and vice versa.
    Result: fluent text, wrong model.

2.  MLA is STORED AS MQA.
    head_count_kv = 1,  key_length = kv_lora + qk_rope = 576.
    A per-layer head_count_kv == 0 is what marks a KDA layer.
    Read it as real MQA -> your shard math divides 1 by tp_size and gets 0.

3.  Routed experts are in a DOWN-PROJECTED space.
    expert_latent_length 3584,  NOT hidden_size 7168.
    Using hidden width for the routed expert input/output axis doubles that axis incorrectly; the FFN intermediate axis remains 3072.
```

A compile error costs you an hour. A silently wrong `full_attn_layers` costs you a week of benchmarking a model that is not the model.

---

## 2. The node and the quant — why the target is what it is

### 2.1 Hopper, on purpose

The historical target is eight H200 SXM GPUs. The advertised capacity is **141 GB per GPU**; eight times that is 1128 **GB**, not 1128 GiB. For an allocation budget, use the actual byte count returned by CUDA and convert by 2³⁰ for GiB. Availability and rental prices change, so this course does not infer today's cheapest node from an August choice.

Hopper lacks Blackwell's native FP4 tensor-core path. That does **not** prohibit FP4/low-bit storage: software may unpack to supported compute types. Storage format, arithmetic instructions, and library version are separate choices. This case uses GGUF integer quantization; it is not evidence that all low-bit formats have the same cost.

The model table follows the [pinned official K3 model card](https://huggingface.co/moonshotai/Kimi-K3/blob/c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721/README.md) and [config](https://huggingface.co/moonshotai/Kimi-K3/blob/c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721/config.json). Its 104B activated count supersedes the engine's legacy A50B label.

### 2.2 The quant ladder, and the difference between "default" and "target"

| quant | GiB | 8× H200 | top-1 vs lossless |
|---|---:|:-:|---:|
| **UD-IQ1_S** *(default)* | **553** | ✅ measured | 78.9% |
| UD-IQ2_XXS | 662 | ✅ | 84.1% |
| **UD-Q2_K_XL** *(accuracy target)* | **802** | ✅ not yet run | **90.4%** |
| UD-Q4_K_XL | 1407 | ❌ | — |

These are the repository's historical quantization-size/quality claims. The percentages lack a reproduced evaluation dataset in this course; they are not general task-accuracy rates or a guarantee that a quant fits at every context. Parity against the same quantized weights is separate from quality relative to the original model.

Two distinct claims, and the repo keeps them separate on purpose:

* **UD-Q2_K_XL is the accuracy knee** — 90.4% top-1 against a lossless reference — and is what the project says it should ultimately be judged on.
* **UD-IQ1_S is the default** because a default that triggers an 802 GiB download present on no machine means every fresh invocation dies before doing anything.

The trap this creates is a *measurement* trap, and the fix is worth stealing. IQ1_S is smaller and therefore decodes faster. Pin an IQ1_S number into an unqualified baseline slot, and every future gain is measured against an inflated reference — understated **forever**. So the slots carry the quant in the name: `KIMI_K3_H200X8_IQ1S_LLAMA_128K`. A number measured under one quant can never be read as the other.

> **Steal this.** Any baseline value whose meaning depends on a configuration axis should carry that axis *in its identifier*, not in a comment. Comments do not survive a copy-paste; names do.

### 2.3 All weights in HBM, enforced

553 / 8 = **69.125 GiB**, an ideal equal share of the file payload. It is not the observed allocation. The source reports roughly **123 GiB resident per rank**; replication, unpacked representations, scratch, and state need a separate allocation inventory. Even the simple 531 GiB sharded + 22 GiB replicated model gives 531/8 + 22 = **88.375 GiB per rank**, before other allocations. That is tight, and the temptation under pressure is to let a little spill. The harness refuses: `kimi_k3_check_fits` rejects a partially-offloaded configuration rather than running one, and it **prices the KV cache in** rather than assuming flat headroom (27 GiB **per rank** for 24×576×2 bytes/token at 1,048,576 tokens; other cache layouts differ).

The benchmark policy excludes offload. Offload can be reproducible, but requires a different declared configuration including host memory, PCIe traffic, residency, and cache state. A precheck based on file sizes cannot prove runtime fit: measure peak allocated/resident bytes per GPU.

---

## 3. Why the baseline is a fork — and what to do when nothing can run your model

Every other model in this engine's family is benchmarked against `ggml-org/llama.cpp` at a pinned commit. The selected historical upstream revision could not load K3 at its 512-expert cap, so this record uses a fork; it makes no claim that current upstream still lacks support.

This is a real and under-discussed situation. The honest options are:

| Option | Problem |
|---|---|
| Compare to a *different* model | Measures the model, not the engine. |
| Compare to yourself over time | No absolute anchor; a slow engine mints big relative wins forever. |
| Compare to a smaller quant that upstream loads | Different weights, different bytes/token. Not the same measurement. |
| **Pin a fork that can load it** | Fork can move under you. Requires provenance machinery. |

The project took the last and built the machinery. The reference is [`unslothai/llama.cpp`](https://github.com/unslothai/llama.cpp) PR #48, pinned by repo + ref + commit + base commit in `bench/scripts/reference.lock`. Four things in that fork are load-bearing rather than cosmetic:

1. `LLAMA_MAX_EXPERTS 1024` — without it the model asserts at load.
2. The `LLM_ARCH_KIMI_K3` graph — hybrid KDA + MLA, latent MoE, `situ`, cross-layer attention residual, MLA output gate, full-rank KDA gate.
3. `graph_max_nodes = max(n_tokens × 160, 64 × n_tensors)` — the generic `× 40` budget shared by other hybrid architectures is exhausted at ubatch 3840.
4. Four **required-not-defaulted** KV keys (`expert_latent_length`, `attn_res.block_size`, both `situ` betas). Silently defaulting them loads cleanly and emits garbage — precisely what a baseline must refuse to do.

### 3.1 The provenance problem, and a fix worth copying

A PR head on someone else's fork is not immutable. It can be force-pushed. Fetch-by-SHA availability depends on reachability and server policy. This harness fetches the PR ref and checks its resolved SHA; it does not rely on the ref being immutable.

The harness fetches the *ref*, then **asserts** it resolves to the pinned commit. A force-push therefore **fails the run** instead of quietly moving the baseline underneath every future comparison.

The same reasoning produced a weekly `pin-audit` job, because the two external things the baseline depends on and cannot control are exactly the two things that drift: the fork's PR branch, and the quantized weights in someone else's Hugging Face repo (a re-upload can change the shard count). Catching drift on a schedule is cheaper than discovering it while paying for a node.

> **The general rule.** If your baseline lives in someone else's repository, you do not have a pinned baseline — you have a *request* for one — until something in your harness fails loudly when it moves.

---

## 4. Where it started

2026-07-31, first measurement, 8× H200, UD-IQ1_S, both engines on the same box:

```text
                    llama.cpp     sparkinfer-k3
  decode @ ctx 64      18.32          3.55        ~5.2x slower
```

The repo's own note on that number is worth quoting, because it is the correct attitude toward a bad first result:

> *"That is the expected shape for a correctness-first fp32 executor against llama.cpp's mature CUDA path (integer dot products, fused ops, graph capture) — but it means the K3 frontier starts far behind."*

Three specific things the reference had and the new engine did not: **integer dot products** (llama.cpp's quantized mat-vec path), **fused ops**, and **graph capture**. Those are not incidental — they are, almost exactly, the subjects of [Lecture 05](Lecture-05.md), [Lecture 05](Lecture-05.md) again, and [Lecture 08](Lecture-08.md). A correctness-first f32 executor is *supposed* to lose to a mature integer path. Knowing which three things you are missing is a roadmap, not an excuse.

### 4.1 The number that was hiding the real one

That `ctx 64` figure was not the workload. It was what the harness measured because `kimi_k3_tp_bench` hardcoded `max_ctx=64` — while the baseline slot was named `_128`, and the contribution guide said 128k.

When the scored context was corrected to a real 131,072:

| 8× H200, UD-IQ1_S, same weights | ctx 64 | ctx 131,072 | lost |
|---|--:|--:|--:|
| llama.cpp | 18.32 | 18.44 | ~0% |
| sparkinfer-k3, as first measured | 10.34 | **1.00** | **−90%** |

Both representations are compressed latent MLA caches. The earlier SparkInfer path used 576 F32 values and one block per head, while the reference used a smaller F16 representation and different parallelization. The reference curve is nearly flat over these measured workloads; this does not make full attention constant-cost in context length.

**Scoring at 64 hid an 18× gap behind a 1.8× one, and pointed the incentive at a context nobody runs.** That single sentence is the reason [Lecture 02](Lecture-02.md) comes before every optimization lecture in this part, and the reason [Lecture 06](Lecture-06.md) exists at all.

---

## 5. The ladder — decode at 128k

The `frontier` is the best number measured on `main` by a sealed eval round. It is stored in `reference.lock`, rewritten by the bot each round, and **raise-only** — so a slow box cannot deflate it and mint easy tiers for everyone behind it. Its commit history is therefore an audit trail of the whole project.

Here it is, in order, at the real 131,072-token scored context:

```text
   1.00 →  4.53 →  9.04 → 16.06 → 17.46 → 18.14 → 18.88 → 20.14 → 21.24
        → 22.12 → 26.09 → 29.93 → 33.58 → 35.63 → 40.02 → 44.85 → 46.48
                                                        ... then 56.82 → 60.17

   llama.cpp on the same box, same weights, throughout:  18.4435
```

Seventeen values, hence **sixteen transitions** in the displayed sequence. Read three things off that sequence:

**There is no 60× change in it.** The first two transitions are 4.53× and 2.00×; the next, 9.04 → 16.06, is another **77.7%**. Later large steps include 22.12 → 26.09 (**17.9%**). The 60× is a *product*, not a discovery. This is the single most important structural fact about real optimization work and the reason the reward system pays marginal gains.

**It crosses the reference around 18.88.** Everything before that is catching up; everything after is a lead. The tier system changes character at that crossing — see [Lecture 02 §4](Lecture-02.md).

**The reference itself moved once, and it was not a speedup.** Between 26.09 and 29.93 sits a commit that corrected the llama.cpp pin from **16.7026 → 18.4435**. The old value was a single rep taken on a *different box*; the new one is a 3-rep median on the box that scores every PR. Nobody got faster or slower. The yardstick was wrong, and correcting it made every subsequent tier ~9.4% smaller.

### 5.1 The largest merged steps, with their own claimed numbers

The frontier is measured on `main` after a merge; a PR's own claim is a separate measurement of a separate tree. They usually agree closely — `#114` claimed 45.38 and `main` then measured 44.85; `#115` claimed 46.49 and `main` measured 46.48 — but they are not the same number, and conflating them is how a ladder starts drifting from reality.

| PR | Tier | Claim, in its own words | Family |
|---|---|---|---|
| [#25](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/25) | `xl` | Q8_0 projection path — 3.54 → 9.94 tok/s, bit-identical | [L05](Lecture-05.md) |
| [#49](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/49) | `xl` | split MLA decode over context — **4.49× at 128k** | [L06](Lecture-06.md) |
| [#57](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/57) | `xl` | cut 128k decode in half — batch MLA heads per block, widen Q8_0 projections | [L06](Lecture-06.md) |
| [#63](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/63) | `xl` | head-shard both attention bands, budget the split cap | [L06](Lecture-06.md) |
| [#89](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/89) | `l` | device-resident decode — per-rank CUDA graphs + split MLA combine (**+22.7%**) | [L08](Lecture-08.md) |
| [#96](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/96) | `l` | 2-D MoE sharding — expert groups × FFN band (**+12.1%**) | [L07](Lecture-07.md) |
| [#107](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/107) | `l` | warp-budget projection tier, banded LM head, one-rendezvous all-reduce (**+15.9%**) | [L04](Lecture-04.md), [L07](Lecture-07.md) |
| [#114](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/114) | `l` | overlap independent work inside each decode layer — 41.31 → 47.00 | [L08](Lecture-08.md) |
| [#127](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/127) | `xl` | widen small-grid launches **and depth-gated approximate expert skipping** — 48.87 → 59.26 | [L04](Lecture-04.md) |

Note which families appear: **attention shape** (three of the four `xl`s), **launch geometry**, **graph residency**, **quantized projection paths**, and **expert sharding**. Not one of them is "a better matmul." That distribution is the actual curriculum of Lectures 04–08.

---

## 6. The ladder — prefill at 32k

On 2026-08-05 the project **changed what it scored**: prefill at 32k became the tier basis, and decode at 128k became a regression guard. The frontier number visibly *drops* at that point — from 46.48 to 40.35 — because it started measuring a different thing.

```text
  40.35 → 53.02 → 59.59 → 66.62 → 69.02  →  (98.80, unsealed)  →  99.68

  llama.cpp on the same box:  143.88 ± 0.23 tok/s  (±0.16%)
```

The reason for the switch, from the contribution guide: decode was the right thing to score while the engine was 18× behind there; at 3.08× ahead the remaining headroom was small, and **the untouched gap was prompt ingestion**. There was no batched prefill at all — every prompt token went through the single-token decode step, so ingesting 32,768 tokens took 812.2 seconds. *Ingesting was not faster than generating.*

[Lecture 09](Lecture-09.md) is that story in full, including the stale-seed accident: 40.35 was hand-seeded and attributed to the wrong commit, so three PRs optimized against a number 31% below reality and reported gains of +25.4%, +24.5% and +5.3% that were actually **−4.6%, −5.3% and −19.8%**.

### 6.1 The current state, honestly stated

| 8× H200 · UD-IQ1_S | llama.cpp | SparkInfer-K3 | |
|---|--:|--:|--:|
| decode @ 128k | 18.44 | **60.17** | **3.26× ahead** |
| prefill @ 32k | 143.88 | **99.68** | 1.44× behind |

Both rows belong in the headline. An engine that reports only the row it wins is doing marketing; the interesting engineering question — what is left — lives in the row it loses.

Note also that 99.68 is **author-reported but unsealed**: no eval round has measured it, so the pinned frontier deliberately holds at 69.02, the last attested value. That distinction between *measured* and *attested* is the subject of [Lecture 02](Lecture-02.md).

---

## 7. Reading a ladder — the compounding arithmetic

Because gains multiply, the intuition most engineers carry into optimization work is wrong in a specific way. Small percentages are not small.

```text
   twenty changes, each +10%          1.10^20  =  6.7x
   twenty changes, each  +5%          1.05^20  =  2.7x
   ONE change of +200%, then nothing              3.0x

   the sixteen displayed decode transitions, compounded:
       1.00 -> 46.48   =   46.48x
       arithmetic mean +40.14%; median +10.39%; geometric mean +27.12%
```

Two consequences for how you spend a month:

* **A 5% win that ships beats a 40% win that does not.** Seven 5% wins is 1.41×; the 40% you never finished is 1.0×.
* **The reward system has to pay marginal gains, or it pays for nothing.** This is why the case-study repo pays `delta / frontier` rather than rank: "copy the leader + ε" pays ≈ ε. Any incentive that rewards *position* rather than *increment* funds the twentieth reimplementation of the first optimization.

And one consequence for how you *report* a month: the honest form of "we got 60×" is the ladder, not the ratio. The ratio can be checked against its endpoints; the ladder additionally lets us audit attribution row by row, which is why it is the artifact this part asks you to produce.

---

## Lab — derive the shape before you profile

Goal: reproduce the *reasoning* of this lecture for a model and node you can actually access, and commit the predictions before you measure anything.

1. **Pick a workload.** A model you can run and a node you can rent or borrow. It does not need to be large — the discipline is scale-free.
2. **Fill the model table** (§1) from the official config, not a blog post. Layers, hidden, vocab, attention type and its KV bytes/token, FFN width, MoE routing if any.
3. **Compute the memory budget.** Weights at your chosen quantization, KV or recurrent state at your target context, activation and compute buffers. Compare to total HBM. State your headroom as a number.
4. **Predict the bandwidth ceiling.** `tokens/s ≤ HBM GB/s ÷ bytes-read-per-token`. For MoE, count only *active* expert bytes. Write the number down; [Lecture 03](Lecture-03.md) will use it.
5. **Count the collectives.** For your intended parallelism, how many all-reduces per token, and how wide is each? Derive it from where the non-linearities sit, not from a diagram.
6. **Name your baseline and pin it.** Repo, commit, weights hash, exact command line. If your baseline cannot load your model, say what you will do instead — and what that costs you (§3).
7. **Write down three traps** in your model's config that would produce plausible-but-wrong output rather than a crash. If you cannot find three, you have not read the converter.

Pass criterion: a one-page `SHAPE.md` in your artifact repo, committed **before** your first profile, containing a predicted bandwidth ceiling and a pinned baseline. [Lecture 03](Lecture-03.md) grades your prediction against a measurement.

---

## Self-check

1. K3 has 896 experts, top-16, with routed experts in a 3584-wide latent space and expert FFN 3072. Estimate the *active* expert weight bytes read per token at ~1.6 bits/weight, then use 8× H200 aggregate HBM bandwidth to bound decode tok/s. Does your bound sit above 60.17? What does it mean if it does not?
2. The engine measured 10.34 tok/s at ctx 64 and 1.00 tok/s at ctx 131,072, while the reference held 18.32 → 18.44 across the same range. From those four numbers alone, what can you conclude about where the engine's time goes at depth — and what would you profile first?
3. A colleague's baseline is a pinned PR branch on a third-party fork. Name two ways that baseline can silently change, and the assertion that catches each.
4. The llama.cpp reference pin was corrected 16.7026 → 18.4435, making every awarded tier ~9.4% smaller. Nobody's code changed. Argue for or against retroactively re-scoring the already-merged PRs.
5. An engine reports "3.26× faster than the reference." What are the four questions you ask before believing it applies to your deployment?
6. Why does pinning an IQ1_S measurement into a quant-unqualified baseline slot bias future gains until the slot is corrected, rather than just once?

## Answers

1. Counting only three routed-expert matrices per expert and 92 MoE layers gives `92×16×3×3584×3072×0.2 = 9.724 GB/token` at a hypothetical uniform 1.6 bits/weight. An ideal 38.4 TB/s aggregate bound is about 3949 tok/s. It omits scales, non-expert traffic, cache state, imbalance, and heterogeneous formats; it is not an attainable target or proof that memory is irrelevant. A bound below a trustworthy measurement means at least one modeling assumption is wrong.
2. Depth-dependent work becomes important; profile attention/cache work first. The four numbers do not identify the specific kernel or prove that the reference has constant-cost attention.
3. The PR ref can move: assert its full SHA. Weight artifacts can be re-uploaded: verify the manifest and every shard hash, not just names/counts.
4. Correct the ledger transparently while retaining the original result and denominator. Whether to change historical rewards is a governance policy, not a mathematical consequence; do not silently rewrite provenance.
5. Ask which model/quantization, input/context/batch and approximation settings, hardware/runtime, and measurement boundary/reference produced the ratio. Check whether the prefix was real or synthetic.
6. Every later comparison reading the mislabeled slot inherits its wrong meaning until repaired. The bias is not literally permanent, and quantization size alone does not guarantee which engine is faster.

---

## References

* **SparkInfer-K3** — [github.com/gittensor-ai-lab/sparkinfer-k3](https://github.com/gittensor-ai-lab/sparkinfer-k3) (MIT). The case study. `docs/technical.md` is the model + shard-policy reference; `bench/scripts/reference.lock` is the pinned-baseline file whose comments and commit history this lecture draws on.
* **The reference engine** — [unslothai/llama.cpp](https://github.com/unslothai/llama.cpp) PR #48 (`kimi-k3-fullsize-vision`), the fork that can load 896 experts.
* **Kimi K2 technical report** — [arXiv:2507.20534](https://arxiv.org/abs/2507.20534) — the published Moonshot AI architecture lineage K3 continues (MLA + large-expert-count MoE).
* **DeepSeek V3 technical report** — [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) — MLA's canonical description; K3's MLA is NoPE-only with a sigmoid output gate.
* **Gated DeltaNet** — [arXiv:2412.06464](https://arxiv.org/abs/2412.06464) — the linear-attention family KDA belongs to; explains the fixed-size recurrent state.

Cross-references:

* [Part 1 Lecture 03 — Roofline, bandwidth, and the memory hierarchy](../Part%201%20-%20Fundamentals/Lecture-03.md) — the ceiling you predict in the lab.
* [Part 3 Lecture 01 — Anatomy of a modern MoE](../Part%203%20-%20MoE%20at%20Blackwell/Lecture-01.md) — MLA and expert routing at 671B, the gentler version of §1.
* [MLSys Deep Dives Lecture 04 — Beyond the dense transformer](../../MLSys%20Deep%20Dives/Lecture-04.md) — why hybrid linear/full attention stacks exist.

---

## Current as of 2026-08

SparkInfer-K3 at `7689cc7`; reference `unslothai/llama.cpp` PR #48 @ `efc8bc38`; weights `Kimi-K3-UD-IQ1_S` (553 GiB, 14 shards); node 8× H200 SXM `sm_90`, CUDA 12.8+. Decode 60.17 tok/s @ 128k, prefill 99.68 tok/s @ 32k, against 18.44 / 143.88. These are frozen historical measurements for one engine on one box — refresh the *reasoning*, not the numbers.

---

## Next

* Next: [Lecture 02 — The scoreboard: a benchmark designed to resist gaming](Lecture-02.md)
* Up: [Part 4 — Optimizing a Real Engine](README.md)
