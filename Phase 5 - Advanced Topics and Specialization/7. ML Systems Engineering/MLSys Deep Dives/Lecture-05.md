# Lecture 05 - The 2026 Frontier as Systems Artifacts: Qwen3, Nemotron Ultra, MiMo, DeepSeek, and MoE

**Collection:** [MLSys Deep Dives](README.md) | **Previous:** [← Lecture 04](Lecture-04.md) | **Next:** [Lecture 06](Lecture-06.md)

---

A model card is a systems spec in disguise. By 2026, the headline parameter count tells you almost nothing about cost — a "235B" model might cost you 22B-worth of compute per token, hold 235B-worth of weights in VRAM, and demand an all-to-all every layer. The skill this lecture builds is **reading a frontier model the way an MLSys engineer reads it**: not "how smart is it" but "what shape is its cost, and what will it do to my GPUs."

We read four of the 2026 frontier as systems artifacts — **Qwen3, Llama Nemotron Ultra 253B, Xiaomi MiMo, DeepSeek V3/R1** — through the one structure that now dominates above ~30B: **Mixture of Experts**, plus the efficiency tricks (MLA, MTP) that travel with it.

---

## Learning objectives

By the end of this lecture, you should be able to:

1. Explain **MoE**: total vs active params, why it decouples capacity from compute, and what cost it shifts onto **memory and interconnect**.
2. Read **Qwen3** (dense + MoE, unified thinking mode), **Nemotron Ultra 253B** (NAS-compressed dense), **MiMo** (MTP, small reasoning), and **DeepSeek V3/R1** (MoE + MLA + MTP + FP8) as systems specs.
3. Identify the **canonical 2026 efficiency stack** and which model exemplifies each piece.
4. Extract a model's **systems fingerprint** from its card: active params, KV behavior, routing, precision, draft mechanism, context.
5. Predict a model's **cost shape** (prefill vs decode, memory- vs comm-bound, rough `$/Mtok`) from that fingerprint.

---

## 1. MoE: the structure that ate the frontier

A dense model runs **every** parameter for **every** token. A **Mixture-of-Experts** model has many "expert" FFNs per layer and a **router** that sends each token to only a few of them. The consequence is the most important number in 2026 model serving:

```text
   DENSE:  token → ALL weights run         compute/token ∝ total params
   MoE:    token → router picks K of N experts → only K run
           ┌─────────────────────────────────────────────────────────┐
           │ TOTAL params  = capacity (must live in VRAM)              │
           │ ACTIVE params = FLOPs per token  (≪ total)               │
           └─────────────────────────────────────────────────────────┘
   e.g.  Qwen3-235B-A22B  → 235B total / 22B active  (~10:1)
         DeepSeek-V3      → 671B total / 37B active  (~18:1)
```

This is why MoE dominates above ~30B (reportedly **>60% of 2025 open releases**, and essentially every model atop the intelligence leaderboards): you get the *quality* of a huge model at the *per-token compute* of a small one.

But nothing is free — MoE **shifts the cost from FLOPs to memory and interconnect**:

```text
   the bill MoE hands the systems engineer:
   • MEMORY:      all experts must be resident → large aggregate VRAM (you hold 235B/671B of weights)
   • INTERCONNECT: experts are scattered across GPUs (expert parallelism) → an ALL-TO-ALL
                   token shuffle (dispatch + combine) on EVERY MoE layer, every forward pass
   • LOAD BALANCE: hot experts create stragglers; routing must stay balanced
```

So an MoE model is **memory- and comm-bound where a dense model is compute-bound**. Dense scaling pays in FLOPs; MoE scaling pays in VRAM and all-to-all bandwidth (NVLink/InfiniBand). Past ~30B, that trade wins — which is why the 2026 frontier is sparse, and why **expert parallelism** and fast all-to-all are core serving skills.

One subtlety decides *how* you serve MoE, and it is worth holding precisely — **expert memory traffic depends on batch size**:

```text
   batch 1:    each token touches only K experts → ~ACTIVE-param bytes stream per step
               (DeepSeek-V3: ~37B of 671B) → the batch-1 bandwidth ceiling (Lec 1, §6)
               looks like a 37B model's, not a 671B model's
   big batch:  different tokens route to DIFFERENT experts → collectively most of the
               N experts are touched every layer → traffic per step approaches TOTAL
               params — but it is now shared across the whole batch → per-token cost collapses
   the middle: enough tokens to touch most experts, too few to amortize them —
               you stream most of 671B for a handful of tokens. the worst regime.
```

So MoE wants to be served at the extremes: **tiny batch** (cheap single stream — the latency-critical case) or **deep batch** (experts amortized — the throughput case), and the awkward middle is where naive deployments bleed money. This is why MoE serving pushes so hard on batch depth and expert parallelism, and why MoE rarely makes sense for small-batch edge deployment (Lecture 7's small dense/hybrid models own that regime). When you run Lecture 1's bandwidth-ceiling check on an MoE, use **active bytes at batch 1** and **total bytes near saturation** — quoting either one alone is how vendor decks mislead you.

---

## 2. Qwen3 — dense and MoE, with a thinking switch

**Qwen3** (Alibaba, Apr 2025) ships as a *family*: dense (0.6B → 32B) and MoE — **Qwen3-30B-A3B** (30B/3B active) and the flagship **Qwen3-235B-A22B** (235B/22B active, **128 experts, 8 activated**), pretrained on ~**36T tokens**, 128K context.

Two systems-relevant facts:

* **The MoE flagship serves at ~22B-active compute per token** but must hold 235B of weights across an expert-parallel deployment. So its *latency/compute* cost is roughly a 22B dense model's; its *memory/interconnect* cost is a 235B model's. That split is the whole MoE story made concrete.
* **Unified thinking mode.** A *single* checkpoint toggles **Thinking vs Non-Thinking** via an `enable_thinking` flag or inline `/think` / `/no_think` tags, with a configurable **thinking budget**. There's no separate reasoning model. For the systems engineer this is a *runtime knob on token count*: thinking mode emits far more tokens (reasoning traces), so it trades latency and `$/request` for accuracy — at request time, per request. You schedule and price the two modes differently.

---

## 3. Llama Nemotron Ultra 253B — architecture search to fit 8 GPUs

**Llama-3.1-Nemotron-Ultra-253B** (NVIDIA, Apr 2025) is the model behind "Nemotron Ultra," and it is a pure MLSys artifact: it was **derived from Meta's Llama-3.1-405B via Neural Architecture Search (NAS) + pruning/distillation**, with one explicit goal — **fit a 405B-quality model onto a single 8×H100 node**.

The NAS produced a **non-uniform, irregular** network (162 layers, non-repeating), using moves that are themselves a lecture in co-design:

```text
   skip-attention:   some blocks drop attention entirely (or replace it with one linear layer)
   variable FFN:     different expansion ratios per block
   FFN fusion:       where attention was skipped, consecutive FFNs are merged into fewer, wider FFNs
   → 405B → 253B, engineered to land inside an 8×H100 memory/latency envelope
```

It is **dense** (not MoE), 128K context, with a **reasoning toggle** via system prompt ("detailed thinking on/off"), emitting `<think>` traces when on. An FP8 variant cuts memory further. The systems lesson: **architecture search is a deployment tool** — you can NAS a model *to a hardware budget*, trading a hand-designed uniform stack for an irregular one that fits the GPUs you actually have. This is the same co-design philosophy as Lecture 4, applied to packing rather than to the KV cache.

---

## 4. Xiaomi MiMo — small, reasoning, and built to be drafted

**MiMo-7B** (Xiaomi, 2025) is the "MiMo" you've heard about, and it punches far above 7B on reasoning (its RL variant reportedly surpasses o1-mini on math/code). Two systems-relevant design choices:

* **Multi-Token Prediction (MTP) in pretraining.** MiMo is trained to predict *several* future tokens, not just the next one. This densifies the training signal **and** — the part that matters here — leaves the model with **built-in draft heads for speculative decoding** (Lecture 6). The architecture ships its own accelerator.
* **Small and single-GPU.** At 7B it is cheap to serve and fits one GPU, the opposite end of the spectrum from the 235B/671B MoEs — and the reason it's a natural fit for edge/on-device (Lecture 7). There are also **MiMo-VL** (vision-language) and **MiMo-Audio** variants.

MiMo also sets up the course's payoff: the **MiMo-V2.5-Pro** line is what a 2026 throughput milestone was demonstrated on — a **1-trillion-parameter MoE pushed past ~1000 tokens/s** on a single 8-GPU node, by stacking MXFP4 quantization + **DFlash** speculative decoding + the **TileRT** megakernel runtime. We dissect that result in Lecture 6; note here that its *first ingredient is the architecture* (MTP-friendly, MoE), and the rest is the systems stack from Lectures 2–3 and 6.

---

## 5. DeepSeek V3 / R1 — the canonical efficiency stack

**DeepSeek-V3** (Dec 2024) is the cleanest single example of "every 2026 efficiency trick at once," and worth memorizing as a template: **671B total / 37B active** MoE (**256 routed + 1 shared expert, 8 routed activated**), 14.8T tokens. Its four stacked techniques map directly onto this course:

```text
   MoE         671B/37B → compute of a 37B, capacity of a 671B            (this lecture, §1)
   MLA         multi-head LATENT attention: cache a low-rank latent        (Lecture 4, §6)
               instead of full per-head K/V → much smaller KV cache
   MTP         multi-token prediction → denser signal + spec-decode draft  (this lecture §4, Lecture 6)
   FP8         trained and largely served in FP8 → memory & throughput     (precision floor)
   + auxiliary-loss-FREE load balancing (bias-based routing) → no aux-loss quality/throughput hit
```

**R1** is the reasoning model built on the V3 base (RL for long chain-of-thought). The reason DeepSeek-V3 could be served at frontier quality for **~$0.14/Mtok** (Lecture 1) is precisely this stack: MoE cut the compute, MLA cut the KV memory, MTP sped decode, FP8 cut the bytes. It is the worked example of model–systems co-design producing an order-of-magnitude cost result — the whole thesis of this course in one model card.

---

## 6. Reading a model card for its systems fingerprint

Here is the reusable skill. Given any 2026 model card, extract **six fields** and you can predict its cost shape without running it:

```text
   ① total / active params  → compute/token (active) AND memory footprint (total)
   ② attention type         → KV-cache behavior:  MHA/GQA = O(L) big · MLA = O(L) small · SSM/hybrid = O(1)
   ③ routing                → dense (no all-to-all) vs MoE (#experts/#active → all-to-all cost)
   ④ dominant precision     → BF16 / FP8 / FP4 / 4-bit → bytes per param, throughput
   ⑤ draft mechanism        → MTP heads / EAGLE / none → decode-speed headroom (Lecture 6)
   ⑥ context length         → KV scaling, prefill cost
   ───────────────────────────────────────────────────────────────────────────────────
   ⇒ COST SHAPE: prefill- vs decode-dominated, memory- vs compute- vs comm-bound, rough $/Mtok class
```

Applied to the four models:

| Model | ① total/active | ② attention/KV | ③ routing | ④ precision | ⑤ draft | ⑥ ctx | Cost shape |
|---|---|---|---|---|---|---|---|
| **Qwen3-235B-A22B** | 235B / 22B | GQA, O(L) | MoE 128/8 | BF16·FP8 | — | 128K | memory+comm-bound (MoE), 22B compute/tok |
| **Nemotron Ultra 253B** | 253B dense | skip-attn (NAS) | dense | BF16·FP8 | — | 128K | compute-bound dense, NAS-fit to 8×H100 |
| **MiMo-7B** | 7B dense | GQA, O(L) | dense | BF16 | **MTP** | 32K | small, single-GPU, decode-accelerable |
| **DeepSeek-V3/R1** | 671B / 37B | **MLA**, O(L) small | MoE 256+1/8 | **FP8** | **MTP** | 128K | the full stack: cheap/tok despite 671B |

That table *is* the lecture. An interviewer who asks "what would it cost to serve model X" is asking you to fill in this row and reason from it.

---

## 7. Hands-on / Measure it

1. **Fingerprint three models.** Pull the real cards for Qwen3-235B-A22B, DeepSeek-V3 (or R1), and one small/hybrid model (MiMo-7B or Falcon-H1). Fill the six-field fingerprint for each.
2. **Predict the cost shape.** For each, state: prefill- or decode-dominated? memory-, compute-, or comm-bound? Then estimate the dominant resource — e.g. for the MoE, the aggregate VRAM to hold all experts and the all-to-all volume per layer; for the dense reasoning model, the FLOPs/token and the extra tokens "thinking" mode emits.
3. **To dollars.** Using Lecture 1's cost model and a reasonable tokens/s for each (from a public benchmark — date it), put a rough `$/Mtok` band on each, and note which lever (MoE compute, MLA memory, MTP decode, FP8 bytes) is doing the most work.

Deliverable: three filled fingerprints, three predicted cost shapes, and three `$/Mtok` bands with the dominant lever named. If your prediction disagrees with a published benchmark, explain the gap — that gap is usually a serving-stack detail (batching, expert parallelism) you didn't model, and finding it is the point.

---

## 8. Mini-lab

Build a one-page **"systems fingerprint" cheat sheet** you'll reuse for the rest of your career: the six-field checklist from §6, with a worked example for each archetype — a dense model (Llama/Nemotron Ultra), an MoE (Qwen3/DeepSeek), a hybrid (Nemotron-H/Falcon-H1), and a small on-device model (MiMo). For each archetype, write the one sentence that predicts its cost shape. Test it on the *next* model release you see: fill the fingerprint from the card alone, predict the cost shape, then check against the first published benchmark.

Deliverable: the cheat sheet + one "prediction vs reality" check on a fresh model card. Getting good at this — pricing a model from its card before anyone benchmarks it — is a senior MLSys signature skill.

---

## Key takeaways

- **MoE** decouples capacity (total params, in VRAM) from compute (active params, FLOPs/token) — the dominant >30B pattern. It **shifts cost from FLOPs to memory + all-to-all interconnect**: MoE is memory/comm-bound where dense is compute-bound.
- **Qwen3** — dense + MoE (235B/22B, 128/8 experts), with a **unified thinking switch** that's a runtime knob on token count and cost.
- **Nemotron Ultra 253B** — **NAS-compressed** from Llama-405B (skip-attention, FFN fusion) to **fit 8×H100**: architecture search as a deployment-to-budget tool.
- **MiMo-7B** — small, reasoning, and **MTP-pretrained** so it ships built-in speculative-decode draft heads; the architecture behind a 1000-tok/s milestone (Lec 6).
- **DeepSeek V3/R1** — the **canonical efficiency stack**: MoE (compute) + **MLA** (KV memory) + **MTP** (decode) + **FP8** (bytes) → frontier quality at ~$0.14/Mtok.
- The reusable skill is the **six-field systems fingerprint** (total/active, attention/KV, routing, precision, draft, context) → predicted **cost shape**. Price a model from its card; that's the senior move.

---

## References

- Qwen3 (Alibaba): [https://qwenlm.github.io/blog/qwen3/](https://qwenlm.github.io/blog/qwen3/)
- Llama-3.1-Nemotron-Ultra-253B (NVIDIA): [https://huggingface.co/nvidia/Llama-3_1-Nemotron-Ultra-253B-v1](https://huggingface.co/nvidia/Llama-3_1-Nemotron-Ultra-253B-v1)
- Xiaomi MiMo, arXiv 2505.07608: [https://arxiv.org/abs/2505.07608](https://arxiv.org/abs/2505.07608) · repo [https://github.com/XiaomiMiMo/MiMo](https://github.com/XiaomiMiMo/MiMo)
- DeepSeek-V3, arXiv 2412.19437: [https://arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437)
- Mixture-of-Experts infrastructure & expert parallelism overview: [https://www.digitalocean.com/community/tutorials/expert-parallelism-in-deep-learning](https://www.digitalocean.com/community/tutorials/expert-parallelism-in-deep-learning)
- *MLSys Deep Dives* — [Lecture 04](Lecture-04.md) (MLA, hybrids) and [Lecture 06](Lecture-06.md) (MTP → speculative decoding).

---

## Current as of

2026-06. Pins: Qwen3 (Apr 2025, 235B-A22B / 30B-A3B, 128 experts/8), Nemotron-Ultra-253B (Apr 2025, NAS from Llama-3.1-405B, 8×H100), MiMo-7B (2025, MTP), DeepSeek-V3 (Dec 2024, 671B/37B, MLA+MTP+FP8). MoE-adoption stats (">60% of 2025 releases") are from an industry blog, illustrative not peer-reviewed. Later Qwen / Nemotron / MiMo releases may supersede these — re-pull the cards.

---

*Next: [Lecture 06 — Making decode fast](Lecture-06.md)*
