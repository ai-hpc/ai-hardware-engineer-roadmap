# Lecture 04 - Beyond the Dense Transformer: Mamba, SSMs, and the Hybrid Wave

**Collection:** [MLSys Deep Dives](README.md) | **Previous:** [← Lecture 03](Lecture-03.md) | **Next:** [Lecture 05](Lecture-05.md)

---

So far we have made the *kernels* and *compilers* faster without touching the model. This lecture attacks the cost from the other side: **redesigning the architecture itself so there is less work to do per token.** This is model–systems co-design, and in 2024–2026 it produced the biggest structural shift since the transformer — the move to **state-space models, linear attention, and hybrids**.

The villain of the story is one data structure: the **KV cache**. Understand why it grows and what kills it, and you understand why Mamba, Jamba, Nemotron-H, Falcon-H1, and MiniMax exist — and why a 2026 long-context model looks nothing like a 2022 one on the inside.

---

## Learning objectives

By the end of this lecture, you should be able to:

1. Explain the **KV-cache problem**: why dense attention is O(L²) compute and O(L) growing memory, and why that caps long-context throughput.
2. Describe an **SSM** (Mamba): a constant-size recurrent state, O(1) memory/compute per token, and the **selection** mechanism.
3. Explain **Mamba-2 / SSD** — the state-space ↔ attention duality — and why it made SSMs tensor-core-friendly and thus the hybrid building block.
4. Place the **linear-attention** family (RWKV, RetNet, GLA) on the same "constant-memory recurrence" idea.
5. Explain the **hybrid wave** (the ~1-attention : ~7–10-SSM pattern) and read Jamba, Nemotron-H, Falcon-H1, MiniMax-01 as instances.
6. Connect each architecture to a **KV/memory number** and therefore to batch depth, tokens/s, and `$/Mtok`.

---

## 1. The villain: the KV cache

Dense self-attention has a property that is wonderful for quality and ruinous for systems: to generate token `t`, it attends over **all previous tokens**. So every token's key and value must be kept around — the **KV cache** — and it grows with the sequence.

```python
# Dense attention: you must keep ALL past keys/values around
K_cache, V_cache = [], []
for x_t in sequence:
    q_t, k_t, v_t = project(x_t)
    K_cache.append(k_t); V_cache.append(v_t)        # the cache GROWS every single token
    y_t = softmax(q_t @ stack(K_cache).T) @ stack(V_cache)
# memory is O(sequence_length) per layer, per request — this is what eats your VRAM
```

The KV-cache size, concretely:

```text
   KV bytes = 2 (K,V) × layers × kv_heads × head_dim × seq_len × dtype_bytes × batch
                                                        ▲                       ▲
                                              grows with context        grows with batch
```

For a 70B-class model at 128K context this is **tens of gigabytes** — often larger than the activations, and it scales with *both* the context length and the batch size. That is the wall. It is why long-context serving is expensive, why batch depth (and therefore TOK/$, from Lecture 1) is capped, and why decode slows as the conversation grows. Two families of architecture exist to tear this wall down: **replace attention with recurrence** (SSMs, §2–4), or **compress the cache** (MLA, §6).

---

## 2. State-space models: trade the cache for a fixed state

A **state space model (SSM)** does what an RNN does — carry a fixed-size **state** forward — but in a form that trains efficiently on GPUs. The entire history is compressed into a constant-size vector; there is **no growing cache**.

```python
# SSM recurrence (conceptual): the whole past compressed into a FIXED-size state h
h = zeros(d_state)
for x_t in sequence:
    h   = A * h + B * x_t          # update fixed-size state — memory does NOT grow
    y_t = C @ h                    # produce the output for this token
# memory is O(d_state), INDEPENDENT of sequence length
```

The problem with classic SSMs was that `A, B, C` were **fixed** (input-invariant), so the model couldn't selectively remember or forget based on *content* — fatal for language. **Mamba** (Albert Gu & Tri Dao, Dec 2023) fixed this with the **selection mechanism**: make `Δ, B, C` **functions of the input** `x_t`, so the model decides, per token, what to keep and what to drop. That single change made SSMs competitive with transformers on language while keeping the systems win:

```text
   attention:  O(L²) compute,  O(L) memory (growing KV cache)
   SSM/Mamba:  O(L)  compute,  O(1) memory per token (fixed state)  ← the whole point
```

The MLSys consequence is enormous: **memory and per-token compute are flat in context length.** A Mamba model's tokens/s does not degrade as the conversation grows, and its memory footprint at 1M tokens looks like its footprint at 1K. That is a different cost curve entirely.

---

## 3. Mamba-2 and the duality that made hybrids possible

Mamba-1 had one systems wart: its selective scan didn't map cleanly onto Tensor Cores (the matmul machinery), so it under-utilized the GPU. **Mamba-2** (Dao & Gu, May 2024) fixed that with a deep result — **Structured State Space Duality (SSD)**:

> A selective SSM is *mathematically equivalent* to a form of **masked attention**, via structured (semiseparable) matrices. So the same layer can be computed either as a linear recurrence **or** as a block of matmuls.

That duality is why Mamba-2 matters for this course. By expressing the SSM as **tensor-core-friendly matmuls** (a block decomposition), Mamba-2 runs **2–8× faster** than Mamba-1 and finally uses the GPU the way a transformer does. It simplified the state matrix `A` to a scalar-times-identity and added **multi-head** structure (like attention heads). The result became the **standard building block** for every hybrid below — you get the constant-state inference win *and* good training-time hardware utilization.

```text
   SSD:  one layer, two ways to compute it
        recurrence form  →  O(1) state, great for INFERENCE (decode)
        matmul form      →  tensor-core-friendly, great for TRAINING (parallel)
```

---

## 4. The linear-attention family (same idea, different gates)

Mamba is the most prominent, but it sits in a family of **sub-quadratic / linear-attention** methods that all share the systems pitch — **constant-memory recurrence instead of a growing KV cache** — and differ mainly in *how they gate* the state:

| Method | Gate / decay | One-line systems point |
|---|---|---|
| **RWKV** | channel-wise time decay | trains parallel like a transformer, runs as an O(1) RNN at inference |
| **RetNet** | fixed exponential decay ("retention") | three modes: parallel (train), recurrent (O(1) infer), chunkwise (long-seq) |
| **GLA** | **data-dependent** matrix gating | input-adaptive forget + a hardware-efficient chunkwise kernel |
| **Mamba-2** | **input-dependent** selective Δ | selection + SSD duality (the matmul form) |

The spectrum to remember: **fixed decay** (RWKV, RetNet — simpler, cheaper) → **input-dependent gating** (GLA, Mamba — adaptive, more expressive). All of them give the same headline: **flat memory and throughput as context grows.** What they give up versus full attention is some exact long-range *recall* — the ability to look up a specific earlier token precisely. Which is exactly the gap hybrids close.

---

## 5. The hybrid wave — the dominant 2025 pattern

Pure SSMs lose a little exact-recall ability; pure attention has the KV wall. The 2025 answer, now everywhere, is to **mix them**: keep a *few* full-attention layers for precise in-context retrieval, and make the *rest* SSM/linear. The empirically recurring ratio is about **1 attention layer per 7–10 recurrent layers**.

```text
   a hybrid stack (schematic, ~1 : 7):
   [SSM][SSM][SSM][SSM][SSM][SSM][SSM][ATTN][SSM][SSM]...[ATTN]
        └──── constant-state, cheap, flat-in-context ────┘  └ a little exact recall ┘

   net effect:  KV cache ≈ (attention-layer fraction) × full-attention KV
                so ~1:7  →  KV cache ~1/8 the size  →  ~deeper batching, 2-3× long-ctx throughput
                quality stays near full-attention
```

The instances you should recognize:

* **Jamba** (AI21, Mar 2024) — the first production-grade hybrid: **52B total / 12B active** (it's *also* MoE), **1 transformer layer per 8**, 256K context, fits **140K tokens on a single 80GB GPU**, ~**3× long-context throughput vs Mixtral 8×7B**. Two KV-reducers stacked: hybrid *and* MoE.
* **NVIDIA Nemotron-H** (Apr 2025) — **8B / 47B / 56B**, most self-attention replaced by **Mamba-2**, attention ≈ **8% of layers**. The 47B is **~2.9× faster than Qwen-2.5-72B / Llama-3.1-70B at 65K context**; trained in **FP8** with <0.1% quality difference vs BF16. This is NVIDIA's reference hybrid and it feeds later Nemotron reasoning models.
* **Falcon-H1** (TII, May 2025) — **this is the "Falcon AI" you've heard about.** Its twist is a **parallel hybrid-head** design: attention and Mamba-2 heads run **in parallel within the same block** (not as separate interleaved layers), with an **independently tunable attention:SSM ratio**. The 34B reportedly rivals 70B-class models; 256K context. The tunable ratio makes the memory/recall trade a *knob*.
* **IBM Bamba** (9B) — interleaves Mamba-2 with attention, **~2.5× throughput** in vLLM — and carries the most important lesson: the speedup is **gated by inference-stack support for SSM state management.** The architecture only pays off once the *serving stack* (vLLM kernels, state handling) supports it. Architecture wins require systems work; they are not free.
* **MiniMax-01** (Jan 2025) — **456B total / 45.9B active** MoE with **Lightning Attention** (an I/O-aware linear attention), **1 softmax-attention layer per 7** lightning layers, trained at **1M-token context, up to 4M at inference**. It demonstrates that **multi-million-token context is only economically serveable with linear/hybrid attention** — a full-attention KV cache at 4M tokens would be absurd.

---

## 6. The other lever: compress the cache (MLA)

Hybrids *remove* attention layers. The complementary approach keeps attention but **shrinks each layer's KV** — **Multi-head Latent Attention (MLA)**, from DeepSeek (detailed in Lecture 5). MLA caches a **low-rank latent vector** instead of the full per-head K and V, then reconstructs them on the fly:

```text
   standard KV cache:  store full K, V per head      → big, O(L)
   MLA:                store a compressed LATENT      → much smaller constant factor, still O(L)
                       reconstruct K,V from it per step
```

So the two families attack the same villain differently: **SSM/hybrid** makes most layers stateless (O(1)); **MLA** makes attention layers cheaper to cache (smaller O(L)). Modern frontier models often combine both ideas with MoE — which is exactly the subject of the next lecture.

---

## 7. Systems-implication table

The whole lecture, as a reference you should be able to reproduce:

| Architecture | Mechanism | Memory vs context | Throughput vs context | Exemplars |
|---|---|---|---|---|
| **Dense attention** | softmax over all past | **O(L)** growing KV | degrades as L grows | Llama, Qwen-dense |
| **SSM (Mamba-2)** | selective recurrence + SSD | **O(1)** fixed state | flat | Mamba |
| **Linear attention** | gated recurrence | **O(1)** fixed state | flat | RWKV, RetNet, GLA |
| **Hybrid (~1:7–10)** | few attn + many SSM | ~fraction × attn KV | 2–3× long-context | Jamba, Nemotron-H, Falcon-H1, MiniMax-01 |
| **MLA** | low-rank latent KV | smaller O(L) | deeper batch | DeepSeek V3/R1 |

And the line that ties it to Lecture 1: **a smaller KV cache → more requests fit in VRAM → deeper batching → higher aggregate tokens/s → lower `$/Mtok`.** Architecture co-design is not an accuracy story; it is a *cost* story. A hybrid that cuts the KV cache 8× can let you batch ~8× deeper at long context, which is a near-order-of-magnitude move on the denominator of the cost equation.

---

## 8. Hands-on / Measure it: plot the wall

Make the KV-cache wall visible, then watch each architecture flatten it.

```python
def kv_cache_gb(layers, kv_heads, head_dim, seq_len, batch, dtype_bytes=2):
    return 2 * layers * kv_heads * head_dim * seq_len * batch * dtype_bytes / 1e9

# dense 70B-ish: watch it explode with context
for L in (4_000, 32_000, 128_000):
    gb = kv_cache_gb(layers=80, kv_heads=8, head_dim=128, seq_len=L, batch=1)
    print(f"dense  ctx={L:>7}: {gb:6.1f} GB KV")

# hybrid at ~1:7 → only ~1/8 the attention layers carry a KV cache
for L in (4_000, 32_000, 128_000):
    gb = kv_cache_gb(layers=10, kv_heads=8, head_dim=128, seq_len=L, batch=1)  # ~10 attn layers
    print(f"hybrid ctx={L:>7}: {gb:6.1f} GB KV   (SSM layers add a flat, tiny state)")
```

The dense column climbs into tens of GB; the hybrid column is a fraction of it, and the SSM layers add a constant that doesn't move with `L`. Then take the freed memory and compute how much **deeper you can batch**, and what that does to aggregate tokens/s and `$/Mtok` (Lecture 1, §6). That chain — KV bytes → batch depth → tokens/s → dollars — is the deliverable.

**The serving caveat (the Bamba lesson):** these wins are only real if your inference stack *implements* SSM state management. Before promising an 8× batching win, confirm vLLM/SGLang (or your runtime) actually supports the architecture's state handling — or you've drawn a chart the hardware can't cash.

---

## 9. Mini-lab

1. **Plot the wall:** reproduce §8 for a dense model, a hybrid (adjust the attention-layer count), and (conceptually) MLA. Make one chart of KV-GB vs context for all three.
2. **Batch math:** for a fixed GPU memory budget, compute max batch size at 128K context for each architecture, then estimate aggregate tokens/s and `$/Mtok` deltas.
3. **Read a real one:** pull the config of **Nemotron-H** or **Falcon-H1**, find the attention:SSM layer ratio, and predict its KV footprint relative to a same-size dense model. Check your prediction against the model card's context/throughput claims.

Deliverable: the KV-vs-context chart, the batch-depth table, and one paragraph: which architecture would you serve at 128K context and why, in `$/Mtok` terms. That argument is the co-design skill.

---

## Key takeaways

- The **KV cache** — O(L) memory that grows with context *and* batch — is the wall that makes long-context serving expensive and caps batch depth (and therefore TOK/$).
- **SSMs (Mamba)** replace the growing cache with a **constant-size recurrent state**: O(1) memory/compute per token, **flat** in context. **Selection** (input-dependent dynamics) made them competitive with attention.
- **Mamba-2 / SSD** proved SSMs are dual to masked attention, giving a **tensor-core-friendly matmul form** — 2–8× faster, and the reason SSMs became the **hybrid building block**.
- The **linear-attention family** (RWKV, RetNet, GLA) shares the constant-memory-recurrence win, differing by gate (fixed decay → input-dependent).
- The **hybrid wave** (~1 attention : 7–10 SSM) keeps a little exact recall while cutting the KV cache to ~the attention-layer fraction: **2–3× long-context throughput** at near-full-attention quality (Jamba, Nemotron-H, Falcon-H1, MiniMax-01).
- **MLA** is the complementary lever — compress the KV per attention layer. Smaller KV (either way) → **deeper batching → more tokens/s → lower `$/Mtok`** — but only if the **serving stack supports the architecture** (the Bamba lesson).

---

## References

- Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," arXiv 2312.00752: [https://arxiv.org/abs/2312.00752](https://arxiv.org/abs/2312.00752)
- Dao & Gu, "Transformers are SSMs (SSD / Mamba-2)," arXiv 2405.21060: [https://arxiv.org/abs/2405.21060](https://arxiv.org/abs/2405.21060)
- AI21, Jamba, arXiv 2403.19887: [https://arxiv.org/abs/2403.19887](https://arxiv.org/abs/2403.19887)
- NVIDIA, Nemotron-H, arXiv 2504.03624 · [https://research.nvidia.com/labs/adlr/nemotronh/](https://research.nvidia.com/labs/adlr/nemotronh/)
- TII, Falcon-H1: [https://falcon-lm.github.io/blog/falcon-h1/](https://falcon-lm.github.io/blog/falcon-h1/)
- IBM, Bamba (SSM-Transformer in vLLM): [https://research.ibm.com/blog/bamba-ssm-transformer-model](https://research.ibm.com/blog/bamba-ssm-transformer-model)
- MiniMax-01 (Lightning Attention), arXiv 2501.08313: [https://arxiv.org/pdf/2501.08313](https://arxiv.org/pdf/2501.08313)

---

## Current as of

2026-06. Architectures pinned: Mamba (Dec 2023), Mamba-2/SSD (May 2024), Jamba (Mar 2024), Nemotron-H 8/47/56B (Apr 2025), Falcon-H1 (May 2025), Bamba-9B, MiniMax-01 (Jan 2025). Throughput multipliers (e.g. Nemotron-H 47B ~2.9× at 65K) are model-card figures at specific context lengths — re-verify against current serving-stack support before relying on them.

---

*Next: [Lecture 05 — The 2026 frontier as systems artifacts](Lecture-05.md)*
