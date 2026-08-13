# Transformer Fundamentals — Attention, Self-Attention, and the Full Block

## Overview

This is the foundational lecture on the **transformer** architecture. It starts from the problem that attention solves, builds up the scaled dot-product attention operation, generalizes to self-attention and multi-head attention, then assembles the full transformer block (attention + FFN + residual + norm) and the three canonical configurations (encoder-only, decoder-only, encoder–decoder).

Read this **before** the Phase 5 inference lectures. Those lectures assume you can read a tensor shape and know which projection corresponds to which letter in `Q, K, V`. This lecture teaches you that, with the math you actually need and with the common misconceptions called out explicitly.

**Companion visualization:** as you read, keep [LLM Visualization (bbycroft.net/llm)](https://bbycroft.net/llm) open in a tab. It walks every tensor of a small GPT through the forward pass in 3D — embeddings, Q/K/V projections, the attention scores grid, softmax, the value-weighted sum, the residual add, layer norm, MLP, and the output projection. When this lecture says "Q is a 2D tensor of shape (T, d_k)," that page lets you literally see the cells move. Use it whenever a shape stops making intuitive sense.

By the end you should be able to:

* Explain why attention exists at all — what bottleneck it removes from older sequence models.
* Derive the scaled dot-product attention formula from "compare a query to many keys."
* Implement self-attention in 30 lines of PyTorch.
* Sketch multi-head attention and explain what each head can specialize in.
* Place a causal mask and a padding mask correctly in the pipeline.
* Identify self-attention vs cross-attention from a block diagram.
* Read the full forward pass of a transformer block and match it to actual production models (BERT, GPT, T5, Llama, Qwen).

---

## 1. Why Attention Exists

### 1.1 The bottleneck attention removes

Before transformers, the dominant sequence models were **recurrent networks** — RNNs, LSTMs, GRUs. They processed sequences token by token, threading a single **hidden state** vector forward:

```
hidden_0 ─► [RNN cell] ─► hidden_1 ─► [RNN cell] ─► hidden_2 ─► ... ─► hidden_T
              │                          │                          │
            x_0                         x_1                         x_T
```

Every token's information had to be packed into the same **fixed-size hidden state**, then passed to the next step. By the time the model reached `x_T`, information about `x_0` had to survive `T` rounds of overwriting.

This is the **fixed-context bottleneck**. For short sequences it works fine. For long sequences (translation of a paragraph, summarization, code) it falls apart — the gradient of the early tokens with respect to a late prediction vanishes, and the model literally forgets the start of the sequence.

LSTMs and GRUs added **gates** that helped but didn't eliminate the problem. The architectures still funneled everything through a single hidden vector.

### 1.2 The attention insight

!!! abstract "The core idea"
    **Not every token matters equally for every decision.**

For the word **bank** in "She sat by the river bank," the word **river** is the single most informative token in the sentence. For the same **bank** in "I deposited money at the bank," it's **money** and **deposited**. The "right" context is **input-dependent**.

Attention makes this concrete: at each output position, the model produces a **weighted combination of every input position's representation**, where the weights are computed dynamically from the data.

```
output_i  =  Σ_j  weight(i, j) · value(j)

where  weight(i, j)  depends on  (query at position i,  key at position j)
```

No hidden-state bottleneck. No information has to survive a chain of overwrites. Every position can pull from every other position directly.

### 1.3 What attention is *not*

!!! warning "Three things attention is not"
    * **Not a full explanation of the model's decision.** Attention weights are useful clues — "the model put more weight on token X" — but the decision also goes through downstream linear projections, FFN layers, residual additions, and many other heads in parallel. Treat attention maps as evidence, not proof.
    * **Not retrieval.** A retrieval system would pick the single most relevant key and return its value. Attention returns a **soft, weighted average**. Even the lowest-scoring token contributes a sliver.
    * **Not free.** Computing `Q @ K^T` for sequence length `N` is `O(N²)` in both time and memory. This is why long-context inference is hard — and why sliding window, sparse attention, FlashAttention, and paged attention all exist.

---

## 2. Queries, Keys, and Values

This is the part of the transformer most worth slowing down for. Once **Q/K/V** clicks, the rest of the architecture is mechanical. Until it clicks, every diagram looks like alphabet soup.

### 2.1 The three roles

| Component | Role | One-line intuition |
|---|---|---|
| **Query (Q)** | What the current position is *looking for* | "What information do I need?" |
| **Key (K)** | What each position *offers for matching* | "What kind of information do I contain?" |
| **Value (V)** | The actual *content* passed along when a match happens | "Here is my information." |

The mechanism, in three steps:

1. Compare a query `q` to every key `k_j`. The comparison produces a **score**.
2. Convert scores to a probability distribution with softmax (weights that are non-negative and sum to 1).
3. Take a weighted sum of values `v_j` using those weights. That weighted sum **is** the attention output.

### 2.2 The library analogy

Imagine you walk into a library with a question in your head. The question is your **query**. The library has thousands of books on shelves. Each book has a **title** printed on its spine and **contents** inside its covers.

* You scan the spines (the **keys**) and judge which titles look most relevant to your query.
* You don't read the contents (the **values**) yet — you just compare your question to the spines.
* Based on how well each title matches your question, you decide how much attention to pay to each book.
* You then pull down the relevant books and skim their contents in proportion to your interest. A book with a perfect-match title gets fully read; one with a barely-related title gets a glance; an irrelevant one is ignored.
* The "answer" you leave with is a **blended summary** of all the books you read, weighted by relevance.

Three crucial properties to notice:

1. **The title and the contents are separate things.** A book about *river ecology* might have the title "Riparian Systems" — title-matching ("river bank?" → "riparian!") and content-reading are different operations. Hence: key ≠ value, even when they describe the same item.
2. **Your query is yours alone.** Two people with different questions standing in the same library would attend to different books. The query is *position-dependent* — each token has its own.
3. **You attend to every book, not just the best one.** You don't pick the single best match and read it; you read all of them in proportion to relevance. Even a poorly-matching book contributes a little. (This is softmax: small weight, not zero.)

### 2.3 Why three vectors and not one

It's a fair question: why does the model need three different vector representations of each token? Why can't it just use the same embedding to play all three roles?

Because the three relationships are **asymmetric**:

* What a token is *looking for* in others (Q) is not the same as what it *offers to others* (K).
* What it *offers to be matched* (K) is not the same as what it *contributes when matched* (V).

Concrete example: the word **"bank"** in "river bank":

* As a **query**, "bank" is looking for context that disambiguates its sense — does it want financial words or geographical words?
* As a **key**, "bank" is offering itself as a possible match target — "I'm the word you might attend to if your query is about money or geography."
* As a **value**, "bank" carries the information another token will pull when it attends to it — the word-sense, the part-of-speech, the semantic content.

These three roles need three different **vector subspaces**. A single shared embedding would conflate them and force the model to compromise. By giving the model **three independent learned projections** of the same input, you give it the flexibility to use one subspace for matching and a different subspace for content delivery.

### 2.4 How Q, K, V are actually produced

For a single token's embedding `x` (a vector of dimension `d_model`), the three vectors come from three different learned weight matrices:

```
q = W_Q · x        W_Q is [d_k × d_model]    →  q is [d_k]
k = W_K · x        W_K is [d_k × d_model]    →  k is [d_k]
v = W_V · x        W_V is [d_v × d_model]    →  v is [d_v]
```

In matrix form, for a whole sequence `X` of shape `[N, d_model]` (`N` tokens):

```
Q = X · W_Q^T       →  Q is [N, d_k]
K = X · W_K^T       →  K is [N, d_k]
V = X · W_V^T       →  V is [N, d_v]
```

(Whether you write `W_Q · x` or `x · W_Q^T` depends on whether the framework treats inputs as row or column vectors. PyTorch uses row vectors, so it's `X @ W_Q^T`. The math is the same.)

The weight matrices `W_Q`, `W_K`, `W_V` are **trained** like any other neural-network weights — via gradient descent on whatever loss the transformer is optimized for. The training signal pushes:

* `W_Q` to project tokens into a space where "looking for" queries land near matching keys.
* `W_K` to project tokens into a space where "what I offer" lands where the right queries can find it.
* `W_V` to project tokens into a space whose **weighted sums** encode useful contextual information.

Crucially, these matrices have **no built-in semantics** at initialization — they start as random numbers. The roles emerge from training. The model figures out on its own what "queries" and "keys" should look like, given the data.

### 2.5 Dimensions — what's free and what's locked

Three dimensions appear in the formulas: `d_model`, `d_k`, `d_v`. Two constraints:

* `d_k` for Q and `d_k` for K **must match** — because the attention score is the inner product `q · k`, which only exists when the two vectors live in the same space.
* `d_v` is **free**. V can be any width. The output of attention has shape `[N, d_v]` — it inherits V's dimension.

In practice, **all production transformers set `d_v = d_k`.** Why? Because the next operation after attention is usually a **linear projection** back to `d_model` (the residual stream dimension), and keeping `d_v = d_k` makes the bookkeeping clean and the parameter count predictable.

For multi-head attention with `H` heads, the per-head dimension is `d_k = d_v = d_model / H`. For Qwen3-4B: `d_model = 2560, H = 32`, so `d_k = d_v = 80`. Wait — actually Qwen specifically uses `head_dim = 128` (it doesn't follow the strict `d_model / H` rule), giving the Q projection an output of `n_heads · head_dim = 32 · 128 = 4096`, which is then projected back down. The point: these dimensions are design choices, and modern models tune them independently.

### 2.6 A concrete numerical walkthrough

Let's run attention on a toy example by hand. 4 tokens, `d_model = 4`, `d_k = d_v = 2` (small for legibility).

The input matrix `X` (4 tokens, each a 4-dim embedding):

```
X = [[1, 0, 1, 0],     # token 0
     [0, 1, 0, 1],     # token 1
     [1, 1, 0, 0],     # token 2
     [0, 0, 1, 1]]     # token 3
```

Three (small) learned projection matrices (`W` shapes: `[d_k, d_model]` = `[2, 4]`):

```
W_Q = [[1, 0, 1, 0],
       [0, 1, 0, 1]]

W_K = [[0, 1, 1, 0],
       [1, 0, 0, 1]]

W_V = [[1, 1, 0, 0],
       [0, 0, 1, 1]]
```

Compute Q, K, V (each `[4, 2]`):

```
Q = X · W_Q^T = [[2, 0],      # token 0's query
                 [0, 2],      # token 1's query
                 [1, 1],      # token 2's query
                 [1, 1]]      # token 3's query

K = X · W_K^T = [[1, 1],      # token 0's key
                 [1, 1],      # token 1's key
                 [1, 1],      # token 2's key
                 [1, 1]]      # token 3's key

V = X · W_V^T = [[1, 1],      # token 0's value
                 [1, 1],      # token 1's value
                 [2, 0],      # token 2's value
                 [0, 2]]      # token 3's value
```

Attention for **token 0** (its query is `[2, 0]`):

```
scores  = Q[0] · K^T = [2, 0]·[[1,1,1,1],[1,1,1,1]] = [2, 2, 2, 2]
                       (each key is identical here, so all scores match)
weights = softmax([2,2,2,2]) = [0.25, 0.25, 0.25, 0.25]
output  = 0.25·V[0] + 0.25·V[1] + 0.25·V[2] + 0.25·V[3]
        = 0.25·[1,1] + 0.25·[1,1] + 0.25·[2,0] + 0.25·[0,2]
        = [1, 1]
```

Token 0's output is `[1, 1]` — the average of all values. Because every key looked identical to its query in this toy, attention spread evenly.

Now token 1 (query `[0, 2]`):

```
scores  = [0,2]·[[1,1,1,1],[1,1,1,1]] = [2, 2, 2, 2]
weights = [0.25, 0.25, 0.25, 0.25]
output  = [1, 1]
```

Same result, because the keys were degenerate. **This is why the model needs *trained* projections** — random or trivial keys produce uniform attention, which is no better than averaging. After training, `W_K` shapes the keys so that different tokens look different from different angles, and softmax produces meaningful weights.

Repeat the §6.2 multi-head exercise after training a small model and you'll see scores like `[8.3, -1.2, 12.7, 0.4]` instead of `[2, 2, 2, 2]` — softmax then concentrates weight on the relevant tokens.

### 2.7 The asymmetry of attention

`Q · K^T` is **not symmetric**. Token A's attention to token B can be very different from B's attention to A. Concretely:

```
score(i, j)  =  Q[i] · K[j]      ≠ in general     Q[j] · K[i]  =  score(j, i)
```

Because Q and K are produced by *different* learned matrices applied to the same input, `Q[i]` and `K[i]` are different vectors. The "asymmetric relationship" is built directly into the math.

Why this matters: in a sentence like "She picked up the book that her teacher had recommended," the word **book** may strongly attend back to **teacher** (to figure out *whose* book), while **teacher** may not need to attend to **book** at all (it already knows what it is). The model learns these one-way dependencies because nothing in the architecture forces symmetry.

### 2.8 K and V are always paired by position

A subtle but critical structural rule: `K[j]` and `V[j]` are produced from the same token `j`. They are not independent slots. When attention weight on position `j` is high, you pull `V[j]` — never `V[k]` for some other `k`.

Why this matters for **cross-attention**: in an encoder–decoder model, the decoder produces queries, and the encoder produces both keys and values. The encoder generates K and V from the same source-sentence tokens, paired by position. The decoder then "reaches across" the diagram, asking "which source token matters here?" (via K) and pulling that token's content (V).

In **self-attention** in modern decoder-only LLMs, all three come from the same sequence — but the position-pairing rule still holds. Token `j`'s key and value always go together.

This is why production runtimes cache K and V as a single **KV cache** indexed by position. It's one logical entry per past token, containing both its key vector and its value vector.

### 2.9 The KV cache — a bridge to inference engineering

A property that doesn't matter at all during training, but matters enormously during inference:

* Once a sequence has been processed, every token's **K and V vectors are fixed**. They don't change when you decode the next token.
* The new token's **Q changes** every step (it's a new query each time).
* So during autoregressive decoding, you compute Q fresh each step, but you **reuse K and V from a cache** rather than re-running `W_K · x` and `W_V · x` for every past token.

This is the **KV cache**. It is the single most important data structure in LLM inference, and it's the reason Q, K, V are designed as separate projections rather than fused into a single representation. If keys and values weren't pre-computable from past tokens, every decoded token would cost O(N²) work — and you couldn't run a 4 k-context chat at acceptable latency.

Concretely for Qwen3-4B at 4 k context, the KV cache for one sequence holds:

```
2 (K and V) × 36 layers × 8 KV heads × 128 head_dim × 4096 positions × 2 bytes (FP16)
  ≈ 576 MB
```

That cache is read on every decoded token. It's the second-biggest bandwidth load (after the weights themselves) on the entire decode path. The Phase 5 inference lectures spend a lot of time on it — see [Edge LLM Inference Internals](../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20C%20-%20Edge%20AI/Edge%20LLM%20Inference%20Internals/Lecture-01.md) §5.

!!! tip "Why the three-vector design exists at all"
    Q/K/V is not just an architectural quirk — it is what makes long-context LLM inference economically viable. The asymmetry between **"Q recomputed per step"** and **"K, V cacheable across steps"** is the entire reason GPT-class models can hold a conversation without recomputing the past from scratch on every token.

### 2.10 Common mental traps

!!! danger "Six misconceptions to unlearn"
    * **"Q, K, V are different types of information."** No — they are three *projections* of the same input embedding. The token's identity is the same; only the role differs.
    * **"K and V are the same thing in self-attention."** No — they're produced by different matrices `W_K` and `W_V`, into different subspaces. They're paired by position, but not equal in content.
    * **"Attention is just retrieval — find the best match and use it."** No — it's a *soft* weighted blend over all positions. Even the worst match contributes a small weighted slice.
    * **"The attention output is a single token's value."** No — it's a **weighted combination of every token's value vector**, with weights summing to 1. The output has shape `[d_v]` regardless of sequence length.
    * **"Each head computes its own Q, K, V from scratch."** Yes, exactly — and that's the point. Each head has its own `W_Q^h`, `W_K^h`, `W_V^h`, so each head can specialize in a different relationship type.
    * **"Why does Q come from the current token but K and V from past tokens? That's weird."** It's weird only if you're thinking of attention as "the current token asking past tokens for help." It's cleaner if you think of it as the library analogy: you (Q) walk in with a question; the books (K and V) are sitting on the shelves regardless of who walks in.

### 2.11 The minimal code, one more time

With all the structure now clear, the per-query attention operation reduces to four lines:

```python
def attention_one_query(q, K, V):
    """
    q: [d_k]              one query vector  (e.g., from the current token)
    K: [n, d_k]           n key vectors     (one per past token)
    V: [n, d_v]           n value vectors   (one per past token, paired with K)
    returns:               [d_v]             a weighted blend of values
    """
    scores  = K @ q                  # [n]  — one score per (q, k_j) pair
    weights = softmax(scores)        # [n]  — non-negative, sum to 1
    return weights @ V               # [d_v] — weighted sum of values
```

That's the entire attention operation for one query. Everything in §3–§9 (scaling, masking, batching, multi-head, the full block, encoder vs decoder) is generalization or wrapping around this four-line core.

---

## 3. The Score Function — Dot Product, Geometrically

The dominant choice for scoring is the **dot product**:

```
score(q, k)  =  q · k  =  Σ_i q_i · k_i
```

Why this works, geometrically:

```
q · k  =  ‖q‖ · ‖k‖ · cos(θ)
```

The dot product captures:

1. **Direction agreement** — `cos(θ)` ranges from −1 to 1.
2. **Magnitude** — longer vectors get bigger scores.

When the model is well-trained, queries and keys end up in a learned embedding space where **vectors pointing in similar directions correspond to semantically related tokens**. The dot product naturally rewards those alignments.

If you normalize `q` and `k` to unit length, the dot product collapses to **cosine similarity**:

```
cos(q, k)  =  (q · k) / (‖q‖ · ‖k‖)
```

Cosine similarity is sometimes used (e.g., in some retrieval embeddings), but standard transformer attention does **not** normalize — letting the model learn to use magnitude as part of the signal.

---

## 4. Scaled Dot-Product Attention

In matrix form, for many queries at once:

```
                          Q · Kᵀ
Attention(Q, K, V)  =  softmax( ─────── ) · V
                           √d_k
```

Where:

* `Q ∈ ℝ^(N × d_k)` — `N` query vectors, each of dimension `d_k`.
* `K ∈ ℝ^(N × d_k)` — `N` key vectors, same dimension.
* `V ∈ ℝ^(N × d_v)` — `N` value vectors, dimension `d_v` (usually `d_v = d_k`).
* `d_k` is the per-head dimension of the keys.

The output has shape `[N, d_v]` — one context-aware vector per query position.

### 4.1 Why divide by sqrt(d_k)

As `d_k` grows, the **variance** of the dot product `q · k` grows roughly linearly with `d_k` (the sum-of-products of two random vectors). Without scaling, the raw scores become large in magnitude, which pushes the softmax into a **highly peaked** regime — one or two entries get nearly all the weight, and gradients with respect to the rest become tiny.

!!! warning "Common misconception, reversed"
    Many beginner explanations say "scaling prevents the softmax from being too flat." This is backwards. Unscaled dot products produce a **too-sharp** softmax (one entry dominates), not a flat one. The `1/√d_k` factor is the fix that **broadens** the distribution back into a useful range.

The fix is to compensate by dividing by the standard deviation of the dot-product distribution, which is `√d_k`:

```
                          q · k         (q · k) / √d_k
Without scaling:  e^(q·k) ─────►        After scaling: e^(scaled) ─►
                          dominant                       balanced
                          single entry                   distribution
```

For `d_k = 64` (a typical per-head dim), `√d_k = 8`. For `d_k = 128` (Qwen), `√d_k ≈ 11.3`. The scaling factor is built into the attention formula and applied before the softmax.

### 4.2 Softmax — turning scores into weights

```
              e^(z_i)
softmax(z)_i = ─────────────
              Σ_j e^(z_j)
```

The softmax guarantees:

* Every output is non-negative.
* All outputs sum to exactly 1.
* Larger inputs become larger outputs (monotonic).

So the output is a valid probability distribution over the `N` key positions. This distribution is the **attention pattern** for this query — sometimes visualized as a heatmap to interpret what the model is "looking at."

### 4.3 PyTorch reference, ~10 lines

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K: [..., N, d_k]
    V:    [..., N, d_v]
    mask: [..., N, N]  (optional, additive — use 0 to allow, -inf to block)
    """
    d_k = Q.size(-1)
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)   # [..., N, N]
    if mask is not None:
        scores = scores + mask
    weights = F.softmax(scores, dim=-1)                    # [..., N, N]
    return weights @ V                                     # [..., N, d_v]
```

Run this once. It's the entire attention mechanism. Every transformer in the world — BERT, GPT, T5, Llama, Qwen — is, at its core, repeated applications of this 10-line function.

---

## 5. Self-Attention

In **self-attention**, Q, K, and V all come from the **same** input sequence. Each token's representation is projected three different ways (with three different learned weight matrices) and used as both a query and a key and a value.

```
input X ∈ ℝ^(N × d_model)
   │
   ├─► [W_Q] ──► Q = X · W_Q      ∈ ℝ^(N × d_k)
   ├─► [W_K] ──► K = X · W_K      ∈ ℝ^(N × d_k)
   └─► [W_V] ──► V = X · W_V      ∈ ℝ^(N × d_v)

Attention(Q, K, V) → output ∈ ℝ^(N × d_v)
```

Now every token can attend to every other token, including itself, all in parallel. The output for position `i` is a context-aware representation that incorporates information from every relevant position in the sequence.

### 5.1 The "river bank" example, mechanized

Given the sentence "She sat by the river bank":

1. Tokenize: `[She, sat, by, the, river, bank]` → 6 token embeddings.
2. Project each through `W_Q`, `W_K`, `W_V`.
3. For the query position of `bank`:
    * Compute scores against each of the 6 keys (including its own).
    * The score against `river` is high (learned: river-like queries match river-like keys).
    * The score against `the` is low.
4. Softmax → weights → weighted sum of values.
5. `bank`'s output vector now contains information from `river` — the disambiguation has happened.

Repeat for every position. Every token's output is a context-aware version of itself.

### 5.2 Self-attention has no notion of order — positional encoding

A subtle problem: self-attention is **permutation-equivariant**. If you shuffle the input tokens, the outputs shuffle the same way, but the **content** of each output is the same. The model literally cannot tell "She sat by the river bank" from "bank river the by sat She."

The fix: inject position information into the input embeddings before they hit attention. Three common schemes:

* **Sinusoidal positional encoding (original Transformer)** — add fixed `sin/cos` waves of different frequencies to each token embedding.
* **Learned positional embeddings (BERT, GPT-2)** — a learned vector per position, added to the token embedding.
* **Rotary Position Embedding (RoPE) (Llama, Qwen, Mistral)** — rotate Q and K inside each attention block (not added to the residual stream). This is what modern LLMs use; see Phase 5 / Qwen Inference Optimization / Lecture 01 §4 for the full treatment.

Without **some** positional signal, the transformer is essentially a bag-of-words model that ignores order. Always one of these schemes is in use.

---

## 6. Multi-Head Attention

A single attention operation has a fixed perspective — one set of `(W_Q, W_K, W_V)` matrices, one "way of looking at" relationships between tokens. **Multi-head attention** runs several attention operations in parallel, each with its own projection matrices:

```
input X
   │
   ├──► head 0: own W_Q, W_K, W_V ──► attention ──► out_0 ∈ ℝ^(N × d_v)
   ├──► head 1: own W_Q, W_K, W_V ──► attention ──► out_1
   ├──► ...
   └──► head H-1                                    ──► out_{H-1}
                                                          │
                       concatenate along feature dim:    [out_0 | out_1 | ... | out_{H-1}]   ∈ ℝ^(N × H·d_v)
                                                          │
                                                 ┌────────┴────────┐
                                                 │   linear W_O    │
                                                 └────────┬────────┘
                                                          ▼
                                                 output ∈ ℝ^(N × d_model)
```

The per-head dim is typically `d_v = d_k = d_model / H`. So multi-head attention costs roughly the same as one large single-head attention — but each head can specialize.

### 6.1 What heads end up doing (empirically)

When researchers probe trained transformers, they often find heads with interpretable specializations:

* **Positional heads** — attend to the previous or next token, regardless of content.
* **Syntactic heads** — attend to syntactic dependencies (subject-verb, modifier-noun).
* **Coreference heads** — attend back to earlier mentions of the same entity.
* **Punctuation / boundary heads** — attend to sentence-ending tokens.
* **Long-range heads** — attend to distant content (especially in deeper layers).

Not every head specializes cleanly, and there's redundancy across heads. But this gives intuition for why multi-head outperforms single-head: a single soft attention pattern is a low-bandwidth bottleneck for capturing multiple relationship types simultaneously.

### 6.2 The PyTorch implementation, with batching

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)   # fused QKV
        self.W_o   = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        # x: [B, N, d_model]
        B, N, _ = x.shape
        qkv = self.W_qkv(x)                                         # [B, N, 3·d_model]
        q, k, v = qkv.chunk(3, dim=-1)                              # each [B, N, d_model]
        # Split into heads: [B, N, n_heads, d_k] → [B, n_heads, N, d_k]
        q = q.view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)    # [B, n_heads, N, N]
        if mask is not None:
            scores = scores + mask
        weights = F.softmax(scores, dim=-1)
        out = weights @ v                                           # [B, n_heads, N, d_k]

        # Recombine heads
        out = out.transpose(1, 2).contiguous().view(B, N, -1)       # [B, N, d_model]
        return self.W_o(out)
```

This is essentially what `torch.nn.MultiheadAttention` does. Real production code (`scaled_dot_product_attention`, FlashAttention) is much faster but mathematically equivalent.

---

## 7. Masking — Padding and Causal

Attention is permissive by default — every position can attend to every other position. Two cases where we need to **block** some attention:

### 7.1 Padding mask

In batched training and inference, sequences in the same batch are padded to a common length. The padding tokens are placeholders and should never contribute to the attention output. The fix: set the scores for padding positions to `-∞` before the softmax:

```
mask[i, j] = -inf  if token j is padding
mask[i, j] =    0  otherwise
```

After softmax, `e^(-∞) = 0`, so padding positions get weight 0. They don't affect the output.

### 7.2 Causal (autoregressive) mask

For a language model that generates tokens left-to-right, each output position must only depend on positions ≤ itself. Otherwise the model would "cheat" at training time by looking at the future, and at inference time the model would predict differently each time you appended a new token.

The causal mask is upper-triangular `-∞`:

```
       j=0   j=1   j=2   j=3
i=0    0    -inf  -inf  -inf
i=1    0     0    -inf  -inf
i=2    0     0     0    -inf
i=3    0     0     0     0
```

Combined with the softmax-makes-`-∞`-go-to-0 trick, this enforces "position i can only see j ≤ i."

This is the mask used by **all decoder-only LLMs**: GPT-2/3/4, Llama, Qwen, Mistral, Phi, Gemma. The output at position `i` is computed from positions `0..i` only.

### 7.3 Combining masks

In practice you add the two masks:

```python
combined_mask = padding_mask + causal_mask
# Both are -inf at "block" positions, 0 at "allow" positions; sum is also -inf or 0.
```

---

## 8. Self-Attention vs Cross-Attention

So far Q, K, V have all come from the same sequence. **Cross-attention** breaks that symmetry: Q comes from one sequence, K and V come from another.

| Type | Q source | K, V source | Common use |
|---|---|---|---|
| Self-attention | sequence A | sequence A | Encoder blocks, decoder-only LLMs |
| Cross-attention | sequence A | sequence B | Encoder–decoder models (translation, T5, Whisper, vision-language) |

In an encoder–decoder transformer (e.g., the original Transformer, T5, BART, Whisper):

```
Input sequence (e.g., source language)
   │
   ▼
┌─────────────────────────┐
│  Encoder (N layers)     │
│   each layer:           │
│   - self-attention      │
│   - FFN                 │
└──────────┬──────────────┘
           │  encoder_output ∈ ℝ^(N_enc × d_model)
           │
Output sequence (e.g., target lang., generated so far)
   │       │
   ▼       │
┌──────────────────────────────┐
│  Decoder (N layers)          │
│   each layer:                │
│   - masked self-attention    │  ← Q, K, V from generated-so-far
│   - cross-attention          │  ← Q from decoder, K & V from encoder_output
│   - FFN                      │
└──────────┬───────────────────┘
           ▼
         output logits
```

Cross-attention is how the decoder pulls information from the encoded source — at each generated position, the decoder "looks at" the source representation to decide what to generate.

Modern decoder-only LLMs (GPT, Llama, Qwen) **don't use cross-attention** — they fold everything into one big self-attention over a concatenated `[prompt | generation]` sequence, with a causal mask. This is simpler and scales better.

---

## 9. The Full Transformer Block

Attention is one piece. A real transformer block also has:

* A **feed-forward network (FFN)** — applied independently per position. Two linear layers with a nonlinearity in between. This is where most of the parameters live.
* **Residual connections** around both the attention and the FFN sub-layers.
* **Layer normalization** (or RMSNorm in modern models) before each sub-layer.

The canonical (modern, pre-norm) block:

```
x ─┬──────────► RMSNorm ──► MultiHeadAttention ──► (residual add) ──┐
   │                                                                 │
   └──────────────────────────────────────────────────────────────┐  │
                                                                  ▼  ▼
                                                                 x + attn_out  =  x'
                                                                       │
                                                                       │
   ┌───────────────────────────────────────────────────────────────────┘
   │
x' ─┬──────────► RMSNorm ──► FFN(x') ──► (residual add) ──► x''
    │
    └─────────────────────────────────────────────►
```

A `D`-layer transformer is just `D` of these blocks stacked. For Qwen3-4B, `D = 36`; for Qwen2.5-72B, `D = 80`; for BERT-base, `D = 12`; for GPT-3, `D = 96`.

### 9.1 The FFN, in detail

The classic FFN, per position:

```
ffn(x) = W_2 · activation(W_1 · x + b_1) + b_2
```

`W_1: [d_model → d_ff]` widens; `W_2: [d_ff → d_model]` projects back. `d_ff` is typically `4 · d_model` (BERT, GPT-2) or `~2.7 · d_model` (Llama, Qwen with SwiGLU).

Modern LLMs use **SwiGLU** (gated linear unit with SiLU activation):

```
ffn(x) = W_down · ( silu(W_gate · x) ⊙ (W_up · x) )
```

Three projections instead of two; the gate and up paths produce vectors of width `d_ff`, multiply element-wise, then project back. Slightly more compute but consistently better quality at the same parameter budget.

### 9.2 Pre-norm vs post-norm

The **original Transformer** put LayerNorm **after** the sub-layer + residual ("post-norm"). Modern models put it **before** ("pre-norm"). Pre-norm is more stable at depth (you can train 100+ layer models without learning-rate warmup) and is now standard.

Forget the original ordering. Pre-norm is what real models use.

### 9.3 The full forward pass

Putting it together for a decoder-only LLM like Qwen3:

```
tokens ─► token_embedding ─┬─► (positional encoding, if any) ─►
                           │
                           ▼
                     ┌────────────────────────┐
                     │  Transformer block × D │
                     │                        │
                     │  RMSNorm → MHA  ⊕      │
                     │  RMSNorm → FFN  ⊕      │
                     └────────────┬───────────┘
                                  │
                                  ▼
                              RMSNorm
                                  │
                                  ▼
                     LM head (Linear: d_model → vocab)
                                  │
                                  ▼
                              logits
                                  │
                                  ▼
                         softmax → sample → next token
```

Append the sampled token to the input, repeat. That's autoregressive generation.

---

## 10. The Three Canonical Configurations

| Configuration | Examples | Mask | Use |
|---|---|---|---|
| **Encoder-only** | BERT, RoBERTa, DeBERTa | Padding only | Classification, embedding, retrieval |
| **Decoder-only** | GPT-2/3/4, Llama, Qwen, Mistral, Phi, Gemma | Causal (+ padding) | Generation, chat, code completion |
| **Encoder–decoder** | Original Transformer, T5, BART, Whisper, mT5 | Encoder: padding. Decoder: causal + cross-attention | Translation, summarization, ASR |

What changed over time:

* **2017–2019** — encoder–decoder was assumed standard.
* **2018–2020** — encoder-only (BERT) dominated NLP benchmarks.
* **2020–present** — decoder-only LLMs scaled up and ate everything.

The reason decoder-only LLMs won: a single causal-masked self-attention model with **enough scale** can do every task an encoder–decoder model can, plus interactive generation. You unify training and inference, you unify pretraining and fine-tuning, and you unify "understanding" tasks with "generation" tasks.

All the models you'll deploy in the inference lectures are decoder-only.

---

## 11. From Architecture to Inference

This lecture is the prerequisite for the Phase 5 inference lectures. The bridge:

* **Phase 5 / Edge AI / Edge LLM Inference Internals** treats the same QKV/FFN math as a **bandwidth problem**. The transformer block you just learned about is what GEMV-decode is computing.
* **Phase 5 / Qwen Inference Optimization (6-lecture series)** specializes the same architecture to two specific Qwen models: Qwen3-4B-Q4_K_M for edge and Qwen2.5-72B-FP16 for datacenter. Same block diagram; different hardware engineering.
* **Phase 5 / Qwen Inference Optimization / Lecture 01 §4** is the full RoPE deep dive — the modern positional-encoding scheme used by Qwen, Llama, Mistral, etc.
* **Phase 5 / Qwen Inference Optimization / Lecture 06** is the cuBLAS GEMM/batched-GEMM lecture — what the attention `Q @ Kᵀ` actually calls under the hood.

!!! tip "Why this lecture unlocks everything else"
    Once you understand the material here, the inference lectures stop being a wall of acronyms:

    * **GQA, MQA, MHA** — choices about whether multi-head attention shares K/V across heads.
    * **AWQ, GPTQ, Q4_K_M** — choices about how to quantize the `W_Q, W_K, W_V, W_O, W_gate, W_up, W_down` matrices.
    * **FlashAttention** — a faster, tiled implementation of the same attention you implemented in §6.2.
    * **Speculative decoding** — running a small autoregressive model alongside a big one.

    Each one slots into the architecture you just learned.

---

## 12. Hands-On Exercises

1. **Implement attention in 30 lines.** Type out the §4.3 reference and verify on a small example (4 tokens, d_k=8). Print the attention weights. Confirm they sum to 1 along the last axis.

2. **Build a single transformer block.** Combine the multi-head attention from §6.2 with a SwiGLU FFN, RMSNorm, and residual connections. Feed in a random `[1, 16, 256]` input and confirm the output shape is `[1, 16, 256]`.

3. **Compare to PyTorch's built-ins.** Replace your manual attention with `torch.nn.functional.scaled_dot_product_attention` and verify the output matches to within 1e-5. Then enable Flash backend (`with torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):`) and confirm the result is still numerically equivalent.

4. **Causal mask in action.** Train a tiny decoder-only transformer (2 layers, d_model=64) on a character-level dataset (Shakespeare, ~1 MB). Sample from it after 1000 steps. Then deliberately remove the causal mask, retrain, and observe how the loss looks much better during training but the model generates gibberish. (Because it learned to "cheat" by looking at the future.)

5. **Visualize attention.** On a pretrained model (load `Qwen/Qwen3-4B-Instruct` via `transformers`), hook into one layer's attention weights for the prompt "She sat by the river bank." Plot the attention heatmap for the head that most strongly connects "bank" to "river." Compare to a different attention head where the pattern is uninterpretable.

6. **Count parameters.** For a transformer with `d_model = 2560, n_heads = 32, n_layers = 36, d_ff = 6912, vocab = 151936`, compute total parameters from scratch. Compare to Qwen3-4B's published 4 B figure. (You should land near 3.2 B — the embedding contributes most of the remainder.)

7. **Permutation-equivariance demo.** Build a single self-attention layer **without** positional encoding. Feed it two inputs that are permutations of each other (same tokens in different order). Confirm the outputs are also permuted (same values, different order). Then add sinusoidal positional encoding and re-run; outputs should now differ in value, not just position.

8. **Read a real model.** Open `Qwen/Qwen3-4B-Instruct`'s `config.json` and identify, by name, every quantity from this lecture: `d_model` (`hidden_size`), `n_heads` (`num_attention_heads`), `n_kv_heads` (`num_key_value_heads` — note GQA), `n_layers` (`num_hidden_layers`), `d_ff` (`intermediate_size`), vocab, RoPE base. Predict the shape of `model.layers[0].self_attn.q_proj.weight` from the config and verify with `print(model.state_dict()['model.layers.0.self_attn.q_proj.weight'].shape)`.

---

## 13. Key Takeaways

| Takeaway | Why it matters |
|---|---|
| Attention removes the fixed-context bottleneck of RNNs | Every position can pull from every other position directly |
| Q/K/V is a three-role view of "look up by similarity, aggregate by softmax" | All attention math reduces to this |
| Scaled dot product divides by √d_k to prevent peaked softmax | This is the *fix*, not the cause — unscaled is too peaked, not too flat |
| Self-attention has no order — you need positional encoding | Without it, a transformer is a bag-of-words model |
| Multi-head attention runs many attention ops in parallel with their own projections | Heads can specialize; total parameter cost is similar to single-head |
| Padding mask zeros out attention to padding; causal mask blocks future positions | Mask is added to scores as `-inf`, then softmax does the rest |
| Modern LLMs are decoder-only with causal mask | One unified architecture for understanding and generation |
| The full block is RMSNorm → attention → ⊕ → RMSNorm → FFN → ⊕ | This pattern repeats `D` times in every transformer you'll deploy |
| Attention weights are clues, not full explanations | Useful for interpretability but never proof of "what the model decided" |
| Self-attention is `O(N²)` in sequence length | This is why long-context inference is hard; all advanced techniques exist to manage it |

---

## 14. The One-Sentence Takeaway

!!! quote ""
    Attention lets a neural network dynamically focus on the most relevant parts of a sequence by comparing queries to keys, using the resulting weights to combine values into context-aware representations — and a transformer is just `D` layers of multi-head attention plus FFN with residual connections, with a mask choice that determines whether it's an encoder, decoder, or encoder–decoder model.

---

## Resources

* **["Attention Is All You Need" (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762):** The original transformer paper. Concise; read it twice.
* **[The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/2018/04/03/attention.html):** A line-by-line PyTorch implementation of the original Transformer paired with the paper text. The single best learning resource.
* **[The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/):** The most accessible visual explainer; great for building first intuition.
* **["Transformers from Scratch" (Brandon Rohrer)](https://e2eml.school/transformers.html):** A second pass through the same material with different intuition.
* **[3Blue1Brown — Neural Networks chapter 5+ (Attention)](https://www.youtube.com/watch?v=eMlx5fFNoYc):** Video treatment with clean diagrams.
* **[Hugging Face Transformers — `modeling_qwen2.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py):** A real production-grade transformer implementation you can read top-to-bottom.
* **[PyTorch — `scaled_dot_product_attention` docs](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html):** The modern fused-attention primitive.
* **[FlashAttention paper (Dao et al., 2022)](https://arxiv.org/abs/2205.14135):** How to compute the same attention 5× faster by tiling. Not required for fundamentals, but the bridge to inference engineering.
* **[Phase 5 — Edge LLM Inference Internals](../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20C%20-%20Edge%20AI/Edge%20LLM%20Inference%20Internals/Lecture-01.md):** The first downstream lecture that applies this material to real hardware.
* **[Phase 5 — Qwen Inference Optimization](../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20C%20-%20Edge%20AI/Qwen%20Inference%20Optimization/README.md):** The 6-lecture series that specializes this architecture to Qwen3-4B and Qwen2.5-72B.
