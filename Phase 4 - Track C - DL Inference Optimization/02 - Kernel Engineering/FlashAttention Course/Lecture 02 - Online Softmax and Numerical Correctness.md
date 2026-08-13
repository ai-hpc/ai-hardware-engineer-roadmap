# Lecture 2 — Online Softmax and Numerical Correctness

**Parent:** [FlashAttention Course](Guide.md)

**One-line purpose:** Derive and implement the blockwise log-sum-exp recurrence that lets FlashAttention produce an exact softmax without materialising the full `[N, N]` score matrix.

**Prerequisites:** Lecture 1. Comfort with the `softmax(x) = exp(x − m) / Σ exp(x − m)` shifted form.

**Artifact:** A NumPy script that proves blockwise-LSE equals one-shot softmax bit-for-bit in fp64, plus a small report of the actual fp16 / bf16 error you see for realistic shapes.

---

## Why it matters

FlashAttention does not approximate softmax. It uses an exact identity that lets you update a partial softmax incrementally as you stream tile after tile of `K` and `V`. If you do not internalise this recurrence you cannot read the FA1 source, you cannot debug a numerics bug, and you cannot port the algorithm to a new backend.

The single trick is: softmax is invariant to a constant shift in its input, and the **log-sum-exp** of two blocks can be combined exactly using a rescaling factor. That's it. Everything else is engineering.

---

## Mental model

### The classical safe softmax

For numerical stability with fp16/bf16, you never compute `exp(x)` directly. You compute:

```
m = max(x)
softmax(x)_i = exp(x_i - m) / sum_j exp(x_j - m)
```

You also keep the **log-sum-exp**:

```
LSE(x) = m + log(sum_j exp(x_j - m))
```

so that `softmax(x)_i = exp(x_i - LSE(x))`. The pair `(m, LSE)` summarises everything you need to combine softmaxes from different blocks.

### Combining two blocks exactly

Suppose you have two blocks `A` and `B` and their statistics:

- `m_A, ℓ_A = sum_j exp(A_j - m_A)`
- `m_B, ℓ_B = sum_j exp(B_j - m_B)`

The combined max is `m = max(m_A, m_B)`. The combined denominator is:

```
ℓ = exp(m_A - m) · ℓ_A + exp(m_B - m) · ℓ_B
```

That's it. The combined softmax over `[A, B]` is `exp(x_i - m) / ℓ`. Equivalent in arithmetic to a one-shot softmax over the concatenation — no information lost.

### Carrying the output along

FlashAttention applies `P · V` while it streams. After processing block `B`, the running output `O` becomes:

```
O_new = (ℓ_old · exp(m_old - m_new) / ℓ_new) · O_old
      + (exp(m_B   - m_new) / ℓ_new) · (P_B · V_B)
```

The first term rescales the already-accumulated output for the new normaliser; the second term adds this block's contribution. After the last block, `O` is the exact `softmax(QKᵀ) · V` row for that query tile. **You never need `S` or `P` in HBM** — just running scalars `(m, ℓ)` per query row and a tile of `V` per step.

### Causal masking just changes which blocks contribute

Causal mask sets `S_{i,j} = -∞` for `j > i`. In the recurrence that means: when block `B` is entirely above the diagonal for the current query tile, you skip it; when it straddles the diagonal you mask the upper-triangular entries of `S_B` to `-∞` before computing `m_B, ℓ_B`. The recurrence stays correct.

---

## Build it

Read FA1 paper Section 3.1 (Algorithm 1 — fused matmul–softmax) and Tri Dao's [online-softmax note](https://github.com/Dao-AILab/flash-attention/blob/main/assets/flashattn_banner.jpg) (the algorithm boxes in the paper are clearer than the README). Then write this exact-equivalence test:

```python
# online_softmax_check.py
import numpy as np

def one_shot_softmax(x):
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)

def online_softmax(x, block):
    """Compute softmax(x) by streaming blocks of size `block`."""
    N = x.shape[-1]
    m = np.full(x.shape[:-1] + (1,), -np.inf)
    ell = np.zeros_like(m)
    for s in range(0, N, block):
        e = x[..., s:s+block]
        m_blk = e.max(axis=-1, keepdims=True)
        ell_blk = np.exp(e - m_blk).sum(axis=-1, keepdims=True)
        m_new = np.maximum(m, m_blk)
        ell = np.exp(m - m_new) * ell + np.exp(m_blk - m_new) * ell_blk
        m = m_new
    # Second pass: produce normalised softmax with the converged (m, ell).
    p = np.empty_like(x)
    for s in range(0, N, block):
        p[..., s:s+block] = np.exp(x[..., s:s+block] - m) / ell
    return p

def online_attention(Q, K, V, block):
    """Same idea but folds P·V into the streaming loop. No N×N materialisation."""
    N = K.shape[-2]
    d = V.shape[-1]
    O = np.zeros(Q.shape[:-1] + (d,))
    m = np.full(Q.shape[:-1] + (1,), -np.inf)
    ell = np.zeros_like(m)
    scale = 1.0 / np.sqrt(Q.shape[-1])
    for s in range(0, N, block):
        Kb = K[..., s:s+block, :]
        Vb = V[..., s:s+block, :]
        Sb = scale * (Q @ Kb.swapaxes(-1, -2))      # [..., q, block]
        m_blk = Sb.max(axis=-1, keepdims=True)
        Pb = np.exp(Sb - m_blk)                     # local probs (unnormalised)
        ell_blk = Pb.sum(axis=-1, keepdims=True)
        m_new = np.maximum(m, m_blk)
        rescale_old = np.exp(m - m_new)
        rescale_new = np.exp(m_blk - m_new)
        O = rescale_old * O + rescale_new * (Pb @ Vb)
        ell = rescale_old * ell + rescale_new * ell_blk
        m = m_new
    return O / ell, m + np.log(ell)  # output and LSE

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    H, N, d = 2, 1024, 64
    Q = rng.standard_normal((H, N, d))
    K = rng.standard_normal((H, N, d))
    V = rng.standard_normal((H, N, d))

    # Reference: one-shot
    S = (Q @ K.swapaxes(-1, -2)) / np.sqrt(d)
    P = one_shot_softmax(S)
    O_ref = P @ V

    O_flash, lse = online_attention(Q, K, V, block=64)
    print("max abs diff (fp64):", np.abs(O_ref - O_flash).max())
```

In fp64 the diff should be at the level of `1e-13` — purely floating-point reduction-order noise. That proves the algorithm is **exact**.

Now repeat the test in fp32 and bf16 (cast `Q, K, V` and run the same code). Record the max absolute error and the relative error at a few percentile cuts (`p50`, `p99`, `max`). You should see fp32 errors around `1e-5` and bf16 errors around `5e-3`. These numbers are the tolerance bands you will use later in your correctness harness (Lecture 7).

---

## Use it in the real stack

Open `flash-attention/flash_attn/flash_attn_interface.py` and look for `flash_attn_func`. The forward pass that runs inside the CUDA kernel implements exactly the recurrence you just wrote in NumPy. The kernel additionally returns the per-row **log-sum-exp** as a side output (`lse`) — that is what makes the backward pass cheap (Lecture 7).

You can verify by reading `csrc/flash_attn/src/flash_fwd_kernel.h`: search for `running_max`, `running_sum`, and `rescale` — these are the in-register names for the `m`, `ℓ`, and rescale factor from your NumPy code.

---

## Measure it

For the lab:

- fp64 reference vs your blockwise-LSE: max abs diff should be at machine precision noise (~1e-13). If it is not, you have a bug — most likely in the `exp(m_old - m_new)` rescale of either `O` or `ℓ`.
- fp32 vs fp32 blockwise: max abs diff ~1e-5.
- bf16 vs bf16 blockwise: max abs diff ~5e-3 to 1e-2, depending on `N` and the score distribution.

Do **not** compare bf16 against fp64 and complain about a 1e-2 error — that is the dtype, not the algorithm.

---

## Ship it

Save the script as `online_softmax_check.py` in your course working directory. It must produce:

1. fp64 max abs diff close to `1e-13` (proves exactness).
2. fp32 and bf16 max abs diff in line with the expected numbers above.
3. A short README note: "the running `(m, ℓ)` and rescale-by-`exp(m_old - m_new)` together implement an exact streaming softmax."

You will reuse this script as the gold reference for every later kernel you build or modify.

---

## Related pages

- [Lecture 1 — Attention bottleneck and roofline](Lecture%2001%20-%20Attention%20Bottleneck%20and%20Roofline.md)
- [Lecture 3 — FlashAttention-1 algorithm](Lecture%2003%20-%20FlashAttention-1%20Algorithm.md)
- [Lecture 7 — Backward pass and validation](Lecture%2007%20-%20Backward%20Pass%20and%20Validation.md)
