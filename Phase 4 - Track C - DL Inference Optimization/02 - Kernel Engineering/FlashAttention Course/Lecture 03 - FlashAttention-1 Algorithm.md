# Lecture 3 — FlashAttention-1 Algorithm

**Parent:** [FlashAttention Course](Guide.md)

**One-line purpose:** Walk the FA1 forward and backward algorithm, explain its IO complexity bound, and connect the tiles in the paper to the variables in the source.

**Prerequisites:** Lectures 1 and 2. You should be able to derive the IO of naive attention and run the blockwise-softmax NumPy check.

**Artifact:** A pseudocode walk-through plus an HBM-byte counter you can run against arbitrary `(N, d, B_r, B_c)` that reproduces the paper's `O(N²d²/M)` bound to within a constant.

---

## Why it matters

FA1 is the first kernel that turned attention from an `O(N²)`-bandwidth operation into an `O(N²d²/M)`-bandwidth one, where `M` is on-chip memory (SRAM). That single change is what made 32k–1M context training affordable. If you understand FA1, FA2 and FA3 become "the same thing, scheduled better."

---

## Mental model

### Tiling Q, K, V

Split the matrices into tiles:

- `Q` into `T_r = ceil(N / B_r)` row blocks of size `B_r × d`.
- `K` and `V` into `T_c = ceil(N / B_c)` column blocks of size `B_c × d`.

Each block must fit, together with running scratch, in SRAM. For Ampere/Hopper that means roughly `B_r · d + B_c · d + B_r · B_c` words.

### The forward loop (Algorithm 1 from the FA1 paper)

```
allocate O ∈ [N, d], LSE ∈ [N], all zero / -inf
for i = 0 .. T_r - 1:                           # outer: over query tiles
    Q_i = load Q rows [i·B_r : (i+1)·B_r]       # into SRAM
    m_i = -inf vector of length B_r              # running max
    ℓ_i =  0 vector of length B_r                # running sum-exp
    O_i =  0 matrix [B_r, d]                     # running output
    for j = 0 .. T_c - 1:                       # inner: over KV tiles
        K_j, V_j = load tile [j·B_c : (j+1)·B_c]
        S_ij = (Q_i @ K_j.T) / sqrt(d)          # SRAM only, [B_r, B_c]
        m_ij = rowmax(S_ij)                      # in-register
        P_ij = exp(S_ij - m_ij)                  # local probs
        ℓ_ij = rowsum(P_ij)
        m_new = max(m_i, m_ij)
        rescale_old = exp(m_i  - m_new)
        rescale_new = exp(m_ij - m_new)
        O_i = rescale_old * O_i + rescale_new * (P_ij @ V_j)
        ℓ_i = rescale_old * ℓ_i + rescale_new * ℓ_ij
        m_i = m_new
    O[i·B_r : (i+1)·B_r] = O_i / ℓ_i             # write back to HBM
    LSE[i·B_r : (i+1)·B_r] = m_i + log(ℓ_i)
```

This is the exact NumPy code from Lecture 2, mapped onto a two-level loop where the outer loop owns the HBM writes of `O` and `LSE`, and the inner loop stays in SRAM.

### Why this is `O(N²d²/M)` bytes

Inner loop loads each `(K_j, V_j)` tile once per outer iteration `i`. That is `T_r · T_c · (2 B_c · d)` bytes for `K + V`, plus `T_r · (B_r · d)` for `Q` and `T_r · (B_r · d)` for `O / LSE` writes.

Substituting `T_r = N/B_r` and `T_c = N/B_c`:

```
KV traffic ≈ (N/B_r) · (N/B_c) · (2 B_c · d · 2 bytes)
           = 4 N² d / B_r           bytes
```

With `B_r = Θ(M / d)` (the largest `B_r` such that a `[B_r, d]` query tile and friends fit in SRAM `M`), this becomes `O(N² d² / M)` bytes — the FA1 bound.

For typical `M = 100 KB` of SRAM per SM, `d = 64`, and `N = 8192`, that is roughly `N² d² / M ≈ (8192² · 64²) / 100e3 ≈ 27 MB` of HBM traffic — vs hundreds of MB for the naive variant. This is the win.

### Backward in one sentence

FA1 does not store `S` or `P`. To compute gradients it **recomputes** them tile by tile using `Q, K, V` and the saved `LSE`. That trades a small amount of extra compute for not having to store the `[N, N]` matrix. The backward pass is structurally identical to the forward pass with a few extra `[B_r, B_c]` matmuls — same `O(N²d²/M)` HBM complexity.

---

## Build it

Read the FA1 paper Section 3 (Algorithm 1 + Section 3.2 on backward). Then write this byte counter:

```python
# fa1_iobytes.py
def naive_bytes(N, d, dtype=2):
    qkv_load = 3 * N * d * dtype
    s_rw     = 2 * N * N * dtype          # write S, read S
    p_rw     = 2 * N * N * dtype          # write P, read P for PV
    o_write  =     N * d * dtype
    return qkv_load + s_rw + p_rw + o_write

def fa1_bytes(N, d, Br, Bc, dtype=2):
    Tr = -(-N // Br)
    Tc = -(-N // Bc)
    q_load   = Tr * (Br * d) * dtype                       # Q once
    kv_load  = Tr * Tc * (2 * Bc * d) * dtype              # K, V per outer i
    out_write = Tr * (Br * d) * dtype                      # O
    lse_write = Tr * Br * 4                                # fp32 LSE
    return q_load + kv_load + out_write + lse_write

if __name__ == "__main__":
    for N in [1024, 2048, 4096, 8192, 16384]:
        for d in [64, 128]:
            Br, Bc = 64, 64
            naive = naive_bytes(N, d) / 1e6
            fa1   = fa1_bytes(N, d, Br, Bc) / 1e6
            ratio = naive / fa1
            print(f"N={N:>5} d={d:>3} naive={naive:>8.1f} MB  fa1={fa1:>8.1f} MB  speedup={ratio:>5.1f}x")
```

Run it; you should see speedups growing roughly linearly in `N` (because the `N²` term dominates naive at the shapes you care about).

Also: try `B_r = B_c = 32, 64, 128`. The smaller the tile, the more KV passes you do per outer `i`, and the higher the HBM cost — match this against what FA1 picks at runtime by reading `csrc/flash_attn/src/flash_fwd_launch_template.h` (the dispatch table by `head_dim`).

---

## Use it in the real stack

Trace one forward call from Python to the kernel. Start at:

`flash-attention/flash_attn/flash_attn_interface.py → flash_attn_func()`
↓
`flash-attention/flash_attn/_C.pyi → mha_fwd(...)`
↓
`flash-attention/csrc/flash_attn/flash_api.cpp → mha_fwd()`
↓
`flash-attention/csrc/flash_attn/src/flash_fwd_launch_template.h → run_mha_fwd_<...>()`
↓
`flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h → flash_fwd_kernel()`

Inside `flash_fwd_kernel.h`, look for:

- The `for (int n_block = ...)` loop — that is the inner KV loop.
- `__syncthreads()` and the `tOrO`, `tOrS`, `tOrP` register variables — the running output and probability tiles.
- `softmax_rescale_o_` and similar — the rescale-by-`exp(m_old - m_new)` step.

Match each FA1-paper variable to a real variable in the kernel. Write the mapping table in your notes (you will need it in Lecture 6 when you compare against FA2's slightly different naming).

---

## Measure it

You cannot easily count HBM bytes from PyTorch directly, but Nsight Compute can:

```
ncu --set full --target-processes all \
    --metrics dram__bytes.sum,sm__sass_thread_inst_executed_op_dfma_pred_on.sum \
    python attention_bench.py
```

Compare `dram__bytes.sum` for a naive SDPA call vs a FlashAttention call at the same `(N, d, H)`. You should see the FA byte count scale roughly as `N · d` rather than `N²`, matching your `fa1_bytes` model within ~20% (driver and L2 traffic make up the rest).

---

## Ship it

Add to your `flash-attn-course/` working dir:

1. `fa1_iobytes.py` — your byte counter, with at least three `(N, d)` rows printed.
2. A short Markdown note mapping FA1 paper symbols (`m_i`, `ℓ_i`, `O_i`, `m_ij`, `ℓ_ij`) to the variable names in `flash_fwd_kernel.h`.
3. One Nsight Compute report at a single shape, showing `dram__bytes.sum` for naive vs FA.

If those three artifacts exist, you have actually read the kernel — not just the paper.

---

## Related pages

- [Lecture 2 — Online softmax and numerical correctness](Lecture%2002%20-%20Online%20Softmax%20and%20Numerical%20Correctness.md)
- [Lecture 4 — GPU kernel performance basics](Lecture%2004%20-%20GPU%20Kernel%20Performance%20Basics.md)
- [Lecture 6 — FlashAttention-2](Lecture%2006%20-%20FlashAttention-2.md)
