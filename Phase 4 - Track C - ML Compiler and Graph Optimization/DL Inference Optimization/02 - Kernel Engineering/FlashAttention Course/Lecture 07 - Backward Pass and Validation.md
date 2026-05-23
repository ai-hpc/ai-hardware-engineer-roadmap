# Lecture 7 — Backward Pass and Validation

**Parent:** [FlashAttention Course](Guide.md)

**One-line purpose:** Derive `dQ`, `dK`, `dV` for FlashAttention, understand the recomputation trick that makes the backward pass affordable, and build a trustworthy correctness harness for any kernel modification.

**Prerequisites:** Lectures 1–6. Comfort with vector calculus and chain rule on matrix operations.

**Artifact:** A correctness harness that compares dQ/dK/dV against a reference implementation across multiple shapes and dtypes, with documented tolerance bands.

---

## Why it matters

The backward pass is where most attention kernel bugs hide. It is also the most numerically fragile part: gradients are smaller than activations by orders of magnitude, and any non-deterministic reduction order will show up as drift between runs. If you ship a forward kernel without a matched backward harness, you will burn weeks of training time debugging convergence issues that turn out to be your kernel.

This lecture builds the harness you will use for the rest of your career.

---

## Mental model

### The math

Given forward `O = softmax(QKᵀ/√d) · V`, the gradients are:

```
dV = Pᵀ · dO                        (where P = softmax(S))
dP = dO · Vᵀ
dS = (dP - rowsum(dP ⊙ P)) ⊙ P     (softmax Jacobian: dS_ij = P_ij(dP_ij - Σ_k P_ik dP_ik))
dQ = (dS · K) / √d
dK = (dSᵀ · Q) / √d
```

Three matmuls plus one elementwise scaling. Straightforward — *if* you have `P` in memory.

### Why naive backward is bad

You either recompute `S` and `P` (extra FLOPs) or store them (extra memory: `O(N²)` per layer per head). For long context training, storage is impossible. So the FA backward **recomputes** `S` and `P` tile by tile using the **saved `LSE`** from the forward.

### The recomputation trick

Forward saves `LSE` (per-row, fp32). For backward, you re-stream tiles of `K, V` against tiles of `Q` and compute `P_ij = exp(S_ij - LSE_i)` directly — no `m` or `ℓ` recurrence needed because `LSE` is already converged. That makes the backward pass structurally similar to the forward pass: stream KV against Q, do two matmuls per tile (for `dV` and `dS`), and accumulate `dQ`, `dK`, `dV`.

Net cost: backward is roughly **2.5× the forward FLOPs** (three matmuls instead of two, plus the recompute), with the same `O(N² d² / M)` HBM complexity. In practice it runs at ~70% the speed of the forward.

### Determinism

Atomic adds (which appear in the FA2 backward for `dK` and `dQ` when warps split work along the same row) are **non-deterministic** in floating-point. FA2 has an `--deterministic` mode that reorders reductions to use shared-memory reductions instead of atomics; you pay a few percent in speed for bit-reproducibility. For training validation harnesses, **always use deterministic mode** when comparing across runs.

### Dropout

FA applies dropout *inside* the kernel: it samples a Bernoulli mask using a Philox RNG seeded by `(seed, offset)`. To verify correctness against a reference, both implementations must use the same RNG with the same seed/offset, otherwise you cannot reproduce the mask. The repo's `tests/test_flash_attn.py` does this; copy that pattern.

### Mask behaviour

- Causal mask just sets `S_ij = -∞` for `j > i`. Backward inherits the mask (gradients above the diagonal are zero).
- Sliding-window mask adds a lower bound on `j` (`j > i - window`) — same masking trick, just two-sided.
- Custom masks via `attn_bias` need a separate testing path because they bypass the fast mask helpers.

---

## Build it

The harness lives in a single file you will use for every kernel change you make:

```python
# fa_correctness_harness.py
import math, json
import torch
import torch.nn.functional as F

DTYPE_TOL = {
    torch.float32: (1e-5, 1e-4),    # (atol, rtol)
    torch.bfloat16: (5e-3, 5e-3),
    torch.float16: (1e-3, 1e-3),
}

def reference_attention(q, k, v, causal=False, sm_scale=None):
    # math-only reference, fp32 throughout
    q32 = q.float(); k32 = k.float(); v32 = v.float()
    sm_scale = sm_scale or 1.0 / math.sqrt(q.size(-1))
    s = q32 @ k32.transpose(-1, -2) * sm_scale
    if causal:
        N = s.size(-1)
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=s.device), diagonal=1)
        s = s.masked_fill(mask, float("-inf"))
    p = s.softmax(-1)
    o = p @ v32
    return o.to(q.dtype)

def check(name, candidate_fn, shape, dtype, causal=False, seed=0):
    torch.manual_seed(seed)
    B, H, N, D = shape
    q = torch.randn(B, H, N, D, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)

    o_ref = reference_attention(q.detach(), k.detach(), v.detach(), causal=causal)
    o_test = candidate_fn(q, k, v, causal=causal)

    # Forward
    fwd_diff = (o_test.detach() - o_ref).abs().max().item()

    # Backward — same dO for both
    dO = torch.randn_like(o_test)
    o_test.backward(dO)
    dq_test, dk_test, dv_test = q.grad.detach(), k.grad.detach(), v.grad.detach()

    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    reference_attention(q2, k2, v2, causal=causal).backward(dO)
    dq_ref, dk_ref, dv_ref = q2.grad, k2.grad, v2.grad

    atol, rtol = DTYPE_TOL[dtype]
    report = {
        "name": name, "shape": shape, "dtype": str(dtype), "causal": causal,
        "fwd_max_abs": fwd_diff,
        "dq_max_abs": (dq_test - dq_ref).abs().max().item(),
        "dk_max_abs": (dk_test - dk_ref).abs().max().item(),
        "dv_max_abs": (dv_test - dv_ref).abs().max().item(),
        "atol_band": atol, "rtol_band": rtol,
        "pass": all(t.allclose(r, atol=atol, rtol=rtol)
                    for t, r in [(o_test, o_ref), (dq_test, dq_ref),
                                 (dk_test, dk_ref), (dv_test, dv_ref)]),
    }
    return report

if __name__ == "__main__":
    from flash_attn import flash_attn_func
    def cand(q, k, v, causal):
        # FA expects [B, S, H, D]; reshape from [B, H, S, D]
        q_ = q.transpose(1, 2); k_ = k.transpose(1, 2); v_ = v.transpose(1, 2)
        o = flash_attn_func(q_, k_, v_, causal=causal)
        return o.transpose(1, 2)

    shapes = [(1, 8, 512, 64), (2, 16, 2048, 128), (1, 4, 8192, 64)]
    rows = []
    for shape in shapes:
        for dtype in [torch.float16, torch.bfloat16]:
            for causal in [False, True]:
                rows.append(check("fa2", cand, shape, dtype, causal=causal))
    print(json.dumps(rows, indent=2))
```

Run it. Every row should `pass: true` with diffs comfortably inside the tolerance band. If any fail, debug *that* row before moving on — do not weaken the tolerance.

---

## Use it in the real stack

The FA repo's own `tests/test_flash_attn.py` is a much more comprehensive version of this harness — it covers dropout, varlen, sliding window, deterministic mode, ALiBi, and more shapes. Read it once end to end; copy the pieces you need into your personal harness.

For a real production patch (Lecture 10), you need at minimum:

- The harness above, expanded with your target shapes.
- A determinism check: run forward+backward twice with the same seed; gradients must be bit-identical when `--deterministic` is on.
- A drift check across long iterations: train a 1-layer toy transformer for 1000 steps with your kernel vs the reference; the loss curves must overlap.

---

## Measure it

For each row in the harness:

- Print `fwd_max_abs`, `dq_max_abs`, `dk_max_abs`, `dv_max_abs`.
- Confirm each is below the dtype's `atol_band`.
- If any is "close but failing", check that you are using the same dropout RNG seed/offset on both paths.

For deterministic mode, run the kernel 10 times with the same inputs and confirm the gradients are bit-identical (`torch.equal`, not `allclose`).

---

## Ship it

Drop into `flash-attn-course/`:

1. `fa_correctness_harness.py` — the script above, expanded for your target shapes.
2. `harness_report.json` — JSON output from one full run.
3. A short README note describing which dtypes / shapes you actually validated and which you skipped (and why).

This harness is your **safety net** for every later patch. Do not modify a kernel without it running green.

---

## Related pages

- [Lecture 6 — FlashAttention-2](Lecture%2006%20-%20FlashAttention-2.md)
- [Lecture 8 — Inference path](Lecture%2008%20-%20Inference%20Path%20KV%20Cache%20and%20Decode.md)
- [Lecture 10 — Capstone](Lecture%2010%20-%20Capstone.md)
