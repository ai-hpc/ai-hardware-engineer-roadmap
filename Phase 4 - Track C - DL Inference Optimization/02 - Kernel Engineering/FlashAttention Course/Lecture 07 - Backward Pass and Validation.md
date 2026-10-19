# Lecture 7 — Backward math, masks and a defensible correctness test

**Course:** [FlashAttention](Guide.md) · **Reviewed:** 18 October 2026

**Goal:** derive gradients and test them independently. **Artifact:** NumPy analytic gradients, finite-difference checks and actual PyTorch CPU autograd comparisons.

## 1. Derive the four gradient products

Let `S = scale × QKᵀ`, `P = softmax(S)` over allowed keys and `O = PV`. Given an upstream gradient `dO`:

```text
dV = Pᵀ dO
dP = dO Vᵀ
dS = P ⊙ (dP − rowsum(P ⊙ dP))
dQ = scale × dS K
dK = scale × dSᵀ Q
```

There are **four** matrix products in these gradient equations. Recomputing `QKᵀ` adds a fifth. For equal dimensions and dense pairs, that explains a rough 2.5× matmul-work ratio to the two-product forward. It does not imply a fixed backward/forward runtime ratio: scheduling, recomputation, memory traffic and reduction costs differ.

Masked probabilities are zero, so their `dS` contribution is zero. For this lab's fully masked row convention, output and corresponding Q gradient are zero. The backward implementation explicitly uses the stable reference probabilities, rather than applying a softmax Jacobian to NaNs.

## 2. Why save normalization statistics

A memory-efficient backward can recompute a score tile and reconstruct probabilities using the saved per-row log-sum-exp. It avoids retaining a dense `[L,S]` probability tensor. The recomputation trades extra arithmetic for memory capacity/traffic. It still needs correct handling of empty rows, masks, scale and input layout. [FA1 recomputation analysis](https://arxiv.org/html/2205.14135v2).

## 3. Run two independent checks

The [CPU lab](CPU-Attention-Lab.md) supplies:

- `backward()` in [attention_reference.py](labs/attention_reference.py), implementing the equations above.
- Finite differences for selected Q/K/V elements, using the scalar objective `sum(O × dO)` and a small perturbation.
- An actual PyTorch SDPA math backward with the same inputs, mask and `dO`, compared against the NumPy analytic result.

Use a small float64 fixture first. A finite difference balances truncation and rounding error; shrinking epsilon indefinitely does not always improve it. Compare multiple selected elements, especially near a tile or mask boundary.

The suite also checks tails, rectangular masks, fully masked rows, noncontiguous inputs, a legal zero scale and grouped-query attention. A shortcut `scale = provided_scale or default` incorrectly replaces the valid value zero; the implementation uses an explicit `None` test.

## 4. Precision, dropout and reproducibility

Pick tolerances for a named fixture/backend/dtype before comparing a patch. Record maximum absolute error and a combined relative/absolute criterion. Relative error near a zero reference can be misleading. A passing tolerance table does not establish training convergence for every workload.

Disable dropout for the first deterministic comparison. Passing the same global seed to unrelated implementations does not guarantee the same dropout mask: random-number assignment and consumption can differ. A dropout test needs a shared mask or a documented compatible RNG mapping, plus statistical/gradient checks appropriate to the implementation.

PyTorch SDPA applies the supplied `dropout_p` even inside `inference_mode`; set it to zero for evaluation. The actual CPU test demonstrates this. [SDPA contract](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention).

A deterministic backward option controls a particular implementation's reduction behavior. It does not guarantee identical bits across compiler versions, devices or different algorithms. Test reproducibility separately from closeness to an oracle.

## 5. Exercise and answers

**Exercise:** use a `[9,2]` lower-right mask. Predict the first seven output/Q-gradient rows before executing: they are zero under the declared convention. Then change alignment to upper-left and explain why the results differ.

**Completion:** retain the 14 optional-test results and identify which tests check math, which check API semantics, and which hardware properties remain untested. GPU synchronization, memory races, FA2/3/4 support and performance need their own selected-hardware evidence.
