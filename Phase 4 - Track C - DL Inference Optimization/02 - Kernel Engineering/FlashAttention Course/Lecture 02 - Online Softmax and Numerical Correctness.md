# Lecture 2 — Online softmax, empty rows and numerical correctness

**Course:** [FlashAttention](Guide.md) · **Reviewed:** 18 October 2026

**Goal:** derive the running normalization and implement it without NaNs on masked rows. **Artifact:** a derivation plus the executed CPU report in [the practical lab](CPU-Attention-Lab.md).

## 1. Keep the scale consistent

For one query row, let its allowed scores be `s_j` and corresponding values be vectors `v_j`. After any prefix, maintain:

```text
m = maximum score seen
ell = Σ exp(s_j − m)
a = Σ exp(s_j − m) × v_j
output = a / ell
```

The numerator `a` is unnormalized. For the next score block, choose `m_new = max(m, max(block))`, and let `alpha = exp(m − m_new)`. Then:

```text
weights = exp(block − m_new)
ell_new = alpha × ell + sum(weights)
a_new = alpha × a + weights @ V_block
```

Both old numerator and denominator must use the new scale. Divide by the final denominator once after all blocks. The algebra is exact over real arithmetic; floating-point order and conversion introduce rounding differences.

A simple counterexample makes the rescaling requirement visible: scores `[0, log(3)]` and values `[2,10]` have weights `[1/4,3/4]` and output **8**. If each block is normalized independently and their outputs are averaged, the result is **6**. The independent blocks lost their relative weights.

## 2. Define what a fully masked row means

A mask removes invalid key positions. Our lab uses `True = allowed`, zero output for a row with no allowed keys, and `LSE = -infinity` for that row. This is an explicit API convention; an unconstrained expression `softmax([-inf,-inf])` does not supply it automatically.

Initialize `m=-inf`, `ell=0`, `a=0`. If both the old prefix and current block are empty, blindly evaluating `exp(-inf - -inf)` yields NaN. The implementation instead tracks whether the row has any valid contributions, uses a safe temporary maximum, and leaves the empty contribution at zero. The final divide uses a harmless denominator only for empty rows, whose numerator is already zero.

Read [attention_reference.py](labs/attention_reference.py), then run the tail/mask and large-max-change tests. It tiles **both** query rows and key columns. It retains the full test mask as a convenience; that dense fixture mask is not a claim of linear total memory. A deployed causal kernel can derive its mask from indices.

## 3. Separate two kinds of numerical error

Compare three outputs:

1. Float64 math on the original input values.
2. Float64 math on inputs already rounded to the target input dtype.
3. The candidate computation on those same rounded inputs.

The first-to-second difference measures input quantization for this fixture. The second-to-third difference measures computation/output rounding for the candidate. Comparing BF16 with float64 is useful when you state which question the comparison answers.

NumPy does not provide a portable built-in BF16 dtype in this lab. Use actual PyTorch BF16 for that stage. Do not cast inputs to FP32 while leaving accumulators silently in float64 and label the whole experiment FP32.

## 4. Execute, vary, explain

The lab checks float64 with `atol=rtol=1e-12` on its small fixtures. Those are fixture thresholds, not universal bounds for every length, score magnitude or device. Floating-point tolerance is `abs(test-ref) ≤ atol + rtol×abs(ref)`; maximum absolute error alone does not evaluate that complete rule.

Try column tiles 1, 4 and larger than the sequence. Then create a mask whose first two blocks are empty but its final block is valid. It should recover a finite output. Finally mask the entire row and explain why output zero and LSE `-inf` are compatible.

**Answer check:** delaying the final division is valid with the unnormalized numerator above. Dropping every intermediate max-rescale factor is not. Lecture 06 builds directly on this distinction.
