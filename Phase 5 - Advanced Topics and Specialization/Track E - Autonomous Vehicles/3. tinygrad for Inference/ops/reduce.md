# Reductions: axes, normalization and information flow

A reduction combines entries along stated axes. Its output shape, accumulator dtype, identity and numerical order are part of the operation's meaning. Begin with [the executed operations lesson](../projects/README.md).

## 1. Choose the axis from the formula

For `A=[[1,2,3],[-1,0,2]]`, summing axis 0 gives `[0,2,5]`; summing axis 1 gives `[6,1]`. A mean additionally divides by the number of reduced entries, giving row means `[2,1/3]`.

Matrix multiplication `C[i,j]=sum_k A[i,k]B[k,j]` reduces the shared inner axis. The manual `[2,3,1] × [1,3,2] → sum(axis=1)` example produces `[2,2]`. Reducing the last axis of that product instead would produce a different shape and meaning.

Changing reduction order can change floating-point rounding. Large/canceling values, low-precision accumulation and parallel trees need tolerances selected for the application. Agreement on a few small integers does not validate every numeric regime.

## 2. RMSNorm and LayerNorm are different

For one row with n values:

```text
RMSNorm(x)[i] = x[i] / sqrt(sum_j x[j]²/n + epsilon)
LayerNorm(x)[i] = (x[i]-mean(x)) / sqrt(sum_j (x[j]-mean(x))²/n + epsilon)
```

Learned scale/bias, when present, are additional named parameters. For a constant row `[3,3]`, RMSNorm remains positive while centered LayerNorm is zero before its affine parameters. With epsilon 7, RMSNorm returns `[0.75,0.75]` exactly.

The custom-operations lesson verifies RMSNorm values, and the autograd lesson checks the gradient of its summed output against finite differences. State whether the normalized axis is channels, tokens or another dimension; changing it changes the model.

If a padded implementation loads invalid lanes as zero, those lanes must also be masked after transformations such as centering. `(0-mean)²` is not zero. The [CUDA fusion lesson](../../../Track%20A%20-%20GPU%20Infrastructure/Nvidia%20GPU/HPC%20Setup/CUDA-Advanced-Optimization/04-Kernel-Fusion.md#4-mask-reductions-after-the-transformation) has an executed counterexample.

## 3. Stabilize softmax and apply masks at the right point

Softmax uses `exp(score-max(score))/sum(exp(score-max(score)))`. Subtracting the maximum preserves the real-valued probabilities while avoiding unnecessary exponential overflow. `[1000,1000]` should yield `[0.5,0.5]`.

Scaled dot-product attention computes `softmax(QK^T/sqrt(d))V`. In a causal sequence, row i may read only keys/values at positions `j<=i`. Apply the future-position mask to scores before softmax, then compute normalization over the allowed positions.

The course fixture has first value `[2,3]` and a large future value `[100,200]`. The first causal output must be exactly `[2,3]`. If it changes when only that future value changes, information leaked through the mask. Masking or slicing after an unmasked softmax cannot repair the earlier normalization.

Every row in this exercise has at least its own valid position. A completely masked row needs a separately defined outcome; ordinary all-negative-infinity softmax can produce undefined/nonfinite results. This fixture does not claim to implement a general padding/empty-row attention kernel. [Attention definition](https://arxiv.org/abs/1706.03762).

## 4. Reduction is not scan

For `[1,2,3,4]`, sum gives `10`, inclusive scan gives `[1,3,6,10]`, and exclusive scan gives `[0,1,3,6]`. A reduction tree alone does not supply every prefix. State the requested output before optimizing synchronization.

**Transfer:** change Q/K/V to sequence length 5 and feature width 3. Compare with the scalar reference, change only a future value and verify earlier rows stay unchanged, then inspect the executed kernels. Preserve correctness before any latency comparison.
