# 20 — Numerical implementation: preserve the meaning of covariance

**Before:** matrix filtering. **After:** write a stable update and test properties that a low trajectory error alone cannot reveal.

Covariance should remain symmetric and positive semidefinite. A small roundoff asymmetry can be symmetrized, but clipping large negative eigenvalues or repeatedly adding arbitrary diagonal jitter hides a deeper problem. Check shapes, units, finite inputs, and matrix conditioning first.

For a matrix observation, compute the gain with a solve:

```python
# P: n×n, H: m×n, R: m×m; all from the same linearization/time.
S = H @ P @ H.T + R
K = np.linalg.solve(S, H @ P).T  # assumes symmetric P and S
A = np.eye(P.shape[0]) - K @ H
P_next = A @ P @ A.T + K @ R @ K.T
P_next = 0.5 * (P_next + P_next.T)
```

The Joseph expression reduces cancellation compared with `(I−KH)P`. It does not guarantee exact positive definiteness in arbitrary floating-point computations; exact measurements or unobservable components may produce semidefinite covariance. If `S` is singular, investigate redundant/noiseless measurements rather than silently replacing it with an inverse of a perturbed matrix. Square-root filters are a useful advanced approach for demanding conditioning.

**A numerical example worth testing:** set `P=1e12`, `R=1e−6`, `H=1`. In double precision the gain may round to 1. The subtraction form gives zero covariance, while Joseph form retains approximately `R=1e−6`. Both may produce the same updated mean; only checking the mean misses the error.

The lab uses scalar innovations and Joseph updates, avoiding a matrix inverse. Tests check covariance eigenvalues, time-step composition, and zero-delay equality of arrival-time and capture-time processing. The complete kernels are linked from [implementation templates](implementation-templates.md).

**Try:** permute a state vector from `[p,v]` to `[v,p]`. What must change? **Answer:** every corresponding row/column of `F,P,Q,H` and all state accesses. An unchanged covariance diagonal with a relabelled state describes a different physical model.

**Pass:** add one independent analytic update test and one invalid-input test; inspect covariance as well as error. Record precision and library versions, and distinguish a tolerance-based numerical test from bitwise reproducibility across machines.

**Navigation:** [Learning path](00-index.md) · [Executable CPU lab](../Camera-Time-Fusion-Lab/Guide.md).
