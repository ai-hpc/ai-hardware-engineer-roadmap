# 19 — Outlier rejection: test a residual against its uncertainty

**Before:** innovation and covariance. **After:** calculate a normalized innovation and recognize when a gate rejects correct data.

Raw pixel or metre thresholds ignore predicted uncertainty. For innovation `r` and its covariance `S`, normalized innovation squared is `NIS=rᵀS⁻¹r`. Under a correctly specified linear Gaussian model, it follows a chi-square distribution with measurement dimension degrees of freedom. The gate must use the innovation covariance, not just measurement noise `R`.

**Worked scalar gate:** `r=0.3 m`, `S=0.01 m²`, so NIS=9. A scalar gate of 9 corresponds to a three-standard-deviation residual in the ideal Gaussian case. The same 0.3 m error with `S=0.09` has NIS=1. For two dimensions, the same probability requires a different chi-square threshold; “three sigma” is not a universal multidimensional radius.

The CPU lab gates camera observations before updating, records rejection counts, and deliberately inserts two 1.5 m association faults after pose recovery. Run the default and `--no-gate` versions on the same seed. Inspect the transient errors near capture times 3.2 and 8.6 seconds; their arrival times include the configured delay.

A gate can fail in both directions. Too small `R` or ignored bias makes valid observations look impossible. Excessive covariance makes a gross outlier look plausible. Wrong timestamps create motion residuals and can cause every camera observation to be rejected. Gating is not a repair for calibration or synchronization.

**Try:** run `--wheel-bias 0.06`. Explain an increase in camera rejections even though the camera did not get noisier. **Answer:** the velocity sensor's persistent bias is not represented in the two-state filter, so the predicted position drifts while its declared covariance remains small. Remedies require an appropriate bias model/observable measurements or revised uncertainty assumptions, evaluated on held-out sequences—not simply widening the gate until all data pass.

For association, gate impossible candidate pairs **before** solving the assignment. Support explicit unmatched choices; solving an unrestricted assignment and deleting bad pairs afterward can lose feasible matches.

**Pass:** calculate NIS by hand, compare gated/ungated traces, and distinguish an injected outlier from systematic model mismatch. Accepted-only residuals are a truncated sample; do not claim they prove full-model Gaussian consistency.

**Navigation:** [Learning path](00-index.md) · [Executable CPU lab](../Camera-Time-Fusion-Lab/Guide.md).
