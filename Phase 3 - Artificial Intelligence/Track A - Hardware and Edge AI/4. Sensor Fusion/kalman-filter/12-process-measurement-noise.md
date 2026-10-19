# 12 — Process noise and measurement noise

**Before:** [matrix filtering](10-matrix-form.md). **After:** you can derive a covariance from a stated stochastic model and change the sample interval without guessing.

A sensor error belongs in `R`; uncertainty about motion between measurements belongs in `Q`. Neither means “how much smoothing I like.” A covariance entry is a variance: position has units m², velocity has units (m/s)², and their cross-covariance has units m²/s. A standard deviation of 0.2 m means `R=0.04 m²`.

For state `x=[position, velocity]`, prediction over `dt` uses:

```text
F = [[1, dt], [0, 1]]
x_new = F x + process_error
P_new = F P Fᵀ + Q
```

There are two common acceleration assumptions; their `Q` matrices differ.

| Assumption | Parameter and units | Q |
|---|---|---|
| One random acceleration held constant during a step | variance σa², m²/s⁴ | σa² [[dt⁴/4,dt³/2],[dt³/2,dt²]] |
| Continuous white acceleration integrated over a step | spectral density q, m²/s³ | q [[dt³/3,dt²/2],[dt²/2,dt]] |

For the first, displacement error is `0.5*a*dt²` and velocity error is `a*dt`; write their outer product to obtain `Q`. For the second, integrate the continuously injected velocity noise. With `q=0.06` and `dt=0.5`, the entries are `[[0.0025,0.0075],[0.0075,0.03]]`. The [CPU lab](../Camera-Time-Fusion-Lab/Guide.md) uses the second model. Predicting for 0.2 then 0.3 seconds gives the same covariance as predicting once for 0.5 seconds. Its test checks that property.

**Correlated errors matter.** Two sensors each with variance 1 and correlation ρ have average variance `(1+ρ)/2`. At ρ=1, averaging does not reduce variance. Reusing one camera result as if it were a new independent sensor makes the filter overconfident. An unmodelled wheel bias is also not fresh independent noise each frame.

**Try:** double `dt` in both formulas. Which entries multiply by 2, 4, 8, and 16? **Answer:** continuous `Qpp×8,Qpv×4,Qvv×2`; step-constant `Qpp×16,Qpv×8,Qvv×4`. Thus the same numeric parameter cannot be silently exchanged between models.

**Pass:** identify units for every state and noise parameter; derive one matrix; run `test_continuous_noise_composes_over_substeps`. See [common mistakes](common-mistakes.md) for bias and correlation.

**Navigation:** [Learning path](00-index.md) · [Executable CPU lab](../Camera-Time-Fusion-Lab/Guide.md).
