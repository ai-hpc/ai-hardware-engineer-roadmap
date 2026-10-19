# 08 — Track motion along one line

**Prerequisites:** arithmetic, variance, and [prediction/correction](07-kalman-filter-idea.md). **Outcome:** update position and velocity correctly from a noisy position measurement.

“One dimensional motion” describes one spatial axis. If both position and velocity are unknown, the filter has **two state components**. Their uncertainties become correlated: a velocity error causes position error after time passes. Keeping only two separate variances loses information needed for a valid velocity correction.

## State, covariance, and units

```text
state = [p, v]                 p in m, v in m/s
P = [[Ppp, Ppv], [Ppv, Pvv]]   units: m², m²/s, (m/s)²
```

A position residual has units metres. Multiplying it by the position gain (dimensionless) updates position. The velocity gain has units 1/second; using the same scalar gain for both, as a former version of this lesson did, is dimensionally wrong.

## Prediction using scalar arithmetic

For interval `dt` and independent fixed per-step noise variances `Qp,Qv`:

```text
p_minus = p + dt*v
v_minus = v
Ppp_minus = Ppp + 2*dt*Ppv + dt²*Pvv + Qp
Ppv_minus = Ppv + dt*Pvv
Pvv_minus = Pvv + Qv
```

Calculate all new covariance entries from the **old** covariance before assigning them. This diagonal per-step noise model is a teaching choice. For a physical acceleration-noise model with variable sampling, use [lesson 12](12-process-measurement-noise.md) and its `Q(dt)`.

## Correction from a position measurement

```text
residual = z − p_minus
S = Ppp_minus + R
Kp = Ppp_minus / S
Kv = Ppv_minus / S
p_plus = p_minus + Kp*residual
v_plus = v_minus + Kv*residual
```

The equivalent covariance entries are `Ppp_plus=Ppp_minus−Kp*Ppp_minus`, `Ppv_plus=Ppv_minus−Kp*Ppv_minus`, and `Pvv_plus=Pvv_minus−Kv*Ppv_minus`. Use the unchanged prior values in all three expressions. The executable implementation uses the Joseph form to reduce numerical cancellation.

**Worked example:** start at position 0 m and velocity 1 m/s, covariance `[[1,0],[0,1]]`, `dt=1`, and no process noise. Prediction gives position 1 and covariance `[[2,1],[1,1]]`. A sensor reports position 3 with variance `R=2`. Then residual=2, `S=4`, `Kp=0.5`, `Kv=0.25`; corrected state is position 2 m, velocity 1.5 m/s. Corrected covariance is `[[1,0.5],[0.5,0.75]]`. The position measurement changes velocity because the prediction created cross-covariance.

## Run the complete example

The maintained implementation is [kalman_1d.py](kalman_1d.py). It includes a seeded trajectory, noisy position measurements, the two-state filter, error statistics, and plots. Use an isolated environment with NumPy and Matplotlib:

```bash
# From this lesson's directory, after activating that environment:
MPLBACKEND=Agg python kalman_1d.py
```

`Agg` saves plots without requiring a display. The script writes its plot beside itself; copy it to a scratch directory if you want to preserve the checked-in illustration. The 1D, 2D, and 6D scripts were executed on 2026-10-18 with Python 3.12.3, NumPy 1.26.4, and Matplotlib 3.6.3; the supplied plots are illustrations, not a physical-robot benchmark.

The [camera/time lab](../Camera-Time-Fusion-Lab/Guide.md) provides a fully pinned NumPy/OpenCV environment and an optional test that extracts this class and checks its entire covariance against an independent matrix implementation. The test also catches the previous in-place cross-covariance bug, which produced an incorrect velocity variance despite plausible position plots.

## Practice and answers

1. Set `Ppv_minus=0` before a position update. Does velocity change? **No:** `Kv=0`; a single position observation has no velocity information without state correlation/model history.
2. Double measurement **standard deviation**. What happens to `R`? **It quadruples**; `R` is variance, not standard deviation.
3. Skip a measurement. Should you insert position zero? **No:** predict without correction and preserve the missing-data event.
4. Use the worked example but set the measurement to the predicted position. **Mean unchanged; covariance still shrinks.**

**Pass:** reproduce the hand calculation, run the script, and explain why position and velocity gains differ. Inspect both error and covariance; a visually smooth trajectory is insufficient. Continue to [matrix tracking](09-tracking-moving-object.md), then [noise models](12-process-measurement-noise.md).
