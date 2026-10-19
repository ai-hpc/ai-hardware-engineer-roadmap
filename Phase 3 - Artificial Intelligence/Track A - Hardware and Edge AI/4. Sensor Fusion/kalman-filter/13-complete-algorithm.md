# 13 — A complete Kalman update, with numbers

**Before:** [noise models](12-process-measurement-noise.md). **After:** you can explain each intermediate value and distinguish a prediction from a corrected estimate.

For linear dynamics and measurements with known, independent, zero-mean Gaussian process/measurement noise, the filter computes the exact Gaussian posterior. More generally, the linear estimator has a minimum mean-square property under its second-moment assumptions; that does not make a wrong sensor model optimal for a real robot.

```text
Prediction:       x⁻ = F x + B u       P⁻ = F P Fᵀ + Q
Innovation:       r = z − H x⁻        S = H P⁻ Hᵀ + R
Gain:             K = P⁻ Hᵀ S⁻¹
State update:     x⁺ = x⁻ + K r
Covariance:       A = I − K H         P⁺ = A P⁻ Aᵀ + K R Kᵀ
```

The last line is the Joseph form. Compute the gain using a linear solve, not a materialized inverse. Superscripts `−` and `+` mean before and after the current measurement, not negative and positive values.

**Worked scalar case:** prior position 4 m with variance 1 m²; a same-time sensor reports 6 m with variance 3 m². With `H=1`, the innovation is 2 m, `S=4 m²`, `K=1/4`, corrected position is 4.5 m, and variance is 0.75 m². The normalized innovation squared is `2²/4=1`. The estimate moves a quarter of the way toward the noisier sensor.

**Now add time:** if velocity is 2 m/s and the observation was captured 0.5 seconds earlier, first compare it with the state at that capture time. Otherwise a 1 m motion displacement masquerades as sensor error. The [replay lab](../Camera-Time-Fusion-Lab/Guide.md) implements this case with a full event history.

A complete loop also validates units, timestamps, finite values, frame agreement, and measurement dimensions; handles dropouts; and records accepted/rejected innovations. Missing data means predict without correction. It does not mean insert a zero-valued measurement.

**Try:** replace `R=3` with `R=1`. **Answer:** `K=0.5`, corrected position 5, corrected variance 0.5. Then remove the measurement entirely: the predicted mean and covariance remain unchanged by a correction step.

**Pass:** reproduce all intermediate numbers by hand, then identify the corresponding lines in `predict`, `update`, and `estimate` in the lab. Explain why velocity/position cross-covariance lets a position measurement adjust velocity.

**Navigation:** [Learning path](00-index.md) · [Executable CPU lab](../Camera-Time-Fusion-Lab/Guide.md).
