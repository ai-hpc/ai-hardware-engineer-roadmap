# Kalman Filter Learning Series

Part of the [AI Hardware Engineer Roadmap](../../../../README.md).

## Learning Path

| Level | Chapters |
|-------|----------|
| **Start** | [00 - Index](00-index.md) |
| **Elementary** | [01](01-what-is-estimation.md) · [02](02-noisy-measurements.md) · [03](03-combining-information.md) |
| **Middle School** | [04](04-averages-and-uncertainty.md) · [05](05-weighted-averages.md) · [06](06-prediction.md) |
| **High School** | [07](07-kalman-filter-idea.md) · [08](08-1d-kalman-filter.md) · [09](09-tracking-moving-object.md) |
| **Undergraduate** | [10](10-matrix-form.md) · [11](11-multidimensional-kf.md) |

## Reference

- [Cheat Sheet](cheat-sheet.md)
- [Sensor Fusion Guide](../Guide.md)

## Example Outputs

![1D](kalman_1d_tracking.png)
![2D](2d_tracking.png)
![6D](kalman_6d_tracking.png)

## Run

```bash
pip install numpy matplotlib
python kalman_1d.py   # or kalman_2d.py, kalman_6d.py
```

## External

- [Kalman Filter (Wikipedia)](https://en.wikipedia.org/wiki/Kalman_filter)
- [Roger Labbe's Book](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)

## Practice and assessment

**Entry diagnostic (prerequisite check):** Write prediction and correction for a one-dimensional state. If this is unfamiliar, use the prerequisite material above or the [roadmap diagnostics](../../../../Learning-Guide.md#diagnose-before-reading) before starting the lab.

1. **Baseline lab:** Estimate a noisy synthetic trajectory and compare with unfiltered measurements.
2. **Guided variation:** Change process/measurement noise and introduce a dropout; inspect covariance and innovation.
3. **Independent transfer:** repeat the method on a new input or configuration and explain where your original assumptions stop applying.

**Evidence to submit:** State error, covariance consistency, residuals, and seed/model assumptions. Keep raw results and reproduction steps; distinguish measured results from estimates or work not run.

**Self-check:** Which observation would disprove your explanation of the result? Identify a competing explanation and the smallest experiment that separates them. A passing answer names a concrete predicted difference, not only another tool to try.

**Exit criteria:** reproduce the baseline, demonstrate the negative/variation check, and meet the [shared completion rubric](../../../../Learning-Guide.md#completion-rubric). Use the [lesson cycle](../../../../Learning-Guide.md#a-repeatable-lesson-cycle) for retrieval practice and spaced review.
