# 18 — Kalman smoothing: use later information to revise an earlier estimate

**Before:** linear Kalman filter. **After:** distinguish filtering, delayed causal replay, and offline smoothing.

A filter at time `t` uses observations available by `t`. A fixed-interval smoother estimates past states after seeing observations from the whole recorded interval. This is useful for trajectory analysis and dataset labels, but its improved past estimate was not available to an online controller at that past time.

The Rauch–Tung–Striebel (RTS) smoother runs backward after the forward Kalman pass. Store filtered means/covariances and each next-step predicted mean/covariance:

```text
C_t = P_filtered_t F_tᵀ (P_predicted_(t+1))⁻¹
x_smoothed_t = x_filtered_t + C_t (x_smoothed_(t+1) − x_predicted_(t+1))
P_smoothed_t = P_filtered_t + C_t (P_smoothed_(t+1) − P_predicted_(t+1)) C_tᵀ
```

Initialize the final smoothed state with the final filtered state. Use a solve for `C_t`, and keep the exact transitions/noise used in the forward pass.

**Worked scalar sequence:** at time 0 the filtered estimate is 0 with variance 1. The next prediction is 0 with variance 2 under `F=1`. At time 1 the corrected estimate becomes 1 with variance 1. Then `C0=1/2`, so the smoothed time-0 mean is 0.5 and variance is `1 + .25*(1−2)=.75`. The last state remains 1 with variance 1. The executable `rts_scalar` test checks these numbers.

The [camera lab](../Camera-Time-Fusion-Lab/Guide.md) uses **causal replay**: at a current output time, it processes only observations that have already arrived, in capture-time order, and predicts back to now. That is not an offline smoother using future arrivals. A fixed-lag smoother makes an additional latency/memory tradeoff by finalizing old states after a chosen lag.

**Try:** compare an online controller with an offline smoothed trajectory. Which error should assess what the controller knew? **Answer:** the original online estimate against truth at each original output time. Report smoother accuracy separately; otherwise future information leaks into the evaluation.

**Pass:** reproduce the scalar calculation; run the test; preserve both original output timestamps and revision timestamps when saving smoothed labels.

**Navigation:** [Learning path](00-index.md) · [Executable CPU lab](../Camera-Time-Fusion-Lab/Guide.md).
