# 15 — Unscented Kalman filter: propagate a set of representative points

**Before:** lesson 14 and covariance. **After:** compare an EKF approximation, an unscented transform, and exact moments on one nonlinear example.

The unscented transform chooses deterministic points around a Gaussian mean, propagates each through the nonlinear function, then recombines them. These are quadrature points, not randomly sampled particles or confidence-bound corners.

For dimension `n`, choose `λ=α²(n+κ)−n`. A matrix square root of `(n+λ)P` gives `2n` offsets around the mean. The centre has mean weight `λ/(n+λ)` and covariance weight `λ/(n+λ)+1−α²+β`; each other point has weight `1/[2(n+λ)]`. Require `n+λ>0` and a valid covariance square root. Some parameter choices yield negative weights; floating-point stability deserves testing.

**Worked nonlinear transformation:** `X~Normal(2,1)` and `Y=X²`. Exact moments are `E[Y]=5` and `Var[Y]=18`. A first-order EKF transform gives mean 4 and variance 16. For `n=1, α=1, β=2, κ=0`, the sigma points are `2,3,1`. Their squared values are `4,9,1`; mean weights are `0,0.5,0.5`; covariance weights are `2,0.5,0.5`. Thus mean is 5 and variance is `2*(4−5)² + .5*(9−5)² + .5*(1−5)² = 18`. Run `unscented_square` and its analytic-moment tests.

A complete additive-noise UKF uses this idea for motion prediction and measurement prediction, adds `Q` and `R` at their corresponding steps, computes state/measurement cross-covariance, then updates with the innovation. Sigma points used for a later step must represent the covariance at that step, including relevant added noise. Augmenting the state with noise is another approach for nonadditive noise.

This exact result for a quadratic example is not a guarantee for arbitrary nonlinear dynamics. A UKF still approximates a potentially multimodal posterior by one mean and covariance; it can fail with ambiguous camera geometry or a badly initialized state.

**Try:** set mean 0, variance 2. **Answer:** `E[X²]=2`, variance 8; the EKF linearization at zero gives zero output variance and misses the uncertainty entirely.

**Pass:** run the three analytic examples in the tests, explain the distinct mean/covariance weights, and give a situation where neither EKF nor UKF represents the posterior well.

**Navigation:** [Learning path](00-index.md) · [Executable CPU lab](../Camera-Time-Fusion-Lab/Guide.md).
