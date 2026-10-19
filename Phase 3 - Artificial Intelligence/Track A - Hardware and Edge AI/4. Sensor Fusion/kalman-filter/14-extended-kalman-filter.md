# 14 — Extended Kalman filter: linearize the right function

**Before:** Jacobians and lesson 13. **After:** implement one nonlinear observation update and identify where it becomes unreliable.

For `x_next=f(x,u)+w` and `z=h(x)+v`, propagate the mean through the nonlinear functions. Linearize only the covariance calculation:

```text
x⁻ = f(x,u)           F = ∂f/∂x evaluated at the previous estimate
P⁻ = F P Fᵀ + Q
r = z − h(x⁻)         H = ∂h/∂x evaluated at x⁻
S = H P⁻ Hᵀ + R      K = P⁻ Hᵀ S⁻¹
x⁺ = x⁻ + K r        covariance: Joseph form
```

This is the additive-noise EKF. Nonadditive noise needs its own noise Jacobians. Evaluating `H*x` in place of `h(x)` is generally wrong.

**Worked range observation:** a robot at estimated `(3,4)` m measures its distance to a landmark at the origin. `h(x,y)=sqrt(x²+y²)=5`, `H=[0.6,0.8]`. Let prior covariance be the identity, observed range 5.4 m, and range variance 0.25. Then `r=0.4`, `S=1.25`, `K=[0.48,0.64]ᵀ`, and the corrected position is `(3.192,4.256)`. A single range constrains the radial direction; the tangential variance remains 1. The executable `ekf_range_update` and its test verify both facts.

At `(0,0)` the range derivative is undefined. The supplied function rejects that case. Adding an arbitrary epsilon may hide the failure without supplying the missing direction information. Large uncertainty, poor initialization, or bearing wrap can invalidate the local approximation. Wrap angle innovations to the chosen interval; never subtract 179° and −179° as a physical 358° turn.

**Try:** predict the update when the range equals 5 exactly. **Answer:** the mean stays `(3,4)`, but radial covariance shrinks because the observation still supplies information. An unchanged mean is not an unused observation.

**Pass:** check the Jacobian against finite differences away from the origin, run the executable range test, and show one failure caused by a near-zero range or an inconsistent angle convention. Continue to [UKF](15-unscented-kalman-filter.md).

**Navigation:** [Learning path](00-index.md) · [Executable CPU lab](../Camera-Time-Fusion-Lab/Guide.md).
