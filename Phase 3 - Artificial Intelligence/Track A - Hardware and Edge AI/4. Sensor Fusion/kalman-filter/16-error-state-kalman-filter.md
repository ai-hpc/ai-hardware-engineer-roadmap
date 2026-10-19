# 16 — Error-state filtering: small corrections around a nominal pose

**Before:** EKF, coordinate frames, rotations. **After:** explain why a 3D orientation error has three components even when its nominal quaternion has four.

A robot's nominal state may contain position, velocity, orientation quaternion, and IMU biases. The filter estimates a small local error around it. For a declared right-multiplicative convention:

```text
R_true = R_nominal Exp([δθ]×)
p_true = p_nominal + δp
v_true = v_nominal + δv
```

Here `[δθ]×` is the skew-symmetric cross-product matrix. `δθ` lives in the local tangent space. With quaternion convention stated, the nominal rotation is updated by quaternion multiplication or the equivalent rotation exponential, not by adding three Euler angles to four quaternion components.

The cycle is: propagate the nominal state with bias-corrected IMU data; propagate error covariance with the corresponding Jacobians; form a measurement residual; estimate the error; inject it into the nominal state; reset the error mean to zero and transform its covariance with the reset Jacobian. Forgetting the reset transformation changes what coordinates the covariance describes. Its exact sign depends on left/right perturbation and quaternion conventions.

**Small worked analogue on SO(2):** nominal yaw 179°, observation −179°, equal variances 0.04 rad². The wrapped innovation is +2°, gain 0.5, and corrected yaw is ±180°, with variance 0.02. Ordinary subtraction would point toward 0°, a completely different direction. The executable `yaw_error_update` verifies this boundary case. SO(2) is a teaching reduction; it does not implement the full 3D ESKF reset Jacobian.

An accelerometer measures specific force, not directly world-frame acceleration: under a common convention, `a_world = R_world_body (a_measured−bias) + g_world`. A stationary accelerometer's reading depends on orientation and gravity convention. Gyroscope biases and timestamps must be handled before integration.

**Try:** explain why normalizing a quaternion alone does not repair a covariance expressed in the wrong tangent frame. **Answer:** normalization enforces unit length of the nominal quaternion; the covariance still requires its coordinate transformation.

**Pass:** document handedness, quaternion component order, perturbation side, and gravity direction; run the yaw test; derive the reset map for your chosen 3D convention before implementing a full inertial estimator. Primary derivations: [Solà, Quaternion kinematics for the error-state Kalman filter](https://arxiv.org/abs/1711.02508).

**Navigation:** [Learning path](00-index.md) · [Executable CPU lab](../Camera-Time-Fusion-Lab/Guide.md).
