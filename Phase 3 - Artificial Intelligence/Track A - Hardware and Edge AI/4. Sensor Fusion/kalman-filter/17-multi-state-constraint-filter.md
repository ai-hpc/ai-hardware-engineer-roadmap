# 17 — MSCKF: use a feature track without keeping every landmark

**Before:** EKF, camera projection, matrix rank. **After:** eliminate a nuisance feature from a linearized residual and explain what information remains.

An MSCKF keeps the current inertial state and a sliding set of camera pose clones, with their cross-covariances. A feature observed from several clones constrains their relative poses. The feature position is estimated for linearization, then removed from the update rather than becoming a permanent filter state.

Stack a track's residuals:

```text
r ≈ H_state δx + H_feature δf + noise
```

If measurement covariance is `R=L Lᵀ`, whiten all three quantities using `L⁻¹`. Find an orthonormal basis `N` for the left nullspace of the whitened feature Jacobian: `N H_feature_whitened=0`. Premultiplication gives:

```text
N r_whitened ≈ N H_state_whitened δx + N noise_whitened
```

The nuisance feature error disappears. Projected noise covariance is identity because `N Nᵀ=I`. Skipping whitening and then assuming identity noise is wrong when observations have unequal variances.

**Worked toy track:** three scalar residuals share one nuisance offset, so `H_feature=[1,1,1]ᵀ`. It has rank 1 and leaves two independent residual combinations. With equal variance, differences such as `r1−r2` remove the shared offset, but raw differences are correlated; an orthonormal nullspace basis handles that covariance cleanly. `nullspace_project` uses Cholesky whitening and SVD. Its test adds a nuisance offset of 7 to all residuals and verifies unchanged projected residuals, with unequal variances `[1,4,9]`.

This algebra is only one component of MSCKF. A working VIO system also needs triangulation, parallax and cheirality checks, camera/IMU extrinsics and time alignment, clone augmentation, feature tracking, marginalization, and careful treatment of unobservable global position/yaw. More features cannot compensate for zero parallax or a wrong clock.

**Try:** suppose a feature contributes six scalar pixel residuals and its Jacobian has rank 3. How many projected constraints remain? **Answer:** three. If the rank is lower, the algebraic dimension changes, but the geometry may also be degenerate; do not interpret more nullspace rows as automatically more useful information.

**Pass:** run the nuisance-invariance test; explain each dimension; identify the coordinate frame of every Jacobian. For full estimator implementation study [OpenVINS nullspace derivation](https://docs.openvins.com/update-null.html) and its linked state/update documentation. The CPU kernel is not a complete VIO implementation.

**Navigation:** [Learning path](00-index.md) · [Executable CPU lab](../Camera-Time-Fusion-Lab/Guide.md).
