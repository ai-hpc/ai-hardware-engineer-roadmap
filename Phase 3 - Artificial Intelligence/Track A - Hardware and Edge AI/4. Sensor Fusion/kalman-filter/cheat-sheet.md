# Kalman Filter Equations Cheat Sheet

## The Five Core Equations

### Prediction (Time Update)

**1. State Prediction**
```
x̂ₖ₊₁|ₖ = F × x̂ₖ|ₖ
```

**2. Covariance Prediction**
```
Pₖ₊₁|ₖ = F × Pₖ|ₖ × Fᵀ + Q
```

### Update (Measurement Update)

**3. Kalman Gain**
```
Kₖ₊₁ = Pₖ₊₁|ₖ × Hᵀ × (H × Pₖ₊₁|ₖ × Hᵀ + R)⁻¹
```

**4. State Update**
```
x̂ₖ₊₁|ₖ₊₁ = x̂ₖ₊₁|ₖ + Kₖ₊₁ × (zₖ₊₁ - H × x̂ₖ₊₁|ₖ)
```

**5. Covariance Update**
```
Pₖ₊₁|ₖ₊₁ = (I - Kₖ₊₁ × H) × Pₖ₊₁|ₖ
```

## Variables

| Symbol | Name | Dimensions | Description |
|--------|------|------------|-------------|
| x̂ | State estimate | n×1 | Best guess of system state |
| P | Covariance | n×n | Uncertainty in state estimate |
| F | State transition | n×n | How state evolves |
| H | Measurement matrix | m×n | Maps state to measurements |
| Q | Process noise | n×n | Model uncertainty |
| R | Measurement noise | m×m | Sensor uncertainty |
| K | Kalman gain | n×m | Optimal weighting |
| z | Measurement | m×1 | Sensor reading |
| I | Identity | n×n | Identity matrix |

## Python Template

Run from this course directory to reuse the shape-checked implementation in [kalman_6d.py](kalman_6d.py), without importing plotting dependencies:

```python
import numpy as np
from kalman_6d import KalmanFilter

kf = KalmanFilter(F=[[1,1],[0,1]], H=[[1,0]], Q=np.zeros((2,2)),
                  R=[[2]], x0=[[0],[1]], P0=np.eye(2))
kf.predict()
kf.update([3])  # explicit flat-to-column conversion
print(kf.x[:, 0])  # [2.0, 1.5]
print(kf.P)        # [[1.0, .5], [.5, .75]]

```

## Common Models

### Constant Velocity (1D)
```python
dt = 0.1
F = np.array([[1, dt],
              [0, 1 ]])
H = np.array([[1, 0]])
Q = q * np.array([[dt**4/4, dt**3/2],
                  [dt**3/2, dt**2  ]])
R = np.array([[sigma_position**2]])  # sigma_position is in metres
```

### Constant Velocity (2D)
```python
F = np.array([[1, 0, dt, 0 ],
              [0, 1, 0,  dt],
              [0, 0, 1,  0 ],
              [0, 0, 0,  1 ]])
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])
```

### Constant Acceleration (1D)
```python
F = np.array([[1, dt, dt**2/2],
              [0, 1,  dt     ],
              [0, 0,  1      ]])
H = np.array([[1, 0, 0]])
```

## Quick Reference

### When to Use What

| Scenario | Model | State |
|----------|-------|-------|
| Stationary object | Constant position | [x] |
| Moving at steady speed | Constant velocity | [x, v] |
| Accelerating object | Constant acceleration | [x, v, a] |
| 2D tracking | 2D constant velocity | [x, y, vx, vy] |
| 3D tracking | 3D constant velocity | [x, y, z, vx, vy, vz] |

### Tuning Guidelines

**Process Noise (Q)**:
- Too small → Filter ignores measurements, slow to adapt
- Too large → Filter jumps around, noisy estimates
- Derive `Q(dt)` from a named motion-noise model and units; see [lesson 12](12-process-measurement-noise.md).

**Measurement Noise (R)**:
- Should match actual sensor noise
- Measure from sensor datasheet or experiments
- R = σ² where σ is standard deviation

**Initial Covariance (P₀)**:
- Represents initial uncertainty
- Large values → Trust first measurements more
- Set each variance and cross-covariance from initial information; position and velocity entries have different units.

## Common Mistakes

❌ **Wrong**: `P = F @ P @ F + Q`
✅ **Right**: `P = F @ P @ F.T + Q`

For one scalar measurement, dividing by scalar `S` is valid. For a matrix innovation covariance, elementwise division is wrong; use `K = np.linalg.solve(S.T, (P @ H.T).T).T`.

Match the implementation's shape contract: the scalar class accepts `5.0`; the general class accepts `[5.0]` or `[[5.0]]`. A flat two-element vector and a `(2,1)` column must not broadcast into a `(2,2)` residual.

## Diagnostics

### Check Filter Health

```python
# General m-dimensional innovation: compare NIS to a declared chi-square gate.
innovation = z - H @ x_pred  # z and x_pred are explicit column vectors
nis = float((innovation.T @ np.linalg.solve(S, innovation)).item())
# For m=1, a threshold of 9 is the squared three-standard-deviation gate.
# For m>1 choose the corresponding chi-square quantile, not per-entry sqrt(S).

# Symmetry and semidefiniteness, allowing small floating-point roundoff.
tolerance = 1e-12 * max(1., np.abs(P).max())
if not np.allclose(P, P.T, rtol=0, atol=tolerance):
    print("Inspect covariance asymmetry")
if np.linalg.eigvalsh((P + P.T) / 2).min() < -tolerance:
    print("Inspect negative covariance eigenvalue")
# There is no universal [0,1] elementwise constraint on a matrix Kalman gain.

```

## Extended Kalman Filter (EKF)

For non-linear systems:

```python
def predict_ekf(x, P, f, F_jacobian, Q):
    """
    f: non-linear state transition function
    F_jacobian: Jacobian of f
    """
    x_pred = f(x)
    F = F_jacobian(x)
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred

def update_ekf(x_pred, P_pred, z, h, H_jacobian, R):
    """
    h: non-linear measurement function
    H_jacobian: Jacobian of h
    """
    H = H_jacobian(x_pred)
    y = z - h(x_pred)
    S = H @ P_pred @ H.T + R
    K = np.linalg.solve(S.T, (P_pred @ H.T).T).T
    x = x_pred + K @ y
    a = np.eye(P_pred.shape[0]) - K @ H
    P = a @ P_pred @ a.T + K @ R @ K.T
    return x, P
```

## Useful Formulas

### Innovation Covariance
```
S = H × P × Hᵀ + R
```

### Posterior Covariance (Joseph Form)
```
P = (I - K×H) × P × (I - K×H)ᵀ + K × R × Kᵀ
```

### Mahalanobis Distance
```
d² = yᵀ × S⁻¹ × y
```

### Log Likelihood
```
log L = -½ × (yᵀ×S⁻¹×y + log|S| + m×log(2π))
```

## Matrix Dimensions Check

For n states and m measurements:

```
x:  n × 1
P:  n × n
F:  n × n
Q:  n × n
H:  m × n
R:  m × m
K:  n × m
z:  m × 1
y:  m × 1  (innovation)
S:  m × m  (innovation covariance)
```

## Performance Tips

1. **Use Joseph form** for covariance update (more stable)
2. **Enforce symmetry**: `P = (P + P.T) / 2`
3. **Check covariance symmetry and semidefiniteness**; a large defect needs diagnosis
4. **Use Cholesky decomposition** instead of matrix inverse when possible
5. **Use a consistent orientation-error model**; quaternion normalization alone does not propagate or reset its covariance

## Further Reading

- Chapter 8: 1D Kalman Filter Implementation
- Chapter 10: Matrix Form Details
- Chapter 14: Extended Kalman Filter
- Chapter 16: Error State Kalman Filter
- Chapter 20: Implementation Tricks
