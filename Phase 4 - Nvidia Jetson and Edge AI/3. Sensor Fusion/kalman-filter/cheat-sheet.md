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

```python
import numpy as np

class KalmanFilter:
    def __init__(self, F, H, Q, R, x0, P0):
        self.F = F  # State transition
        self.H = H  # Measurement matrix
        self.Q = Q  # Process noise
        self.R = R  # Measurement noise
        self.x = x0  # Initial state
        self.P = P0  # Initial covariance
        self.I = np.eye(F.shape[0])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P
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
R = np.array([[σ²]])
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
- Start with: Q = 0.01 × I

**Measurement Noise (R)**:
- Should match actual sensor noise
- Measure from sensor datasheet or experiments
- R = σ² where σ is standard deviation

**Initial Covariance (P₀)**:
- Represents initial uncertainty
- Large values → Trust first measurements more
- Start with: P₀ = 10 × I

## Common Mistakes

❌ **Wrong**: `P = F @ P @ F + Q`
✅ **Right**: `P = F @ P @ F.T + Q`

❌ **Wrong**: `K = P @ H.T / S`
✅ **Right**: `K = P @ H.T @ np.linalg.inv(S)`

❌ **Wrong**: `z = 5.0` (scalar)
✅ **Right**: `z = np.array([[5.0]])` (column vector)

## Diagnostics

### Check Filter Health

```python
# 1. Innovation should be small
innovation = z - H @ x_pred
if np.abs(innovation) > 3 * np.sqrt(S):
    print("Warning: Large innovation!")

# 2. Covariance should be positive definite
eigenvalues = np.linalg.eigvals(P)
if np.any(eigenvalues <= 0):
    print("Warning: P not positive definite!")

# 3. Kalman gain should be 0 < K < 1
if np.any(K < 0) or np.any(K > 1):
    print("Warning: K out of range!")
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
    K = P_pred @ H.T @ np.linalg.inv(S)
    x = x_pred + K @ y
    P = (I - K @ H) @ P_pred
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
3. **Check positive definiteness** regularly
4. **Use Cholesky decomposition** instead of matrix inverse when possible
5. **Normalize quaternions** if tracking orientation

## Further Reading

- Chapter 8: 1D Kalman Filter Implementation
- Chapter 10: Matrix Form Details
- Chapter 14: Extended Kalman Filter
- Chapter 16: Error State Kalman Filter
- Chapter 20: Implementation Tricks
