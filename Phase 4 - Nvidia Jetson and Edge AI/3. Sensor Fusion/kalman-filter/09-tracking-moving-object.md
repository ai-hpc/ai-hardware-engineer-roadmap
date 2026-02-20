# Tracking a Moving Object

**Level: High School (Ages 16-18)**

## Introduction

In the previous chapter, we tracked position using a 1D Kalman filter. Now let's track BOTH position AND velocity simultaneously. This is more powerful and realistic!

## The State Vector

Instead of tracking position and velocity separately, we combine them into a **state vector**:

```
x = [position]
    [velocity]
```

This is a 2×1 matrix (2 rows, 1 column).

### Example
```
x = [100]  means: position = 100m
    [20 ]         velocity = 20m/s
```

## The Motion Model (Matrix Form)

For constant velocity motion:
```
position(t+Δt) = position(t) + velocity(t) × Δt
velocity(t+Δt) = velocity(t)
```

In matrix form:
```
x(t+Δt) = F × x(t)

where F = [1  Δt]
          [0   1]
```

### Example Calculation
```
Current state: x = [100]
                   [20 ]

Time step: Δt = 1 second

F = [1  1]
    [0  1]

New state: x_new = F × x
                 = [1  1] × [100]
                   [0  1]   [20 ]
                 = [100 + 20×1]
                   [20       ]
                 = [120]
                   [20 ]
```

Position updated to 120m, velocity stayed 20m/s!

## The Covariance Matrix

Uncertainty is now a 2×2 matrix called the **covariance matrix** P:

```
P = [σ²_position      σ_position,velocity]
    [σ_position,velocity    σ²_velocity   ]
```

### What Each Element Means

- **P[0,0]**: Position variance (uncertainty in position)
- **P[1,1]**: Velocity variance (uncertainty in velocity)
- **P[0,1] = P[1,0]**: Covariance (how position and velocity errors are related)

### Example
```
P = [4   0]
    [0   1]
```

This means:
- Position uncertainty: ±2m (√4)
- Velocity uncertainty: ±1m/s (√1)
- No correlation between errors (off-diagonal = 0)

## Complete 2D Kalman Filter

### Initialization

```python
import numpy as np

# State vector [position, velocity]
x = np.array([[0.0],    # position
              [1.0]])   # velocity

# Covariance matrix
P = np.array([[10.0, 0.0],
              [0.0,  1.0]])

# Process noise covariance
Q = np.array([[0.01, 0.0],
              [0.0,  0.01]])

# Measurement noise (scalar, we only measure position)
R = 4.0
```

### Predict Step

```python
def predict(x, P, F, Q):
    """
    Predict next state

    Args:
        x: state vector (2×1)
        P: covariance matrix (2×2)
        F: state transition matrix (2×2)
        Q: process noise covariance (2×2)

    Returns:
        x_pred, P_pred
    """
    # State prediction
    x_pred = F @ x

    # Covariance prediction
    P_pred = F @ P @ F.T + Q

    return x_pred, P_pred
```

### Update Step

```python
def update(x_pred, P_pred, z, H, R):
    """
    Update state with measurement

    Args:
        x_pred: predicted state (2×1)
        P_pred: predicted covariance (2×2)
        z: measurement (scalar)
        H: measurement matrix (1×2)
        R: measurement noise (scalar)

    Returns:
        x_new, P_new
    """
    # Innovation
    y = z - H @ x_pred

    # Innovation covariance
    S = H @ P_pred @ H.T + R

    # Kalman Gain
    K = P_pred @ H.T / S

    # State update
    x_new = x_pred + K * y

    # Covariance update
    I = np.eye(2)
    P_new = (I - K @ H) @ P_pred

    return x_new, P_new
```

## The Measurement Matrix

We measure position but not velocity directly. The **measurement matrix** H extracts position from the state:

```
H = [1  0]
```

This means: measurement = 1×position + 0×velocity

### Example
```
State: x = [100]
           [20 ]

Measurement: z = H × x
              = [1  0] × [100]
                         [20 ]
              = 100
```

## Complete Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter2D:
    def __init__(self, dt=1.0):
        """
        Initialize 2D Kalman Filter for position and velocity

        Args:
            dt: time step
        """
        self.dt = dt

        # State vector [position, velocity]
        self.x = np.array([[0.0],
                          [1.0]])

        # Covariance matrix
        self.P = np.array([[10.0, 0.0],
                          [0.0,  1.0]])

        # State transition matrix
        self.F = np.array([[1.0, dt],
                          [0.0, 1.0]])

        # Measurement matrix (measure position only)
        self.H = np.array([[1.0, 0.0]])

        # Process noise covariance
        self.Q = np.array([[0.01, 0.0],
                          [0.0,  0.01]])

        # Measurement noise
        self.R = 4.0

        # History
        self.history = {
            'x': [self.x[0, 0]],
            'v': [self.x[1, 0]],
            'P_x': [self.P[0, 0]],
            'P_v': [self.P[1, 1]]
        }

    def predict(self):
        """Prediction step"""
        # State prediction
        self.x = self.F @ self.x

        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x

    def update(self, z):
        """
        Update step with measurement

        Args:
            z: position measurement (scalar)
        """
        # Innovation
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman Gain
        K = self.P @ self.H.T / S

        # State update
        self.x = self.x + K * y

        # Covariance update
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P

        # Save history
        self.history['x'].append(self.x[0, 0])
        self.history['v'].append(self.x[1, 0])
        self.history['P_x'].append(self.P[0, 0])
        self.history['P_v'].append(self.P[1, 1])

        return self.x

    def get_state(self):
        """Get current state"""
        return self.x[0, 0], self.x[1, 0]


def simulate_tracking():
    """Simulate tracking with 2D Kalman filter"""

    # True system
    true_x = 0.0
    true_v = 1.0
    dt = 1.0

    # Create Kalman filter
    kf = KalmanFilter2D(dt=dt)

    # Storage
    times = [0]
    true_positions = [true_x]
    true_velocities = [true_v]
    measurements = []
    est_positions = [kf.x[0, 0]]
    est_velocities = [kf.x[1, 0]]

    # Simulate for 20 seconds
    np.random.seed(42)
    for t in range(1, 21):
        # True system evolves
        true_x = true_x + true_v * dt

        # Noisy measurement
        measurement = true_x + np.random.normal(0, 2.0)

        # Kalman filter
        kf.predict()
        kf.update(measurement)

        # Store results
        times.append(t)
        true_positions.append(true_x)
        true_velocities.append(true_v)
        measurements.append(measurement)
        est_positions.append(kf.x[0, 0])
        est_velocities.append(kf.x[1, 0])

    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Position plot
    ax = axes[0]
    ax.plot(times, true_positions, 'g-', label='True Position', linewidth=2)
    ax.plot(times[1:], measurements, 'r.', label='Measurements', markersize=8)
    ax.plot(times, est_positions, 'b-', label='Kalman Estimate', linewidth=2)

    # Uncertainty bounds
    P_x = np.array(kf.history['P_x'])
    est_pos_array = np.array(est_positions)
    ax.fill_between(times,
                     est_pos_array - 2*np.sqrt(P_x),
                     est_pos_array + 2*np.sqrt(P_x),
                     alpha=0.3, color='blue', label='95% Confidence')

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Position (meters)')
    ax.set_title('Position Tracking')
    ax.legend()
    ax.grid(True)

    # Velocity plot
    ax = axes[1]
    ax.plot(times, true_velocities, 'g-', label='True Velocity', linewidth=2)
    ax.plot(times, est_velocities, 'b-', label='Estimated Velocity', linewidth=2)

    P_v = np.array(kf.history['P_v'])
    est_vel_array = np.array(est_velocities)
    ax.fill_between(times,
                     est_vel_array - 2*np.sqrt(P_v),
                     est_vel_array + 2*np.sqrt(P_v),
                     alpha=0.3, color='blue', label='95% Confidence')

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity Estimation (Not Directly Measured!)')
    ax.legend()
    ax.grid(True)

    # Uncertainty plot
    ax = axes[2]
    ax.plot(times, np.sqrt(P_x), 'b-', label='Position Uncertainty', linewidth=2)
    ax.plot(times, np.sqrt(P_v), 'r-', label='Velocity Uncertainty', linewidth=2)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Uncertainty (std dev)')
    ax.set_title('Uncertainty Over Time')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('kalman_2d_tracking.png', dpi=150)
    plt.show()

    # Statistics
    pos_errors = np.array(est_positions) - np.array(true_positions)
    vel_errors = np.array(est_velocities) - np.array(true_velocities)

    print(f"Position RMSE: {np.sqrt(np.mean(pos_errors**2)):.3f} meters")
    print(f"Velocity RMSE: {np.sqrt(np.mean(vel_errors**2)):.3f} m/s")
    print(f"Final position estimate: {est_positions[-1]:.3f} m (true: {true_positions[-1]:.3f} m)")
    print(f"Final velocity estimate: {est_velocities[-1]:.3f} m/s (true: {true_velocities[-1]:.3f} m/s)")


if __name__ == "__main__":
    simulate_tracking()
```

## Key Observations

### 1. Velocity Estimation Without Direct Measurement!

The amazing thing: We NEVER measure velocity directly, but the filter estimates it accurately!

**How?** By observing how position changes over time.

### 2. Covariance Coupling

Position and velocity uncertainties are coupled through the covariance matrix. Knowing position better helps estimate velocity better!

### 3. Faster Convergence

The 2D filter converges faster than the 1D filter because it uses the relationship between position and velocity.

## Matrix Dimensions Summary

```
State vector:           x (2×1)
Covariance matrix:      P (2×2)
State transition:       F (2×2)
Process noise:          Q (2×2)
Measurement matrix:     H (1×2)
Measurement noise:      R (1×1 or scalar)
Kalman Gain:            K (2×1)
```

## Experiments to Try

### Experiment 1: Changing Velocity

Modify the simulation so velocity changes:

```python
# Add acceleration
true_a = 0.5  # m/s²
true_v = true_v + true_a * dt
```

What happens to the filter? (Hint: It will lag because it assumes constant velocity!)

### Experiment 2: Measure Velocity Too

Add velocity measurements:

```python
# Measure both position and velocity
H = np.array([[1.0, 0.0],
              [0.0, 1.0]])

R = np.array([[4.0, 0.0],
              [0.0, 1.0]])
```

How does this improve performance?

### Experiment 3: Correlated Noise

Add correlation to process noise:

```python
Q = np.array([[0.01, 0.005],
              [0.005, 0.01]])
```

What effect does this have?

### Experiment 4: Different Time Steps

Try dt = 0.1 seconds (faster updates) or dt = 5.0 seconds (slower updates).

How does this affect uncertainty growth?

## Practice Problems

### Problem 1: Matrix Multiplication

Given:
```
F = [1  2]    x = [10]
    [0  1]        [5 ]
```

Calculate F × x by hand.

### Problem 2: Covariance Prediction

Given:
```
P = [4  0]    F = [1  1]    Q = [0.1  0  ]
    [0  1]        [0  1]        [0    0.1]
```

Calculate P_pred = F × P × F^T + Q

### Problem 3: Kalman Gain

Given:
```
P_pred = [5  0]    H = [1  0]    R = 2
         [0  1]
```

Calculate K = P_pred × H^T × (H × P_pred × H^T + R)^(-1)

### Problem 4: Design Challenge

Design a Kalman filter for tracking a car that:
- Measures position every 0.5 seconds (±3m accuracy)
- Measures velocity every 2 seconds (±1m/s accuracy)
- Has process noise of ±0.5m and ±0.2m/s

What are your F, H, Q, and R matrices?

## Common Pitfalls

### Pitfall 1: Wrong Matrix Dimensions

```python
# WRONG: H should be 1×2, not 2×1
H = np.array([[1.0],
              [0.0]])

# RIGHT:
H = np.array([[1.0, 0.0]])
```

### Pitfall 2: Forgetting Transpose

```python
# WRONG: Missing .T
P_pred = F @ P @ F + Q

# RIGHT:
P_pred = F @ P @ F.T + Q
```

### Pitfall 3: Scalar vs Matrix

```python
# WRONG: R should be scalar or 1×1 matrix
R = np.array([[4.0, 0.0],
              [0.0, 4.0]])

# RIGHT (for single measurement):
R = 4.0
# or
R = np.array([[4.0]])
```

## Key Takeaways

1. **State vector** combines multiple variables
2. **Covariance matrix** tracks uncertainty and correlations
3. **Matrix form** generalizes to any number of dimensions
4. **Velocity can be estimated** without direct measurement
5. **Coupled variables** help each other converge faster

## What's Next?

The next chapter introduces the full matrix notation and generalizes to N dimensions. This is the complete, general Kalman filter!

---

**Key Vocabulary**
- **State Vector**: Column matrix containing all state variables
- **Covariance Matrix**: Matrix of variances and covariances
- **State Transition Matrix (F)**: Describes how state evolves
- **Measurement Matrix (H)**: Extracts measurements from state
- **Kalman Gain Matrix (K)**: Optimal weighting for each state variable
