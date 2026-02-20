# Multidimensional Kalman Filter

**Level: Undergraduate (College)**

## Introduction

Now we'll explore practical multidimensional Kalman filters for real-world applications. We'll build filters for increasingly complex scenarios.

## Example 1: 2D Position Tracking (4 States)

Track an object moving in a plane with position (x, y) and velocity (vx, vy).

### State Vector

```
x = [x ]    ← x position
    [y ]    ← y position
    [vx]    ← x velocity
    [vy]    ← y velocity
```

### State Transition Matrix

Constant velocity model in 2D:

```python
dt = 0.1  # Time step

F = np.array([[1, 0, dt, 0 ],
              [0, 1, 0,  dt],
              [0, 0, 1,  0 ],
              [0, 0, 0,  1 ]])
```

This means:
- x(k+1) = x(k) + vx(k) × dt
- y(k+1) = y(k) + vy(k) × dt
- vx(k+1) = vx(k)
- vy(k+1) = vy(k)

### Measurement Matrix

If we measure both x and y positions:

```python
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])
```

### Complete Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter2DTracking:
    def __init__(self, dt=0.1):
        """
        2D position and velocity tracking

        State: [x, y, vx, vy]
        """
        self.dt = dt

        # State transition matrix
        self.F = np.array([[1, 0, dt, 0 ],
                          [0, 1, 0,  dt],
                          [0, 0, 1,  0 ],
                          [0, 0, 0,  1 ]])

        # Measurement matrix (measure x and y)
        self.H = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0]])

        # Process noise covariance
        q = 0.1
        self.Q = q * np.array([[dt**4/4, 0,       dt**3/2, 0      ],
                               [0,       dt**4/4, 0,       dt**3/2],
                               [dt**3/2, 0,       dt**2,   0      ],
                               [0,       dt**3/2, 0,       dt**2  ]])

        # Measurement noise covariance
        self.R = np.array([[1.0, 0.0],
                          [0.0, 1.0]])

        # Initial state [x, y, vx, vy]
        self.x = np.array([[0.0],
                          [0.0],
                          [1.0],
                          [1.0]])

        # Initial covariance
        self.P = 10.0 * np.eye(4)

        # Identity matrix
        self.I = np.eye(4)

    def predict(self):
        """Prediction step"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        """
        Update with measurement

        Args:
            z: [x_measured, y_measured] (2×1)
        """
        # Innovation
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update
        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P

        return self.x


def simulate_2d_tracking():
    """Simulate 2D object tracking"""

    dt = 0.1
    kf = KalmanFilter2DTracking(dt)

    # True trajectory (circular motion)
    t = np.linspace(0, 10, 100)
    radius = 10
    omega = 0.5

    true_x = radius * np.cos(omega * t)
    true_y = radius * np.sin(omega * t)

    # Storage
    est_x, est_y = [], []
    meas_x, meas_y = [], []

    for i in range(len(t)):
        # Noisy measurement
        z = np.array([[true_x[i] + np.random.normal(0, 1)],
                     [true_y[i] + np.random.normal(0, 1)]])

        meas_x.append(z[0, 0])
        meas_y.append(z[1, 0])

        # Kalman filter
        kf.predict()
        kf.update(z)

        est_x.append(kf.x[0, 0])
        est_y.append(kf.x[1, 0])

    # Plot
    plt.figure(figsize=(10, 10))
    plt.plot(true_x, true_y, 'g-', label='True Path', linewidth=2)
    plt.plot(meas_x, meas_y, 'r.', label='Measurements', alpha=0.5)
    plt.plot(est_x, est_y, 'b-', label='Kalman Estimate', linewidth=2)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('2D Object Tracking')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('2d_tracking.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    simulate_2d_tracking()
```

## Example 2: Constant Acceleration Model (6 States)

Track position, velocity, AND acceleration in 2D.

### State Vector

```
x = [x ]    ← x position
    [y ]    ← y position
    [vx]    ← x velocity
    [vy]    ← y velocity
    [ax]    ← x acceleration
    [ay]    ← y acceleration
```

### State Transition Matrix

```python
dt = 0.1

F = np.array([[1, 0, dt, 0,  dt**2/2, 0      ],
              [0, 1, 0,  dt, 0,       dt**2/2],
              [0, 0, 1,  0,  dt,      0      ],
              [0, 0, 0,  1,  0,       dt     ],
              [0, 0, 0,  0,  1,       0      ],
              [0, 0, 0,  0,  0,       1      ]])
```

This implements:
- x(k+1) = x(k) + vx(k)×dt + ax(k)×dt²/2
- vx(k+1) = vx(k) + ax(k)×dt
- ax(k+1) = ax(k)

## Example 3: Coordinated Turn Model

For tracking vehicles that turn, we need a different model.

### State Vector

```
x = [x ]    ← x position
    [y ]    ← y position
    [v ]    ← speed
    [θ ]    ← heading angle
    [ω ]    ← turn rate
```

### Non-Linear State Transition

```
x(k+1) = x(k) + v(k) × cos(θ(k)) × dt
y(k+1) = y(k) + v(k) × sin(θ(k)) × dt
v(k+1) = v(k)
θ(k+1) = θ(k) + ω(k) × dt
ω(k+1) = ω(k)
```

This is NON-LINEAR! We'll need the Extended Kalman Filter (next chapters).

## Example 4: Sensor Fusion (GPS + IMU)

Combine GPS (position) and IMU (acceleration) measurements.

### State Vector

```
x = [x ]    ← position
    [v ]    ← velocity
    [a ]    ← acceleration
```

### Two Measurement Models

**GPS measurement** (position only):
```python
H_gps = np.array([[1, 0, 0]])
R_gps = np.array([[25.0]])  # GPS noise: ±5m
```

**IMU measurement** (acceleration only):
```python
H_imu = np.array([[0, 0, 1]])
R_imu = np.array([[0.01]])  # IMU noise: ±0.1m/s²
```

### Implementation

```python
class SensorFusionKF:
    def __init__(self, dt=0.01):
        self.dt = dt

        # State: [position, velocity, acceleration]
        self.x = np.array([[0.0], [0.0], [0.0]])

        # State transition
        self.F = np.array([[1, dt, dt**2/2],
                          [0, 1,  dt     ],
                          [0, 0,  1      ]])

        # Process noise
        self.Q = 0.1 * np.eye(3)

        # Covariance
        self.P = 10.0 * np.eye(3)

        # Measurement matrices
        self.H_gps = np.array([[1, 0, 0]])
        self.H_imu = np.array([[0, 0, 1]])

        # Measurement noise
        self.R_gps = np.array([[25.0]])
        self.R_imu = np.array([[0.01]])

        self.I = np.eye(3)

    def predict(self):
        """Prediction step"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update_gps(self, z_gps):
        """Update with GPS measurement"""
        H = self.H_gps
        R = self.R_gps

        y = z_gps - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (self.I - K @ H) @ self.P

    def update_imu(self, z_imu):
        """Update with IMU measurement"""
        H = self.H_imu
        R = self.R_imu

        y = z_imu - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (self.I - K @ H) @ self.P
```

## Example 5: Multi-Sensor Tracking

Track an aircraft with:
- Radar (range and bearing)
- GPS (position)
- Barometer (altitude)

### State Vector (3D position + velocity)

```
x = [x, y, z, vx, vy, vz]ᵀ
```

### Different Measurement Models

**GPS** (measures x, y, z):
```python
H_gps = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0]])
```

**Barometer** (measures z only):
```python
H_baro = np.array([[0, 0, 1, 0, 0, 0]])
```

**Radar** (non-linear - measures range and bearing):
```python
# Range: r = sqrt(x² + y²)
# Bearing: θ = atan2(y, x)
# This requires Extended Kalman Filter!
```

## Handling Multiple Measurement Rates

Different sensors update at different rates:

```python
class MultiRateKF:
    def __init__(self):
        # ... initialization ...
        self.last_update_time = 0

    def process_measurement(self, t, sensor_type, z):
        """Process measurement from any sensor"""

        # Predict to current time
        dt = t - self.last_update_time
        if dt > 0:
            self.predict(dt)
            self.last_update_time = t

        # Update based on sensor type
        if sensor_type == 'GPS':
            self.update_gps(z)
        elif sensor_type == 'IMU':
            self.update_imu(z)
        elif sensor_type == 'BARO':
            self.update_baro(z)
```

## Asynchronous Measurements

```python
def run_filter_async():
    """Run filter with asynchronous measurements"""

    kf = SensorFusionKF(dt=0.01)

    # Measurement schedule
    gps_rate = 1.0    # 1 Hz
    imu_rate = 0.01   # 100 Hz

    t = 0
    next_gps = 0
    next_imu = 0

    while t < 10.0:
        # Always predict
        kf.predict()

        # GPS update?
        if t >= next_gps:
            z_gps = get_gps_measurement()
            kf.update_gps(z_gps)
            next_gps += gps_rate

        # IMU update?
        if t >= next_imu:
            z_imu = get_imu_measurement()
            kf.update_imu(z_imu)
            next_imu += imu_rate

        t += kf.dt
```

## Process Noise Tuning

### Method 1: Physical Modeling

For a car with maximum acceleration aₘₐₓ:

```python
# Assume acceleration is uniformly distributed
q = (aₘₐₓ / 3)**2  # 3-sigma rule

Q = q * np.array([[dt**4/4, dt**3/2],
                  [dt**3/2, dt**2  ]])
```

### Method 2: Empirical Tuning

```python
def tune_process_noise(measurements, q_values):
    """
    Try different Q values and pick best

    Args:
        measurements: List of measurements
        q_values: List of q values to try

    Returns:
        best_q: Optimal q value
    """
    best_q = None
    best_error = float('inf')

    for q in q_values:
        kf = create_filter(q)
        errors = []

        for z in measurements:
            kf.predict()
            kf.update(z)
            errors.append(compute_error(kf.x, true_state))

        rmse = np.sqrt(np.mean(np.array(errors)**2))

        if rmse < best_error:
            best_error = rmse
            best_q = q

    return best_q
```

### Method 3: Adaptive Estimation

```python
class AdaptiveKF:
    """Kalman filter with adaptive process noise"""

    def __init__(self):
        # ... standard initialization ...
        self.innovation_history = []

    def update(self, z):
        """Update and adapt Q based on innovations"""

        # Standard update
        y = z - self.H @ self.x
        self.innovation_history.append(y)

        # ... rest of update ...

        # Adapt Q if innovations are too large
        if len(self.innovation_history) > 10:
            recent_innovations = self.innovation_history[-10:]
            innovation_var = np.var(recent_innovations)

            # If innovations are large, increase Q
            if innovation_var > threshold:
                self.Q *= 1.1
            # If innovations are small, decrease Q
            elif innovation_var < threshold / 2:
                self.Q *= 0.9
```

## Practice Problems

### Problem 1: 3D Tracking

Design a Kalman filter for tracking a drone in 3D space:
- State: [x, y, z, vx, vy, vz]
- Measurements: GPS gives [x, y, z] every 1 second
- dt = 0.1 seconds

Write the F, H, Q, and R matrices.

### Problem 2: Sensor Fusion

You have:
- GPS: ±5m accuracy, 1 Hz
- Odometer: ±0.1m/s accuracy, 10 Hz

Design a filter to fuse both sensors for 1D position tracking.

### Problem 3: Multi-Rate Processing

Implement a filter that handles:
- Sensor A: Updates every 0.1s
- Sensor B: Updates every 1.0s
- Sensor C: Updates every 0.05s

How do you structure the predict-update cycle?

### Problem 4: Process Noise Design

A car can accelerate up to ±3 m/s². Design Q for a [position, velocity] state with dt=0.1s.

### Problem 5: Measurement Selection

You have 3 sensors measuring the same quantity:
- Sensor A: ±1m, 10 Hz
- Sensor B: ±5m, 1 Hz
- Sensor C: ±0.5m, 0.1 Hz

Which should you trust most? Design appropriate R values.

## Common Patterns

### Pattern 1: Decoupled Dimensions

For independent x and y motion:

```python
# Can use two separate 1D filters
kf_x = KalmanFilter1D()
kf_y = KalmanFilter1D()

# Or one 2D filter with block-diagonal matrices
F = np.block([[F_x, np.zeros((2,2))],
              [np.zeros((2,2)), F_y]])
```

### Pattern 2: Hierarchical Filtering

```python
# Low-level filter (fast, local)
kf_local = KalmanFilter(dt=0.01)

# High-level filter (slow, global)
kf_global = KalmanFilter(dt=1.0)

# Fuse estimates
combined_estimate = fuse(kf_local.x, kf_global.x)
```

### Pattern 3: Bank of Filters

For uncertain models, run multiple filters:

```python
filters = [
    KalmanFilter(model='constant_velocity'),
    KalmanFilter(model='constant_acceleration'),
    KalmanFilter(model='coordinated_turn')
]

# Update all
for kf in filters:
    kf.predict()
    kf.update(z)

# Pick best based on likelihood
best_filter = max(filters, key=lambda kf: kf.likelihood)
```

## Key Takeaways

1. **State vector** can include any variables you want to estimate
2. **Multiple sensors** can be fused by using different H matrices
3. **Asynchronous measurements** are handled by predicting to measurement time
4. **Process noise tuning** is critical for performance
5. **Decoupled dimensions** can simplify implementation

## What's Next?

The next chapter dives deep into process and measurement noise modeling - how to choose Q and R properly for your application.

---

**Key Vocabulary**
- **Sensor Fusion**: Combining multiple sensors optimally
- **Asynchronous Measurements**: Sensors updating at different times
- **Multi-Rate Processing**: Handling different update rates
- **Process Noise Tuning**: Choosing appropriate Q matrix
- **Decoupled Dimensions**: Independent state variables
