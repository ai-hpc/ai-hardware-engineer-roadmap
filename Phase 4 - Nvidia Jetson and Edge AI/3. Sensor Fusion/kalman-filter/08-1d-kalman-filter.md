# One Dimensional Kalman Filter

**Level: High School (Ages 16-18)**

## Introduction

Time to code! We'll implement a complete 1D Kalman filter in Python. This tracks a single variable (like position) with simple math.

## The Problem

Track the position of an object moving at constant velocity using noisy GPS measurements.

**Given**:
- Initial position: 0 meters
- Initial velocity: 1 meter/second
- GPS measurements every second (noisy!)

**Goal**: Estimate true position over time

## The Complete Algorithm

### Variables We Track

```python
# State
x = 0.0          # position estimate (meters)
v = 1.0          # velocity estimate (m/s)

# Uncertainty
P_x = 1.0        # position uncertainty (variance)
P_v = 1.0        # velocity uncertainty (variance)

# Process noise
Q_x = 0.01       # position process noise
Q_v = 0.01       # velocity process noise

# Measurement noise
R = 4.0          # GPS measurement noise (variance)
```

### Step 1: Predict

```python
def predict(x, v, P_x, P_v, dt):
    """
    Predict next state based on motion model

    Args:
        x: current position
        v: current velocity
        P_x: position uncertainty
        P_v: velocity uncertainty
        dt: time step

    Returns:
        x_pred, v_pred, P_x_pred, P_v_pred
    """
    # State prediction (constant velocity model)
    x_pred = x + v * dt
    v_pred = v

    # Uncertainty prediction (grows!)
    P_x_pred = P_x + P_v * dt**2 + Q_x
    P_v_pred = P_v + Q_v

    return x_pred, v_pred, P_x_pred, P_v_pred
```

### Step 2: Update

```python
def update(x_pred, v_pred, P_x_pred, P_v_pred, z, R):
    """
    Update state with measurement

    Args:
        x_pred: predicted position
        v_pred: predicted velocity
        P_x_pred: predicted position uncertainty
        P_v_pred: predicted velocity uncertainty
        z: measurement (position)
        R: measurement noise

    Returns:
        x_new, v_new, P_x_new, P_v_new
    """
    # Innovation (measurement - prediction)
    y = z - x_pred

    # Kalman Gain
    K = P_x_pred / (P_x_pred + R)

    # State update
    x_new = x_pred + K * y
    v_new = v_pred + K * y  # velocity also updated!

    # Uncertainty update (decreases!)
    P_x_new = (1 - K) * P_x_pred
    P_v_new = P_v_pred  # simplified for 1D

    return x_new, v_new, P_x_new, P_v_new
```

## Complete Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter1D:
    def __init__(self, x0=0.0, v0=1.0, P_x0=1.0, P_v0=1.0,
                 Q_x=0.01, Q_v=0.01, R=4.0):
        """
        Initialize 1D Kalman Filter

        Args:
            x0: initial position
            v0: initial velocity
            P_x0: initial position uncertainty
            P_v0: initial velocity uncertainty
            Q_x: position process noise
            Q_v: velocity process noise
            R: measurement noise
        """
        # State
        self.x = x0
        self.v = v0

        # Uncertainty
        self.P_x = P_x0
        self.P_v = P_v0

        # Noise parameters
        self.Q_x = Q_x
        self.Q_v = Q_v
        self.R = R

        # History (for plotting)
        self.history = {
            'x': [x0],
            'v': [v0],
            'P_x': [P_x0],
            'P_v': [P_v0]
        }

    def predict(self, dt=1.0):
        """Prediction step"""
        # State prediction
        self.x = self.x + self.v * dt
        # velocity stays constant

        # Uncertainty prediction
        self.P_x = self.P_x + self.P_v * dt**2 + self.Q_x
        self.P_v = self.P_v + self.Q_v

        return self.x, self.v

    def update(self, z):
        """Update step with measurement z"""
        # Innovation
        y = z - self.x

        # Kalman Gain
        K = self.P_x / (self.P_x + self.R)

        # State update
        self.x = self.x + K * y
        self.v = self.v + K * y  # velocity correction

        # Uncertainty update
        self.P_x = (1 - K) * self.P_x
        # P_v stays same (simplified)

        # Save history
        self.history['x'].append(self.x)
        self.history['v'].append(self.v)
        self.history['P_x'].append(self.P_x)
        self.history['P_v'].append(self.P_v)

        return self.x, self.v

    def get_state(self):
        """Get current state estimate"""
        return self.x, self.v, self.P_x, self.P_v


# Simulation
def simulate_tracking():
    """Simulate tracking an object with noisy GPS"""

    # True system
    true_x = 0.0
    true_v = 1.0
    dt = 1.0

    # Create Kalman filter
    kf = KalmanFilter1D(x0=0.0, v0=1.0, P_x0=10.0, P_v0=1.0,
                        Q_x=0.01, Q_v=0.01, R=4.0)

    # Storage
    times = [0]
    true_positions = [true_x]
    measurements = []
    estimates = [kf.x]
    uncertainties = [np.sqrt(kf.P_x)]

    # Simulate for 20 seconds
    for t in range(1, 21):
        # True system evolves
        true_x = true_x + true_v * dt

        # Noisy measurement (GPS)
        measurement = true_x + np.random.normal(0, np.sqrt(4.0))

        # Kalman filter: Predict
        kf.predict(dt)

        # Kalman filter: Update
        x_est, v_est = kf.update(measurement)

        # Store results
        times.append(t)
        true_positions.append(true_x)
        measurements.append(measurement)
        estimates.append(x_est)
        uncertainties.append(np.sqrt(kf.P_x))

    # Plot results
    plt.figure(figsize=(12, 8))

    # Position plot
    plt.subplot(2, 1, 1)
    plt.plot(times, true_positions, 'g-', label='True Position', linewidth=2)
    plt.plot(times[1:], measurements, 'r.', label='GPS Measurements', markersize=8)
    plt.plot(times, estimates, 'b-', label='Kalman Estimate', linewidth=2)

    # Uncertainty bounds
    estimates_array = np.array(estimates)
    uncertainties_array = np.array(uncertainties)
    plt.fill_between(times,
                     estimates_array - 2*uncertainties_array,
                     estimates_array + 2*uncertainties_array,
                     alpha=0.3, color='blue', label='95% Confidence')

    plt.xlabel('Time (seconds)')
    plt.ylabel('Position (meters)')
    plt.title('1D Kalman Filter: Position Tracking')
    plt.legend()
    plt.grid(True)

    # Uncertainty plot
    plt.subplot(2, 1, 2)
    plt.plot(times, uncertainties, 'b-', linewidth=2)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Position Uncertainty (meters)')
    plt.title('Uncertainty Over Time')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('kalman_1d_tracking.png', dpi=150)
    plt.show()

    # Print statistics
    errors = np.array(estimates) - np.array(true_positions)
    rmse = np.sqrt(np.mean(errors**2))
    print(f"Root Mean Square Error: {rmse:.3f} meters")
    print(f"Final uncertainty: ±{uncertainties[-1]:.3f} meters")
    print(f"Final estimate: {estimates[-1]:.3f} meters")
    print(f"True position: {true_positions[-1]:.3f} meters")


if __name__ == "__main__":
    simulate_tracking()
```

## Running the Code

Save the code above and run it:

```bash
python kalman_1d.py
```

You'll see:
1. A plot showing true position, noisy measurements, and Kalman estimate
2. Uncertainty bounds (95% confidence interval)
3. How uncertainty decreases over time
4. Statistics on accuracy

## Understanding the Results

### What You'll Observe

1. **Measurements are noisy** (red dots scatter around true line)
2. **Kalman estimate is smooth** (blue line follows true position closely)
3. **Uncertainty decreases** (filter becomes more confident)
4. **Estimate is better than any single measurement**

### Why It Works

The Kalman filter:
- Uses the motion model (constant velocity)
- Doesn't trust any single noisy measurement
- Combines all information optimally
- Adapts to changing uncertainty

## Experiments to Try

### Experiment 1: Change Measurement Noise

```python
# Very noisy GPS
kf = KalmanFilter1D(R=16.0)  # was 4.0

# Very accurate GPS
kf = KalmanFilter1D(R=0.25)  # was 4.0
```

**Question**: How does this affect the Kalman Gain and final uncertainty?

### Experiment 2: Change Process Noise

```python
# Very predictable motion
kf = KalmanFilter1D(Q_x=0.001, Q_v=0.001)

# Very unpredictable motion
kf = KalmanFilter1D(Q_x=1.0, Q_v=1.0)
```

**Question**: How does this affect how much the filter trusts predictions vs measurements?

### Experiment 3: Wrong Initial Estimate

```python
# Start with wrong position
kf = KalmanFilter1D(x0=50.0)  # true is 0.0

# Start with high uncertainty
kf = KalmanFilter1D(P_x0=100.0)
```

**Question**: How quickly does the filter converge to the true value?

### Experiment 4: Missing Measurements

```python
# Only update every other measurement
for t in range(1, 21):
    true_x = true_x + true_v * dt
    kf.predict(dt)

    if t % 2 == 0:  # Only update every 2 seconds
        measurement = true_x + np.random.normal(0, 2.0)
        kf.update(measurement)
```

**Question**: What happens to uncertainty when measurements are missing?

## Step-by-Step Example

Let's trace through the first few iterations:

### t=0 (Initial)
```
State: x=0.0m, v=1.0m/s
Uncertainty: P_x=10.0, P_v=1.0
```

### t=1 (Predict)
```
x_pred = 0.0 + 1.0×1 = 1.0m
v_pred = 1.0m/s
P_x_pred = 10.0 + 1.0×1² + 0.01 = 11.01
P_v_pred = 1.0 + 0.01 = 1.01
```

### t=1 (Update)
```
Measurement: z = 1.5m (true=1.0, noise=+0.5)
Innovation: y = 1.5 - 1.0 = 0.5m
Kalman Gain: K = 11.01/(11.01+4.0) = 0.733

x_new = 1.0 + 0.733×0.5 = 1.37m
v_new = 1.0 + 0.733×0.5 = 1.37m/s
P_x_new = (1-0.733)×11.01 = 2.94
```

### t=2 (Predict)
```
x_pred = 1.37 + 1.37×1 = 2.74m
P_x_pred = 2.94 + 1.01×1² + 0.01 = 3.96
```

And so on...

## Common Issues and Solutions

### Issue 1: Filter Diverges

**Symptom**: Estimates get worse over time

**Causes**:
- Process noise too small
- Measurement noise too small
- Wrong motion model

**Solution**: Increase Q or R, check your model

### Issue 2: Filter Too Slow

**Symptom**: Takes many measurements to converge

**Causes**:
- Initial uncertainty too small
- Measurement noise too large

**Solution**: Increase P_x0 or decrease R

### Issue 3: Filter Too Jumpy

**Symptom**: Estimate jumps around with each measurement

**Causes**:
- Process noise too large
- Measurement noise too small

**Solution**: Decrease Q or increase R

## Practice Problems

### Problem 1: Hand Calculation

Given:
- x_pred = 10.0, P_x_pred = 4.0
- Measurement z = 12.0, R = 2.0

Calculate:
a) Kalman Gain K
b) Updated estimate x_new
c) Updated uncertainty P_x_new

### Problem 2: Code Modification

Modify the code to track an object with:
- Constant acceleration (not constant velocity)
- Initial velocity = 0
- Acceleration = 0.5 m/s²

### Problem 3: Tuning

You have a system where:
- GPS updates every 1 second with ±5m accuracy
- Object moves at roughly constant velocity
- Occasional sudden stops

What values would you choose for Q_x, Q_v, and R?

### Problem 4: Analysis

Run the simulation 100 times and calculate:
a) Average RMSE
b) Percentage of time true value is within 95% confidence bounds
c) How does this compare to using raw measurements?

## Key Takeaways

1. **Predict-Update cycle** is the core of Kalman filtering
2. **Kalman Gain** automatically balances prediction vs measurement
3. **Uncertainty oscillates** but generally decreases
4. **Tuning parameters** (Q, R) affects performance
5. **Initial conditions** matter but filter converges

## What's Next?

The next chapter extends this to track BOTH position AND velocity simultaneously using matrix notation. This is the full multidimensional Kalman filter!

---

**Key Vocabulary**
- **Innovation**: Difference between measurement and prediction
- **Kalman Gain**: Optimal weighting factor (0 to 1)
- **Process Noise (Q)**: Uncertainty in motion model
- **Measurement Noise (R)**: Uncertainty in sensor
- **Convergence**: Filter settling to accurate estimates
