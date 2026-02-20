import os
import numpy as np
import matplotlib.pyplot as plt

# Save plots next to this script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class KalmanFilter1D:
    """
    1D Kalman filter with state [position, velocity].
    Uses full 2-state formulation for correct velocity updates from position-only measurements.
    """
    def __init__(self, x0=0.0, v0=1.0, P_x0=1.0, P_v0=1.0, P_xv0=0.0,
                 Q_x=0.01, Q_v=0.01, R=4.0):
        """
        Initialize 1D Kalman Filter

        Args:
            x0: initial position
            v0: initial velocity
            P_x0: initial position variance
            P_v0: initial velocity variance
            P_xv0: initial position-velocity covariance (default 0)
            Q_x: position process noise
            Q_v: velocity process noise
            R: measurement noise variance
        """
        # State [x, v]
        self.x = x0
        self.v = v0

        # Covariance matrix [[P_xx, P_xv], [P_xv, P_vv]]
        self.P_xx = P_x0
        self.P_xv = P_xv0
        self.P_vv = P_v0

        # Noise parameters
        self.Q_x = Q_x
        self.Q_v = Q_v
        self.R = R

        # History (for plotting)
        self.history = {'x': [x0], 'v': [v0], 'P_x': [np.sqrt(P_x0)], 'P_v': [np.sqrt(P_v0)]}

    def predict(self, dt=1.0):
        """Prediction step (constant velocity model)"""
        # State: x = x + v*dt, v = v
        self.x = self.x + self.v * dt

        # Covariance: P = F @ P @ F.T + Q
        # F = [[1, dt], [0, 1]]
        P_xx_new = self.P_xx + 2*dt*self.P_xv + dt**2*self.P_vv + self.Q_x
        P_xv_new = self.P_xv + dt*self.P_vv
        P_vv_new = self.P_vv + self.Q_v

        self.P_xx = P_xx_new
        self.P_xv = P_xv_new
        self.P_vv = P_vv_new

        return self.x, self.v

    def update(self, z):
        """Update step with position measurement z"""
        # Innovation
        y = z - self.x

        # H = [1, 0], S = H*P*H' + R = P_xx + R
        S = self.P_xx + self.R

        # Kalman gain K = P*H'/S = [P_xx; P_xv]/S
        K_x = self.P_xx / S
        K_v = self.P_xv / S

        # State update
        self.x = self.x + K_x * y
        self.v = self.v + K_v * y

        # Covariance update: P = (I - K*H)*P
        self.P_xx = (1 - K_x) * self.P_xx
        self.P_xv = (1 - K_x) * self.P_xv
        self.P_vv = self.P_vv - K_v * self.P_xv

        self.history['x'].append(self.x)
        self.history['v'].append(self.v)
        self.history['P_x'].append(np.sqrt(self.P_xx))
        self.history['P_v'].append(np.sqrt(self.P_vv))

        return self.x, self.v

    def get_state(self):
        """Get current state estimate"""
        return self.x, self.v, np.sqrt(self.P_xx), np.sqrt(self.P_vv)


# Simulation
def simulate_tracking():
    """Simulate tracking an object with noisy GPS"""

    np.random.seed(42)  # Reproducible results

    # True system
    true_x = 0.0
    true_v = 1.0
    dt = 1.0

    # Create Kalman filter (P_x0=10: high initial position uncertainty)
    kf = KalmanFilter1D(x0=0.0, v0=1.0, P_x0=10.0, P_v0=1.0,
                        Q_x=0.01, Q_v=0.01, R=4.0)

    # Storage
    times = [0]
    true_positions = [true_x]
    measurements = []
    estimates = [kf.x]
    uncertainties = [np.sqrt(kf.P_xx)]

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
        uncertainties.append(np.sqrt(kf.P_xx))

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
    out_path = os.path.join(_SCRIPT_DIR, 'kalman_1d_tracking.png')
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved as '{out_path}'")
    plt.show()

    # Print statistics
    errors = np.array(estimates) - np.array(true_positions)
    rmse = np.sqrt(np.mean(errors**2))
    print(f"Root Mean Square Error: {rmse:.3f} meters")
    print(f"Final uncertainty: +/-{uncertainties[-1]:.3f} meters")
    print(f"Final estimate: {estimates[-1]:.3f} meters")
    print(f"True position: {true_positions[-1]:.3f} meters")


if __name__ == "__main__":
    simulate_tracking()