#!/usr/bin/env python3
"""
6D Kalman Filter Example: Track a drone in 3D space

This example demonstrates:
- 6-dimensional state tracking (3D position + 3D velocity)
- GPS measurements (position only)
- Velocity estimation without direct measurement
- Spiral trajectory tracking
- Comprehensive visualization

State vector: [x, y, z, vx, vy, vz]
Measurements: [x, y, z] from GPS

Author: Kalman Filter Learning Series
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Save plots next to this script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class KalmanFilter:
    """
    General N-dimensional Kalman Filter
    """
    def __init__(self, F, H, Q, R, x0, P0):
        """
        Initialize Kalman Filter

        Args:
            F: State transition matrix (n×n)
            H: Measurement matrix (m×n)
            Q: Process noise covariance (n×n)
            R: Measurement noise covariance (m×m)
            x0: Initial state estimate (n×1)
            P0: Initial covariance estimate (n×n)
        """
        self.F = F  # State transition
        self.H = H  # Measurement matrix
        self.Q = Q  # Process noise
        self.R = R  # Measurement noise

        self.x = x0  # State estimate
        self.P = P0  # Covariance estimate

        # Dimensions
        self.n = F.shape[0]  # Number of states
        self.m = H.shape[0]  # Number of measurements

        # Identity matrix
        self.I = np.eye(self.n)

    def predict(self):
        """
        Prediction step (time update)

        Returns:
            x_pred, P_pred
        """
        # State prediction
        self.x = self.F @ self.x

        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x, self.P

    def update(self, z):
        """
        Update step (measurement update)

        Args:
            z: Measurement vector (m×1)

        Returns:
            x_updated, P_updated
        """
        # Innovation
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update (Joseph form for numerical stability)
        I_KH = self.I - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return self.x, self.P

    def get_state(self):
        """Get current state estimate"""
        return self.x.copy()

    def get_covariance(self):
        """Get current covariance estimate"""
        return self.P.copy()


def run_6d_tracking():
    """
    Complete 6D tracking example: Track a drone in 3D space
    State: [x, y, z, vx, vy, vz]
    """

    dt = 0.1  # 10 Hz update rate

    # State transition matrix (constant velocity model in 3D)
    F = np.array([[1, 0, 0, dt, 0,  0 ],
                  [0, 1, 0, 0,  dt, 0 ],
                  [0, 0, 1, 0,  0,  dt],
                  [0, 0, 0, 1,  0,  0 ],
                  [0, 0, 0, 0,  1,  0 ],
                  [0, 0, 0, 0,  0,  1 ]])

    # Measurement matrix (GPS measures position only)
    H = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0]])

    # Process noise covariance (continuous white noise model)
    q = 0.1  # Process noise magnitude
    Q = q * np.array([
        [dt**4/4, 0,       0,       dt**3/2, 0,       0      ],
        [0,       dt**4/4, 0,       0,       dt**3/2, 0      ],
        [0,       0,       dt**4/4, 0,       0,       dt**3/2],
        [dt**3/2, 0,       0,       dt**2,   0,       0      ],
        [0,       dt**3/2, 0,       0,       dt**2,   0      ],
        [0,       0,       dt**3/2, 0,       0,       dt**2  ]
    ])

    # Measurement noise covariance (GPS noise)
    R = np.diag([2.0, 2.0, 3.0])  # x, y more accurate than z

    # Initial state [x, y, z, vx, vy, vz]
    x0 = np.array([[0.0],   # x position
                   [0.0],   # y position
                   [0.0],   # z position
                   [1.0],   # vx velocity
                   [0.5],   # vy velocity
                   [0.2]])  # vz velocity

    # Initial covariance (high uncertainty)
    P0 = np.diag([10.0, 10.0, 10.0, 5.0, 5.0, 5.0])

    # Create Kalman filter
    kf = KalmanFilter(F, H, Q, R, x0, P0)

    # Simulate true trajectory (spiral upward)
    n_steps = 200
    true_states = []
    measurements = []
    estimates = []

    # True initial state
    true_x = x0.copy()

    print("Running 6D Kalman Filter Simulation...")
    print("=" * 50)

    np.random.seed(42)  # For reproducibility

    for i in range(n_steps):
        # True system evolution (with some acceleration for interesting trajectory)
        # Add slight circular motion
        omega = 0.1  # Angular velocity
        true_x[3, 0] = 1.0 * np.cos(omega * i * dt)  # vx
        true_x[4, 0] = 1.0 * np.sin(omega * i * dt)  # vy
        true_x[5, 0] = 0.2  # vz (constant upward)

        # Update true position
        true_x[0:3] = true_x[0:3] + true_x[3:6] * dt

        true_states.append(true_x.copy())

        # Generate noisy GPS measurement
        noise = np.array([[np.random.normal(0, np.sqrt(R[0, 0]))],
                         [np.random.normal(0, np.sqrt(R[1, 1]))],
                         [np.random.normal(0, np.sqrt(R[2, 2]))]])
        z = true_x[0:3] + noise
        measurements.append(z.copy())

        # Kalman filter predict and update
        kf.predict()
        kf.update(z)

        estimates.append(kf.x.copy())

        # Print progress every 50 steps
        if (i + 1) % 50 == 0:
            pos_error = np.linalg.norm(kf.x[0:3] - true_x[0:3])
            vel_error = np.linalg.norm(kf.x[3:6] - true_x[3:6])
            print(f"Step {i+1}/{n_steps}: Pos Error = {pos_error:.3f}m, "
                  f"Vel Error = {vel_error:.3f}m/s")

    # Convert to arrays for plotting
    true_states = np.array([s.flatten() for s in true_states])
    measurements = np.array([m.flatten() for m in measurements])
    estimates = np.array([e.flatten() for e in estimates])

    # Calculate errors
    pos_errors = np.linalg.norm(estimates[:, 0:3] - true_states[:, 0:3], axis=1)
    vel_errors = np.linalg.norm(estimates[:, 3:6] - true_states[:, 3:6], axis=1)

    print("\n" + "=" * 50)
    print("RESULTS:")
    print(f"Mean Position Error: {np.mean(pos_errors):.3f} m")
    print(f"Mean Velocity Error: {np.mean(vel_errors):.3f} m/s")
    print(f"Final Position Error: {pos_errors[-1]:.3f} m")
    print(f"Final Velocity Error: {vel_errors[-1]:.3f} m/s")
    print("=" * 50)

    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 12))

    # 3D trajectory plot
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(true_states[:, 0], true_states[:, 1], true_states[:, 2],
             'g-', label='True Path', linewidth=2)
    ax1.scatter(measurements[:, 0], measurements[:, 1], measurements[:, 2],
                c='r', marker='.', s=1, alpha=0.3, label='GPS Measurements')
    ax1.plot(estimates[:, 0], estimates[:, 1], estimates[:, 2],
             'b-', label='Kalman Estimate', linewidth=2)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    ax1.grid(True)

    # X-Y plane view
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(true_states[:, 0], true_states[:, 1], 'g-', label='True', linewidth=2)
    ax2.scatter(measurements[:, 0], measurements[:, 1], c='r', s=1, alpha=0.3, label='GPS')
    ax2.plot(estimates[:, 0], estimates[:, 1], 'b-', label='Estimate', linewidth=2)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top View (X-Y Plane)')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')

    # Position errors over time
    ax3 = fig.add_subplot(2, 3, 3)
    time = np.arange(n_steps) * dt
    ax3.plot(time, pos_errors, 'b-', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position Error (m)')
    ax3.set_title('Position Error Over Time')
    ax3.grid(True)

    # Individual position components
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(time, true_states[:, 0], 'g-', label='True X', linewidth=2)
    ax4.plot(time, estimates[:, 0], 'b--', label='Est X', linewidth=2)
    ax4.plot(time, true_states[:, 1], 'g:', label='True Y', linewidth=2)
    ax4.plot(time, estimates[:, 1], 'b:', label='Est Y', linewidth=2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Position (m)')
    ax4.set_title('X and Y Position')
    ax4.legend()
    ax4.grid(True)

    # Altitude (Z) tracking
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(time, true_states[:, 2], 'g-', label='True Z', linewidth=2)
    ax5.scatter(time, measurements[:, 2], c='r', s=1, alpha=0.3, label='GPS Z')
    ax5.plot(time, estimates[:, 2], 'b-', label='Est Z', linewidth=2)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Altitude (m)')
    ax5.set_title('Altitude Tracking')
    ax5.legend()
    ax5.grid(True)

    # Velocity estimation (not directly measured!)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(time, vel_errors, 'r-', linewidth=2)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Velocity Error (m/s)')
    ax6.set_title('Velocity Error (Not Directly Measured!)')
    ax6.grid(True)

    plt.tight_layout()
    out_path = os.path.join(_SCRIPT_DIR, 'kalman_6d_tracking.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as '{out_path}'")
    plt.show()

    return kf, true_states, estimates


if __name__ == "__main__":
    # Run the 6D tracking example
    kf, true_states, estimates = run_6d_tracking()

    # Print final state
    print("\nFinal Estimated State:")
    print(f"Position: [{kf.x[0,0]:.2f}, {kf.x[1,0]:.2f}, {kf.x[2,0]:.2f}] m")
    print(f"Velocity: [{kf.x[3,0]:.2f}, {kf.x[4,0]:.2f}, {kf.x[5,0]:.2f}] m/s")

    # Print final covariance (uncertainty)
    print("\nFinal Position Uncertainty (std dev):")
    print(f"σ_x = {np.sqrt(kf.P[0,0]):.3f} m")
    print(f"σ_y = {np.sqrt(kf.P[1,1]):.3f} m")
    print(f"σ_z = {np.sqrt(kf.P[2,2]):.3f} m")

    print("\nFinal Velocity Uncertainty (std dev):")
    print(f"σ_vx = {np.sqrt(kf.P[3,3]):.3f} m/s")
    print(f"σ_vy = {np.sqrt(kf.P[4,4]):.3f} m/s")
    print(f"σ_vz = {np.sqrt(kf.P[5,5]):.3f} m/s")
