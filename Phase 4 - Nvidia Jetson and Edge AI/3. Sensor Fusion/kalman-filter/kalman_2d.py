import os
import numpy as np
import matplotlib.pyplot as plt

# Save plots next to this script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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
    out_path = os.path.join(_SCRIPT_DIR, '2d_tracking.png')
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved as '{out_path}'")
    plt.show()


if __name__ == "__main__":
    simulate_2d_tracking()