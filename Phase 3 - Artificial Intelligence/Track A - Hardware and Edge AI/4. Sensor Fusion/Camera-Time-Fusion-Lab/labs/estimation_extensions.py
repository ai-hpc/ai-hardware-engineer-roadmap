"""Small NumPy kernels for Kalman lessons 12–20; explicit teaching assumptions."""
import numpy as np


def ekf_range_update(x, p, measured_range, variance):
    """One 2D range measurement to the origin, no motion prediction."""
    x, p = np.asarray(x, dtype=float), np.asarray(p, dtype=float)
    distance = np.linalg.norm(x)
    if x.shape != (2,) or p.shape != (2, 2) or distance < 1e-9 or variance <= 0:
        raise ValueError('range Jacobian requires nonzero 2D position and positive variance')
    h = x / distance
    innovation = measured_range - distance
    s = h @ p @ h + variance
    k = p @ h / s
    a = np.eye(2) - np.outer(k, h)
    return x + k * innovation, a @ p @ a.T + np.outer(k, k) * variance


def unscented_square(mean, variance):
    """Scalar unscented transform of x^2, alpha=1, beta=2, kappa=0."""
    if variance <= 0:
        raise ValueError('positive input variance required')
    points = np.array([mean, mean + np.sqrt(variance), mean - np.sqrt(variance)])
    transformed = points**2
    mean_y = np.array([0., .5, .5]) @ transformed
    var_y = np.array([2., .5, .5]) @ (transformed - mean_y)**2
    return float(mean_y), float(var_y)


def wrap_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


def yaw_error_update(nominal, variance, measured, measurement_variance):
    """SO(2) example; a scalar yaw error is not a complete 3D ESKF."""
    if variance <= 0 or measurement_variance <= 0:
        raise ValueError('positive variances required')
    innovation = wrap_angle(measured - nominal)
    gain = variance / (variance + measurement_variance)
    return wrap_angle(nominal + gain * innovation), (1-gain) * variance


def nullspace_project(residual, state_jacobian, feature_jacobian, covariance):
    """Whiten then remove feature error from a linearized residual via SVD."""
    residual = np.asarray(residual, dtype=float)
    hs, hf = np.asarray(state_jacobian, dtype=float), np.asarray(feature_jacobian, dtype=float)
    chol = np.linalg.cholesky(covariance)
    rw, hsw, hfw = (np.linalg.solve(chol, a) for a in (residual, hs, hf))
    u, singular, _ = np.linalg.svd(hfw, full_matrices=True)
    tolerance = max(hfw.shape) * np.finfo(float).eps * (singular[0] if len(singular) else 0.)
    rank = int(np.sum(singular > tolerance))
    left_null = u[:, rank:].T
    return left_null @ rw, left_null @ hsw, left_null


def rts_scalar(filtered_mean, filtered_variance, predicted_mean, predicted_variance, transition):
    """predicted[t+1] is the forward prediction from filtered[t]; index 0 unused."""
    means, variances = np.array(filtered_mean, dtype=float), np.array(filtered_variance, dtype=float)
    if np.any(variances <= 0) or np.any(np.asarray(predicted_variance)[1:] <= 0):
        raise ValueError('positive covariance required')
    for t in range(len(means)-2, -1, -1):
        gain = filtered_variance[t] * transition[t] / predicted_variance[t+1]
        means[t] += gain * (means[t+1] - predicted_mean[t+1])
        variances[t] += gain**2 * (variances[t+1] - predicted_variance[t+1])
    return means, variances
