"""CPU experiment: known-board pose -> delayed position + wheel-speed fusion.

Original synthetic teaching fixture, under this repository's MIT license.
No camera, network, GPU, downloaded weights, or physical robot required.
"""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import sys

import cv2
import numpy as np

from contracts import Measurement, visible_measurements

K = np.array([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]])
DIST = np.array([-.05, .01, .001, -.001, 0.])
BOARD = np.array([(x, y, 0.) for y in np.linspace(-.3, .3, 5)
                  for x in np.linspace(-.4, .4, 6)], dtype=np.float64)
BOARD_WORLD_Z = 8.


def true_state(t):
    return np.array([.45 * t + .08 * np.sin(t), .45 + .08 * np.cos(t)])


def project_board(position, rng, pixel_noise=.35):
    distance = BOARD_WORLD_Z - position
    if distance <= 0 or pixel_noise < 0:
        raise ValueError('board must be in front of camera; noise must be nonnegative')
    pixels, _ = cv2.projectPoints(BOARD, np.zeros(3), np.array([0., 0., distance]), K, DIST)
    return pixels.reshape(-1, 2) + rng.normal(0., pixel_noise, (len(BOARD), 2))


def recover_position(pixels, board_scale=1.):
    """solvePnP gives board-to-camera transform, not camera-to-board pose."""
    pixels = np.asarray(pixels, dtype=np.float64)
    if pixels.shape != (len(BOARD), 2) or not np.all(np.isfinite(pixels)):
        raise ValueError('need finite matching board corners in pixel coordinates')
    if not np.isfinite(board_scale) or board_scale <= 0:
        raise ValueError('board dimensions must be positive and expressed in metres')
    ok, rvec, tvec = cv2.solvePnP(BOARD * board_scale, pixels, K, DIST,
                                 flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or not np.all(np.isfinite(tvec)):
        raise ValueError('pose recovery failed')
    rotation, _ = cv2.Rodrigues(rvec)
    depths = ((rotation @ (BOARD * board_scale).T) + tvec)[2]
    if np.any(depths <= 0):
        raise ValueError('recovered corner behind camera')
    projected, _ = cv2.projectPoints(BOARD * board_scale, rvec, tvec, K, DIST)
    reprojection_rmse = np.sqrt(np.mean(np.sum((projected.reshape(-1, 2)-pixels)**2, axis=1)))
    # Camera centre in board coordinates is -R.T @ t, then board->world translation.
    camera_in_board = -rotation.T @ tvec
    return float(BOARD_WORLD_Z + camera_in_board[2, 0]), float(reprojection_rmse)


def predict(x, p, dt, q=.06):
    """Continuous white acceleration spectral density q, in m^2/s^3."""
    if dt < -1e-10 or q < 0:
        raise ValueError('cannot predict backward; nonnegative process spectral density required')
    dt = max(0., dt)
    f = np.array([[1., dt], [0., 1.]])
    qmat = q * np.array([[dt**3 / 3., dt**2 / 2.], [dt**2 / 2., dt]])
    return f @ x, f @ p @ f.T + qmat


def update(x, p, measurement, gate=9.):
    h = np.array([1., 0.]) if measurement.kind == 'camera' else np.array([0., 1.])
    innovation = measurement.value - h @ x
    s = float(h @ p @ h + measurement.variance)
    nis = float(innovation**2 / s)
    # Gate only camera observations; wheel bias is deliberately unmodelled.
    if measurement.kind == 'camera' and nis > gate:
        return x, p, nis, False
    gain = p @ h / s
    corrected = x + gain * innovation
    a = np.eye(2) - np.outer(gain, h)
    covariance = a @ p @ a.T + np.outer(gain, gain) * measurement.variance
    return corrected, (covariance + covariance.T) / 2., nis, True


def estimate(measurements, now, use_arrival_time=False, gate=9.):
    """Causal full-history replay from a fixed prior; not a fixed-lag implementation.

    Gate decisions are recomputed in timestamp order after delayed arrivals.
    Returned rejection count is for this reconstruction, not a cumulative total.
    """
    available = visible_measurements(measurements, now)
    if use_arrival_time:
        available.sort(key=lambda m: (m.arrival_time, m.sequence))
    x, p, previous = np.array([0., .45]), np.diag([.05**2, .1**2]), 0.
    rejected, camera_nis = 0, []
    for m in available:
        stamp = m.arrival_time if use_arrival_time else m.sample_time
        x, p = predict(x, p, stamp - previous)
        x, p, nis, accepted = update(x, p, m, gate)
        rejected += int(not accepted)
        if m.kind == 'camera':
            camera_nis.append(nis)
        previous = stamp
    x, p = predict(x, p, now - previous)
    return x, p, rejected, camera_nis


def generate(seed=11, delay=.6, pixel_noise=.35, board_scale=1., outliers=True, wheel_bias=.02):
    if not all(np.isfinite(v) for v in (delay, pixel_noise, board_scale, wheel_bias)):
        raise ValueError('finite configuration required')
    if delay < 0 or pixel_noise < 0 or board_scale <= 0:
        raise ValueError('invalid delay, pixel noise, or board scale')
    rng = np.random.default_rng(seed)
    times = np.arange(241, dtype=float) * .05
    observations, camera_rows = [], []
    for i, t in enumerate(times):
        truth = true_state(t)
        observations.append(Measurement(float(t), float(t), 'wheel',
                            float(truth[1] + wheel_bias + rng.normal(0, .025)), .025**2, 2*i))
        if i % 4 == 0:
            pixels = project_board(truth[0], rng, pixel_noise)
            z, residual = recover_position(pixels, board_scale)
            dropout = 5. <= t <= 7.
            injected_fault = outliers and i in (64, 172)
            if injected_fault:
                z += 1.5  # Inject a post-pose wrong-target/association fault.
            camera_rows.append({'sample_time': float(t), 'arrival_time': float(t+delay),
                                'position': z, 'truth': float(truth[0]),
                                'reprojection_rmse_px': residual,
                                'dropout': bool(dropout), 'injected_fault': bool(injected_fault)})
            if not dropout:
                observations.append(Measurement(float(t), float(t+delay), 'camera',
                                                z, .05**2, 2*i+1))
    return times, observations, camera_rows


def run(seed=11, delay=.6, pixel_noise=.35, board_scale=1., outliers=True, gate=9., wheel_bias=.02):
    times, observations, camera_rows = generate(seed, delay, pixel_noise, board_scale, outliers, wheel_bias)
    rows, summaries = [], {}
    for method in ('wheel_only', 'arrival_time', 'capture_time_replay'):
        events = [m for m in observations if m.kind == 'wheel'] if method == 'wheel_only' else observations
        errors, drop_errors, sigmas = [], [], []
        for t in times:
            x, p, rejected, nis = estimate(events, float(t), method == 'arrival_time', gate)
            truth = true_state(t)
            error = float(x[0] - truth[0])
            errors.append(error)
            sigmas.append(float(np.sqrt(p[0, 0])))
            if 5. <= t <= 7.:
                drop_errors.append(error)
            rows.append({'method': method, 'time': float(t), 'true_position': float(truth[0]),
                         'estimated_position': float(x[0]), 'estimated_velocity': float(x[1]),
                         'position_sigma': sigmas[-1], 'position_error': error,
                         'rejected_camera_in_reconstruction': rejected})
        summaries[method] = {'rmse_m': float(np.sqrt(np.mean(np.square(errors)))),
                             'dropout_rmse_m': float(np.sqrt(np.mean(np.square(drop_errors)))),
                             'final_error_m': errors[-1], 'final_position_sigma_m': sigmas[-1],
                             'final_rejected_camera': rejected,
                             'within_two_sigma_fraction': float(np.mean(np.abs(errors) <= 2*np.array(sigmas)))}
    clean_camera = [r for r in camera_rows if not r['dropout'] and not r['injected_fault']]
    report = {'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'opencv': cv2.__version__},
              'configuration': {'seed': seed, 'delay_s': delay, 'pixel_noise_px': pixel_noise,
                                'board_scale': board_scale, 'outliers': outliers, 'wheel_bias_m_s': wheel_bias,
                                'camera_nis_gate': gate if np.isfinite(gate) else None},
              'frame_contract': {'length_unit': 'metre', 'time_unit': 'second',
                                 'camera_axes': 'x right, y down, z forward',
                                 'world_axes': 'parallel to initial camera; board centre at (0,0,8)',
                                 'timestamp': 'capture and arrival in one synthetic monotonic clock'},
              'counts': {'outputs_per_method': len(times), 'generated_camera': len(camera_rows),
                         'dropout_camera': sum(r['dropout'] for r in camera_rows),
                         'camera_arrived_by_end': int(sum(m.kind == 'camera' and m.arrival_time <= times[-1]+1e-10 for m in observations))},
              'camera_clean_pose_rmse_m': float(np.sqrt(np.mean([(r['position']-r['truth'])**2 for r in clean_camera]))),
              'mean_reprojection_rmse_px': float(np.mean([r['reprojection_rmse_px'] for r in camera_rows])),
              'methods': summaries,
              'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ('camera_fusion.py', 'contracts.py')},
              'limitations': ['Synthetic known-board correspondences, not image detection or calibration.',
                              'Fixed camera variance is a teaching assumption, not a PnP uncertainty estimate.',
                              'Wheel bias is unmodelled; covariance is not a verified physical accuracy guarantee.',
                              'Full-history replay is causal but grows in cost; no runtime deadline claim.']}
    return report, rows, camera_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/camera-time-fusion'))
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--delay', type=float, default=.6)
    parser.add_argument('--pixel-noise', type=float, default=.35)
    parser.add_argument('--board-scale', type=float, default=1.)
    parser.add_argument('--wheel-bias', type=float, default=.02)
    parser.add_argument('--no-outliers', action='store_true')
    parser.add_argument('--no-gate', action='store_true')
    args = parser.parse_args()
    report, rows, camera_rows = run(args.seed, args.delay, args.pixel_noise, args.board_scale,
                                    not args.no_outliers, np.inf if args.no_gate else 9., args.wheel_bias)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    for name, data in [('states.csv', rows), ('camera.csv', camera_rows)]:
        with (args.output/name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    canvas = np.full((480, 640, 3), 240, dtype=np.uint8)
    for u, v in project_board(0., np.random.default_rng(args.seed), args.pixel_noise):
        cv2.circle(canvas, (round(u), round(v)), 3, (0, 0, 180), -1)
    cv2.imwrite(str(args.output/'synthetic-corners.png'), canvas)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
