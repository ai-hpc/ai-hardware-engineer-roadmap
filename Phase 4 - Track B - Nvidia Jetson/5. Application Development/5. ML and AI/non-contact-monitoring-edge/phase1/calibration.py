"""Read the dataset's explicit camera-centre convention, failing on bad data."""
from pathlib import Path
import numpy as np


def load_paras_txt(paras_path):
    lines = [s.split() for s in Path(paras_path).read_text().splitlines() if s.strip()]
    if not lines or len(lines) % 5:
        raise ValueError("expected complete five-line camera records")
    labels = (("camera_id", 2), ("resolution", 3), ("K_matrix", 5), ("R_matrix", 10), ("world_position", 4))
    cameras, ids = [], set()
    for start in range(0, len(lines), 5):
        block = lines[start:start+5]
        if any(row[0] != key or len(row) != size for row, (key, size) in zip(block, labels)):
            raise ValueError("invalid calibration record fields")
        cid = int(block[0][1]); w, h = map(int, block[1][1:])
        fx, fy, cx, cy = map(float, block[2][1:])
        R = np.asarray(block[3][1:], dtype=float).reshape(3, 3)
        centre = np.asarray(block[4][1:], dtype=float)
        if cid in ids or w <= 0 or h <= 0 or fx <= 0 or fy <= 0:
            raise ValueError("duplicate camera or invalid dimensions/focal length")
        if not np.isfinite([fx, fy, cx, cy]).all() or not np.isfinite(R).all() or not np.isfinite(centre).all():
            raise ValueError("nonfinite calibration")
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R), 1, atol=1e-5):
            raise ValueError("R must be a proper rotation")
        ids.add(cid)
        cameras.append({"camera_id": cid, "width": w, "height": h,
                        "K": np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]]), "R": R, "t": centre})
    return cameras


def get_camera(cameras, camera_index=0):
    if not 0 <= camera_index < len(cameras):
        raise IndexError("camera index outside calibration records")
    return cameras[camera_index]


def get_intrinsics(camera):
    return camera["K"], camera["width"], camera["height"]


def project_world_to_image(K, R, t, point_3d):
    """Xc=R@(Xw-C); t is camera centre C, not OpenCV translation -R@C.

    Inputs are distortion-free. Reject points at/behind the camera instead
    of hiding the invalid projection with an epsilon denominator.
    """
    p = np.asarray(point_3d, dtype=float)
    single = p.ndim == 1
    p = np.atleast_2d(p)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("expected finite [3] or [N,3] points")
    Xc = (np.asarray(R) @ (p - np.asarray(t)).T).T
    if np.any(Xc[:, 2] <= 0):
        raise ValueError("point at/behind camera plane")
    Xp = (np.asarray(K) @ Xc.T).T
    uv = Xp[:, :2] / Xp[:, 2:3]
    return uv[0] if single else uv
