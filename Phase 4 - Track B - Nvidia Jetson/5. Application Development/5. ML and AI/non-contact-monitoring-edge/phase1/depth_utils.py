"""Decode dataset disparity; metric scale must come from external evidence."""
import math
import numpy as np

FB = 32504.0
MIN_DEPTH = 40.0
MAX_DEPTH = 150.0


def gray_to_depth(gray, *, encoding="byte", units_to_meters=None):
    """Return dataset depth units, or metres with an explicitly supplied scale.

    byte: grayscale [0,255]; normalized: [0,1]. Never infer encoding from
    one frame's maximum. In the published mapping zero is far depth, not
    automatically a missing-pixel marker. Supply any validity mask separately.
    """
    p = np.asarray(gray, dtype=np.float64)
    if p.ndim != 2 or not p.size or not np.isfinite(p).all():
        raise ValueError("nonempty finite 2-D grayscale frame required")
    if encoding not in ("byte", "normalized"):
        raise ValueError("encoding must be byte or normalized")
    upper = 255 if encoding == "byte" else 1
    if np.any((p < 0) | (p > upper)):
        raise ValueError("grayscale outside declared encoding range")
    disparity = p / upper * (FB/MIN_DEPTH - FB/MAX_DEPTH) + FB/MAX_DEPTH
    result = FB / disparity
    if units_to_meters is not None:
        if not math.isfinite(units_to_meters) or units_to_meters <= 0:
            raise ValueError("metric scale must be finite and positive")
        result *= units_to_meters
    return result


def depth_in_roi(depth_map, bbox, valid_min=0, valid_max=float("inf"), use_median=True):
    """Median/mean of finite positive depth in clipped half-open xyxy box."""
    d = np.asarray(depth_map)
    if d.ndim != 2 or len(bbox) != 4 or not all(math.isfinite(v) for v in bbox):
        raise ValueError("2-D depth and finite xyxy box required")
    if not 0 <= valid_min < valid_max:
        raise ValueError("invalid depth bounds")
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    h, w = d.shape
    x1, x2 = max(0, math.floor(x1)), min(w, math.ceil(x2))
    y1, y2 = max(0, math.floor(y1)), min(h, math.ceil(y2))
    if x1 >= x2 or y1 >= y2:
        return None
    roi = d[y1:y2, x1:x2]
    vals = roi[np.isfinite(roi) & (roi > 0) & (roi >= valid_min) & (roi <= valid_max)]
    return None if not len(vals) else float(np.median(vals) if use_median else np.mean(vals))
