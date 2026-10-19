"""Uniformly sampled synthetic signal exercise; not a physiology estimator."""
import numpy as np


def peak_in_band(times, values, low_hz=.8, high_hz=3.):
    t, y = np.asarray(times, dtype=float), np.asarray(values, dtype=float)
    if t.ndim != 1 or t.shape != y.shape or len(t) < 8 or not np.isfinite(t).all() or not np.isfinite(y).all():
        raise ValueError("aligned finite 1-D time and signal arrays required")
    dt = np.diff(t)
    if np.any(dt <= 0) or not np.allclose(dt, dt.mean(), rtol=1e-3, atol=1e-9):
        raise ValueError("uniform increasing timestamps required; do not silently use average FPS")
    fs = 1 / dt.mean()
    if not 0 < low_hz < high_hz < fs/2:
        raise ValueError("band must lie strictly below Nyquist")
    centered = y - np.mean(y)
    if np.std(centered) < 1e-12:
        raise ValueError("constant signal has no identifiable peak")
    power = np.abs(np.fft.rfft(centered * np.hanning(len(y)))) ** 2
    frequencies = np.fft.rfftfreq(len(y), d=dt.mean())
    inside = (frequencies >= low_hz) & (frequencies <= high_hz)
    if not inside.any():
        raise ValueError("record too short for requested frequency band")
    peak = float(frequencies[inside][np.argmax(power[inside])])
    return {"peak_hz": peak, "sample_rate_hz": float(fs), "bin_spacing_hz": float(fs/len(t)),
            "samples": len(t), "scope": "spectral peak; no physiological or thermal validation"}
