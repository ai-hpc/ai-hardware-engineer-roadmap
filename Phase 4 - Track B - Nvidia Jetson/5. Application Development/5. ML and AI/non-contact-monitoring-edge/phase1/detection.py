"""
Real-time object detection for Phase 1 pipeline.
Uses OpenCV built-in face (Haar) and person (HOG) detectors; no extra model files.
"""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

# (x1, y1, x2, y2), label, raw score (not calibrated probability)
Detection = tuple[tuple[int, int, int, int], str, float | None]


def _find_opencv_data() -> Path | None:
    """Try to locate OpenCV data dir for haarcascades."""
    for base in [Path(cv2.__file__).parent, Path(".")]:
        d = base / "data" / "haarcascade_frontalface_default.xml"
        if d.exists():
            return d.parent
        d = base / "share" / "opencv4" / "haarcascades" / "haarcascade_frontalface_default.xml"
        if d.exists():
            return d.parent
    return None


class FaceDetector:
    """OpenCV Haar cascade face detector (built-in)."""

    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5, min_size: tuple[int, int] = (30, 30)):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        data_dir = _find_opencv_data()
        if data_dir is None:
            raise FileNotFoundError(
                "OpenCV haarcascade not found. Install opencv-python or opencv-contrib-python."
            )
        path = data_dir / "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(str(path))
        if self._cascade.empty():
            raise RuntimeError(f"Failed to load cascade from {path}")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return list of (bbox, label, confidence). Bbox = (x1, y1, x2, y2)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        rects = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )
        out: list[Detection] = []
        for (x, y, w, h) in rects:
            bbox = (int(x), int(y), int(x + w), int(y + h))
            out.append((bbox, "face", None))
        return out


class PersonDetector:
    """OpenCV HOG person detector (built-in)."""

    def __init__(self, hit_threshold: float = 0.0, win_stride: tuple[int, int] = (8, 8)):
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.hit_threshold = hit_threshold
        self.win_stride = win_stride

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return list of (bbox, label, confidence)."""
        if frame.shape[0] < 128 or frame.shape[1] < 64:
            return []
        rects, weights = self._hog.detectMultiScale(
            frame,
            hitThreshold=self.hit_threshold,
            winStride=self.win_stride,
        )
        out: list[Detection] = []
        for i, (x, y, w, h) in enumerate(rects):
            conf = float(np.asarray(weights).reshape(-1)[i]) if weights is not None and len(weights) > i else None
            bbox = (int(x), int(y), int(x + w), int(y + h))
            out.append((bbox, "person", conf))
        return out


class CompositeDetector:
    """Keep face and person hypotheses; their boxes represent different objects."""
    def __init__(self, detect_face=True, detect_person=True):
        self._face = FaceDetector() if detect_face else None
        self._person = PersonDetector() if detect_person else None

    def detect(self, frame):
        results = []
        for detector in (self._face, self._person):
            if detector is not None:
                results.extend(detector.detect(frame))
        return results
