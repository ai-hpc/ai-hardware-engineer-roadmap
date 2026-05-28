# Multi-Object Tracking: Hungarian Algorithm + Kalman Filter

<div class="course-identity auto-course" style="--course-accent: #475569; --course-accent-rgb: 71, 85, 105;" markdown="1">
<div class="course-identity__icon">MOTH</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · AI Workloads</p>
<p class="course-identity__title">Specialized course identity for Multi-Object Tracking: Hungarian Algorithm + Kalman Filter.</p>
<p class="course-identity__meta">Artifact: model or workload study · Measure: accuracy, latency, memory, throughput</p>
</div>
</div>


> **Reference implementation:** [srianant/kalman_filter_multi_object_tracking](https://github.com/srianant/kalman_filter_multi_object_tracking)
> **Goal:** Understand and build a complete MOT system from scratch — Kalman filter for prediction, Hungarian algorithm for assignment, track lifecycle management — then integrate with YOLO detection, BEVFusion 3D output, and ROS2.

---


## 1. The Multi-Object Tracking Problem

### Detection vs Tracking

A **detector** answers: "What objects are in this frame?"
A **tracker** answers: "Which object in this frame is the same object I saw last frame?"

```
Frame 1:   [car_A at (100,200)]  [car_B at (400,300)]  [person_C at (250,180)]
Frame 2:   [det_1 at (110,205)]  [det_2 at (415,305)]  [det_3 at (258,185)]

Question:  det_1 = car_A? det_2 = car_B? det_3 = person_C?
           How do we match consistently?
```

Without tracking:
- Each frame gives independent detections with no identity
- Impossible to measure velocity, predict position, count unique objects

With tracking:
- Each object gets a persistent ID across frames
- Velocity and heading are estimated
- Occlusions and missed detections are handled gracefully
- Downstream: path planning, collision avoidance, behavior prediction

### The Two Core Sub-Problems

```
1. Motion prediction (Kalman Filter):
   "Given where object X was, where will it be in the next frame?"
   Used to narrow the search region and build the cost matrix.

2. Data association (Hungarian Algorithm):
   "Given N predictions and M detections, which pairs match?"
   Globally optimal assignment that minimizes total cost.
```

---

## 2. System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│  Frame t                                                        │
│                                                                 │
│  [Detection]    boxes = detector(frame)                         │
│      │          [(x1,y1,x2,y2,score,class), ...]               │
│      ▼                                                          │
│  [Prediction]   for each track: KF.predict()                   │
│                 predicted positions from last frame             │
│      │                                                          │
│      ▼                                                          │
│  [Cost Matrix]  distance(prediction_i, detection_j)            │
│                 shape: [N_tracks × M_detections]                │
│      │                                                          │
│      ▼                                                          │
│  [Assignment]   Hungarian algorithm → optimal matching          │
│                 matched pairs, unmatched tracks, unmatched dets │
│      │                                                          │
│      ▼                                                          │
│  [Update]       for matched: KF.update(detection)               │
│                 for unmatched tracks: age++, maybe delete       │
│                 for unmatched dets:  create new track           │
│      │                                                          │
│      ▼                                                          │
│  [Output]       confirmed tracks with IDs and states           │
└────────────────────────────────────────────────────────────────┘
```

---

## 3. Kalman Filter for Object Motion

### Why Kalman Filter?

Detectors are noisy — the bounding box for a car jitters frame-to-frame even if the car moves smoothly. The Kalman filter:
- **Predicts** where an object should be (based on physics model)
- **Corrects** the prediction with noisy measurements
- Maintains a **smooth, optimal estimate** of true position and velocity

### State Vector

For 2D bounding box tracking, the state includes position, size, and their velocities:

```
State: x = [cx, cy, s, r, ċx, ċy, ṡ]

cx, cy  : bounding box center (x, y)
s       : box area (scale)
r       : aspect ratio (w/h) — assumed constant
ċx, ċy  : velocity of center
ṡ       : velocity of scale (object growing/shrinking = approaching/receding)
```

### The Kalman Filter Equations

```
── Prediction step ──────────────────────────────────────
x̂_{k|k-1} = F · x̂_{k-1|k-1}          (state prediction)
P_{k|k-1}  = F · P_{k-1|k-1} · Fᵀ + Q (covariance prediction)

── Update step ──────────────────────────────────────────
y_k = z_k - H · x̂_{k|k-1}             (innovation: measurement minus prediction)
S_k = H · P_{k|k-1} · Hᵀ + R          (innovation covariance)
K_k = P_{k|k-1} · Hᵀ · S_k⁻¹          (Kalman gain)
x̂_{k|k}   = x̂_{k|k-1} + K_k · y_k    (state update)
P_{k|k}    = (I - K_k · H) · P_{k|k-1} (covariance update)

Matrices:
  F : state transition (constant velocity model)
  H : observation model (we observe [cx, cy, s, r] from detector)
  Q : process noise (how much we trust the motion model)
  R : measurement noise (how much we trust the detector)
  P : state covariance (current uncertainty)
  K : Kalman gain (how much to trust measurement vs prediction)
```

### Constant Velocity Model — F Matrix

```
State:  [cx, cy, s, r, ċx, ċy, ṡ]
         0   1   2  3   4    5   6

F = [[1, 0, 0, 0, dt, 0,  0 ],   cx  ← cx + ċx·dt
     [0, 1, 0, 0,  0, dt, 0 ],   cy  ← cy + ċy·dt
     [0, 0, 1, 0,  0,  0, dt],   s   ← s  + ṡ·dt
     [0, 0, 0, 1,  0,  0,  0],   r   ← r   (constant)
     [0, 0, 0, 0,  1,  0,  0],   ċx  ← ċx  (constant velocity)
     [0, 0, 0, 0,  0,  1,  0],   ċy  ← ċy
     [0, 0, 0, 0,  0,  0,  1]]   ṡ   ← ṡ

With dt=1 (frame-to-frame):
F = I + dt * [[0,0,0,0,1,0,0],
              [0,0,0,0,0,1,0],
              [0,0,0,0,0,0,1],
              ...]
```

### Observation Matrix — H

The detector gives us [cx, cy, s, r] — not velocities. H extracts these from state:

```
H = [[1, 0, 0, 0, 0, 0, 0],   observe cx
     [0, 1, 0, 0, 0, 0, 0],   observe cy
     [0, 0, 1, 0, 0, 0, 0],   observe s
     [0, 0, 0, 1, 0, 0, 0]]   observe r
```

---

## 4. The Hungarian Algorithm — Data Association

### The Assignment Problem

Given N tracks (with predicted positions) and M detections, find the **globally optimal one-to-one assignment** that minimizes total cost.

```
Example:
Tracks:     T1=(100,200)  T2=(400,300)  T3=(250,180)
Detections: D1=(110,205)  D2=(415,305)  D3=(258,185)

Cost matrix (Euclidean distance):
        D1      D2      D3
T1    11.18  317.5   158.1     ← T1 is close to D1
T2   306.1    19.4   166.4     ← T2 is close to D2
T3   147.6   174.9    12.0     ← T3 is close to D3

Greedy (wrong): T1→D1 (11.18), T2→D2 (19.4), T3→D3 (12.0) ✓ happens to work here
                but with more objects, greedy is suboptimal

Hungarian (always optimal):
  Find permutation of columns that minimizes sum of diagonal elements
  Result: T1→D1, T2→D2, T3→D3  (total cost: 42.58)
```

### How the Hungarian Algorithm Works

The Hungarian algorithm solves the linear sum assignment problem in O(n³):

```
Step 1: Row reduction — subtract row minimum from each row
Step 2: Column reduction — subtract column minimum from each column
Step 3: Cover all zeros with minimum number of lines
Step 4: If number of lines = n → optimal assignment found
Step 5: Otherwise: find minimum uncovered element,
        subtract from uncovered, add to double-covered, repeat from 3
```

In practice, use `scipy.optimize.linear_sum_assignment` — it implements this efficiently.

### Example: Why Greedy Fails

```
Cost matrix:                  Greedy:          Hungarian:
      D1   D2                 T1→D1 (1)  ✓    T1→D2 (4)
T1  [  1    4  ]              T2→D2 (2)  ✓    T2→D1 (3)
T2  [  3    2  ]              Total: 3         Total: 7  ← WAIT

Hmm, greedy wins here. But:

      D1   D2
T1  [  1   10  ]              Greedy: T1→D1(1) → T2→D2(10)  Total: 11
T2  [  2    3  ]              Hungarian: T1→D2(10)? No...
                               T1→D1(1), T2→D2(3)  Total: 4  ← Hungarian wins!

Real failure case (3×3):
      D1   D2   D3
T1  [  1    2    3 ]
T2  [  4    5    6 ]          Greedy: T1→D1(1), T2→D2(5), T3→D3(9)  = 15
T3  [  7    8    9 ]          Hungarian: same result here

      D1   D2   D3
T1  [  4    1    3 ]
T2  [  2    0    5 ]          Greedy: T2→D2(0), T1→D1(4), T3→D3(9)  = 13
T3  [  3    2    2 ]          Hungarian: T1→D2(1), T2→D1(2), T3→D3(2) = 5 ✓
```

---

## 5. Cost Matrix: Euclidean vs IoU

### Euclidean Distance (Position-based)

```python
import numpy as np

def euclidean_cost(predictions, detections):
    """
    predictions: [N, 2] — predicted (cx, cy) for each track
    detections:  [M, 2] — detected  (cx, cy) for each detection
    Returns cost matrix [N, M]
    """
    N, M = len(predictions), len(detections)
    cost = np.zeros((N, M))
    for i, pred in enumerate(predictions):
        for j, det in enumerate(detections):
            cost[i, j] = np.sqrt((pred[0]-det[0])**2 + (pred[1]-det[1])**2)
    return cost
    # Or vectorized:
    # return np.linalg.norm(predictions[:, None] - detections[None, :], axis=2)
```

**Good for:** position-only state, when boxes have similar sizes.
**Bad for:** objects at different scales (large car vs small pedestrian).

### IoU-Based Cost (Box Overlap)

IoU (Intersection over Union) measures how much two boxes overlap. Using `1 - IoU` as cost:

```python
def iou(box_a, box_b):
    """
    box_a, box_b: [x1, y1, x2, y2]
    Returns IoU ∈ [0, 1]
    """
    # Intersection
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter_area = inter_w * inter_h

    area_a = (box_a[2]-box_a[0]) * (box_a[3]-box_a[1])
    area_b = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])
    union_area = area_a + area_b - inter_area

    return inter_area / (union_area + 1e-6)

def iou_cost(predicted_boxes, detected_boxes):
    """
    predicted_boxes: [N, 4] — [x1,y1,x2,y2] for each track
    detected_boxes:  [M, 4] — [x1,y1,x2,y2] for each detection
    Returns cost matrix [N, M] where cost = 1 - IoU
    """
    N, M = len(predicted_boxes), len(detected_boxes)
    cost = np.ones((N, M))   # default: no overlap = cost 1.0
    for i, pb in enumerate(predicted_boxes):
        for j, db in enumerate(detected_boxes):
            cost[i, j] = 1.0 - iou(pb, db)
    return cost
```

**Good for:** accurate bounding boxes, objects of varying size.
**Bad for:** large displacement between frames (no overlap → cost always 1.0).

### Choosing Your Cost

| Scenario                             | Recommended Cost     |
|--------------------------------------|----------------------|
| Fast camera, small motion per frame  | IoU                  |
| Slow camera or large motion          | Euclidean + threshold|
| 3D tracking (BEVFusion output)       | 3D Euclidean         |
| Mixed (robust)                       | Combination          |

```python
def combined_cost(pred_boxes, det_boxes, alpha=0.5):
    """Weighted combination of Euclidean + IoU costs"""
    pred_centers = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in pred_boxes])
    det_centers  = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in det_boxes])

    # Normalize Euclidean to [0, 1] by diagonal of frame
    frame_diag = np.sqrt(1920**2 + 1080**2)
    euc = np.linalg.norm(pred_centers[:, None] - det_centers[None, :], axis=2) / frame_diag

    iou_c = iou_cost(pred_boxes, det_boxes)

    return alpha * euc + (1 - alpha) * iou_c
```

---

## 6. Track Lifecycle Management

### Track States

```
                 New detection (unmatched)
                        │
                        ▼
                  ┌──────────┐
                  │TENTATIVE │  hit_streak < min_hits
                  └──────────┘
                        │ matched for min_hits consecutive frames
                        ▼
                  ┌──────────┐
                  │CONFIRMED │  ← output to downstream
                  └──────────┘
                        │
             ┌──────────┴──────────┐
    matched  │                     │ unmatched for max_age frames
             ▼                     ▼
      stay CONFIRMED          ┌──────────┐
                              │ DELETED  │
                              └──────────┘
```

### Parameters

```python
MIN_HITS  = 3    # frames an object must be detected before being confirmed
MAX_AGE   = 5    # frames a track can survive without a detection
DIST_THRESHOLD = 50  # pixels — max cost to accept an assignment
```

**Why MIN_HITS?** Prevents false-positive detections from becoming tracks. A real object will appear in multiple consecutive frames; a spurious detection usually won't.

**Why MAX_AGE?** Objects can be temporarily occluded. The tracker keeps predicting for MAX_AGE frames, waiting for the object to reappear. If it doesn't, the track is deleted.

---

## 7. Complete Implementation (Python 3)

### kalman_filter.py

Based on the srianant reference, rewritten for Python 3 with `[cx, cy, s, r, ċx, ċy, ṡ]` state:

```python
# kalman_filter.py
import numpy as np

class KalmanFilter:
    """
    Kalman Filter for 2D bounding box tracking.

    State vector: [cx, cy, s, r, vcx, vcy, vs]
      cx, cy : bounding box center
      s      : scale (area = w * h)
      r      : aspect ratio (w / h) — assumed constant
      vcx    : velocity of cx
      vcy    : velocity of cy
      vs     : velocity of s (approaching/receding)

    Observation: [cx, cy, s, r]  (from detector)
    """

    def __init__(self):
        dt = 1.0   # time step between frames

        # State dimension = 7, observation dimension = 4
        self.dim_x = 7
        self.dim_z = 4

        # State transition matrix F (constant velocity model)
        self.F = np.array([
            [1, 0, 0, 0, dt, 0,  0 ],
            [0, 1, 0, 0,  0, dt, 0 ],
            [0, 0, 1, 0,  0,  0, dt],
            [0, 0, 0, 1,  0,  0,  0],
            [0, 0, 0, 0,  1,  0,  0],
            [0, 0, 0, 0,  0,  1,  0],
            [0, 0, 0, 0,  0,  0,  1],
        ], dtype=float)

        # Observation matrix H (observe position and size, not velocity)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=float)

        # Process noise covariance Q
        # Higher = trust motion model less, follow measurements more
        self.Q = np.eye(self.dim_x, dtype=float)
        self.Q[4:, 4:] *= 0.01   # velocities have lower process noise

        # Measurement noise covariance R
        # Higher = trust detector less
        self.R = np.eye(self.dim_z, dtype=float)
        self.R[2:, 2:] *= 10.0   # scale and ratio less reliable

        # Initial state covariance P
        self.P = np.eye(self.dim_x, dtype=float)
        self.P[4:, 4:] *= 1000.0 # high uncertainty for initial velocities

        # State estimate (initialized when first detection arrives)
        self.x = np.zeros((self.dim_x, 1), dtype=float)

    def initialize(self, measurement):
        """
        Initialize state from first detection.
        measurement: [cx, cy, s, r]
        """
        self.x[:4] = np.array(measurement, dtype=float).reshape(4, 1)
        # Velocities initialized to zero

    def predict(self):
        """
        Prediction step: project state forward.
        Returns predicted observation [cx, cy, s, r].
        """
        # x_{k|k-1} = F * x_{k-1|k-1}
        self.x = self.F @ self.x

        # P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Prevent scale from going negative
        if self.x[2] < 0:
            self.x[2] = 0

        return (self.H @ self.x).flatten()   # predicted [cx, cy, s, r]

    def update(self, measurement):
        """
        Update step: correct prediction with measurement.
        measurement: [cx, cy, s, r] from detector.
        """
        z = np.array(measurement, dtype=float).reshape(self.dim_z, 1)

        # Innovation: y = z - H * x
        y = z - self.H @ self.x

        # Innovation covariance: S = H * P * H^T + R
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain: K = P * H^T * S^{-1}
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update: x = x + K * y
        self.x = self.x + K @ y

        # Covariance update: P = (I - K * H) * P
        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P
```

### track.py

```python
# track.py
import numpy as np
from kalman_filter import KalmanFilter

class TrackState:
    TENTATIVE = 1
    CONFIRMED = 2
    DELETED   = 3

class Track:
    """
    Represents a single tracked object.

    Maintains:
      - Kalman filter for state estimation
      - Track ID (unique, monotonically increasing)
      - Hit streak (consecutive matched frames)
      - Time since last update (frames without detection)
      - Trace history (list of past positions)
    """
    _next_id = 1

    def __init__(self, detection, min_hits=3):
        """
        detection: [cx, cy, s, r]  (center_x, center_y, area, aspect_ratio)
        """
        self.kf = KalmanFilter()
        self.kf.initialize(detection)

        self.track_id   = Track._next_id
        Track._next_id += 1

        self.state      = TrackState.TENTATIVE
        self.hits       = 1          # total detection count
        self.hit_streak = 1          # consecutive hits
        self.age        = 1          # total frames alive
        self.time_since_update = 0   # frames since last detection

        self.min_hits = min_hits
        self.trace    = [detection[:2].copy()]  # (cx, cy) history

    def predict(self):
        """Advance Kalman filter one step. Call once per frame."""
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.kf.predict()

    def update(self, detection):
        """Update track with a matched detection."""
        self.kf.update(detection)
        self.hits       += 1
        self.hit_streak += 1
        self.time_since_update = 0
        self.trace.append(detection[:2].copy())

        # Promote tentative → confirmed
        if self.state == TrackState.TENTATIVE and self.hit_streak >= self.min_hits:
            self.state = TrackState.CONFIRMED

    def mark_missed(self):
        """Called when no detection matched this track."""
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.DELETED

    @property
    def is_confirmed(self):
        return self.state == TrackState.CONFIRMED

    @property
    def is_deleted(self):
        return self.state == TrackState.DELETED

    def get_state(self):
        """Return current Kalman state as bounding box [cx, cy, s, r]."""
        return self.kf.x[:4].flatten()

    def get_box_xyxy(self):
        """Convert state [cx, cy, s, r] → [x1, y1, x2, y2]."""
        cx, cy, s, r = self.get_state()
        s = max(s, 1e-6)
        w = np.sqrt(s * r)
        h = s / w
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
```

### tracker.py

```python
# tracker.py
import numpy as np
from scipy.optimize import linear_sum_assignment
from track import Track, TrackState

def box_to_obs(box_xyxy):
    """Convert [x1,y1,x2,y2] → [cx, cy, s, r] for Kalman filter."""
    x1, y1, x2, y2 = box_xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w  = x2 - x1
    h  = y2 - y1
    s  = w * h          # area
    r  = w / float(h)   # aspect ratio
    return np.array([cx, cy, s, r])

def iou(box_a, box_b):
    """IoU between two boxes [x1,y1,x2,y2]."""
    ix1 = max(box_a[0], box_b[0]);  iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2]);  iy2 = min(box_a[3], box_b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    area_a = (box_a[2]-box_a[0]) * (box_a[3]-box_a[1])
    area_b = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])
    return inter / (area_a + area_b - inter + 1e-6)

def iou_cost_matrix(predicted_boxes, detected_boxes):
    """Cost matrix using 1 - IoU. Shape: [N_tracks, M_dets]."""
    N = len(predicted_boxes)
    M = len(detected_boxes)
    cost = np.ones((N, M), dtype=float)
    for i, pb in enumerate(predicted_boxes):
        for j, db in enumerate(detected_boxes):
            cost[i, j] = 1.0 - iou(pb, db)
    return cost

class Tracker:
    """
    Multi-object tracker using Kalman Filter + Hungarian Algorithm.

    Reference: github.com/srianant/kalman_filter_multi_object_tracking
    Modernized to Python 3, IoU-based cost, cleaner track lifecycle.
    """

    def __init__(self,
                 max_age=5,
                 min_hits=3,
                 iou_threshold=0.3):
        """
        max_age:       max frames a track survives without detection
        min_hits:      min consecutive detections before track is confirmed
        iou_threshold: min IoU for a valid assignment (cost < 1 - iou_threshold)
        """
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: list[Track] = []
        self.frame_count = 0

    def update(self, detections):
        """
        Run one tracking step.

        detections: list of [x1, y1, x2, y2, score, class_id]
                    or numpy array shape [M, 6]

        Returns: list of confirmed tracks as
                 [x1, y1, x2, y2, track_id, class_id]
        """
        self.frame_count += 1
        detections = np.array(detections, dtype=float) if len(detections) else np.zeros((0, 6))

        # ── Step 1: Predict ────────────────────────────────────────────
        predicted_boxes = []
        for track in self.tracks:
            pred = track.predict()          # KF predict → [cx, cy, s, r]
            # Convert back to xyxy for IoU cost
            cx, cy, s, r = pred
            s = max(s, 1.0)
            w = np.sqrt(s * r)
            h = s / w
            predicted_boxes.append([cx-w/2, cy-h/2, cx+w/2, cy+h/2])

        # ── Step 2: Build cost matrix ──────────────────────────────────
        if len(self.tracks) > 0 and len(detections) > 0:
            det_boxes = detections[:, :4]   # [M, 4]
            cost = iou_cost_matrix(predicted_boxes, det_boxes)

            # ── Step 3: Hungarian assignment ───────────────────────────
            row_ind, col_ind = linear_sum_assignment(cost)

            # ── Step 4: Filter out low-IoU assignments ─────────────────
            matched_tracks, matched_dets = [], []
            unmatched_tracks = list(range(len(self.tracks)))
            unmatched_dets   = list(range(len(detections)))

            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < (1.0 - self.iou_threshold):
                    matched_tracks.append(r)
                    matched_dets.append(c)
                    if r in unmatched_tracks: unmatched_tracks.remove(r)
                    if c in unmatched_dets:   unmatched_dets.remove(c)
        else:
            matched_tracks, matched_dets = [], []
            unmatched_tracks = list(range(len(self.tracks)))
            unmatched_dets   = list(range(len(detections)))

        # ── Step 5: Update matched tracks ─────────────────────────────
        for t_idx, d_idx in zip(matched_tracks, matched_dets):
            obs = box_to_obs(detections[d_idx, :4])
            self.tracks[t_idx].update(obs)

        # ── Step 6: Handle unmatched tracks ───────────────────────────
        for t_idx in unmatched_tracks:
            self.tracks[t_idx].mark_missed()

        # ── Step 7: Create new tracks for unmatched detections ─────────
        for d_idx in unmatched_dets:
            obs = box_to_obs(detections[d_idx, :4])
            new_track = Track(obs, min_hits=self.min_hits)
            self.tracks.append(new_track)

        # ── Step 8: Delete dead tracks ─────────────────────────────────
        self.tracks = [t for t in self.tracks
                       if not t.is_deleted
                       and t.time_since_update <= self.max_age]

        # ── Step 9: Collect confirmed track outputs ────────────────────
        outputs = []
        for track in self.tracks:
            if track.is_confirmed:
                box = track.get_box_xyxy()
                outputs.append([
                    box[0], box[1], box[2], box[3],
                    track.track_id,
                    detections[matched_dets[matched_tracks.index(
                        self.tracks.index(track))], 5]
                    if self.tracks.index(track) in matched_tracks else -1
                ])

        return outputs
```

### Simple output helper

```python
def get_confirmed_tracks(tracker_outputs):
    """
    Format tracker outputs for display/downstream use.
    Returns list of dicts.
    """
    results = []
    for t in tracker_outputs:
        results.append({
            'x1': int(t[0]), 'y1': int(t[1]),
            'x2': int(t[2]), 'y2': int(t[3]),
            'id': int(t[4]),
            'class': int(t[5]) if t[5] >= 0 else -1,
        })
    return results
```

---

## 8. Integration with YOLO Detections

### End-to-End: YOLO + Kalman-Hungarian Tracker

```python
# yolo_tracker.py
import cv2
import numpy as np
import time
from tracker import Tracker
from ultralytics import YOLO    # pip install ultralytics

class YOLOTracker:
    def __init__(self, model_path='yolov8n.pt', device='cuda'):
        self.detector = YOLO(model_path)
        self.detector.to(device)
        self.tracker  = Tracker(max_age=5, min_hits=3, iou_threshold=0.3)

        # Assign consistent colors per track ID
        self._colors = {}

    def _get_color(self, track_id):
        if track_id not in self._colors:
            np.random.seed(track_id * 7)
            self._colors[track_id] = tuple(np.random.randint(50, 255, 3).tolist())
        return self._colors[track_id]

    def run_frame(self, frame):
        """
        Run YOLO detection + tracking on a single frame.
        Returns frame with drawn boxes + track info.
        """
        t0 = time.perf_counter()

        # YOLO detection
        results = self.detector(frame, verbose=False)[0]
        boxes   = results.boxes

        # Format detections: [x1, y1, x2, y2, score, class_id]
        dets = []
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                score   = float(box.conf[0])
                class_id = int(box.cls[0])
                if score > 0.3:
                    dets.append([x1, y1, x2, y2, score, class_id])

        # Tracker update
        tracks = self.tracker.update(dets)

        dt = (time.perf_counter() - t0) * 1000

        # Draw results
        for t in tracks:
            x1, y1, x2, y2, tid, cls = int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[4]), int(t[5])
            color = self._get_color(tid)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{tid}"
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw trace (past positions)
            track_obj = next((tr for tr in self.tracker.tracks if tr.track_id == tid), None)
            if track_obj and len(track_obj.trace) > 1:
                for i in range(1, len(track_obj.trace)):
                    p1 = (int(track_obj.trace[i-1][0]), int(track_obj.trace[i-1][1]))
                    p2 = (int(track_obj.trace[i][0]),   int(track_obj.trace[i][1]))
                    cv2.line(frame, p1, p2, color, 1)

        cv2.putText(frame, f"{dt:.1f}ms | {len(tracks)} tracks",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        return frame

# Run on webcam / video
tracker = YOLOTracker('yolov8n.pt')
cap = cv2.VideoCapture(0)    # 0 = webcam, or path to video file

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = tracker.run_frame(frame)
    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()
```

### TensorRT-Accelerated YOLO + Tracker on Jetson

```python
# jetson_tracker.py — use TensorRT engine instead of PyTorch YOLO
from tracker import Tracker
import numpy as np
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

class JetsonYOLOTracker:
    """
    On Orin Nano: YOLO inference via TensorRT FP16 + Kalman-Hungarian tracker.
    Targets >25 FPS for 640×640 input.
    """
    def __init__(self, engine_path, input_size=(640, 640)):
        self.input_size = input_size
        self.tracker = Tracker(max_age=5, min_hits=2, iou_threshold=0.3)

        # Load TensorRT engine
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self._allocate_buffers()

    def _allocate_buffers(self):
        self.stream = cuda.Stream()
        self.inputs, self.outputs, self.bindings = [], [], []
        for binding in self.engine:
            size  = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host  = cuda.pagelocked_empty(size, dtype)
            dev   = cuda.mem_alloc(host.nbytes)
            self.bindings.append(int(dev))
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host, 'device': dev})
            else:
                self.outputs.append({'host': host, 'device': dev})

    def preprocess(self, frame):
        img = cv2.resize(frame, self.input_size)
        img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.ascontiguousarray(img[None])   # [1, 3, H, W]

    def infer(self, data):
        np.copyto(self.inputs[0]['host'], data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        return self.outputs[0]['host'].copy()

    def postprocess(self, raw_output, orig_shape, conf_thresh=0.3):
        """Parse YOLO output format [1, 84, 8400] → list of [x1,y1,x2,y2,score,class]."""
        # raw_output shape depends on YOLO version — adjust as needed
        output = raw_output.reshape(84, -1).T   # [8400, 84]
        boxes_cx = output[:, 0]; boxes_cy = output[:, 1]
        boxes_w  = output[:, 2]; boxes_h  = output[:, 3]
        class_scores = output[:, 4:]

        scores    = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)
        mask = scores > conf_thresh

        cx, cy, w, h = boxes_cx[mask], boxes_cy[mask], boxes_w[mask], boxes_h[mask]
        sc, cls      = scores[mask], class_ids[mask]

        # Scale to original image
        sx = orig_shape[1] / self.input_size[1]
        sy = orig_shape[0] / self.input_size[0]
        x1 = (cx - w/2) * sx;  x2 = (cx + w/2) * sx
        y1 = (cy - h/2) * sy;  y2 = (cy + h/2) * sy

        return np.stack([x1, y1, x2, y2, sc, cls.astype(float)], axis=1)

    def run_frame(self, frame):
        inp  = self.preprocess(frame)
        raw  = self.infer(inp)
        dets = self.postprocess(raw, frame.shape)
        return self.tracker.update(dets)
```

---

## 9. 3D Object Tracking (BEVFusion Output)

Extend the tracker to 3D state for BEVFusion detections. The 3D state includes position, size, heading, and their velocities.

### 3D State Vector

```
State: [x, y, z, w, l, h, θ, vx, vy, vz]

x, y, z  : center position (metric)
w, l, h  : box dimensions
θ        : heading angle (yaw)
vx, vy, vz: velocities
```

### 3D Kalman Filter

```python
# kalman_filter_3d.py
import numpy as np

class KalmanFilter3D:
    """
    Kalman filter for 3D bounding box tracking (BEVFusion output).
    State: [x, y, z, w, l, h, theta, vx, vy, vz]
    Observation: [x, y, z, w, l, h, theta]
    """
    def __init__(self):
        self.dim_x = 10
        self.dim_z = 7

        dt = 0.1    # 10 Hz LiDAR scan rate

        # State transition: position += velocity * dt, rest constant
        self.F = np.eye(self.dim_x)
        self.F[0, 7] = dt   # x += vx * dt
        self.F[1, 8] = dt   # y += vy * dt
        self.F[2, 9] = dt   # z += vz * dt

        # Observation: observe [x,y,z,w,l,h,theta]
        self.H = np.zeros((self.dim_z, self.dim_x))
        self.H[:7, :7] = np.eye(7)

        # Noise matrices
        self.Q = np.eye(self.dim_x) * 0.1
        self.Q[7:, 7:] *= 0.01     # velocity noise

        self.R = np.eye(self.dim_z)
        self.R[3:6, 3:6] *= 0.5    # dimension noise

        self.P = np.eye(self.dim_x)
        self.P[7:, 7:] *= 100.0    # initial velocity uncertainty

        self.x = np.zeros((self.dim_x, 1))

    def initialize(self, detection):
        """detection: [x, y, z, w, l, h, theta]"""
        self.x[:7] = np.array(detection).reshape(7, 1)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return (self.H @ self.x).flatten()

    def update(self, detection):
        z = np.array(detection).reshape(self.dim_z, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim_x) - K @ self.H) @ self.P
```

### 3D Cost Matrix (Euclidean in BEV plane)

```python
def bev_distance_cost(predicted_3d, detected_3d):
    """
    Cost based on 2D (BEV) center distance — ignore Z for association.
    predicted_3d, detected_3d: arrays of [x, y, z, w, l, h, theta]
    """
    pred_xy = np.array([[p[0], p[1]] for p in predicted_3d])   # [N, 2]
    det_xy  = np.array([[d[0], d[1]] for d in detected_3d])    # [M, 2]

    # [N, M] Euclidean distance in BEV plane
    diff = pred_xy[:, None, :] - det_xy[None, :, :]   # [N, M, 2]
    return np.linalg.norm(diff, axis=2)

# Max distance threshold for 3D association
MAX_3D_DIST = 5.0   # meters — objects must be within 5m of prediction
```

### 3D Tracker — Full Implementation

```python
# tracker_3d.py
import numpy as np
from scipy.optimize import linear_sum_assignment
from kalman_filter_3d import KalmanFilter3D

class Track3D:
    _next_id = 1

    def __init__(self, detection, class_id, min_hits=2):
        # detection: [x, y, z, w, l, h, theta]
        self.kf       = KalmanFilter3D()
        self.kf.initialize(detection)
        self.track_id = Track3D._next_id
        Track3D._next_id += 1
        self.class_id = class_id
        self.hits     = 1
        self.hit_streak = 1
        self.time_since_update = 0
        self.min_hits = min_hits
        self.confirmed = False
        self.trace = [detection[:3].copy()]   # xyz history

    def predict(self):
        self.time_since_update += 1
        if self.time_since_update > 1:
            self.hit_streak = 0
        return self.kf.predict()

    def update(self, detection):
        self.kf.update(detection[:7])
        self.hits       += 1
        self.hit_streak += 1
        self.time_since_update = 0
        self.trace.append(self.kf.x[:3].flatten().copy())
        if self.hit_streak >= self.min_hits:
            self.confirmed = True

    def get_state(self):
        return self.kf.x[:7].flatten()  # [x,y,z,w,l,h,theta]

class Tracker3D:
    def __init__(self, max_age=3, min_hits=2, max_dist=5.0):
        self.max_age  = max_age
        self.min_hits = min_hits
        self.max_dist = max_dist
        self.tracks: list[Track3D] = []

    def update(self, detections):
        """
        detections: list of {'box': [x,y,z,w,l,h,theta], 'class': int, 'score': float}
        Returns: list of confirmed tracks with IDs
        """
        # Predict
        predicted = [t.predict() for t in self.tracks]

        # Cost matrix
        if self.tracks and detections:
            pred_states = [p for p in predicted]
            det_states  = [d['box'] for d in detections]
            cost = bev_distance_cost(pred_states, det_states)

            # Hungarian assignment
            row_ind, col_ind = linear_sum_assignment(cost)

            matched_t, matched_d = [], []
            unmatched_t = list(range(len(self.tracks)))
            unmatched_d = list(range(len(detections)))

            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < self.max_dist:
                    matched_t.append(r)
                    matched_d.append(c)
                    unmatched_t.remove(r)
                    unmatched_d.remove(c)
        else:
            matched_t, matched_d = [], []
            unmatched_t = list(range(len(self.tracks)))
            unmatched_d = list(range(len(detections)))

        # Update matched
        for t_idx, d_idx in zip(matched_t, matched_d):
            self.tracks[t_idx].update(detections[d_idx]['box'])

        # Mark unmatched
        for t_idx in unmatched_t:
            pass   # time_since_update already incremented in predict()

        # New tracks
        for d_idx in unmatched_d:
            t = Track3D(detections[d_idx]['box'],
                        detections[d_idx]['class'],
                        min_hits=self.min_hits)
            self.tracks.append(t)

        # Delete old tracks
        self.tracks = [t for t in self.tracks
                       if t.time_since_update <= self.max_age]

        # Return confirmed tracks
        results = []
        for t in self.tracks:
            if t.confirmed:
                state = t.get_state()
                results.append({
                    'id':      t.track_id,
                    'class':   t.class_id,
                    'x': state[0], 'y': state[1], 'z': state[2],
                    'w': state[3], 'l': state[4], 'h': state[5],
                    'theta':   state[6],
                    'vx': t.kf.x[7,0], 'vy': t.kf.x[8,0],
                    'trace':   t.trace,
                })
        return results
```

---

## 10. ROS2 Integration

### Multi-Object Tracking ROS2 Node

```python
#!/usr/bin/env python3
# mot_node.py

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray, Detection3DArray
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import numpy as np
from tracker import Tracker
from tracker_3d import Tracker3D

class MOTNode(Node):
    """
    Multi-object tracking node.
    Subscribes to Detection2DArray (from YOLO) or Detection3DArray (from BEVFusion).
    Publishes tracked objects with persistent IDs.
    """
    def __init__(self):
        super().__init__('mot_node')

        self.declare_parameter('mode', '3d')       # '2d' or '3d'
        self.declare_parameter('max_age', 5)
        self.declare_parameter('min_hits', 3)
        self.declare_parameter('iou_threshold', 0.3)
        self.declare_parameter('max_dist_3d', 5.0)

        mode      = self.get_parameter('mode').value
        max_age   = self.get_parameter('max_age').value
        min_hits  = self.get_parameter('min_hits').value

        if mode == '2d':
            self.tracker = Tracker(max_age=max_age, min_hits=min_hits,
                                   iou_threshold=self.get_parameter('iou_threshold').value)
            self.sub = self.create_subscription(
                Detection2DArray, '/detections_2d', self.cb_2d, 10)
        else:
            self.tracker = Tracker3D(max_age=max_age, min_hits=min_hits,
                                     max_dist=self.get_parameter('max_dist_3d').value)
            self.sub = self.create_subscription(
                Detection3DArray, '/bevfusion/detections', self.cb_3d, 10)

        self.track_pub  = self.create_publisher(Detection3DArray, '/tracks', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/track_markers', 10)

        self.get_logger().info(f'MOT node started in {mode} mode')

    def cb_3d(self, msg):
        """Process BEVFusion Detection3DArray detections."""
        dets = []
        for det in msg.detections:
            c = det.bbox.center
            s = det.bbox.size
            # Extract yaw from quaternion
            import math
            qz = c.orientation.z
            qw = c.orientation.w
            yaw = 2 * math.atan2(qz, qw)

            dets.append({
                'box': [c.position.x, c.position.y, c.position.z,
                        s.x, s.y, s.z, yaw],
                'class': 0,
                'score': 1.0
            })

        tracks = self.tracker.update(dets)

        # Publish tracked detections as Detection3DArray
        out = Detection3DArray()
        out.header = msg.header
        # ... fill Detection3D messages from tracks ...
        self.track_pub.publish(out)

        # Publish RViz markers
        self.publish_markers(tracks, msg.header)

    def publish_markers(self, tracks, header):
        """Publish 3D bounding box markers and trace lines for RViz2."""
        markers = MarkerArray()

        for track in tracks:
            # Box marker
            box_marker = Marker()
            box_marker.header = header
            box_marker.ns     = 'tracks'
            box_marker.id     = track['id']
            box_marker.type   = Marker.CUBE
            box_marker.action = Marker.ADD

            box_marker.pose.position.x = track['x']
            box_marker.pose.position.y = track['y']
            box_marker.pose.position.z = track['z']

            import math
            yaw = track['theta']
            box_marker.pose.orientation.z = math.sin(yaw / 2)
            box_marker.pose.orientation.w = math.cos(yaw / 2)

            box_marker.scale.x = track['w']
            box_marker.scale.y = track['l']
            box_marker.scale.z = track['h']

            # Color by ID (deterministic)
            np.random.seed(track['id'] * 7)
            rgb = np.random.rand(3)
            box_marker.color = ColorRGBA(r=rgb[0], g=rgb[1], b=rgb[2], a=0.5)
            box_marker.lifetime.sec = 0
            box_marker.lifetime.nanosec = 200_000_000   # 0.2s

            markers.markers.append(box_marker)

            # Trace line marker
            if len(track['trace']) > 1:
                line = Marker()
                line.header = header
                line.ns     = 'traces'
                line.id     = track['id'] + 10000
                line.type   = Marker.LINE_STRIP
                line.action = Marker.ADD
                line.scale.x = 0.05   # line width
                line.color   = ColorRGBA(r=rgb[0], g=rgb[1], b=rgb[2], a=0.9)
                line.lifetime.sec = 0
                line.lifetime.nanosec = 200_000_000

                for pt in track['trace'][-20:]:   # last 20 positions
                    p = Point()
                    p.x, p.y, p.z = float(pt[0]), float(pt[1]), float(pt[2])
                    line.points.append(p)
                markers.markers.append(line)

        # Add DELETE markers for tracks that disappeared
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        # markers.markers.insert(0, delete_marker)  # uncomment to clear all first

        self.marker_pub.publish(markers)

def main():
    rclpy.init()
    rclpy.spin(MOTNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## 11. Evaluation Metrics (HOTA, MOTA, IDF1)

Understanding tracking metrics is essential for comparing trackers and tuning hyperparameters.

### MOTA (Multiple Object Tracking Accuracy)

```
MOTA = 1 - (FP + FN + IDSW) / GT

FP   : False Positives — tracker outputs a box where no object is
FN   : False Negatives — tracker misses a real object
IDSW : ID Switches     — tracker assigns wrong ID to a correctly detected object
GT   : total ground-truth object count

Range: (-∞, 1.0]  — can be negative with many errors
Higher = better
```

### IDF1

```
IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)

IDTP : ID True Positives  — correct detections with correct ID
IDFP : ID False Positives — detections with wrong ID
IDFN : ID False Negatives — missed detections

Measures: how consistently the correct ID is maintained
Range: [0, 1]
```

### HOTA (Higher Order Tracking Accuracy) — Preferred Modern Metric

```
HOTA = √(DetA × AssA)

DetA : Detection Accuracy    — how well objects are detected
AssA : Association Accuracy  — how consistently IDs are maintained

HOTA balances detection and association quality equally.
Range: [0, 1]
```

### Running Evaluation with TrackEval

```bash
pip install trackeval

# Convert your tracker output to MOT Challenge format:
# Each line: frame_id, track_id, x1, y1, w, h, score, -1, -1, -1

python3 -c "
import trackeval
evaluator = trackeval.Evaluator({'USE_PARALLEL': False, 'NUM_PARALLEL_CORES': 1})
dataset    = trackeval.datasets.MotChallenge2DBox({'GT_FOLDER': 'gt/', 'TRACKERS_FOLDER': 'trackers/'})
metrics    = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
evaluator.evaluate([dataset], metrics)
"
```

---

## 12. Advanced Trackers Overview

The Kalman + Hungarian tracker is the **baseline** (SORT tracker). Modern approaches improve on it:

### SORT (Simple Online and Realtime Tracking)

This guide's implementation. Fast, simple, good for offline use.
- **Pro:** fast, understandable, easy to tune
- **Con:** many ID switches during occlusion, no appearance features

### DeepSORT

SORT + **Re-ID (appearance embedding)** from a separate CNN:

```
Cost = α × IoU cost + (1-α) × appearance distance

Appearance distance = cosine distance of 128-dim embedding vectors

Pro: far fewer ID switches — appearance helps through occlusions
Con: needs a separate Re-ID model (adds latency)
```

### ByteTrack

Associates **both high and low confidence detections** in two stages:

```
Stage 1: associate confirmed tracks with HIGH conf detections (IoU)
Stage 2: associate remaining tracks with LOW conf detections
         (catches occluded objects that detector is uncertain about)

Pro: best MOTA on MOT17, very fast (no appearance model)
Con: slightly more FP from low-confidence detections
```

### OC-SORT (Observation-Centric SORT)

Fixes SORT's drift during occlusion by re-initializing velocity from observations rather than propagating stale KF state. Drop-in improvement over SORT.

### Tracker Comparison on MOT17

| Tracker     | HOTA  | MOTA  | IDF1  | FPS    |
|-------------|-------|-------|-------|--------|
| SORT        | 55.1  | 74.6  | 61.4  | 143    |
| DeepSORT    | 61.2  | 75.4  | 71.8  | 40     |
| ByteTrack   | 63.1  | 80.3  | 77.3  | 30     |
| OC-SORT     | 63.9  | 78.0  | 77.5  | 38     |
| StrongSORT  | 64.4  | 79.6  | 79.5  | 8      |

For Jetson Orin Nano: **ByteTrack** offers the best accuracy/speed tradeoff with no appearance model.

---

## 13. Projects

### Project 1: Reproduce srianant's Tracker
Clone [srianant/kalman_filter_multi_object_tracking](https://github.com/srianant/kalman_filter_multi_object_tracking), run it, then rewrite it in Python 3 using the code in this guide. Verify outputs match on the same video.

### Project 2: YOLO + Tracker Benchmark
Run four trackers on the same video (SORT, DeepSORT, ByteTrack, OC-SORT). Measure FPS and MOTA on a short labeled clip. Plot the accuracy-speed Pareto curve.

### Project 3: ID Switch Analysis
Intentionally cause ID switches by running a video with heavy occlusion. Log every ID switch. Implement the "re-identification window" — when a deleted track comes back near its last known position, restore its original ID.

### Project 4: 3D Tracker + BEVFusion Integration
Connect the `Tracker3D` from Section 9 to the BEVFusion ROS2 node from the [BEVFusion guide](../BEVFusion/Guide.md). Visualize persistent 3D tracks in RViz2 with trace lines. Measure end-to-end latency: LiDAR scan → BEVFusion detection → Kalman prediction → assigned track output.

### Project 5: Velocity Estimation and Prediction
Use the tracked state `[vx, vy]` from the Kalman filter to predict where each object will be in 0.5 seconds. Draw the predicted future position as a cross. Compare: Kalman predicted position vs actual position (measure pixel error) over 100 frames.

### Project 6: Jetson Deployment Benchmark
Deploy YOLO TensorRT + Kalman-Hungarian tracker on Orin Nano 8GB. Measure:
- Detector FPS (TRT FP16)
- Tracker overhead (Hungarian on N tracks × M detections)
- Total pipeline FPS
- CPU core usage during tracker (tracker is CPU-bound, detector is GPU-bound → they can run in parallel)

---

## 14. Resources

### Reference Repository
- **srianant/kalman_filter_multi_object_tracking** — github.com/srianant/kalman_filter_multi_object_tracking: The original Python 2 implementation this guide is based on. Clean, readable, good for understanding the core idea.

### Tracker Implementations
- **SORT** — github.com/abewley/sort: Original SORT paper implementation
- **DeepSORT** — github.com/nwojke/deep_sort: SORT + appearance features
- **ByteTrack** — github.com/ifzhang/ByteTrack: State-of-the-art fast tracker
- **OC-SORT** — github.com/noahcao/OC_SORT: Observation-centric improvement

### Papers
- **"Simple Online and Realtime Tracking"** (Bewley et al., ICIP 2016): the SORT paper. Read this before anything else — it's 4 pages and explains Kalman + Hungarian for tracking clearly.
- **"Simple Online and Realtime Tracking with a Deep Association Metric"** (Wojke et al., ICIP 2017): DeepSORT paper. Shows how appearance embedding improves ID consistency.
- **"ByteTrack: Multi-Object Tracking by Associating Every Detection Box"** (Zhang et al., ECCV 2022): explains two-stage association and why low-confidence detections help.
- **"HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking"** (Luiten et al., IJCV 2021): why MOTA is misleading and how HOTA fixes it.

### Hungarian Algorithm Theory
- **"The Hungarian Method for the Assignment Problem"** (Kuhn, 1955): the original paper. Surprisingly readable.
- **scipy.optimize.linear_sum_assignment docs**: the practical tool you'll use. Worth reading the source.

### Kalman Filter Background
- See [../kalman-filter/README.md](../kalman-filter/README.md) — the learning series in this repo covers 1D → 2D → 6D Kalman filters with code before this guide's 7D tracker state.

---

*Up: [Sensor Fusion Guide](../Guide.md)*
*See also: [BEVFusion](../BEVFusion/Guide.md) — 3D detection output that feeds into the 3D tracker*
*See also: [Kalman Filter Series](../kalman-filter/README.md) — prerequisite for this guide*
