# BEVFusion — Camera + LiDAR Fusion in Bird's Eye View

<div class="course-identity auto-course" style="--course-accent: #475569; --course-accent-rgb: 71, 85, 105;" markdown="1">
<div class="course-identity__icon">BCLF</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · AI Workloads</p>
<p class="course-identity__title">Specialized course identity for BEVFusion — Camera + LiDAR Fusion in Bird's Eye View.</p>
<p class="course-identity__meta">Artifact: model or workload study · Measure: accuracy, latency, memory, throughput</p>
</div>
</div>


> **What:** Multi-modal 3D object detection that fuses camera images and LiDAR point clouds in a shared **Bird's Eye View (BEV)** feature space.
> **Why:** Cameras see color/texture; LiDAR sees accurate depth/geometry. Fusion in BEV can combine their strengths; improvement must be established with matched ablations and calibration checks.
> **First hardware target:** CPU for the checked geometry kernels below. Full learned detection requires the selected implementation's supported training/inference environment; fit and speed on Orin Nano are not established here.

---


## 1. Why BEV Fusion?

### The Problem with Earlier Fusion Approaches

**Early fusion (raw data):**
```
Camera pixels + LiDAR points → concatenate → model
Problem: very different data formats, hard to align
```

**Late fusion (prediction-level):**
```
Camera model → 2D boxes
LiDAR model  → 3D boxes
                → merge/NMS boxes
Problem: each modality's errors compound; no shared features
```

**BEV Fusion (feature-level in BEV space):**
```
Camera  → image encoder → lift to BEV features
LiDAR   → point encoder → voxelize to BEV features
                        → concatenate in BEV
                        → unified 3D detection head
```

Each modality contributes its strengths at the feature level before any decision is made.

### What Each Sensor Contributes

| Sensor   | Strength                              | Weakness                              |
|----------|---------------------------------------|---------------------------------------|
| Camera   | Rich semantics (color, texture, class)| No depth; affected by lighting        |
| LiDAR    | Precise 3D geometry, range up to 100m | No texture; sparse at distance        |
| **BEV Fusion** | **Geometry + Semantics**        | **More complex, higher compute**      |

### Compare fusion with a controlled ablation

Do not subtract scores from unrelated PointPillars, BEVDet, and BEVFusion configurations and attribute the entire difference to fusion. Match dataset split, backbone, training, augmentations, and evaluation settings. The [authors' repository](https://github.com/mit-han-lab/bevfusion) separates validation and test results and provides checkpoint-specific metrics.

Measure a camera-only, LiDAR-only, and fused configuration on the same held-out scenes. Then inject a calibration offset or sensor dropout. Fusion can degrade when inputs are misaligned. Published Jetson Orin results do not establish performance on Orin Nano; identify the exact board, memory capacity, model, and runtime.

---

## 2. BEV Space Explained

### What is Bird's Eye View?

BEV is a top-down 2D grid that represents a region of the 3D world:

```
Real world:            BEV grid:
      z (up)           y (forward) ↑
      ↑                │   [car][car]
      │       →        │       [ego]
 ─────┼─────  →    ────┼────────────→ x (right)
 lidar│             │  [pedestrian]
      │             │
      y (forward)   BEV cell = e.g. 0.1m × 0.1m

Range: typically x ∈ [-51.2m, 51.2m], y ∈ [-51.2m, 51.2m], z ∈ [-5m, 3m]
Resolution: 0.1m/voxel → 1024×1024 BEV grid
```

### Why BEV?

- **Useful metric layout**: perspective scale changes are reduced, but objects can still overlap in XY, including bridges, shelves, and stacked structures
- **Easy to read geometry from LiDAR**: a simple projection collapses Z into the BEV plane
- **Planning-friendly output**: self-driving plans routes in the XY plane, same as BEV
- **Metric grid**: equal physical sizes occupy comparable grid extents; measurement density and uncertainty still vary with range

---

## 3. BEVFusion Architecture

### High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  N cameras (e.g., 6 cameras: front, rear, left, right ×2)      │
│  Each: [3, H, W] image                                          │
└─────────────────────┬───────────────────────────────────────────┘
                      │
              [Image Encoder]           e.g., Swin-T or ResNet-50
              backbone + FPN
                      │
              [View Transformer]        Lift-Splat-Shoot
              2D features → 3D BEV     shape: [C, X, Y]
                      │
                      ▼
┌────────────────[BEV Feature Map]──────────────────────────────┐
│   Camera BEV features [C_cam, X, Y]                           │
│          +                                                    │
│   LiDAR BEV features  [C_lidar, X, Y]                        │
│          │                                                    │
│   [Fusion: concat → ConvNet]                                  │
│          ↓                                                    │
│   Fused BEV features  [C_fused, X, Y]                        │
└──────────────────────────┬────────────────────────────────────┘
                           │
                  [Detection Head]      CenterPoint / TransFusion
                           │
                  3D bounding boxes:
                  (x, y, z, w, l, h, heading, class, score)

┌────────────────────────────────────────────────────────────────┐
│  LiDAR point cloud: [N_points, 5]  (x, y, z, intensity, ring) │
└────────────────────────┬───────────────────────────────────────┘
                         │
                [Point Encoder]         VoxelNet or PointPillars
                pillarize + sparse conv
                         │
                LiDAR BEV [C_lidar, X, Y]
```

### Two Main Open-Source Implementations

| Repo                      | Focus                        | Paper         |
|---------------------------|------------------------------|---------------|
| **MIT HAN Lab BEVFusion** | Camera + LiDAR, nuScenes SoTA | ICRA 2023    |
| **NVIDIA CUDA-BEVFusion** | Deployment implementation of selected model components | Repository-specific runtime |

This guide uses the **MIT HAN Lab** version as the primary reference, with TensorRT export optimizations from the NVIDIA implementation.

---

## 4. Camera → BEV: Lift-Splat-Shoot

### The Core Problem

A camera image is a 2D projection. To get 3D position from a pixel, you need depth — but cameras don't measure depth directly.

**Lift-Splat-Shoot (LSS)** solves this by predicting a discrete depth distribution per pixel, then "lifting" each pixel into 3D space at multiple candidate depths:

### Step 1: Lift (Pixel → 3D Points)

The complete checked implementation is [labs/bev_geometry.py](labs/bev_geometry.py), function `lift_pixels`. It accepts pixel coordinates, **optical-axis depth in metres**, intrinsic matrices, image augmentation matrices, and camera-to-ego transforms. Undo image resize/crop in pixel space first, then form the calibrated ray and multiply by depth:

```text
p_original = inverse(A_image_augmentation) * [u_aug, v_aug, 1]
ray_camera = inverse(K) * p_original
point_camera = depth_metres * ray_camera
point_ego = R_ego_camera * point_camera + t_ego_camera
```

The executable uses linear solves. It assumes pinhole-compatible or correctly undistorted input pixels. Do not apply a pinhole inverse directly to distorted fisheye coordinates. Camera-to-ego is the inverse direction of a camera-from-ego calibration; name transforms explicitly.

**Worked check:** `fx=fy=100`, principal point `(50,40)`, original pixel `(60,60)`, depth 2 m gives camera point `(0.2,0.4,2)`. An ego translation `(1,0,0)` gives `(1.2,0.4,2)`. Doubling depth gives `(1.4,0.8,4)` after the same translation. A 2× image resize followed by crop `(10,20)` changes the observed pixel to `(110,100)` but must not change the recovered metric point. The CPU test checks both depths and the augmentation.

The same file's `bev_sum` maps points to grid cells with **floor**, preserves batch IDs, filters bounds/height, and sums features into `[B,C,Y,X]`. Truncating a negative coordinate toward zero can wrongly admit an out-of-bounds point. Flattening batches before scatter without keeping their batch IDs mixes different scenes. Both faults have dedicated tests, including a gradient check for accepted/rejected contributions.

### Step 2: Depth Distribution Prediction

The network predicts a **D-dimensional softmax** over depth bins for each pixel:

```python
import torch
from torch import nn

class DepthFeatureHead(nn.Module):
    """Complete head for already computed backbone features; no pretrained detector."""
    def __init__(self, in_channels, depth_bins, feature_channels):
        super().__init__()
        self.depth_bins = depth_bins
        self.projection = nn.Conv2d(in_channels, depth_bins + feature_channels, 1)

    def forward(self, backbone_features):
        raw = self.projection(backbone_features)       # [B,D+C,H,W]
        probability = raw[:, :self.depth_bins].softmax(dim=1)
        feature = raw[:, self.depth_bins:]
        return probability[:, None] * feature[:, :, None]  # [B,C,D,H,W]
```

The weighted features for each depth hypothesis must remain aligned with the geometry used by the pooling step. A depth softmax is a learned discrete distribution, not automatically calibrated metric depth. A pretrained checkpoint also fixes the backbone/FPN, channel sizes, depth bins, resizing, and loss conventions; substituting an arbitrary classifier backbone does not recreate it.


---

## 5. LiDAR → BEV: Voxelization and Pillars

### PointPillars (Fast, Jetson-Friendly)

PointPillars divides the 3D space into vertical columns (pillars), processes each pillar with a small MLP, then scatters features into a 2D BEV grid:

A pillar encoder has three complete responsibilities:

1. Filter metric points to the configured bounds and assign XY cell indices with floor. Preserve batch indices. Define maximum points/pillars and the deterministic or random truncation policy.
2. Construct per-point features such as `(x,y,z,intensity)`, offsets from the **mean of points in that pillar**, and offsets from the pillar's geometric XY centre. Padding masks must affect features and normalization consistently; masking only after batch normalization does not remove padding's influence on batch statistics.
3. Pool per-point features into one feature per occupied pillar and scatter to `[B,C,Y,X]`, then apply the BEV backbone. Empty cells remain zero; input coordinate order must match the scatter convention.

The tested `bev_sum` provides a small, differentiable reference for the final accumulation. For a trained PointPillars or sparse-voxel encoder, use the selected repository's complete voxelization/feature code and weights together. A missing voxelizer cannot be replaced by an empty `pass` method while calling the result a working detector.

### VoxelNet / Sparse Convolution (Higher Accuracy)

For higher accuracy (at cost of more compute), use sparse 3D convolutions before collapsing to BEV:

```
Points → Voxelize [X, Y, Z, C]
       → Sparse 3D Conv (only on occupied voxels)
       → Collapse Z dimension → BEV [X, Y, C*Z]
```

MIT BEVFusion uses **VoxelNet with sparse convolutions** (via `spconv` library) for the LiDAR branch.

---

## 6. BEV Feature Fusion

### Concatenation + Convolutional Fusion

```python
# bev_fusion.py
import torch
import torch.nn as nn

class BEVFusionLayer(nn.Module):
    """
    Fuse camera BEV features and LiDAR BEV features.

    Camera BEV:  [B, C_cam, X, Y]   — semantic, lower spatial accuracy
    LiDAR BEV:   [B, C_lid, X, Y]   — precise geometry

    Both must have the same spatial resolution (X, Y).
    """
    def __init__(self, C_cam=80, C_lid=128, C_out=256):
        super().__init__()

        # Align feature channels before fusion
        self.cam_proj  = nn.Sequential(
            nn.Conv2d(C_cam, C_out // 2, 1, bias=False),
            nn.BatchNorm2d(C_out // 2), nn.ReLU()
        )
        self.lid_proj  = nn.Sequential(
            nn.Conv2d(C_lid, C_out // 2, 1, bias=False),
            nn.BatchNorm2d(C_out // 2), nn.ReLU()
        )

        # Fusion convolutions
        self.fusion = nn.Sequential(
            nn.Conv2d(C_out, C_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_out), nn.ReLU(),
            nn.Conv2d(C_out, C_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_out), nn.ReLU(),
        )

    def forward(self, cam_bev, lidar_bev):
        # Project to common channel count
        cam_feat   = self.cam_proj(cam_bev)     # [B, C_out/2, X, Y]
        lidar_feat = self.lid_proj(lidar_bev)   # [B, C_out/2, X, Y]

        # Concatenate along channel axis
        fused = torch.cat([cam_feat, lidar_feat], dim=1)   # [B, C_out, X, Y]
        return self.fusion(fused)                           # [B, C_out, X, Y]
```

### Alignment: Handling Different BEV Resolutions

Camera BEV and LiDAR BEV may have different resolutions. The interpolation example below assumes the same metric bounds, axes, origin, time, and compatible cell-centre convention. Equal tensor shape alone does not establish geometric alignment. If grids differ in extent or frame, construct the coordinate warp explicitly. Image-backbone stride does not by itself determine BEV metric resolution:

```python
class ResolutionAlignedFusion(nn.Module):
    def __init__(self, C_cam, C_lid, C_out, cam_stride=2):
        super().__init__()
        # Camera BEV is typically lower resolution after FPN downsampling
        # Upsample camera to match LiDAR resolution
        self.cam_upsample = nn.Sequential(
            nn.Upsample(scale_factor=cam_stride, mode='bilinear', align_corners=False),
            nn.Conv2d(C_cam, C_cam, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_cam), nn.ReLU()
        )
        self.fuse = BEVFusionLayer(C_cam, C_lid, C_out)

    def forward(self, cam_bev, lidar_bev):
        cam_bev_aligned = self.cam_upsample(cam_bev)
        return self.fuse(cam_bev_aligned, lidar_bev)
```

---

## 7. 3D Detection Head

### CenterPoint-style head: an architecture exercise

CenterPoint-style heads predict object-centre heatmaps and box attributes. This small head/decoder illustrates those tensors; it is not the pinned MIT fusion checkpoint, whose detection configuration uses TransFusion. It omits training losses, velocity output, dataset-specific class handling, and final rotated/circle NMS, so it is not a complete nuScenes detector.

```python
# centerpoint_head.py
import torch
import torch.nn as nn

class CenterPointHead(nn.Module):
    """
    Anchor-free 3D detection head.
    Input:  BEV feature map [B, C, X, Y]
    Output: Per-class heatmaps + regression maps
    """
    def __init__(self, in_channels=256, num_classes=10):
        super().__init__()

        # Heatmap head: predicts center probability per class
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, num_classes, 1)    # sigmoid applied in loss
        )

        # Regression heads: offset, height, size, rotation, velocity
        self.offset_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 2, 1)    # (Δx, Δy) sub-voxel offset
        )
        self.height_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 1, 1)    # z center
        )
        self.size_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 3, 1)    # (log w, log l, log h)
        )
        self.rot_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 2, 1)    # (sin θ, cos θ)
        )

    def forward(self, bev_features):
        return {
            'heatmap': self.heatmap_head(bev_features).sigmoid(),
            'offset':  self.offset_head(bev_features),
            'height':  self.height_head(bev_features),
            'size':    self.size_head(bev_features),
            'rot':     self.rot_head(bev_features),
        }

def decode_predictions(preds, bev_range, voxel_size, score_threshold=0.3, top_k=200, output_stride=1):
    """
    Convert raw CenterPoint predictions to 3D bounding boxes.

    Returns list of:
      (x, y, z, w, l, h, heading_rad, class_id, score)
    """
    heatmap = preds['heatmap']   # [B, num_classes, X, Y]
    B, num_cls, X, Y = heatmap.shape

    results = []
    for b in range(B):
        boxes_batch = []
        for cls in range(num_cls):
            heat = heatmap[b, cls]    # [X, Y]

            # Find peaks above threshold
            # Non-maximum suppression via max pooling (peak = local max)
            heat_nms = torch.max_pool2d(heat.unsqueeze(0).unsqueeze(0), 3, 1, 1)
            heat_nms = heat_nms[0, 0]
            peaks = (heat == heat_nms) & (heat > score_threshold)

            ys, xs = peaks.nonzero(as_tuple=True)   # BEV grid coords

            if len(xs) == 0:
                continue

            # Top-K selection
            scores = heat[ys, xs]
            if len(scores) > top_k:
                topk_idx = scores.topk(top_k).indices
                xs, ys, scores = xs[topk_idx], ys[topk_idx], scores[topk_idx]

            # Decode box position from grid to metric coordinates
            offset = preds['offset'][b, :, ys, xs]   # [2, N]
            x_metric = (xs.float() + offset[0]) * voxel_size[0] * output_stride + bev_range[0]
            y_metric = (ys.float() + offset[1]) * voxel_size[1] * output_stride + bev_range[1]
            z_metric = preds['height'][b, 0, ys, xs]

            # Box size
            size = preds['size'][b, :, ys, xs].exp()  # [3, N] (w, l, h)

            # Heading
            rot = preds['rot'][b, :, ys, xs]           # [2, N] (sin, cos)
            heading = torch.atan2(rot[0], rot[1])

            for i in range(len(xs)):
                boxes_batch.append({
                    'x': x_metric[i].item(),
                    'y': y_metric[i].item(),
                    'z': z_metric[i].item(),
                    'w': size[0, i].item(),
                    'l': size[1, i].item(),
                    'h': size[2, i].item(),
                    'heading': heading[i].item(),
                    'class': cls,
                    'score': scores[i].item(),
                })
        results.append(boxes_batch)
    return results
```

---

## 8. nuScenes Dataset

nuScenes is the **standard benchmark for BEVFusion** and all modern multi-modal 3D detection methods. Every BEVFusion paper reports results on nuScenes. Understanding its structure is required before training or evaluating.

### What nuScenes Is

```
nuScenes — by nuTonomy (acquired by Aptiv / Motional), released 2019
Location:  Boston (USA) + Singapore  ← day/night, rain, different traffic rules
License:   CC BY-NC-SA 4.0 (non-commercial research)
Paper:     Caesar et al., CVPR 2020 — "nuScenes: A Multimodal Dataset for AD"
```

### Sensor Suite

nuScenes captures the **full autonomous vehicle sensor stack** — every sensor you'd deploy on a real AV:

```
┌─────────────────────────────────────────────────────────────┐
│                    Vehicle Top View                         │
│                                                             │
│            CAM_FRONT_LEFT  CAM_FRONT  CAM_FRONT_RIGHT       │
│                    ↖           ↑           ↗                │
│                                                             │
│  CAM_BACK_LEFT ←─────────[LIDAR_TOP]─────────→ CAM_BACK_RIGHT │
│                                                             │
│                    ↙           ↓           ↘                │
│                         CAM_BACK                           │
│                                                             │
│  RADAR:  FRONT + FRONT_LEFT + FRONT_RIGHT + BACK_LEFT + BACK_RIGHT │
└─────────────────────────────────────────────────────────────┘
```

| Sensor | Model | Specs |
|--------|-------|-------|
| **Camera ×6** | Basler acA1600-60gc | 1600×900 px, 12 Hz, 360° surround |
| **LiDAR ×1** | Velodyne HDL-32E | 32 beams, 20 Hz, 70m range, 360° |
| **RADAR ×5** | Continental ARS 408-21 | 13 Hz, 250m range, Doppler velocity |
| **IMU** | Applanix POS LV | 6-axis, 100 Hz |
| **GPS** | Applanix POS LV | RTK, cm-level accuracy |

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Scenes | **1,000** (each 20 seconds) |
| Keyframes | **40,000** (annotated at 2 Hz = 1 per 0.5s) |
| Camera images | **1.4 million** (6 cameras × keyframes + sweeps) |
| LiDAR sweeps | **390,000** (20 Hz × 20s × 1000 scenes) |
| RADAR sweeps | **1.4 million** (5 radars × 13 Hz) |
| 3D bounding boxes | **1.4 million** |
| Detection classes | **10** (for detection task) |
| Full class set | **23** (including rare and static objects) |
| Attributes | **8** (visibility, activity state) |
| vs KITTI | 7× more annotations, 100× more images |

### Data Splits

| Split | Scenes | Keyframes | Use |
|-------|--------|-----------|-----|
| **train** | 700 | 28,130 | Training |
| **val** | 150 | 6,019 | Validation during development |
| **test** | 150 | 6,008 | Hidden labels — submit to leaderboard |
| **mini** | 10 | 404 | Getting started, debugging |

**Start with mini** — it downloads in minutes, has the same structure as full, and is enough to verify your pipeline works before committing to the 300 GB full download.

### Detection Classes

BEVFusion is evaluated on these **10 detection classes** (subset of 23 full classes):

```python
NUSCENES_DETECTION_CLASSES = [
    'car',                   # sedan, SUV, pickup
    'truck',                 # box truck, delivery
    'construction_vehicle',  # crane, bulldozer
    'bus',                   # city bus, tour bus
    'trailer',               # semi trailer (may be detached)
    'barrier',               # jersey barrier, temporary divider
    'motorcycle',
    'bicycle',
    'pedestrian',            # person on foot
    'traffic_cone',
]
```

### Download

```bash
# 1. Register at https://www.nuscenes.org/ (free academic license)
# 2. Go to "Download" → agree to terms → download via browser or wget

# nuScenes mini (10 scenes, ~4 GB) — START HERE
# From nuscenes.org download page, get the wget command with your token:
wget -O nuScenes-v1.0-mini.tar.bz2 \
  "https://d36yt3mvayqg5m.cloudfront.net/public/v1.0/v1.0-mini.tgz"

# Extract
mkdir -p /data/nuscenes
tar -xjf nuScenes-v1.0-mini.tar.bz2 -C /data/nuscenes/

# nuScenes full (300 GB, split into ~30 blobs) — for real training
# Use the s3 CLI tool provided on nuscenes.org after registration
pip install awscli
# They provide a script: nuScenes-v1.0-data.sh
bash nuScenes-v1.0-data.sh --target /data/nuscenes/
```

### Dataset Structure on Disk

```
/data/nuscenes/
├── v1.0-mini/                   ← or v1.0-trainval/ for full dataset
│   ├── scene.json               ← 10 scenes (mini) or 850 (trainval)
│   ├── sample.json              ← keyframes (one every 0.5s)
│   ├── sample_data.json         ← all sensor readings (cameras, LiDAR, radar)
│   ├── sample_annotation.json   ← 3D bounding boxes per keyframe
│   ├── calibrated_sensor.json   ← extrinsics + intrinsics per sensor
│   ├── ego_pose.json            ← vehicle pose at each timestamp
│   ├── instance.json            ← object instances (unique IDs across frames)
│   ├── category.json            ← class taxonomy
│   ├── attribute.json           ← object attributes (moving, stopped, etc.)
│   ├── visibility.json          ← annotation visibility levels
│   ├── log.json                 ← drive metadata (date, location, vehicle)
│   └── map.json                 ← map metadata
├── samples/                     ← keyframe sensor data (annotated)
│   ├── CAM_FRONT/               ← 1600×900 JPEG images
│   ├── CAM_FRONT_LEFT/
│   ├── CAM_FRONT_RIGHT/
│   ├── CAM_BACK/
│   ├── CAM_BACK_LEFT/
│   ├── CAM_BACK_RIGHT/
│   ├── LIDAR_TOP/               ← .bin files (N×5: x,y,z,intensity,ring)
│   └── RADAR_FRONT/             ← .pcd files
├── sweeps/                      ← intermediate frames (not annotated)
│   ├── CAM_FRONT/               ← denser temporal data between keyframes
│   ├── LIDAR_TOP/               ← LiDAR runs at 20 Hz, keyframes at 2 Hz
│   └── ...                      ← so 10 LiDAR sweeps between each keyframe
└── maps/                        ← HD maps as PNG + JSON
    ├── boston-seaport.png
    ├── singapore-onenorth.png
    └── *.json                   ← vectorized road/lane graphs
```

### Database Schema (Relational)

nuScenes uses a relational database stored as JSON files. Understanding the links is key to loading data correctly:

```
scene (20s clip)
  └── sample[] (keyframe every 0.5s, annotated)
        ├── sample_data[] (one per sensor: 6 cameras + LiDAR + 5 radars)
        │     ├── calibrated_sensor → intrinsics + extrinsics (T_sensor_ego)
        │     └── ego_pose          → vehicle pose in world (T_ego_world)
        └── sample_annotation[] (one 3D box per object per keyframe)
              ├── instance → tracks the same object across frames
              ├── category → class name
              └── attribute → moving/stopped/etc.
```

### nuScenes Devkit Python API

```python
from nuscenes.nuscenes import NuScenes

# Initialize — loads all JSON tables into memory (~2 GB RAM for full)
nusc = NuScenes(version='v1.0-mini', dataroot='/data/nuscenes', verbose=True)

# ── Scene ──────────────────────────────────────────────────────────────────
scene = nusc.scene[0]
print(scene['name'])           # 'scene-0061'
print(scene['description'])    # 'Parked truck, motorcycle, ped...'
print(scene['nbr_samples'])    # 39 keyframes in this scene

# ── Sample (keyframe) ───────────────────────────────────────────────────────
first_sample_token = scene['first_sample_token']
sample = nusc.get('sample', first_sample_token)
print(sample['timestamp'])     # Unix microseconds
print(sample.keys())
# dict_keys(['token', 'timestamp', 'prev', 'next', 'scene_token', 'data', 'anns'])

# ── Access Camera Image ─────────────────────────────────────────────────────
cam_token = sample['data']['CAM_FRONT']
cam_data = nusc.get('sample_data', cam_token)
img_path = nusc.get_sample_data_path(cam_token)
# img_path = '/data/nuscenes/samples/CAM_FRONT/n015-...-CAM_FRONT__1531281445162460.jpg'

import cv2
img = cv2.imread(img_path)  # 1600×900

# ── Access LiDAR Point Cloud ────────────────────────────────────────────────
lidar_token = sample['data']['LIDAR_TOP']
lidar_path, boxes, camera_intrinsic = nusc.get_sample_data(lidar_token)
import numpy as np
pts = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
# pts[:, 0:3] = XYZ, pts[:, 3] = intensity, pts[:, 4] = ring index

# ── Get 3D Annotations ──────────────────────────────────────────────────────
for ann_token in sample['anns']:
    ann = nusc.get('sample_annotation', ann_token)
    print(ann['category_name'])     # 'vehicle.car'
    print(ann['translation'])       # [x, y, z] in ego frame at annotation time
    print(ann['size'])              # [width, length, height] in meters
    print(ann['rotation'])          # quaternion [w, x, y, z]
    print(ann['num_lidar_pts'])     # how many LiDAR points hit this object

# ── Calibrated Sensor (camera intrinsics + extrinsics) ─────────────────────
cam_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
cs = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
K = np.array(cs['camera_intrinsic'])      # 3×3 intrinsic matrix
T_cam_ego = np.array(cs['translation'])   # camera position in ego frame
R_cam_ego = cs['rotation']                # quaternion

# ── Ego Pose (vehicle pose in world frame) ──────────────────────────────────
ep = nusc.get('ego_pose', cam_data['ego_pose_token'])
T_ego_world = np.array(ep['translation'])  # [x, y, z] in global map frame
R_ego_world = ep['rotation']               # quaternion

# ── Iterate Through a Scene ─────────────────────────────────────────────────
def iter_scene(nusc, scene):
    """Yield all samples in a scene in temporal order."""
    token = scene['first_sample_token']
    while token:
        sample = nusc.get('sample', token)
        yield sample
        token = sample['next']   # empty string at end

for sample in iter_scene(nusc, scene):
    print(sample['timestamp'])
```

### Visualization with Devkit

```python
# Render a sample — annotated BEV + 6 camera images + LiDAR
nusc.render_sample(sample['token'])

# Render just the LiDAR point cloud in BEV with GT boxes
nusc.render_sample_data(sample['data']['LIDAR_TOP'],
                        with_anns=True, axes_limit=50)

# Render camera image with projected LiDAR points + 2D boxes
nusc.render_sample_data(sample['data']['CAM_FRONT'],
                        with_anns=True)

# Render a full scene as video
nusc.render_scene_channel(scene['token'], channel='CAM_FRONT',
                          out_path='scene_front.avi')

# nuScenes-lidarseg: render semantic segmentation labels on point cloud
from nuscenes.eval.lidarseg.utils import colormap_to_colors
nusc.render_pointcloud_in_image(sample['token'],
                                pointsensor_channel='LIDAR_TOP',
                                camera_channel='CAM_FRONT',
                                render_intensity=False)
```

### Coordinate Systems

nuScenes uses three coordinate frames — understanding them is critical for BEVFusion:

```
Global (world) frame:
  Origin: fixed point in the map (city-level)
  Used for: ego_pose, global object tracks, HD maps

Ego (vehicle) frame:
  Origin: rear axle midpoint
  Used for: sample_annotation translations are in THIS frame
  +X: forward, +Y: left, +Z: up

Sensor frame:
  Origin: sensor mounting position
  Used for: raw point cloud data is in THIS frame
  calibrated_sensor gives T_sensor→ego transform

Conversion chain:
  LiDAR points (sensor frame)
    → apply calibrated_sensor.T → ego frame
    → apply ego_pose.T → global frame
    → (subtract target_ego_pose.T) → another ego frame
```

```python
from pyquaternion import Quaternion
import numpy as np

def lidar_to_ego(pts_lidar: np.ndarray, cs_record: dict) -> np.ndarray:
    """Transform LiDAR points from sensor frame to ego frame."""
    R = Quaternion(cs_record['rotation']).rotation_matrix   # 3×3
    t = np.array(cs_record['translation'])                  # (3,)
    return (R @ pts_lidar[:, :3].T).T + t

def ego_to_global(pts_ego: np.ndarray, ep_record: dict) -> np.ndarray:
    """Transform ego-frame points to global frame."""
    R = Quaternion(ep_record['rotation']).rotation_matrix
    t = np.array(ep_record['translation'])
    return (R @ pts_ego.T).T + t
```

### Evaluation Metrics (NDS)

BEVFusion papers report **NDS (nuScenes Detection Score)** — a composite metric:

```
NDS = (5 × mAP + sum(max(0, 1 - error) for error in TP_errors)) / 10
TP_errors = [mATE, mASE, mAOE, mAAE, mAVE]

where:
  mAP  = mean Average Precision (matching threshold: BEV center distance, not IoU)
  mATE = mean Average Translation Error  [meters]   ← lower is better
  mASE = mean Average Scale Error        [1 - IoU]  ← lower is better
  mAOE = mean Average Orientation Error  [radians]  ← lower is better
  mAAE = mean Average Attribute Error    [1 - acc]  ← lower is better
  mAVE = mean Average Velocity Error     [m/s]      ← lower is better

Key difference from KITTI:
  nuScenes uses center-distance matching (2D BEV), not 3D IoU.
  Threshold: 0.5m, 1.0m, 2.0m, 4.0m → average over thresholds.
  Distance thresholds measure centre localization error; they do not automatically
  apply a safety weighting to objects near the ego vehicle.
```

### Interpret NDS correctly

Use normalized mAP (0–1) in the formula, and clip each error contribution at zero. If mAP=0.5 and all five error values are 0.2, NDS is `(2.5+4)/10=0.65`. The earlier denominator of 5 would produce an invalid score above 1. Class exclusions, true-positive matching thresholds, and error definitions belong to the [official nuScenes evaluator](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/README.md).

Published results compare specific checkpoint/configuration combinations. The original MIT repository's modality ablations use a particular heavier camera baseline and a TransFusion-L LiDAR baseline; unrelated PointPillars/BEVDet scores are not controlled evidence for the effect of fusion.

---

## 9. First reproduce the CPU geometry

Use the isolated, hash-pinned environment from the [PyTorch CPU robot lab](../../../2.%20Deep%20Learning%20Frameworks/PyTorch/CPU-Robot-Stop-Lab/Guide.md). It includes CPU PyTorch 2.14.0. From the repository root:

```bash
/tmp/course-pytorch-cpu-venv/bin/python -m unittest discover -s "Phase 3 - Artificial Intelligence/Track A - Hardware and Edge AI/4. Sensor Fusion/BEVFusion/labs/optional" -v
```

Three real CPU tests check depth/augmentation, batch isolation, boundary flooring, accumulation gradients, and invalid grid configuration. They passed on 2026-10-18. No downloaded detector or dataset is required. Read the 60-line [geometry implementation](labs/bev_geometry.py), change one batch's feature values, and predict which output cells should change before running it.

**Independent exercise:** rotate a camera 90° about the ego z axis, preserving its translation. Derive the expected ego point from the worked camera point, then add the assertion. Next halve image resolution and adjust the augmentation/intrinsics consistently; the same physical ray must remain the same ray.

## 10. Reproduce a published detector in its own environment

The original MIT implementation is a **historical research stack**. Its README at immutable revision [`326653dc06e0938edf1aae7d01efcd158ba83de5`](https://github.com/mit-han-lab/bevfusion/tree/326653dc06e0938edf1aae7d01efcd158ba83de5) specifies Python 3.8, PyTorch 1.9–1.10.2, MMCV 1.4.0, MMDetection 2.20.0, OpenMPI 4.0.4, and mpi4py 3.0.3, plus its remaining requirements/custom extensions. Installing current PyTorch, MMCV 2.x, and arbitrary MMDetection3D packages together does not reproduce that environment.

```bash
git clone https://github.com/mit-han-lab/bevfusion.git
cd bevfusion
git checkout 326653dc06e0938edf1aae7d01efcd158ba83de5
# Follow THIS revision's README and docker/Dockerfile in an isolated GPU environment.
# Build its custom extensions and record image digest, drivers, and pip freeze.
```

A newer implementation is [MMDetection3D's BEVFusion project](https://github.com/open-mmlab/mmdetection3d/tree/main/projects/BEVFusion). Treat it as a separate stack: pin its revision, configuration, checkpoint, dependency set, and evaluation code. Do not mix its coordinate conventions or prepared data files with the old MIT fork. The MIT README explicitly warns that its data preparation predates a coordinate-system refactor.

Use the exact model-zoo download link from the pinned README, record the checkpoint checksum/license, and preserve the paired config. Start with nuScenes mini to validate paths, dimensions, transforms, and evaluation code. Mini is a pipeline smoke test, not a representative quality estimate or a substitute for the full validation split. Dataset access/license terms and available downloads must be checked at the official dataset site.

For context, the pinned MIT README's full fusion evaluation command is:

```bash
torchpack dist-run -np 8 python tools/test.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  pretrained/bevfusion-det.pth --eval bbox
```

This is the upstream eight-process GPU recipe, **not an executed course result or a CPU command**. Adapt worker count only within the selected implementation's supported workflow, then record the change. Train/evaluate using that repository's actual configuration hierarchy; the former `BEVFusion.from_pretrained` and `configs/bevfusion/bevfusion-det.yaml` examples did not identify valid APIs/configs for this fork.

## 11. Export by verified subgraph boundaries

Export the chosen checkpoint's actual camera backbone/neck, geometry/pooling, LiDAR encoder, fusion block, and detection head according to that runtime's implementation. Sparse convolution and custom pooling may need supported custom kernels/plugins. A successful ONNX export of one block is not evidence that the whole detector can run in ONNX Runtime or TensorRT.

For every boundary, save a real reference input/output pair and compare shape, layout, dtype, absolute/relative error, and final decoded detections. Explicitly preserve `[B,C,Y,X]`, feature stride, box dimensions/order, yaw convention, depth discretization, and preprocessing. Dictionary/list outputs need a stable named wrapper; inventing five tensor output names does not flatten a real detection head correctly.

The [NVIDIA CUDA-BEVFusion implementation](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/CUDA-BEVFusion) is a separate deployment route. Follow its pinned export/build instructions and supported versions. Its historical 2023 Orin throughput is a particular published configuration, not proof of performance or memory fit on Orin Nano. TensorRT engine portability depends on build/runtime/device compatibility; downloaded engines are not universally reusable.

## 12. TensorRT and Jetson version boundary

As checked 2026-10-18, TensorRT **11.4** documentation describes a different API/support boundary from the JetPack-supported **10.x** stack. TensorRT 11.4 does not support JetPack; its strongly typed network behavior and removed APIs must not be mixed with old `--fp16`, implicit quantization, or legacy binding examples. Use [NVIDIA's Jetson migration guidance](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-jetson.html), the installed JetPack release notes, and the [archived TensorRT 10.x documentation](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/python-api-docs.html) for an actual 10.x target.

Start with the [Jetson inference course](../../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Real-Time-Inference/Guide.md) for the target runtime workflow. Precision support is per operation/plugin and platform. An INT8 flag without representative calibrated ranges or explicit Q/DQ semantics is not a validated quantized model. Compare accuracy before reporting throughput.

## 13. ROS 2 observation contract

A complete integration must parse actual message fields, synchronize the appropriate capture times, transform points into one declared ego frame, apply the checkpoint's exact preprocessing, execute every model branch, decode its outputs, and publish boxes with a frame/time contract. A ROS subscriber plus an empty postprocessor is not a detector.

Use a bag replay first. Record camera/point-cloud capture times and clock domains, transform revision, modality validity, observation age, and output stamp. If cameras and LiDAR are asynchronous, compensate ego motion to a common reference time when the model supports it. Queuing the latest image from each topic is not synchronization. A stale camera feature cache also needs an explicit age and geometric treatment.

Verify one known point through camera→ego→camera, one moving-target time-offset case, one dropout, and one out-of-order message. Then compare replay outputs against the offline reference. Hardware transport/ROS integration was not executed in this course revision.

## 14. Measure memory and quality before optimization

A float32 BEV tensor with shape `[1,256,128,128]` occupies `1*256*128*128*4 = 16 MiB`. Doubling both spatial dimensions makes that one tensor four times larger. The full peak also includes activations, camera features, sparse data structures, engine workspace, buffers, and the rest of the application.

A lower resolution, fewer cameras, lighter backbone, or reduced precision changes the experiment. Retraining/fine-tuning may be needed; changing an input shape alone does not preserve a checkpoint's geometry or quality. Use matched validation scenes and measure small/far objects, modality dropout, calibration shift, and latency/memory under sustained load. Report end-to-end capture-to-output age, not only a fast inference block.

## 15. Completion project: explain one fusion failure

First pass the CPU kernels and [timestamped camera-fusion lab](../Camera-Time-Fusion-Lab/Guide.md). Then choose either a documented full-detector reproduction with suitable hardware/data or a deeper geometry experiment on CPU.

For the full detector, submit its pinned environment/config/checkpoint/data split, a real evaluator report, camera-only/LiDAR-only/fused comparison with controlled training settings where feasible, and one deliberately injected frame/time/calibration fault. If exact matched training is unaffordable, label checkpoint comparisons as such rather than attributing their difference solely to fusion.

For the CPU route, add and verify a non-identity extrinsic, image crop/resize, height filtering, and independent batches; measure how a wrong extrinsic or stale pose moves occupied grid cells. Include an analytic expected output and a failing implementation that your test detects. This completes the geometry objective; it does not claim trained 3D detection accuracy.

**Sources and execution record:** API/metadata checks and CPU tests on 2026-10-18. The original model's GPU training, nuScenes evaluation, TensorRT conversion, Jetson deployment, and ROS integration were not executed. Preserve those boundaries in your own report.

---

## 16. Resources

### Papers
- **"BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation"** (MIT HAN Lab, ICRA 2023): the primary paper. Read Sections 3 and 4 carefully — the architecture, LSS, and fusion are all explained.
- **"BEVFusion: A Simple and Robust LiDAR-Camera Fusion Framework"** (NVIDIA, NeurIPS 2022): the NVIDIA variant with TensorRT optimizations.
- **"Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D"** (Philion & Fidler, ECCV 2020): the foundation of the camera → BEV transform used in BEVFusion. Read this before the BEVFusion paper.
- **"PointPillars: Fast Encoders for Object Detection from Point Clouds"** (Lang et al., CVPR 2019): the LiDAR branch backbone.
- **"Center-based 3D Object Detection and Tracking"** (CenterPoint, CVPR 2021): the detection head used in BEVFusion.

### Code
- **MIT HAN Lab BEVFusion**: github.com/mit-han-lab/bevfusion
- **NVIDIA CUDA-BEVFusion** (Jetson-optimized, TensorRT native): github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/CUDA-BEVFusion
- **nuScenes devkit**: github.com/nutonomy/nuscenes-devkit — data loading, evaluation, visualization

### Datasets
- **nuScenes**: nuscenes.org — 1000 scenes, 6 cameras, 1 LiDAR, full 3D annotations. The standard benchmark for BEVFusion.
- **nuScenes mini** (free, 10 scenes): start here before downloading the full dataset.
- **KITTI**: for simpler camera+LiDAR tasks (frontal only, 2D+3D boxes).

### Visualization
```bash
# nuScenes-devkit viewer: render a scene with camera + LiDAR + GT boxes
python3 -c "
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes', verbose=True)
nusc.render_sample('scene-0001')
"
```

---

*Up: [Sensor Fusion Guide](../Guide.md)*
*See also: [4. Sensor Fusion / kalman-filter](../kalman-filter/README.md)*


## Carry this course into the physical-AI system

**Start from:** Calibrated camera/LiDAR samples and a checkpoint-specific configuration.

**Build and challenge:** Compare matched single-modality and fused runs on one split. Inject a transform offset and a missing sensor; report quality, memory, and latency rather than importing another board’s FPS.

**Pass forward:** A calibrated fusion workload and a modality-ablation report. Continue with [the connected course](../../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20D%20-%20Robotics/VLA%20Foundations%20to%20Frontier%202026/README.md) and keep the artifact in the [Physical AI Design Studio](../../../../Physical-AI-Design-Studio.md).
