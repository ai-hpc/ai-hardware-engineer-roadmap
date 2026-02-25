# Computer Vision — Complete Guide

> **Goal:** Build a comprehensive understanding of computer vision from image processing fundamentals through modern deep learning architectures, with practical skills in annotation tools, dataset formats, and deployment on edge hardware.

---

## Table of Contents

1. [What is Computer Vision?](#1-what-is-computer-vision)
2. [Image Processing Fundamentals](#2-image-processing-fundamentals)
3. [Feature Extraction](#3-feature-extraction)
4. [Image Segmentation](#4-image-segmentation) — threshold, watershed, semantic segmentation, instance segmentation
5. [Object Detection](#5-object-detection)
6. [Object Tracking](#6-object-tracking)
7. [3D Vision](#7-3d-vision) — calibration, stereo depth, pose, multi-camera ADAS (openpilot near/wide)
8. [Advanced Deep Learning Architectures](#8-advanced-deep-learning-architectures)
9. [OpenCV — Core to Advanced](#9-opencv--core-to-advanced)
10. [Annotation Tools](#10-annotation-tools)
11. [Dataset Formats](#11-dataset-formats)
12. [Model Training Pipeline](#12-model-training-pipeline)
13. [Projects](#13-projects)
14. [Resources](#14-resources)

---

## 1. What is Computer Vision?

Computer vision gives machines the ability to interpret and understand visual information from the world — images, videos, depth maps, point clouds.

```
Input: raw pixels / point cloud / depth frame
         ↓
  Preprocessing (resize, normalize, augment)
         ↓
  Feature extraction (learned or hand-crafted)
         ↓
  Task-specific head (detection / segmentation / depth)
         ↓
Output: bounding boxes / masks / pose / 3D structure
```

### Core Tasks

| Task | Input | Output | Example |
|------|-------|--------|---------|
| Classification | Image | Class label | "this is a cat" |
| Object Detection | Image | Boxes + labels | "car at (x1,y1,x2,y2)" |
| Semantic Segmentation | Image | Per-pixel class | every pixel = road/car/sky |
| Instance Segmentation | Image | Per-object mask | each car gets its own mask |
| Depth Estimation | RGB | Depth map | distance per pixel |
| Pose Estimation | Image | Keypoints | skeleton of a person |
| 3D Detection | Point Cloud | 3D boxes | LiDAR object detection |
| Optical Flow | Video | Motion field | pixel displacement between frames |

### Why Computer Vision is Hard

```
Challenges:
  Viewpoint variation    — same object, different camera angle
  Illumination           — shadows, overexposure, nighttime
  Occlusion              — object partly hidden by another
  Scale variation        — same object at different distances
  Intra-class variation  — all chairs look different
  Background clutter     — object blends into background
  Deformable objects     — humans, animals change shape
```

---

## 2. Image Processing Fundamentals

### 2.1 Color Spaces

```python
import cv2
import numpy as np

img_bgr = cv2.imread('image.jpg')           # OpenCV reads as BGR

img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
img_hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
img_lab  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

# HSV is useful for color-based segmentation (hue is color, S=saturation, V=brightness)
# LAB is perceptually uniform — good for distance-based color similarity
# Gray reduces to 1 channel — reduces computation for non-color tasks
```

### 2.2 Filtering

#### Spatial Filters

```python
# Gaussian blur — smooths noise
blurred = cv2.GaussianBlur(img_gray, ksize=(5, 5), sigmaX=1.5)

# Median blur — removes salt-and-pepper noise (edge-preserving)
median = cv2.medianBlur(img_gray, ksize=5)

# Bilateral filter — edge-preserving smoothing (keeps sharp edges, blurs flat regions)
bilateral = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=75, sigmaSpace=75)

# Sharpening via custom kernel
kernel_sharpen = np.array([[ 0, -1,  0],
                            [-1,  5, -1],
                            [ 0, -1,  0]])
sharpened = cv2.filter2D(img_bgr, -1, kernel_sharpen)
```

#### Adaptive Filtering

```python
# Adaptive threshold — threshold varies per region (handles uneven lighting)
adaptive_thresh = cv2.adaptiveThreshold(
    img_gray,
    maxValue=255,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY,
    blockSize=11,   # neighborhood size (must be odd)
    C=2             # constant subtracted from mean
)

# CLAHE — Contrast Limited Adaptive Histogram Equalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(img_gray)
```

#### Frequency Domain Filtering

```python
import numpy as np

def frequency_domain_filter(img_gray: np.ndarray, low_pass: bool = True,
                             radius: int = 30) -> np.ndarray:
    """Apply low-pass or high-pass filter in frequency domain."""
    rows, cols = img_gray.shape
    crow, ccol = rows // 2, cols // 2

    # Fourier transform + shift DC to center
    dft = np.fft.fft2(img_gray.astype(np.float32))
    dft_shift = np.fft.fftshift(dft)

    # Create circular mask
    Y, X = np.ogrid[:rows, :cols]
    dist = np.sqrt((X - ccol)**2 + (Y - crow)**2)
    mask = (dist <= radius).astype(np.float32)
    if not low_pass:
        mask = 1 - mask   # invert for high-pass

    # Apply mask, inverse FFT
    filtered = np.fft.ifftshift(dft_shift * mask)
    result = np.fft.ifft2(filtered)
    return np.abs(result).clip(0, 255).astype(np.uint8)

# Low-pass: removes high-freq noise (blurring effect)
# High-pass: keeps edges only (sharpening effect)
```

#### Wavelet Transform

```python
# pip install PyWavelets
import pywt

def wavelet_denoise(img_gray: np.ndarray, wavelet: str = 'db1',
                    level: int = 3, threshold: float = 20.0) -> np.ndarray:
    """Denoise image using wavelet soft-thresholding."""
    coeffs = pywt.wavedec2(img_gray.astype(float), wavelet, level=level)

    # Soft-threshold all detail coefficients (not approximation)
    new_coeffs = [coeffs[0]]  # keep approximation
    for detail_tuple in coeffs[1:]:
        new_coeffs.append(tuple(
            pywt.threshold(d, threshold, mode='soft') for d in detail_tuple
        ))

    denoised = pywt.waverec2(new_coeffs, wavelet)
    return denoised.clip(0, 255).astype(np.uint8)
```

### 2.3 Edge Detection

```python
# Canny edge detector — gradient magnitude + non-max suppression + hysteresis
edges = cv2.Canny(img_gray, threshold1=50, threshold2=150)

# Sobel — gradient in X and Y separately
sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, dx=1, dy=0, ksize=3)
sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, dx=0, dy=1, ksize=3)
magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

# Laplacian — second derivative (finds blobs + edges)
laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
```

### 2.4 Morphological Operations

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

erosion  = cv2.erode(binary_img, kernel, iterations=1)   # shrinks white regions
dilation = cv2.dilate(binary_img, kernel, iterations=1)  # grows white regions
opening  = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)   # erosion→dilation (removes noise)
closing  = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)  # dilation→erosion (fills holes)
gradient = cv2.morphologyEx(binary_img, cv2.MORPH_GRADIENT, kernel)  # dilation−erosion = outline
```

---

## 3. Feature Extraction

### 3.1 Classical Descriptors

```python
# ── ORB (fast, free, rotation/scale invariant) ──────────────────────────────
orb = cv2.ORB_create(nfeatures=500)
keypoints, descriptors = orb.detectAndCompute(img_gray, mask=None)
# descriptors: (N, 32) uint8 — binary descriptor

# ── SIFT (scale/rotation invariant, patent expired 2020) ────────────────────
sift = cv2.SIFT_create()
kp, des = sift.detectAndCompute(img_gray, None)
# des: (N, 128) float32

# ── Feature Matching ─────────────────────────────────────────────────────────
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)   # for ORB
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# FLANN matcher (faster for large descriptor sets)
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Lowe's ratio test — filter bad matches
good = [m for m, n in matches if m.distance < 0.75 * n.distance]
```

### 3.2 Texture Features

```python
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

# LBP — Local Binary Patterns (rotation-invariant texture)
radius = 3
n_points = 8 * radius
lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2,
                       range=(0, n_points + 2), density=True)

# GLCM — Gray-Level Co-occurrence Matrix
glcm = graycomatrix(img_gray, distances=[1, 3], angles=[0, np.pi/4, np.pi/2],
                    levels=256, symmetric=True, normed=True)
contrast    = graycoprops(glcm, 'contrast').mean()
homogeneity = graycoprops(glcm, 'homogeneity').mean()
energy      = graycoprops(glcm, 'energy').mean()
correlation = graycoprops(glcm, 'correlation').mean()
```

### 3.3 HOG — Histogram of Oriented Gradients

```python
from skimage.feature import hog

features, hog_image = hog(
    img_gray,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    visualize=True,
    channel_axis=None
)
# features: 1D vector describing gradient orientation histogram
# Classic use: pedestrian detection with SVM classifier
```

---

## 4. Image Segmentation

### 4.1 Threshold-Based

```python
# Global Otsu threshold — automatic threshold selection
_, otsu = cv2.threshold(img_gray, 0, 255,
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

### 4.2 Watershed

```python
def watershed_segment(img_bgr: np.ndarray) -> np.ndarray:
    """Segment touching objects using Watershed algorithm."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background (dilated)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Sure foreground (via distance transform)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.7 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Markers for watershed
    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(img_bgr, markers)
    img_bgr[markers == -1] = [0, 0, 255]   # boundaries in red
    return markers
```

### 4.3 Semantic Segmentation

Semantic segmentation assigns a class label to every pixel in the image. Unlike object detection (bounding boxes) or instance segmentation (per-object masks), semantic segmentation treats all instances of a class as one region.

```
Input:  RGB image (H, W, 3)
Output: label map  (H, W)    — each pixel = class index (0=background, 1=road, 2=car …)
```

#### Architectures Overview

```
FCN (2015)           — first fully-convolutional net; bilinear upsampling
SegNet (2016)        — encoder-decoder with pooling indices for upsampling
U-Net (2015)         — encoder-decoder with skip connections; standard for medical/satellite
PSPNet (2017)        — pyramid pooling module captures multi-scale context
DeepLab v3+ (2018)   — atrous (dilated) convolutions + ASPP + decoder; PASCAL/Cityscapes SOTA
SegFormer (2021)     — transformer encoder + lightweight MLP decoder; strong on ADE20K
Mask2Former (2022)   — unified architecture for semantic/instance/panoptic
```

#### Architecture Detail: DeepLab v3+

```
Input image
    ↓
Encoder (ResNet/MobileNet backbone with dilated conv)
    ↓
ASPP — Atrous Spatial Pyramid Pooling
  ├── 1×1 conv
  ├── dilated conv rate=6
  ├── dilated conv rate=12
  ├── dilated conv rate=18
  └── global average pooling
  → concatenate → 1×1 conv → features (H/16, W/16, 256)
    ↓
Decoder
  ├── bilinear upsample ×4
  ├── concat with low-level encoder features (H/4 resolution)
  └── 3×3 conv → 1×1 conv → (H/4, W/4, num_classes)
    ↓
Bilinear upsample ×4 → logits (H, W, num_classes)
    ↓
argmax → per-pixel class label
```

#### segmentation_models_pytorch — The Standard Library

```bash
pip install segmentation-models-pytorch
```

```python
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

# ── Build model ──────────────────────────────────────────────────────────────
model = smp.DeepLabV3Plus(
    encoder_name='resnet50',        # backbone: resnet18/34/50, efficientnet-b*, mit_b*
    encoder_weights='imagenet',     # pretrained weights
    in_channels=3,
    classes=19,                     # Cityscapes has 19 evaluation classes
)

# Other architectures (same API):
# smp.Unet(...)         — classic U-Net (best for small datasets)
# smp.FPN(...)          — Feature Pyramid Network decoder
# smp.PSPNet(...)       — Pyramid Scene Parsing
# smp.SegFormer(...)    — transformer encoder + MLP decoder

# ── Loss functions ────────────────────────────────────────────────────────────
# smp provides standard losses that work with logits (no sigmoid/softmax needed)
dice_loss = smp.losses.DiceLoss(mode='multiclass')
ce_loss   = nn.CrossEntropyLoss(ignore_index=255)   # 255 = void/ignore label

def combined_loss(logits, targets):
    return 0.5 * ce_loss(logits, targets) + 0.5 * dice_loss(logits, targets)

# ── Training step ─────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=100, power=0.9)

model.train()
for images, masks in train_loader:
    images = images.cuda()   # (B, 3, H, W) float32
    masks  = masks.cuda()    # (B, H, W) long  — class indices

    logits = model(images)   # (B, num_classes, H, W)
    loss   = combined_loss(logits, masks)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
scheduler.step()

# ── Inference ─────────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    logits = model(images)                  # (B, C, H, W)
    probs  = torch.softmax(logits, dim=1)   # (B, C, H, W)
    preds  = probs.argmax(dim=1)            # (B, H, W)
```

#### Metrics: mIoU and Pixel Accuracy

mIoU (mean Intersection-over-Union) is the standard metric. It is computed per-class then averaged.

```python
import torch

def compute_miou(preds: torch.Tensor, targets: torch.Tensor,
                 num_classes: int, ignore_index: int = 255) -> dict:
    """
    Compute per-class IoU and mIoU.

    Args:
        preds:   (B, H, W) long — predicted class per pixel
        targets: (B, H, W) long — ground truth class per pixel
    Returns:
        dict with 'miou', 'per_class_iou', 'pixel_acc'
    """
    mask = targets != ignore_index
    preds   = preds[mask]
    targets = targets[mask]

    per_class_iou = []
    for cls in range(num_classes):
        pred_cls   = preds == cls
        target_cls = targets == cls

        intersection = (pred_cls & target_cls).sum().float()
        union        = (pred_cls | target_cls).sum().float()

        if union == 0:
            per_class_iou.append(float('nan'))   # class not present
        else:
            per_class_iou.append((intersection / union).item())

    valid_iou = [v for v in per_class_iou if not torch.isnan(torch.tensor(v))]
    miou = sum(valid_iou) / len(valid_iou) if valid_iou else 0.0
    pixel_acc = (preds == targets).float().mean().item()

    return {
        'miou': miou,
        'per_class_iou': per_class_iou,
        'pixel_acc': pixel_acc,
    }

# Usage
metrics = compute_miou(preds, targets, num_classes=19)
print(f"mIoU: {metrics['miou']:.4f}  |  Pixel Acc: {metrics['pixel_acc']:.4f}")
```

```
Common mIoU benchmarks:
  Cityscapes val:  DeepLabV3+(R101) = 81.3%
                   SegFormer(B5)    = 84.0%
  ADE20K val:      SegFormer(B5)    = 51.8%
  PASCAL VOC:      DeepLabV3+       = 89.0%
```

#### YOLOv8-seg — Semantic + Instance in One Model

```python
from ultralytics import YOLO
import numpy as np
import cv2

# YOLOv8-seg does instance segmentation — masks per detected object
# For ADAS road segmentation, use a class-specific semantic projection
model = YOLO('yolov8n-seg.pt')   # n/s/m/l/x variants

results = model('frame.jpg', conf=0.4)

for r in results:
    if r.masks is not None:
        # masks.data: (N_instances, H, W) float32, values 0–1
        masks = r.masks.data.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        # Build semantic map by projecting instance masks
        H, W = r.orig_shape
        semantic_map = np.zeros((H, W), dtype=np.uint8)
        for mask, cls_id in zip(masks, classes):
            binary = (cv2.resize(mask, (W, H)) > 0.5)
            semantic_map[binary] = cls_id + 1   # 0 = background

# Fine-tune on custom segmentation dataset
model = YOLO('yolov8n-seg.pt')
model.train(
    data='seg_dataset.yaml',   # same format as detection, but images have masks
    epochs=100,
    imgsz=640,
    batch=8,
    device='cuda',
)
```

#### Dataset Format for Segmentation Training (YOLO)

```yaml
# seg_dataset.yaml
path: /data/road_seg
train: images/train
val:   images/val

nc: 5
names:
  0: road
  1: sidewalk
  2: car
  3: person
  4: vegetation
```

```
# Polygon mask label (YOLO seg format)
# labels/frame001.txt
# class_id  x1 y1  x2 y2  x3 y3  ...  (normalized polygon vertices)
0  0.10 0.95  0.45 0.55  0.90 0.55  0.95 0.95   ← road polygon
2  0.30 0.40  0.45 0.40  0.45 0.70  0.30 0.70   ← car bounding polygon
```

#### ADAS Road Segmentation with Cityscapes Classes

```python
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp

# Cityscapes 19-class palette (RGB)
CITYSCAPES_PALETTE = np.array([
    [128, 64, 128],   # 0  road
    [244, 35, 232],   # 1  sidewalk
    [ 70, 70, 70],   # 2  building
    [102, 102, 156],  # 3  wall
    [190, 153, 153],  # 4  fence
    [153, 153, 153],  # 5  pole
    [250, 170,  30],  # 6  traffic light
    [220, 220,   0],  # 7  traffic sign
    [107, 142,  35],  # 8  vegetation
    [152, 251, 152],  # 9  terrain
    [ 70, 130, 180],  # 10 sky
    [220,  20,  60],  # 11 person
    [255,   0,   0],  # 12 rider
    [  0,   0, 142],  # 13 car
    [  0,   0,  70],  # 14 truck
    [  0,  60, 100],  # 15 bus
    [  0,  80, 100],  # 16 train
    [  0,   0, 230],  # 17 motorcycle
    [119,  11,  32],  # 18 bicycle
], dtype=np.uint8)

CITYSCAPES_MEAN = np.array([0.485, 0.456, 0.406])
CITYSCAPES_STD  = np.array([0.229, 0.224, 0.225])


def preprocess_frame(frame_bgr: np.ndarray, size=(1024, 512)) -> torch.Tensor:
    """Resize and normalize a BGR frame for Cityscapes-trained model."""
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    img = (img - CITYSCAPES_MEAN) / CITYSCAPES_STD
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()


def colorize(label_map: np.ndarray) -> np.ndarray:
    """Map (H, W) class indices → (H, W, 3) RGB color image."""
    color = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for cls_id, rgb in enumerate(CITYSCAPES_PALETTE):
        color[label_map == cls_id] = rgb
    return color


def segment_frame(model, frame_bgr: np.ndarray, device='cuda') -> np.ndarray:
    """Run full segmentation pipeline on one frame. Returns color overlay."""
    H, W = frame_bgr.shape[:2]
    inp = preprocess_frame(frame_bgr).to(device)

    with torch.no_grad():
        logits = model(inp)                        # (1, 19, h, w)
        pred   = logits.argmax(1).squeeze().cpu().numpy()  # (h, w)

    # Upsample prediction to original resolution
    pred_full = cv2.resize(pred.astype(np.uint8), (W, H),
                           interpolation=cv2.INTER_NEAREST)
    color_mask = colorize(pred_full)

    # Blend with original frame
    overlay = cv2.addWeighted(frame_bgr, 0.5,
                              cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR), 0.5, 0)
    return overlay, pred_full


# Load Cityscapes pretrained model
model = smp.DeepLabV3Plus(
    encoder_name='resnet50',
    encoder_weights='imagenet',
    classes=19,
).cuda().eval()

# Run on video
cap = cv2.VideoCapture('dashcam.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break
    overlay, pred = segment_frame(model, frame)
    road_pixels = (pred == 0).sum()
    total_pixels = pred.size
    print(f"Road coverage: {100 * road_pixels / total_pixels:.1f}%")
    cv2.imshow('Semantic Segmentation', overlay)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

#### Quick Comparison: Semantic vs Instance vs Panoptic

```
Semantic segmentation:
  Output: (H, W) label map — every pixel has one class
  All cars = same "car" region
  Models: DeepLab, SegFormer, FCN

Instance segmentation:
  Output: N masks, one per object — same class objects get separate masks
  Car A mask, Car B mask, Car C mask
  Models: Mask R-CNN, YOLOv8-seg, SOLO, SOLOv2

Panoptic segmentation:
  Output: combines both — stuff (road, sky) as semantic, things (car, person) as instances
  Complete scene understanding
  Models: Panoptic FPN, Mask2Former, DETR (panoptic head)

ADAS typically needs:
  Drivable area      → semantic (road class)
  Obstacle avoidance → instance (each car/pedestrian separately)
  Full understanding → panoptic
```

### 4.4 Instance Segmentation with Mask R-CNN (PyTorch)

```python
import torchvision
import torch

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

from torchvision.transforms.functional import to_tensor
img_tensor = to_tensor(img_rgb).unsqueeze(0)

with torch.no_grad():
    predictions = model(img_tensor)[0]

# predictions keys: 'boxes', 'labels', 'scores', 'masks'
for i, score in enumerate(predictions['scores']):
    if score > 0.5:
        box  = predictions['boxes'][i].int().numpy()    # [x1, y1, x2, y2]
        mask = predictions['masks'][i, 0].numpy() > 0.5  # bool mask (H, W)
        label = predictions['labels'][i].item()
```

---

## 5. Object Detection

### 5.1 Detection Paradigms

```
Two-stage (high accuracy, slower):
  Region proposals → classify each proposal
  Examples: Faster R-CNN, Mask R-CNN

One-stage (real-time, slightly lower accuracy):
  Grid/anchor predictions in single pass
  Examples: YOLO, SSD, RetinaNet, FCOS

Anchor-free (modern, simpler):
  No pre-defined anchor boxes
  Examples: CenterNet, FCOS, RT-DETR
```

### 5.2 YOLO — You Only Look Once

```python
# YOLOv8 — Ultralytics (easiest modern YOLO)
# pip install ultralytics

from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n.pt')   # nano (fastest), also: s, m, l, x

# Inference
results = model('image.jpg', conf=0.4, iou=0.45)

for r in results:
    for box in r.boxes:
        xyxy  = box.xyxy[0].numpy()   # [x1, y1, x2, y2]
        conf  = float(box.conf)
        cls   = int(box.cls)
        label = model.names[cls]
        print(f"{label}: {conf:.2f} at {xyxy}")

# Fine-tune on custom dataset
model.train(
    data='custom_dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device='cuda',
    patience=20,        # early stopping
    optimizer='AdamW',
    lr0=0.001
)

# Export to ONNX / TensorRT
model.export(format='onnx', opset=12, simplify=True)
model.export(format='engine', device=0, half=True)  # TensorRT FP16
```

### 5.3 Custom YOLO Dataset YAML

```yaml
# custom_dataset.yaml
path: /data/my_dataset      # root path
train: images/train
val: images/val
test: images/test

nc: 3                       # number of classes
names:
  0: forklift
  1: person
  2: pallet
```

### 5.4 Detection Metrics

| Metric | Definition |
|--------|-----------|
| **IoU** | Intersection over Union — box overlap quality |
| **Precision** | TP / (TP + FP) — of all detections, how many are correct |
| **Recall** | TP / (TP + FN) — of all GT objects, how many were found |
| **AP** | Area under Precision-Recall curve (per class) |
| **mAP@50** | Mean AP at IoU threshold 0.5 |
| **mAP@50:95** | Mean AP averaged over IoU 0.5–0.95 (COCO standard) |

```python
# Evaluate with Ultralytics
metrics = model.val(data='custom_dataset.yaml')
print(metrics.box.map)      # mAP@50:95
print(metrics.box.map50)    # mAP@50
print(metrics.box.mp)       # mean precision
print(metrics.box.mr)       # mean recall
```

---

## 6. Object Tracking

### 6.1 Classic Trackers in OpenCV

```python
# Create tracker — options: CSRT (accurate), KCF (fast), MIL
tracker = cv2.TrackerCSRT_create()

# Initialize with first frame + bounding box
ret, frame = cap.read()
bbox = cv2.selectROI('Select Object', frame, fromCenter=False)
tracker.init(frame, bbox)

while True:
    ret, frame = cap.read()
    ok, bbox = tracker.update(frame)
    if ok:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    else:
        cv2.putText(frame, 'Tracking failure', (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
```

### 6.2 ByteTrack — Multi-Object Tracking

ByteTrack is the current standard for real-time MOT (pairs with any detector).

```bash
pip install lap  # linear assignment
pip install git+https://github.com/ifzhang/ByteTrack.git
```

```python
from ultralytics import YOLO

# Ultralytics has ByteTrack built-in
model = YOLO('yolov8n.pt')

# Track across video frames
results = model.track(
    source='video.mp4',
    tracker='bytetrack.yaml',
    persist=True,       # keep track IDs between frames
    conf=0.3,
    iou=0.5
)

for r in results:
    if r.boxes.id is not None:
        track_ids = r.boxes.id.int().tolist()
        boxes = r.boxes.xyxy.tolist()
        for tid, box in zip(track_ids, boxes):
            print(f"Track {tid}: {box}")
```

### 6.3 Optical Flow

```python
# Lucas-Kanade sparse optical flow — tracks specific points
feature_params = dict(maxCorners=100, qualityLevel=0.3,
                      minDistance=7, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

while True:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

# Farneback dense optical flow — flow for every pixel
flow = cv2.calcOpticalFlowFarneback(
    old_gray, frame_gray, None,
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
)
mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
```

---

## 7. 3D Vision

### 7.1 Camera Model and Calibration

```python
# Checkerboard calibration
import cv2
import numpy as np

CHECKERBOARD = (9, 6)   # inner corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints, imgpoints = [], []
for fname in calibration_images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)
# K: 3×3 intrinsic matrix [[fx,0,cx],[0,fy,cy],[0,0,1]]
# dist: [k1,k2,p1,p2,k3] distortion coefficients

# Undistort an image
h, w = img.shape[:2]
newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))
undistorted = cv2.undistort(img, K, dist, None, newK)
```

### 7.2 Stereo Vision — Depth from Two Cameras

```python
# After calibrating both cameras separately, stereo calibrate
ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    obj_pts, img_pts_left, img_pts_right,
    K1, D1, K2, D2, gray.shape[::-1],
    flags=cv2.CALIB_FIX_INTRINSIC
)

# Rectify — make epipolar lines horizontal
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K1, D1, K2, D2, gray.shape[::-1], R, T
)

# Compute disparity map
stereo = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=128, blockSize=11,
    P1=8*3*11**2, P2=32*3*11**2,
    disp12MaxDiff=1, uniquenessRatio=15,
    speckleWindowSize=100, speckleRange=32
)
disparity = stereo.compute(left_rect, right_rect).astype(np.float32) / 16.0

# Convert disparity to depth
# depth = (baseline × focal_length) / disparity
baseline = np.linalg.norm(T)          # meters between cameras
depth = (baseline * K1[0, 0]) / (disparity + 1e-6)

# Or use the Q matrix from stereoRectify
points_3D = cv2.reprojectImageTo3D(disparity, Q)  # (H, W, 3) XYZ per pixel
```

### 7.3 Pose Estimation (solvePnP)

```python
# Given 3D object points and their 2D image projections, find camera pose
# object_points: (N, 3) float32 — 3D coordinates in object frame
# image_points:  (N, 2) float32 — corresponding 2D pixel coordinates

success, rvec, tvec = cv2.solvePnP(
    object_points, image_points, K, dist,
    flags=cv2.SOLVEPNP_ITERATIVE
)

# rvec: rotation vector (Rodrigues format) → convert to matrix
R, _ = cv2.Rodrigues(rvec)   # 3×3 rotation matrix
# tvec: translation vector (camera-to-object)

# Project 3D points back to image
projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)

# Draw axes on object
cv2.drawFrameAxes(img, K, dist, rvec, tvec, length=0.05)
```

### 7.4 Multi-Camera ADAS: openpilot Near + Wide Camera

openpilot (comma.ai) is the reference open-source ADAS stack. It uses two forward-facing cameras with deliberately different fields of view — a design decision that reveals the fundamental trade-off in ADAS perception.

#### Why Two Cameras?

```
Single camera problem:
  Wide FOV (120°) → low focal length → poor resolution at distance
                  → lane lines 50m ahead are only a few pixels wide
                  → lead vehicle at 80m is too small to detect reliably

  Narrow FOV (60°) → high focal length → good long-range resolution
                   → misses adjacent lanes and near-field cut-ins
                   → blind to pedestrians at crossings

Solution: run both simultaneously, fuse their outputs per task
```

```
comma 3 / 3X camera layout:
  ┌─────────────────────────────────────────────┐
  │         windshield mount (top center)        │
  │                                              │
  │  [Wide road cam]   [Road cam]   [Driver cam] │
  │   ~120° HFOV       ~60° HFOV   (inward)     │
  │   1.71 mm lens     2.2 mm lens               │
  └─────────────────────────────────────────────┘
```

#### Camera Specifications

| Property | Road (Narrow) | Wide Road |
|----------|--------------|-----------|
| Sensor | OV8856 | OV8856 |
| Focal length (approx) | 2.2 mm | 1.71 mm |
| HFOV | ~60° | ~120° |
| Native resolution | 1928×1208 | 1928×1208 |
| Model input crop | 1152×1152 | 1152×1152 |
| Resized for model | 512×256 → 128×256 | 512×256 → 128×256 |
| Primary use | Lead vehicle, lane geometry, long range | Adjacent lanes, near-field, wider view |

#### Intrinsic Matrix for Each Camera

```python
import numpy as np

# comma 3 road camera intrinsics (approximate — calibrated per device)
# K = [[fx,  0, cx],
#       [ 0, fy, cy],
#       [ 0,  0,  1]]

K_road = np.array([
    [2648.0,    0.0,  964.0],
    [   0.0, 2648.0,  604.0],
    [   0.0,    0.0,    1.0],
], dtype=np.float64)

K_wide = np.array([
    [1036.0,    0.0,  964.0],
    [   0.0, 1036.0,  604.0],
    [   0.0,    0.0,    1.0],
], dtype=np.float64)

# Focal lengths: road (2648 px) vs wide (1036 px)
# Lower fx/fy = shorter focal length = wider field of view
# Both cameras at same image center (cx=964, cy=604) assuming no lens offset

def pixel_to_ray(u: float, v: float, K: np.ndarray) -> np.ndarray:
    """Convert pixel (u, v) to unit direction ray in camera frame."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    return ray / np.linalg.norm(ray)

# Road camera: pixel at image center-right (u=1200, v=604)
ray_road = pixel_to_ray(1200, 604, K_road)
angle_road = np.degrees(np.arctan2(ray_road[0], ray_road[2]))
print(f"Road cam angle: {angle_road:.1f}°")   # ~8.7° off-center

# Wide camera: same pixel
ray_wide = pixel_to_ray(1200, 604, K_wide)
angle_wide = np.degrees(np.arctan2(ray_wide[0], ray_wide[2]))
print(f"Wide cam angle: {angle_wide:.1f}°")   # ~22.5° off-center
```

#### YUV420 Frame Packing — supercombo Input

The supercombo model takes `input_imgs` of shape `(1, 12, 128, 256)`. These 12 channels encode two consecutive YUV420 frames (temporal context for motion estimation).

```
YUV420 memory layout for one frame (H=128, W=256):
  Y  plane: 128 × 256 = 32768 bytes  (full luma)
  U  plane:  64 × 128 =  8192 bytes  (half-res chroma)
  V  plane:  64 × 128 =  8192 bytes  (half-res chroma)
  Total: 49152 bytes per frame

Packed into 6 channels at (128, 256):
  ch 0: Y[0::2, 0::2]   — even rows, even cols  (64×128, tiled to 128×256)
  ch 1: Y[0::2, 1::2]   — even rows, odd cols
  ch 2: Y[1::2, 0::2]   — odd rows, even cols
  ch 3: Y[1::2, 1::2]   — odd rows, odd cols
  ch 4: U resized to 128×256 via nearest-neighbor
  ch 5: V resized to 128×256 via nearest-neighbor

Two frames (t and t-1) → 6 × 2 = 12 channels
```

```python
import numpy as np
import cv2


def yuv420_to_6ch(yuv420_frame: np.ndarray, out_h: int = 128, out_w: int = 256) -> np.ndarray:
    """
    Convert a YUV420 frame to the 6-channel format used by supercombo.

    Args:
        yuv420_frame: raw YUV420 bytes as (H*3//2, W) uint8 array,
                      OR (H + H//2, W) array (standard cv2 YUV420 layout).
                      For comma 3: H=886 (after crop), W=1152.
        out_h, out_w: target spatial size (128, 256 for supercombo)

    Returns:
        (6, out_h, out_w) float32 array, values in [0, 1]
    """
    h_full = yuv420_frame.shape[0] * 2 // 3
    w_full = yuv420_frame.shape[1]

    # Split Y, U, V planes
    Y = yuv420_frame[:h_full, :]                                  # (H, W)
    uv = yuv420_frame[h_full:, :].reshape(h_full // 2, w_full)
    U = uv[:h_full // 4, :]                                      # (H/2, W/2) after reshape
    V = uv[h_full // 4:, :]

    # Resize Y to target
    Y_resized = cv2.resize(Y, (out_w, out_h), interpolation=cv2.INTER_AREA)

    # Sub-sample Y into 4 interleaved channels
    ch0 = Y_resized[0::2, 0::2]   # (out_h//2, out_w//2)
    ch1 = Y_resized[0::2, 1::2]
    ch2 = Y_resized[1::2, 0::2]
    ch3 = Y_resized[1::2, 1::2]

    # Resize to full (out_h, out_w) via repeat
    ch0 = np.repeat(np.repeat(ch0, 2, axis=0), 2, axis=1)
    ch1 = np.repeat(np.repeat(ch1, 2, axis=0), 2, axis=1)
    ch2 = np.repeat(np.repeat(ch2, 2, axis=0), 2, axis=1)
    ch3 = np.repeat(np.repeat(ch3, 2, axis=0), 2, axis=1)

    # Resize U and V to (out_h, out_w)
    ch4 = cv2.resize(U, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    ch5 = cv2.resize(V, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    channels = np.stack([ch0, ch1, ch2, ch3, ch4, ch5], axis=0)  # (6, H, W) uint8
    return channels.astype(np.float32) / 255.0


def build_supercombo_input(frame_t: np.ndarray, frame_t1: np.ndarray) -> np.ndarray:
    """
    Stack two consecutive YUV420 frames into supercombo input_imgs.

    Args:
        frame_t:  current frame YUV420 raw array
        frame_t1: previous frame YUV420 raw array

    Returns:
        (1, 12, 128, 256) float32 — ready for supercombo inference
    """
    ch_t  = yuv420_to_6ch(frame_t)    # (6, 128, 256)
    ch_t1 = yuv420_to_6ch(frame_t1)   # (6, 128, 256)
    stacked = np.concatenate([ch_t, ch_t1], axis=0)   # (12, 128, 256)
    return stacked[np.newaxis]                         # (1, 12, 128, 256)


# Load two consecutive dashcam frames (BGR → YUV420)
frame_bgr_0 = cv2.imread('frame_000.jpg')
frame_bgr_1 = cv2.imread('frame_001.jpg')

yuv_0 = cv2.cvtColor(frame_bgr_0, cv2.COLOR_BGR2YUV_I420)
yuv_1 = cv2.cvtColor(frame_bgr_1, cv2.COLOR_BGR2YUV_I420)

input_imgs = build_supercombo_input(yuv_0, yuv_1)
print(input_imgs.shape)  # (1, 12, 128, 256)
print(input_imgs.min(), input_imgs.max())  # 0.0  1.0
```

#### Road vs Wide: Per-Task Routing

```
Task                     Primary camera       Reason
──────────────────────────────────────────────────────────────────────
Lane detection (far)     Road (narrow)        High focal length → lane lines at 50m are wider
Lead vehicle distance    Road (narrow)        Accurate size → distance estimation
Adjacent lane cuts       Wide                 FOV covers 2 lanes either side
Near-field pedestrians   Wide                 Pedestrians at <10m outside road cam FOV
Traffic sign reading     Road (narrow)        Signs need text resolution at distance
Blind spot monitoring    Wide                 ~120° catches what road cam misses at sides
Ego-motion estimation    Road (narrow)        Stable horizon → cleaner optical flow
```

openpilot's supercombo model receives **only the road camera** frame as `input_imgs`. The wide camera feeds a separate path (e.g., driver monitoring, wide-angle object detection). The two outputs are fused in `controlsd` — the lateral controller uses road-camera-derived lane predictions while wide-angle detections update the obstacle map.

#### Projecting Between Camera Coordinate Systems

```python
import numpy as np


def project_road_to_wide(
    pts_road_cam: np.ndarray,   # (N, 3) 3D points in road camera frame
    R_r2w: np.ndarray,          # (3, 3) rotation: road cam → wide cam
    t_r2w: np.ndarray,          # (3,) translation: road cam → wide cam
    K_wide: np.ndarray,         # (3, 3) wide cam intrinsics
    D_wide: np.ndarray = None,  # (5,) wide cam distortion (optional)
) -> np.ndarray:
    """
    Project 3D points seen in road camera frame into wide camera pixel coordinates.
    Used to cross-validate detections across cameras.

    Returns:
        (N, 2) pixel coordinates in wide camera image
    """
    # Transform points to wide camera frame
    pts_wide_cam = (R_r2w @ pts_road_cam.T).T + t_r2w   # (N, 3)

    # Project to wide image plane
    if D_wide is not None:
        pts_2d, _ = cv2.projectPoints(
            pts_wide_cam.astype(np.float64),
            np.zeros(3), np.zeros(3),   # identity (already in cam frame)
            K_wide, D_wide
        )
        return pts_2d.squeeze()   # (N, 2)
    else:
        # Simple pinhole projection (no distortion)
        z = pts_wide_cam[:, 2:3].clip(0.01, None)
        xy = pts_wide_cam[:, :2] / z
        uv = (K_wide[:2, :2] @ xy.T + K_wide[:2, 2:3]).T
        return uv   # (N, 2)


# Example: project lead vehicle center from road cam 3D into wide cam 2D
# Lead vehicle detected at 30m ahead, ~0m lateral, ~1.5m height
lead_vehicle_3d = np.array([[30.0, 0.0, 1.5]])   # (N, 3) in road cam frame

# Extrinsic: wide cam is ~5mm to the left, same orientation (approx)
R_r2w = np.eye(3)   # cameras are nearly parallel
t_r2w = np.array([-0.005, 0.0, 0.0])   # 5mm lateral offset

uv_in_wide = project_road_to_wide(lead_vehicle_3d, R_r2w, t_r2w, K_wide)
print(f"Lead vehicle projects to wide cam pixel: {uv_in_wide}")
```

#### Camera-to-Ground Homography (Flat Road Assumption)

```python
def camera_to_ground_homography(K: np.ndarray,
                                 camera_height_m: float = 1.22,
                                 pitch_deg: float = 0.0) -> np.ndarray:
    """
    Compute homography mapping image pixels → ground plane (Z=0) coordinates.

    Assumes flat road. Used for:
      - Lane width estimation in meters
      - Lead vehicle distance from pixel height
      - Free-space estimation

    Args:
        K:                camera intrinsics
        camera_height_m:  camera height above road (comma 3: ~1.22m)
        pitch_deg:        downward pitch of camera (positive = looking down)

    Returns:
        H: (3, 3) homography — pixel (u,v,1) → ground (X, Y, 1) in meters
    """
    pitch = np.radians(pitch_deg)
    # Rotation: camera tilted downward by pitch
    R = np.array([
        [1,           0,            0],
        [0,  np.cos(pitch), -np.sin(pitch)],
        [0,  np.sin(pitch),  np.cos(pitch)],
    ])
    t = np.array([0, -camera_height_m, 0])   # camera above ground

    # Ground plane normal in world: [0, 1, 0], d = 0
    # Homography from image to ground (standard derivation)
    # P = K [R | t], solve for intersection with Y=0 plane
    P = K @ np.hstack([R, t[:, None]])
    # Columns 0, 2, 3 of P (drop Y column since Y=0 on ground)
    H = P[:, [0, 2, 3]]
    return np.linalg.inv(H)


def pixel_to_ground(u: float, v: float, H: np.ndarray):
    """Convert image pixel to ground plane coordinates (meters from camera)."""
    p = H @ np.array([u, v, 1.0])
    return p[0] / p[2], p[1] / p[2]   # (X_meters, Z_meters)


H_road = camera_to_ground_homography(K_road, camera_height_m=1.22, pitch_deg=1.5)
H_wide = camera_to_ground_homography(K_wide, camera_height_m=1.22, pitch_deg=1.5)

# Where does bottom-center of image touch the road in road cam?
x, z = pixel_to_ground(964, 1100, H_road)
print(f"Road cam bottom-center → ({x:.2f}m lateral, {z:.2f}m ahead)")

# Same pixel in wide cam
x_w, z_w = pixel_to_ground(964, 1100, H_wide)
print(f"Wide cam bottom-center → ({x_w:.2f}m lateral, {z_w:.2f}m ahead)")
```

#### openpilot Model Input Summary

```
supercombo inputs:
  input_imgs:             (1, 12, 128, 256)  ← 2 consecutive road cam YUV420 frames
  desire:                 (1, 8)             ← one-hot: lane change L/R, keep, turn L/R
  traffic_convention:     (1, 2)             ← [RHD, LHD] traffic direction
  lateral_control_params: (1, 2)             ← [v_ego, roll]
  nav_features:           (1, 64)            ← map/route embedding
  nav_instructions:       (1, 150)           ← turn-by-turn instruction vector

supercombo output:
  outputs: flat float32 vector (~6504 values)
  Parsed by openpilot modeldata.py into:
    lead:         lead vehicle position, velocity, acceleration
    path:         lane line polynomials (4 points × 33 time steps)
    desire_state: predicted driver intent
    meta:         model confidence, disengagement probability
    pose:         ego velocity and orientation

Wide camera (separate model / separate input stream):
  Feeds wide-angle obstacle detection (pedestrians, cyclists at ±60° off-center)
  Output merged in controlsd with supercombo lead predictions
```

---

## 8. Advanced Deep Learning Architectures

### 8.1 Vision Transformer (ViT)

```python
# ViT splits image into fixed patches, treats each as a "token"
# Self-attention captures global relationships between any two patches

from transformers import ViTForImageClassification, ViTImageProcessor
import torch

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
model.eval()

from PIL import Image
img = Image.open('image.jpg')
inputs = processor(images=img, return_tensors='pt')

with torch.no_grad():
    logits = model(**inputs).logits
pred_class = logits.argmax(-1).item()
print(model.config.id2label[pred_class])
```

### 8.2 DETR — Detection Transformer

```python
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image

processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')

img = Image.open('image.jpg')
inputs = processor(images=img, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)

# Post-process to get boxes and labels
target_sizes = torch.tensor([img.size[::-1]])
results = processor.post_process_object_detection(
    outputs, threshold=0.9, target_sizes=target_sizes)[0]

for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
    print(f"{model.config.id2label[label.item()]}: {score:.2f} {box.tolist()}")
```

### 8.3 RT-DETR — Real-Time Detection Transformer

RT-DETR combines transformer accuracy with YOLO-level speed (runs in real-time on GPU).

```python
from ultralytics import RTDETR

model = RTDETR('rtdetr-l.pt')   # large variant
results = model.predict('image.jpg', conf=0.4)
model.export(format='engine')   # TensorRT export
```

### 8.4 SAM — Segment Anything Model

```python
# pip install segment-anything
from segment_anything import sam_model_registry, SamPredictor
import numpy as np

sam = sam_model_registry['vit_b'](checkpoint='sam_vit_b_01ec64.pth')
predictor = SamPredictor(sam)

predictor.set_image(img_rgb)

# Prompt with point (x, y) + foreground/background label
input_point = np.array([[500, 375]])
input_label = np.array([1])   # 1=foreground, 0=background

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,  # returns 3 masks (small/medium/large)
)
best_mask = masks[np.argmax(scores)]   # (H, W) bool
```

---

## 9. OpenCV — Core to Advanced

### 9.1 DNN Module — Run Models Without PyTorch/TF

```python
# Load any ONNX model
net = cv2.dnn.readNetFromONNX('yolov8n.onnx')

# Use CUDA backend on GPU
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Or use OpenVINO for Intel CPUs
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Preprocess image into blob
blob = cv2.dnn.blobFromImage(
    img_bgr,
    scalefactor=1.0/255,
    size=(640, 640),
    mean=(0, 0, 0),
    swapRB=True,        # BGR→RGB
    crop=False
)
net.setInput(blob)
outputs = net.forward(net.getUnconnectedOutLayersNames())
```

### 9.2 CUDA Acceleration

```python
# Check CUDA support at runtime
print(cv2.cuda.getCudaEnabledDeviceCount())   # 0 if no CUDA build

# Upload mat to GPU
gpu_mat = cv2.cuda_GpuMat()
gpu_mat.upload(img_gray)

# GPU operations
gpu_blurred = cv2.cuda.createGaussianFilter(
    cv2.CV_8UC1, cv2.CV_8UC1, (5, 5), 0
).apply(gpu_mat)

gpu_edges = cv2.cuda.createCannyEdgeDetector(50, 150).detect(gpu_mat)

# Download back to CPU
result = gpu_edges.download()
```

### 9.3 Video Pipeline with GStreamer

```python
# CSI camera on Jetson (zero-copy path)
pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink"
)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

# RTSP stream input
rtsp_pipeline = "rtspsrc location=rtsp://192.168.1.50/stream ! decodebin ! videoconvert ! appsink"
cap = cv2.VideoCapture(rtsp_pipeline, cv2.CAP_GSTREAMER)
```

### 9.4 G-API — Graph API for Pipelines

G-API lets you define your vision pipeline as a computation graph, enabling automatic optimization and backend switching.

```python
import cv2 as cv

# Define the graph
g_in  = cv.GMat()
g_gray = cv.gapi.BGR2Gray(g_in)
g_blur = cv.gapi.gaussianBlur(g_gray, (5, 5), 0)
g_edges = cv.gapi.Canny(g_blur, 50, 150)

# Compile
pipeline = cv.GComputation(cv.GIn(g_in), cv.GOut(g_edges))
compiled = pipeline.compileStreaming()

compiled.setSource(cv.gin(cv.gapi.wip.make_capture_src(0)))
compiled.start()

while True:
    ok, (edges,) = compiled.pull()
    if not ok:
        break
    cv2.imshow('Edges', edges)
```

---

## 10. Annotation Tools

Data annotation is the foundation of supervised learning. The quality and format of your annotations directly determines model performance. Choosing the right tool depends on your scale, team size, and output format requirements.

### Overview Comparison

| Tool | Type | Price | AI-Assist | Formats | Best for |
|------|------|-------|-----------|---------|----------|
| **X-AnyLabeling** | Desktop | Free | ✓ GPU | COCO, VOC, YOLO, DOTA, MOT, MASK | AI-assisted single/batch annotation |
| **CVAT** | Web / Self-hosted | Free | ✓ | Pascal VOC, COCO, YOLO, Datumaro | Team collaboration, video tracking |
| **LabelImg** | Desktop | Free | ✗ | Pascal VOC, YOLO | Quick single-annotator labeling |
| **RectLabel** | Desktop (macOS) | Paid | ✓ | COCO, VOC, YOLO | macOS users, fast polygon annotation |
| **VGG VIA** | Browser | Free | ✗ | JSON, CSV | Lightweight, no install needed |
| **Labelbox** | Cloud SaaS | Freemium | ✓ | COCO, VOC, YOLO + custom | Enterprise scale, team management |
| **Roboflow** | Cloud | Freemium | ✓ | YOLO, COCO, VOC, TFRecord, TensorFlow | Dataset versioning, augmentation, train & deploy, edge export |
| **COCO Annotator** | Web / Self-hosted | Free | ✓ (limited) | COCO JSON | COCO-format segmentation datasets |

---

### 10.1 X-AnyLabeling

X-AnyLabeling is the most feature-rich open-source annotation tool — it embeds AI inference engines directly, enabling one-click predictions on images and videos.

**GitHub:** [github.com/CVHub520/X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling)

#### Key Features

```
AI-Assisted Annotation:
  Embedded models: YOLOv5/v8/v9/v10, SAM, GroundingDINO, DINO,
                   RT-DETR, CLIP, DepthAnything, and more
  GPU acceleration via ONNX Runtime with CUDA EP
  Single-frame prediction → manually correct → next frame
  Batch prediction → annotate entire folder in one click

Annotation Types:
  Polygons                  (fine-grained segmentation)
  Rectangles                (bounding boxes)
  Rotated boxes             (DOTA / aerial imagery)
  Circles
  Lines / polylines
  Points / keypoints

Format Support:
  Import:  COCO, VOC, YOLO, DOTA, MOT, MASK, LabelMe
  Export:  COCO, VOC, YOLO, DOTA, MOT, MASK, LabelMe, ODVG

Video Support:
  Frame-by-frame annotation
  Auto-tracking using embedded trackers
  Batch inference on video files
```

#### Installation

```bash
# Via pip (recommended)
pip install anylabeling

# With GPU support (ONNX Runtime CUDA)
pip install anylabeling[gpu]

# Or via release binary:
# Download from github.com/CVHub520/X-AnyLabeling/releases
# Available: .exe (Windows), .AppImage (Linux), .dmg (macOS)

# Run
anylabeling
```

#### Workflow

```
1. Open X-AnyLabeling
2. File → Open Dir → select your image folder
3. AI Models → select a model (e.g., YOLOv8n, SAM)
   - First use: model downloads automatically
4. Predict:
   - Single frame: Ctrl+M or click "Run Model"
   - Batch: Tools → Auto Labeling → Run on All Images
5. Review predictions, correct mistakes manually
6. File → Export → choose format (COCO / YOLO / VOC)
```

#### Export Formats from X-AnyLabeling

```bash
# YOLO format export example output:
# labels/image001.txt
# 0 0.512 0.334 0.241 0.189     ← class cx cy w h (normalized 0–1)

# COCO format export:
# annotations.json — single file with all boxes + images list

# VOC format export:
# Annotations/image001.xml — one XML per image

# DOTA format (rotated boxes for aerial imagery):
# labels/image001.txt
# 100 200 150 200 150 250 100 250 vehicle 0   ← 4 corner points + class + difficult
```

---

### 10.2 CVAT

CVAT (Computer Vision Annotation Tool) is the industry-standard open-source annotation platform, developed by Intel and used at scale by major ML teams.

**GitHub:** [github.com/cvat-ai/cvat](https://github.com/cvat-ai/cvat)
**Hosted version:** [app.cvat.ai](https://app.cvat.ai)

#### Key Features

```
Annotation Types:
  Bounding boxes (2D + 3D)
  Polygons / polylines / points
  Masks (brush tool, superpixel segmentation)
  Keypoints + skeletons (pose estimation)
  Cuboids (3D objects in 2D images)
  LiDAR point cloud annotation (3D bboxes)

Video Annotation:
  Frame-by-frame OR annotate keyframes → interpolation fills gaps
  Semi-automatic tracking (SiamMask, OpenCV trackers)
  Object track IDs across frames (MOT format)

AI Assistance:
  SAM (Segment Anything) integration — click → mask
  Detection models via Nuclio serverless functions
  Interactors: DEXTR, f-BRS for polygon assistance

Collaboration:
  Role-based access: annotator / reviewer / owner
  Task assignment across multiple annotators
  Review workflow with accept/reject per annotation

Format Support:
  Export: Pascal VOC, COCO JSON, YOLO, TFRecord, MOT,
          Cityscapes, KITTI, LFW, Wider Face, VGGFace2
  Import: All of the above
```

#### Self-Hosted Setup

```bash
# Clone and start with Docker Compose
git clone https://github.com/cvat-ai/cvat.git
cd cvat

# Start all services
docker compose up -d

# Create admin user
docker exec -it cvat_server python manage.py createsuperuser

# Access at http://localhost:8080
```

#### CVAT Python SDK — Programmatic Annotation

```python
# pip install cvat-sdk

from cvat_sdk import make_client
from cvat_sdk.models import TaskWriteRequest, DataRequest

with make_client(host='localhost', credentials=('user', 'password')) as client:
    # Create task
    task = client.tasks.create(TaskWriteRequest(
        name='My Detection Task',
        labels=[
            {'name': 'car',    'color': '#ff0000'},
            {'name': 'person', 'color': '#00ff00'},
        ]
    ))

    # Upload images
    task.upload_data(DataRequest(
        image_quality=95,
        server_files=['path/to/images/']
    ))

    # Export annotations
    task.export_dataset(
        format_name='COCO 1.0',
        filename='annotations.zip',
        include_images=False
    )
```

#### Export from CVAT — YOLO Format Structure

```
cvat_export/
├── obj_train_data/
│   ├── image001.txt       ← YOLO label file
│   ├── image002.txt
│   └── ...
├── obj.data               ← paths config
├── obj.names              ← class names, one per line
└── train.txt              ← list of image paths
```

---

### 10.3 LabelImg

LabelImg is the classic, lightweight, single-annotator tool — simple, battle-tested, and universally supported.

**GitHub:** [github.com/HumanSignal/labelImg](https://github.com/HumanSignal/labelImg)

#### Installation

```bash
pip install labelImg

# Or from source
git clone https://github.com/HumanSignal/labelImg.git
cd labelImg
pip install pyqt5 lxml
python labelImg.py
```

#### Key Shortcuts

| Key | Action |
|-----|--------|
| `W` | Create rectangle box |
| `D` | Next image |
| `A` | Previous image |
| `Ctrl+S` | Save annotation |
| `Ctrl+R` | Change save directory |
| `Del` | Delete selected box |

#### Workflow

```
1. Open Dir → select image folder
2. Change Save Dir → set label output folder
3. Toggle format: PascalVOC or YOLO (bottom-left of window)
4. Press W → draw box → enter class name → confirm
5. Ctrl+S → save → D (next image) → repeat
6. Result:
   YOLO:  labels/image001.txt   (class cx cy w h normalized)
   VOC:   Annotations/image001.xml
```

#### Pascal VOC XML Format

```xml
<!-- Annotations/image001.xml -->
<annotation>
  <folder>images</folder>
  <filename>image001.jpg</filename>
  <size>
    <width>1920</width><height>1080</height><depth>3</depth>
  </size>
  <object>
    <name>car</name>
    <difficult>0</difficult>
    <bndbox>
      <xmin>452</xmin><ymin>78</ymin>
      <xmax>929</xmax><ymax>534</ymax>
    </bndbox>
  </object>
</annotation>
```

---

### 10.4 RectLabel

RectLabel is a polished commercial annotation tool for macOS, focused on speed for individual annotators.

**Website:** [rectlabel.com](https://rectlabel.com)

#### Key Features

```
Annotation Types:
  Bounding boxes
  Polygons
  Polylines
  Points
  Keypoints with skeleton templates
  Oriented bounding boxes

Unique Capabilities:
  Core ML integration — annotate with Apple Neural Engine
  Pre-labeling via Core ML detection models
  Tracking between video frames
  Label history / auto-complete class names
  Attribute support (color, material, truncated, occluded)

Export Formats:
  COCO JSON
  Pascal VOC XML
  YOLO
  CreateML JSON
  CSV

macOS Features:
  Native SwiftUI app (fast, no Electron overhead)
  Drag-and-drop image/video import
  Full keyboard shortcut customization
```

#### When to Use RectLabel

```
✓ Solo annotator on macOS
✓ Need fast Core ML pre-labeling (uses Apple Silicon Neural Engine)
✓ Dataset has oriented/rotated objects (satellite, aerial)
✓ Need keypoint/skeleton annotation (sports, medical)

✗ Team collaboration (single user license)
✗ Linux/Windows (macOS only)
✗ Free budget required
```

---

### 10.5 VGG Image Annotator (VIA)

VIA is a zero-install browser-based annotator — just open an HTML file and start labeling. Developed by Oxford VGG.

**Website:** [robots.ox.ac.uk/~vgg/software/via/](https://www.robots.ox.ac.uk/~vgg/software/via/)
**GitHub:** [github.com/ox-vgg/via](https://github.com/ox-vgg/via)

#### Key Features

```
No installation:
  Single HTML file — open in any browser, works offline
  Load local images directly (no server upload needed)
  All data stays on your machine

Annotation Types:
  Bounding boxes
  Polygons
  Polylines
  Points
  Circles / ellipses

Attributes:
  Define custom attributes per region (class, color, quality, etc.)
  Checkbox / radio / text / number attribute types

Export Formats:
  VIA JSON (native format)
  CSV (one row per annotation)
  COCO JSON (via VIA3 version)
```

#### Usage

```
1. Download via.html from robots.ox.ac.uk/~vgg/software/via/
2. Open in browser (no server needed)
3. Add Files → select images from your disk
4. Define attributes (e.g., "class": type=radio, options=car,person,bike)
5. Draw regions: select shape → draw → set attribute values
6. Annotations → Export → JSON or CSV
```

#### Parse VIA JSON in Python

```python
import json

with open('via_annotations.json') as f:
    via_data = json.load(f)

for file_key, file_data in via_data['_via_img_metadata'].items():
    filename = file_data['filename']
    for region in file_data['regions']:
        shape = region['shape_attributes']
        attrs = region['region_attributes']
        class_name = attrs.get('class', 'unknown')

        if shape['name'] == 'rect':
            x, y = shape['x'], shape['y']
            w, h = shape['width'], shape['height']
            print(f"{filename}: {class_name} box ({x},{y}) {w}×{h}")
        elif shape['name'] == 'polygon':
            xs = shape['all_points_x']
            ys = shape['all_points_y']
            print(f"{filename}: {class_name} polygon with {len(xs)} points")
```

---

### 10.6 Labelbox

Labelbox is an enterprise cloud platform for large-scale data labeling — collaborative, API-driven, and integrated with MLOps pipelines.

**Website:** [labelbox.com](https://labelbox.com)

#### Key Features

```
Scale:
  Manage thousands of annotators
  Built-in quality control (review workflows, consensus)
  Nested task queues with priority

Automation:
  Model-assisted labeling (MAL): run your model, human corrects
  DINO, SAM integration for pre-labeling
  Active learning: prioritize uncertain/valuable examples

API-First:
  Full REST + GraphQL API
  Python SDK for programmatic dataset management
  Webhooks for real-time events

Export Formats:
  COCO, Pascal VOC, YOLO
  Custom format via API
  Direct integration: HuggingFace, Roboflow, AWS SageMaker

Pricing:
  Free: 5 users, 1 project, limited storage
  Team: $0.06–0.12/label
  Enterprise: custom pricing
```

#### Labelbox Python SDK

```python
# pip install labelbox

import labelbox as lb

client = lb.Client(api_key='YOUR_API_KEY')

# Create dataset
dataset = client.create_dataset(name='MyDetectionDataset')

# Upload images from URLs
dataset.create_data_rows([
    {'row_data': 'https://example.com/image001.jpg', 'global_key': 'img001'},
    {'row_data': 'https://example.com/image002.jpg', 'global_key': 'img002'},
])

# Create labeling project
ontology_builder = lb.OntologyBuilder(
    tools=[
        lb.Tool(tool=lb.Tool.Type.BOUNDING_BOX, name='car'),
        lb.Tool(tool=lb.Tool.Type.BOUNDING_BOX, name='person'),
        lb.Tool(tool=lb.Tool.Type.POLYGON, name='road'),
    ]
)
ontology = client.create_ontology('Detection Ontology', ontology_builder.asdict())

project = client.create_project(
    name='Warehouse Detection',
    media_type=lb.MediaType.Image
)
project.setup_editor(ontology)
project.create_batch('batch-1', dataset.export_data_rows(), 5)

# Export completed labels
export_task = project.export(params={
    'data_row_details': True,
    'metadata_fields': True,
    'attachments': False,
    'project_details': True,
    'performance_details': True,
    'label_details': True
})
export_task.wait_till_done()

for row in export_task.get_buffered_stream():
    label = row.json
    for annotation in label.get('projects', {}).get(project.uid, {}).get('labels', []):
        for obj in annotation.get('annotations', {}).get('objects', []):
            print(obj['value'], obj['bounding_box'])
```

---

### 10.7 Roboflow

Roboflow is a cloud platform for building computer vision datasets and pipelines — annotate, augment, version, and export to YOLO/COCO/VOC, then train in the cloud or export for local/edge deployment (Jetson, TensorRT, ONNX, TFLite).

**Website:** [roboflow.com](https://roboflow.com)

#### Key Features

```
Dataset Management:
  Upload images/video; annotate in browser (boxes, polygons, keypoints)
  Dataset versioning — track changes and augmentations per version
  Train/valid/test split with one click
  Public dataset catalog (COCO, Open Images, etc.) and custom uploads

Augmentation & Preprocessing:
  Built-in augmentations (flip, rotate, brightness, mosaic, etc.)
  Auto-orientation and resize for target model input
  Generate new versions with different augmentations without re-labeling

Export & Deployment:
  Export: YOLO (v5/v8/v11), COCO, Pascal VOC, TFRecord, TensorFlow, Create ML
  Train in Roboflow (YOLOv8, etc.) or download dataset for local training
  Deploy: Roboflow API, ONNX, TensorRT, TFLite, Core ML, browser (JavaScript)
  Edge: direct export for Jetson, Raspberry Pi, and embedded targets

API & Integrations:
  Python SDK (roboflow) for upload, annotation, version, and inference
  REST API for pipelines and MLOps
  Integrates with Labelbox, CVAT (import/export), and major frameworks
```

#### When to Use Roboflow

- You want one place to version datasets, augment, and export to YOLO/COCO for training and edge deployment.
- You need quick iteration: annotate → augment → export → train (locally or in cloud) → deploy to Jetson/edge.
- You prefer a managed pipeline over self-hosting CVAT or managing raw files and scripts.

#### Quick Start (Python)

```python
# pip install roboflow

from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("my-workspace").project("my-project")
dataset = project.version(1).download("yolov8")   # or "coco", "voc", etc.

# Inference with hosted model
model = project.version(1).model
pred = model.predict("image.jpg")
print(pred.json())
```

---

### 10.8 COCO Annotator

COCO Annotator is an open-source web-based annotation tool built specifically for the COCO dataset format — supports segmentation masks natively.

**GitHub:** [github.com/jsbroks/coco-annotator](https://github.com/jsbroks/coco-annotator)

#### Key Features

```
Annotation Types:
  Bounding boxes
  Segmentation masks (polygon)
  Keypoints

COCO-Native:
  Stores directly in COCO JSON format
  Supports supercategory hierarchy
  Proper instance segmentation with "iscrowd" flag

Smart Tools:
  Magic Wand (color-based region fill)
  Superpixel-based selection (SLIC segments)
  Polygon simplification

Collaboration:
  User accounts with dataset permissions
  Multi-user concurrent annotation
  Export at any time (live COCO JSON)
```

#### Docker Setup

```bash
git clone https://github.com/jsbroks/coco-annotator.git
cd coco-annotator

# Start with Docker Compose (includes MongoDB + Flask + Vue frontend)
docker-compose up -d

# Access at http://localhost:5000
# Default credentials: admin / admin

# Mount your images directory
# Edit docker-compose.yml:
# volumes:
#   - /path/to/your/images:/datasets
```

#### Export and Load COCO JSON

```python
import json

# Load COCO format annotations
with open('annotations.json') as f:
    coco = json.load(f)

# COCO structure:
# {
#   "images":      [...],   ← image metadata (id, file_name, width, height)
#   "annotations": [...],   ← boxes/masks (image_id, category_id, bbox, segmentation, area)
#   "categories":  [...],   ← class list (id, name, supercategory)
# }

# Build lookup maps
cat_map = {c['id']: c['name'] for c in coco['categories']}
img_map = {i['id']: i['file_name'] for i in coco['images']}

# Iterate annotations
for ann in coco['annotations']:
    img_name = img_map[ann['image_id']]
    class_name = cat_map[ann['category_id']]
    x, y, w, h = ann['bbox']   # COCO bbox: x_min, y_min, width, height
    area = ann['area']
    segmentation = ann.get('segmentation', [])   # list of polygon vertex lists
    is_crowd = ann.get('iscrowd', 0)             # 1=RLE mask, 0=polygon

    print(f"{img_name}: {class_name} at [{x:.0f},{y:.0f},{x+w:.0f},{y+h:.0f}]")
```

---

### 10.9 Annotation Workflow Best Practices

#### Quality Control

```python
# Check annotation quality: count boxes per image, flag outliers
import json
from collections import Counter

with open('annotations.json') as f:
    coco = json.load(f)

boxes_per_image = Counter()
for ann in coco['annotations']:
    boxes_per_image[ann['image_id']] += 1

# Flag images with too few or too many boxes
avg = sum(boxes_per_image.values()) / len(boxes_per_image)
for img_id, count in boxes_per_image.items():
    if count < avg * 0.2 or count > avg * 3:
        img = next(i for i in coco['images'] if i['id'] == img_id)
        print(f"REVIEW: {img['file_name']} has {count} boxes (avg={avg:.1f})")
```

#### Dataset Splitting

```python
import random
from pathlib import Path
import shutil

def split_dataset(image_dir: str, label_dir: str,
                  train=0.7, val=0.2, test=0.1, seed=42):
    """Split YOLO-format dataset into train/val/test."""
    images = list(Path(image_dir).glob('*.jpg')) + \
             list(Path(image_dir).glob('*.png'))
    random.seed(seed)
    random.shuffle(images)

    n = len(images)
    splits = {
        'train': images[:int(n*train)],
        'val':   images[int(n*train):int(n*(train+val))],
        'test':  images[int(n*(train+val)):],
    }

    for split_name, imgs in splits.items():
        img_out = Path(f'dataset/{split_name}/images')
        lbl_out = Path(f'dataset/{split_name}/labels')
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in imgs:
            lbl_path = Path(label_dir) / (img_path.stem + '.txt')
            shutil.copy(img_path, img_out / img_path.name)
            if lbl_path.exists():
                shutil.copy(lbl_path, lbl_out / lbl_path.name)

    print(f"Split: train={len(splits['train'])}, "
          f"val={len(splits['val'])}, test={len(splits['test'])}")

split_dataset('images', 'labels')
```

---

## 11. Dataset Formats

### 11.1 COCO JSON

The most complete format — supports detection, segmentation, keypoints, captions.

```json
{
  "info": {"description": "My dataset", "version": "1.0", "year": 2025},
  "licenses": [],
  "categories": [
    {"id": 1, "name": "car",    "supercategory": "vehicle"},
    {"id": 2, "name": "person", "supercategory": "human"}
  ],
  "images": [
    {"id": 1, "file_name": "image001.jpg", "width": 1920, "height": 1080}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [452, 78, 477, 456],
      "area": 217332,
      "segmentation": [[452,78, 929,78, 929,534, 452,534]],
      "iscrowd": 0
    }
  ]
}
```

### 11.2 YOLO Format

One `.txt` file per image. Class indices are 0-based. All values normalized 0–1.

```
# labels/image001.txt
# class_id  cx      cy      width   height
0            0.5130  0.2833  0.2474  0.4222
1            0.1250  0.5370  0.0625  0.1667
```

```python
# Convert YOLO label to pixel coordinates
def yolo_to_pixel(cx, cy, w, h, img_w, img_h):
    x1 = int((cx - w/2) * img_w)
    y1 = int((cy - h/2) * img_h)
    x2 = int((cx + w/2) * img_w)
    y2 = int((cy + h/2) * img_h)
    return x1, y1, x2, y2
```

### 11.3 Pascal VOC XML

```xml
<annotation>
  <folder>images</folder>
  <filename>image001.jpg</filename>
  <size><width>1920</width><height>1080</height><depth>3</depth></size>
  <object>
    <name>car</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox><xmin>452</xmin><ymin>78</ymin><xmax>929</xmax><ymax>534</ymax></bndbox>
  </object>
</annotation>
```

### 11.4 MOT Format (Multi-Object Tracking)

```
# mot_labels/image001.txt
# frame_id, track_id, x, y, w, h, confidence, class, visibility
1, 1, 452, 78, 477, 456, 1.0, 1, 1.0
1, 2, 120, 200,  60, 180, 0.9, 2, 0.8
```

### 11.5 Format Conversion Utilities

```python
# Roboflow — free web tool for format conversion
# Or use supervision library:
# pip install supervision

import supervision as sv

# Load COCO
ds = sv.DetectionDataset.from_coco(
    images_directory_path='images/',
    annotations_path='annotations.json'
)

# Save as YOLO
ds.as_yolo(
    images_directory_path='yolo/images/',
    annotations_directory_path='yolo/labels/',
    data_yaml_path='yolo/data.yaml'
)

# Save as Pascal VOC
ds.as_pascal_voc(
    images_directory_path='voc/images/',
    annotations_directory_path='voc/Annotations/'
)
```

---

## 12. Model Training Pipeline

### End-to-End Workflow

```
1. Collect images
      ↓
2. Annotate (X-AnyLabeling / CVAT)
      ↓
3. Export in target format (YOLO / COCO)
      ↓
4. Split dataset (70/20/10 train/val/test)
      ↓
5. Augment (Albumentations / YOLOv8 built-in)
      ↓
6. Train (YOLOv8 / TAO Toolkit / custom PyTorch)
      ↓
7. Evaluate (mAP@50, mAP@50:95, confusion matrix)
      ↓
8. Optimize (TensorRT / tinygrad quantization)
      ↓
9. Deploy (Jetson / DeepStream / ROS2)
```

### Augmentation with Albumentations

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.RandomResizedCrop(640, 640, scale=(0.5, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.MotionBlur(blur_limit=5, p=0.2),
    A.RandomRain(p=0.1),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
```

---

## 13. Projects

### Project 1: Full Annotation-to-Deployment Pipeline

Build a warehouse safety detector (person, forklift, hard hat) end-to-end.

```
1. Collect 500 images from warehouse video
2. Annotate with X-AnyLabeling (YOLOv8 pre-label → human correction)
3. Export as YOLO format, split 70/20/10
4. Fine-tune YOLOv8n for 100 epochs
5. Evaluate: target mAP@50 > 85%
6. Export to TensorRT INT8 for Jetson
7. Run at 30+ FPS on Jetson Orin Nano with ROS2 publisher

Deliverable: GitHub repo + annotated dataset on Roboflow Universe
```

### Project 2: Multi-Camera People Re-Identification

Track the same person across 4 cameras without overlapping FOVs.

```
Components:
  4 cameras → YOLOv8 person detection
           → Re-ID embedding (OSNet, MobileNetV2 backbone)
           → Hungarian algorithm cross-camera matching
           → Redis store for active identities

Annotation: label 1000 person crops per camera with CVAT
Training: contrastive loss on person pairs (same/different identity)
```

### Project 3: Aerial Object Detection with Rotated Boxes

Detect vehicles and buildings in satellite imagery using oriented bounding boxes.

```
Dataset: DOTA dataset (aerial images, 15 categories, rotated boxes)
Tool: X-AnyLabeling DOTA export
Model: YOLOv8-OBB (oriented bounding box variant)
Metric: mAP@50 with rotated IoU

uv run ultralytics train model=yolov8n-obb.pt data=dota.yaml epochs=100
```

### Project 4: Real-Time Pose Estimation

```python
# YOLOv8 pose — 17 keypoints (COCO skeleton)
from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')
results = model('video.mp4', stream=True)

for r in results:
    if r.keypoints is not None:
        kpts = r.keypoints.xy.numpy()   # (N_persons, 17, 2)
        # kpts[0, 0] = nose, [0, 5] = left shoulder, [0, 11] = left hip
        for person_kpts in kpts:
            nose = person_kpts[0]
            left_shoulder = person_kpts[5]
            right_shoulder = person_kpts[6]
            # Compute angle, detect fall, measure posture
```

### Project 5: 3D Scene Understanding with PointNet

```python
# pip install torch-geometric
import torch
import torch.nn as nn

class PointNet(nn.Module):
    """Classify a point cloud into one of N classes."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):  # x: (B, 3, N)
        x = self.mlp1(x)
        x = x.max(dim=-1).values   # global max pooling: (B, 1024)
        return self.fc(x)
```

---

## 14. Resources

### Foundational

- **"Computer Vision: Algorithms and Applications"** — Richard Szeliski (free online): the most complete CV textbook, covers everything from pinhole cameras to neural nets
- **"Programming Computer Vision with Python"** — Jan Erik Solem: practical OpenCV projects
- **CS231n: Convolutional Neural Networks for Visual Recognition** (Stanford, free): the canonical CNN course, lecture notes are excellent
- **"Deep Learning for Vision Systems"** — Mohamed Elgendy: CNN architectures with practical code

### Papers

| Topic | Paper |
|-------|-------|
| YOLO original | Redmon et al., "You Only Look Once" (CVPR 2016) |
| YOLOv8 | Jocher et al., Ultralytics YOLOv8 (2023) |
| Faster R-CNN | Ren et al., "Faster R-CNN" (NeurIPS 2015) |
| ViT | Dosovitskiy et al., "An Image is Worth 16×16 Words" (ICLR 2021) |
| DETR | Carion et al., "End-to-End Object Detection with Transformers" (ECCV 2020) |
| SAM | Kirillov et al., "Segment Anything" (ICCV 2023) |
| DeepLab v3+ | Chen et al., "Encoder-Decoder with Atrous Separable Convolution" (ECCV 2018) |
| SegFormer | Xie et al., "SegFormer: Simple and Efficient Design for Semantic Segmentation" (NeurIPS 2021) |
| Mask2Former | Cheng et al., "Masked-attention Mask Transformer" (CVPR 2022) |
| BEVFusion | Liu et al., "BEVFusion" (ICRA 2023) |
| ByteTrack | Zhang et al., "ByteTrack" (ECCV 2022) |

### Tools and Libraries

- **OpenCV** — docs.opencv.org — core library
- **Ultralytics YOLOv8** — docs.ultralytics.com — modern detection/segmentation/pose
- **Albumentations** — albumentations.ai — augmentation
- **Supervision** — supervision.roboflow.com — detection utilities, visualization
- **Roboflow** — roboflow.com — dataset hosting, format conversion, augmentation

### Annotation Tools Summary

| Tool | Link |
|------|------|
| X-AnyLabeling | github.com/CVHub520/X-AnyLabeling |
| CVAT | app.cvat.ai or github.com/cvat-ai/cvat |
| LabelImg | github.com/HumanSignal/labelImg |
| RectLabel | rectlabel.com |
| VGG VIA | robots.ox.ac.uk/~vgg/software/via |
| Labelbox | labelbox.com |
| Roboflow | roboflow.com |
| COCO Annotator | github.com/jsbroks/coco-annotator |

### Conferences

- **CVPR** — IEEE/CVF Conference on Computer Vision and Pattern Recognition
- **ICCV** — International Conference on Computer Vision
- **ECCV** — European Conference on Computer Vision
- **NeurIPS** — ML/AI (strong CV track)

---

*Previous: [Phase 3 — Advanced FPGA and Acceleration](../Guide.md)*
