# Computer Vision — Complete Guide

> **Goal:** Build a comprehensive understanding of computer vision from image processing fundamentals through modern deep learning architectures, with practical skills in annotation tools, dataset formats, and deployment on edge hardware.

---

## Table of Contents

1. [What is Computer Vision?](#1-what-is-computer-vision)
2. [Image Processing Fundamentals](#2-image-processing-fundamentals)
3. [Feature Extraction](#3-feature-extraction)
4. [Image Segmentation](#4-image-segmentation)
5. [Object Detection](#5-object-detection)
6. [Object Tracking](#6-object-tracking)
7. [3D Vision](#7-3d-vision)
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

### 4.3 Deep Learning Segmentation

```python
# Semantic segmentation with OpenCV DNN (DeepLab v3+)
net = cv2.dnn.readNetFromTensorflow(
    'deeplabv3_mnv2_pascal_train_aug.pb'
)

blob = cv2.dnn.blobFromImage(
    img_bgr, scalefactor=1/127.5, size=(513, 513),
    mean=(127.5, 127.5, 127.5), swapRB=True
)
net.setInput(blob)
output = net.forward()   # (1, num_classes, H, W)
seg_map = np.argmax(output[0], axis=0)  # per-pixel class index
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

### 10.7 COCO Annotator

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

### 10.8 Annotation Workflow Best Practices

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
| COCO Annotator | github.com/jsbroks/coco-annotator |

### Conferences

- **CVPR** — IEEE/CVF Conference on Computer Vision and Pattern Recognition
- **ICCV** — International Conference on Computer Vision
- **ECCV** — European Conference on Computer Vision
- **NeurIPS** — ML/AI (strong CV track)

---

*Previous: [Phase 3 — Advanced FPGA and Acceleration](../Guide.md)*
