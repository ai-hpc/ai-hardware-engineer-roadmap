# NVIDIA TAO Toolkit — Concrete Guide

> **Goal:** Use NVIDIA TAO Toolkit to take a pre-trained model from NGC, fine-tune it on a custom dataset, prune and quantize it, export to TensorRT, and deploy on Jetson Orin Nano — without training from scratch.

---

## Table of Contents

1. [What is TAO Toolkit?](#1-what-is-tao-toolkit)
2. [Why TAO Instead of Training from Scratch?](#2-why-tao-instead-of-training-from-scratch)
3. [Installation](#3-installation)
4. [Pre-Trained Models from NGC](#4-pre-trained-models-from-ngc)
5. [Dataset Preparation](#5-dataset-preparation)
6. [Transfer Learning — Fine-Tuning a Model](#6-transfer-learning--fine-tuning-a-model)
7. [Pruning](#7-pruning)
8. [Quantization-Aware Training (QAT)](#8-quantization-aware-training-qat)
9. [Exporting to TensorRT](#9-exporting-to-tensorrt)
10. [Deploying on Jetson Orin Nano](#10-deploying-on-jetson-orin-nano)
11. [DeepStream Integration](#11-deepstream-integration)
12. [TAO vs Manual Pipeline Comparison](#12-tao-vs-manual-pipeline-comparison)
13. [Projects](#13-projects)
14. [Resources](#14-resources)

---

## 1. What is TAO Toolkit?

**TAO** = **Train, Adapt, Optimize** — NVIDIA's no-code/low-code framework for building production-grade AI models with transfer learning, without needing to write training loops from scratch.

```
TAO Toolkit Architecture:
─────────────────────────────────────────────────────────────────
NGC Model Zoo                                     Your Custom Model
(pre-trained weights)                             (optimized for edge)
       │                                                  │
       ▼                                                  ▲
  tao download ──► tao train ──► tao prune ──► tao export
                   (fine-tune)   (compress)   (TensorRT engine)
                       │
               Your custom dataset
               (KITTI / COCO format)
─────────────────────────────────────────────────────────────────
```

### What TAO provides:
| Feature | What it does |
|---------|-------------|
| **NGC pre-trained models** | ResNet, EfficientNet, YOLO, SSD with ImageNet/COCO weights |
| **Transfer learning** | Freeze backbone, train detection head on your data |
| **Auto-pruning** | Magnitude-based channel pruning with one command |
| **QAT** | Fake quantization during training → INT8 TRT engine |
| **TensorRT export** | Generates calibrated `.engine` files directly |
| **DeepStream plugins** | Drop-in `nvinfer` config for video pipelines |

### TAO Toolkit Versions

| Version | Key feature | Python API |
|---------|------------|------------|
| TAO 3.x | PyTorch + TF2 backends, Docker-based | `tao_toolkit` |
| TAO 4.x | Unified launcher, 100+ models | `nvidia-tao` pip |
| TAO 5.x (current) | Cloud-native, Triton support | `nvidia-tao` pip |

---

## 2. Why TAO Instead of Training from Scratch?

### The Core Problem

Training a detection model from scratch requires:

```
From scratch:
  Dataset needed:  100,000–1,000,000 labeled images
  GPU time:        72–200 hours (A100)
  Cost:            $500–5,000 cloud compute
  Expertise:       Deep learning researcher

TAO transfer learning:
  Dataset needed:  500–5,000 labeled images       ← 100× less data
  GPU time:        2–8 hours (RTX 3080)           ← 10× faster
  Cost:            $10–50 cloud compute            ← 100× cheaper
  Expertise:       Engineer following docs
```

### Why It Works: Transfer Learning

Pre-trained backbones have learned to detect edges → textures → shapes → objects. Your fine-tuning only needs to adjust the final layers to recognize your specific objects.

```
ResNet-50 trained on ImageNet (1.2M images, 1000 classes):
  Layer 1:  detects edges, gradients
  Layer 2:  detects corners, textures
  Layer 3:  detects patterns (wheels, faces, windows)
  Layer 4:  detects object parts (car front, person torso)
  Layer 5:  classifies ImageNet objects  ← replace this

After TAO fine-tuning on 2,000 custom images:
  Layers 1-4: frozen (ImageNet features still useful)
  Layer 5:  retrained on "forklift", "pallet", "person"

Result: 94% mAP after 4 hours of training
```

### TAO Optimization Pipeline Output

```
Original ResNet-50 + SSD head:
  Model size:      98 MB
  Jetson FPS:      8 FPS (FP32)

After TAO:
  Prune 60%:       39 MB
  Retrain + QAT:   ACC within 1% of baseline
  Export INT8:     9.8 MB TRT engine
  Jetson FPS:      47 FPS (INT8 TRT)   ← 6× speedup
```

---

## 3. Installation

### Option A: pip (Recommended for workstation)

```bash
# Requires Python 3.8+, CUDA 11.x+
pip install nvidia-tao

# Verify
tao --version
# NVIDIA TAO Toolkit, version 5.x.x

# Install model-specific dependencies (e.g., object detection)
tao model yolo_v4 --help
```

### Option B: Docker (Recommended for reproducibility)

```bash
# Pull the TAO container (replace 5.x.x with latest)
docker pull nvcr.io/nvidia/tao/tao-toolkit:5.5.0-tf2.11.0

# Run with GPU access and workspace mounted
docker run --gpus all -it \
  -v $(pwd)/workspace:/workspace \
  -v ~/.tao:/root/.tao \
  nvcr.io/nvidia/tao/tao-toolkit:5.5.0-tf2.11.0 \
  /bin/bash
```

### NGC API Key Setup

TAO downloads models from NGC — you need an API key.

```bash
# 1. Create account at ngc.nvidia.com
# 2. Generate API key: Setup → API key → Generate API key

# 3. Configure NGC CLI
pip install ngccli
ngc config set
# Prompts for API key and org

# 4. TAO uses NGC automatically after this
tao ngc-registry model list --filter_str object_detection
```

### Workspace Structure

```
workspace/
├── data/
│   ├── train/           # training images + labels
│   ├── val/             # validation images + labels
│   └── test/            # test images (no labels needed)
├── models/
│   └── pretrained/      # downloaded NGC weights
├── specs/               # TAO YAML spec files
├── results/             # training outputs (checkpoints)
└── export/              # TensorRT engines
```

---

## 4. Pre-Trained Models from NGC

### Browse and Download Models

```bash
# List available object detection models
ngc registry model list nvidia/tao/pretrained_object_detection:*

# Common models:
#   pretrained_object_detection:resnet18      (fast, Jetson-friendly)
#   pretrained_object_detection:resnet50      (balanced)
#   pretrained_object_detection:efficientdet_ef0  (compact)
#   pretrained_classification:efficientnet_b0
#   pretrained_segmentation:vanilla_unet_resnet25

# Download ResNet-18 SSD backbone
ngc registry model download-version \
  nvidia/tao/pretrained_object_detection:resnet18 \
  --dest workspace/models/pretrained

# Directory after download:
# workspace/models/pretrained/
#   resnet18.hdf5   (Keras weights)
#   OR
#   resnet18.pth    (PyTorch weights, newer models)
```

### Supported Task Models (TAO 5.x)

| Task | Models available |
|------|-----------------|
| Object Detection | YOLOv4, YOLO-NAS, SSD, RetinaNet, EfficientDet, DINO, Grounding DINO |
| Classification | ResNet-{18,34,50,101}, EfficientNet-{B0-B7}, MobileNet-V2, ViT |
| Segmentation | UNet, MaskRCNN, SegFormer |
| Face Detection | FaceDetect, FPENet (facial landmarks) |
| Re-identification | ReID (person tracking across cameras) |
| NLP | BERT, QuestionAnswering, IntentSlot |
| Multi-modal | CLIP, DINO-v2 |

---

## 5. Dataset Preparation

TAO supports **KITTI format** (primary) and **COCO JSON format**.

### KITTI Format (Object Detection)

```
data/train/
├── images/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
└── labels/
    ├── 000001.txt   ← one label file per image, same stem name
    ├── 000002.txt
    └── ...
```

**Label file format** (one object per line):
```
# class_name truncated occluded alpha xmin ymin xmax ymax h w l x y z ry
# For 2D detection, only class_name xmin ymin xmax ymax are required:

forklift 0 0 0 452 78 929 534 0 0 0 0 0 0 0
person 0 0 0 120 200 180 380 0 0 0 0 0 0 0
pallet 0 0 0 300 400 600 520 0 0 0 0 0 0 0
```

### Convert COCO to KITTI with Python

```python
import json
import os
from pathlib import Path

def coco_to_kitti(coco_json_path: str, output_label_dir: str):
    """Convert COCO annotation JSON to KITTI format label files."""
    with open(coco_json_path) as f:
        coco = json.load(f)

    # Build category ID → name map
    cat_map = {cat['id']: cat['name'] for cat in coco['categories']}

    # Group annotations by image ID
    annotations_by_image: dict[int, list] = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        annotations_by_image.setdefault(img_id, []).append(ann)

    os.makedirs(output_label_dir, exist_ok=True)

    for img_info in coco['images']:
        img_id = img_info['id']
        stem = Path(img_info['file_name']).stem
        label_path = os.path.join(output_label_dir, f"{stem}.txt")

        lines = []
        for ann in annotations_by_image.get(img_id, []):
            x, y, w, h = ann['bbox']  # COCO: x_min, y_min, width, height
            xmin, ymin = x, y
            xmax, ymax = x + w, y + h
            class_name = cat_map[ann['category_id']]

            # KITTI format: class trunc occ alpha x1 y1 x2 y2 h w l x y z ry
            lines.append(
                f"{class_name} 0 0 0 {xmin:.2f} {ymin:.2f} {xmax:.2f} {ymax:.2f} "
                f"0 0 0 0 0 0 0"
            )

        with open(label_path, 'w') as f:
            f.write('\n'.join(lines))

    print(f"Converted {len(coco['images'])} images to KITTI format in {output_label_dir}")

# Usage:
coco_to_kitti("annotations/instances_train2017.json", "data/train/labels")
```

### Dataset Validation

```bash
# TAO can validate your dataset before training
tao model yolo_v4 dataset_convert \
  -e specs/yolov4_train.yaml \
  --validate_only

# Output shows:
#   ✓ Found 2,847 images
#   ✓ Found 2,847 label files
#   ✓ Class distribution: forklift=1204, person=3891, pallet=892
#   ✗ 3 images missing labels → listed by name
```

---

## 6. Transfer Learning — Fine-Tuning a Model

### Step 1: Write a Spec File (YAML)

TAO uses YAML spec files to configure everything. This is the core interface.

```yaml
# specs/yolov4_train.yaml
dataset_config:
  data_sources:
    - label_directory_path: workspace/data/train/labels
      image_directory_path: workspace/data/train/images
  target_class_mapping:
    forklift: forklift
    person: person
    pallet: pallet
  validation_data_sources:
    - label_directory_path: workspace/data/val/labels
      image_directory_path: workspace/data/val/images

model_config:
  pretrained_model_file: workspace/models/pretrained/resnet18.hdf5
  num_layers: 18                    # ResNet-18
  arch: resnet
  input_image_size: "3,544,960"     # C,H,W
  output_stride: 16
  freeze_bn: True                   # freeze BatchNorm during fine-tune
  freeze_blocks: [0, 1]             # freeze first 2 residual blocks

augmentation_config:
  output_width: 960
  output_height: 544
  randomize_input_shape_period: 0
  hue: 0.1
  saturation: 1.5
  exposure: 1.5
  random_flip: 1
  random_rotate: False
  jitter: 0.3

training_config:
  batch_size_per_gpu: 16
  num_epochs: 80
  learning_rate: 0.00261
  lr_schedule: "cosine"
  warmup_epochs: 3
  weight_decay: 0.0005
  momentum: 0.9
  pretrain_model_path: workspace/models/pretrained/resnet18.hdf5
  enable_qat: False

bbox_handler_config:
  kitti_box_utils:
    num_small_boxes_rejected: 0
  target_class_config:
    - name: forklift
      coverage_threshold: 0.005    # min object area fraction
    - name: person
      coverage_threshold: 0.005
    - name: pallet
      coverage_threshold: 0.005
```

### Step 2: Train

```bash
# Single GPU training
tao model yolo_v4 train \
  -e specs/yolov4_train.yaml \
  -r workspace/results/yolov4_resnet18 \
  --gpus 1

# Multi-GPU (if available)
tao model yolo_v4 train \
  -e specs/yolov4_train.yaml \
  -r workspace/results/yolov4_resnet18 \
  --gpus 4

# Training output structure:
# workspace/results/yolov4_resnet18/
#   train/
#     events.out.tfevents.*   ← TensorBoard logs
#   weights/
#     yolov4_resnet18_epoch_010.hdf5
#     yolov4_resnet18_epoch_020.hdf5
#     ...
#     yolov4_resnet18_epoch_080.hdf5  ← best kept
```

### Step 3: Evaluate

```bash
tao model yolo_v4 evaluate \
  -e specs/yolov4_eval.yaml \
  -m workspace/results/yolov4_resnet18/weights/yolov4_resnet18_epoch_080.hdf5

# Output:
# class_name       ap
# forklift         91.3%
# person           88.7%
# pallet           85.2%
# mAP:             88.4%
```

### Step 4: Visualize with TensorBoard

```bash
tensorboard --logdir workspace/results/yolov4_resnet18/train \
            --port 6006 --bind_all
# Open http://localhost:6006
# Shows: loss curves, mAP per epoch, learning rate schedule
```

### Training Config Knobs

| Parameter | Too low | Good range | Too high |
|-----------|---------|------------|----------|
| `batch_size_per_gpu` | slow GPU util | 8–32 | OOM |
| `learning_rate` | never converges | 1e-4 – 1e-2 | diverges |
| `freeze_blocks` | overfits small data | [0,1] or [0,1,2] | underfits |
| `num_epochs` | underfits | 50–150 | overfits |
| `jitter` | no augmentation | 0.2–0.4 | too distorted |

---

## 7. Pruning

TAO pruning uses **magnitude-based channel pruning**: removes entire feature map channels whose L1-norm falls below a threshold. Result: a smaller model with the same architecture family but fewer channels.

### Why Prune?

```
Before pruning:
  ResNet-18 + SSD:  98 MB, 47 FPS INT8 on Orin Nano

After 60% pruning + retrain:
  ResNet-18 + SSD:  38 MB, 62 FPS INT8 on Orin Nano
  mAP drop:         < 1%
```

### Prune Command

```bash
# Spec file for pruning
cat > specs/yolov4_prune.yaml << 'EOF'
pruning_config:
  method: "l1_norm"        # magnitude of weight channels
  prune_ratio: 0.6         # remove 60% of channels by weight
  equalization_criterion: "union"
  granularity: 8           # prune in groups of 8 channels (CUDA efficiency)
  min_num_filters: 16      # never go below 16 channels per layer
  threshold: 0.1           # alternative: threshold instead of ratio
EOF

tao model yolo_v4 prune \
  -e specs/yolov4_prune.yaml \
  -m workspace/results/yolov4_resnet18/weights/yolov4_resnet18_epoch_080.hdf5 \
  -o workspace/results/yolov4_pruned/yolov4_pruned.hdf5

# Output shows:
# Original filters: 512
# Pruned filters:   198  (61.3% reduction)
# Model size: 98 MB → 37 MB
```

### Retrain After Pruning

Pruning disrupts learned features — you must retrain the pruned model to recover accuracy.

```yaml
# specs/yolov4_retrain.yaml — same as train spec but:
training_config:
  pretrain_model_path: workspace/results/yolov4_pruned/yolov4_pruned.hdf5
  num_epochs: 40           # fewer epochs — recovering, not learning from scratch
  learning_rate: 0.00065   # lower LR — fine adjustment
```

```bash
tao model yolo_v4 train \
  -e specs/yolov4_retrain.yaml \
  -r workspace/results/yolov4_retrained \
  --gpus 1

# After retraining, re-evaluate:
tao model yolo_v4 evaluate \
  -e specs/yolov4_eval.yaml \
  -m workspace/results/yolov4_retrained/weights/yolov4_resnet18_epoch_040.hdf5
# mAP: 87.8% (was 88.4%) — only 0.6% drop after 60% pruning
```

### Iterative Pruning Strategy

```
Round 1: Prune 30% → Retrain → Check mAP drop
Round 2: Prune 30% more → Retrain → Check mAP drop
Round 3: Prune 20% more → Retrain → mAP drops >2% → stop

Final: ~65% total reduction, mAP within 1.5% of baseline
```

This is better than single-step aggressive pruning because the model adapts gradually.

---

## 8. Quantization-Aware Training (QAT)

TAO can insert fake quantization nodes during training so the model learns to be robust to INT8 precision before export.

### Enable QAT in Training Spec

```yaml
# specs/yolov4_qat.yaml
training_config:
  pretrain_model_path: workspace/results/yolov4_retrained/weights/yolov4_resnet18_epoch_040.hdf5
  num_epochs: 15          # short fine-tune — model already trained
  learning_rate: 0.0001   # very low — just adapting to quantization noise
  enable_qat: True        # THE KEY FLAG
  batch_size_per_gpu: 8   # reduce if OOM during QAT (2× memory overhead)
```

```bash
tao model yolo_v4 train \
  -e specs/yolov4_qat.yaml \
  -r workspace/results/yolov4_qat \
  --gpus 1

# During QAT, TAO automatically:
# 1. Inserts FakeQuantize nodes after Conv/Linear layers
# 2. Learns quantization scale/zero-point as parameters
# 3. Backprop through STE (Straight-Through Estimator)
```

### QAT vs PTQ Comparison

```
Post-Training Quantization (PTQ):
  Workflow:  train FP32 → calibrate on 1000 samples → INT8 engine
  mAP drop:  ~2–4% typical
  Time:      +30 minutes calibration

Quantization-Aware Training (QAT):
  Workflow:  train FP32 → QAT fine-tune 15 epochs → INT8 engine
  mAP drop:  ~0.5–1% typical
  Time:      +6 hours QAT training

Use QAT when:
  - mAP is critical and budget allows extra training time
  - PTQ drops below acceptable threshold
  - Model has depthwise separable convolutions (very sensitive to quantization)
```

---

## 9. Exporting to TensorRT

### Export to ONNX + TRT Engine

```yaml
# specs/yolov4_export.yaml
export_config:
  # Input model
  model: workspace/results/yolov4_qat/weights/yolov4_resnet18_epoch_015.hdf5
  # OR for non-QAT:
  # model: workspace/results/yolov4_retrained/weights/yolov4_resnet18_epoch_040.hdf5

  # Output
  output_file: workspace/export/yolov4_resnet18.onnx

  # Target precision
  data_type: int8               # "fp32", "fp16", "int8"

  # Calibration (for PTQ INT8 only, not needed after QAT)
  cal_image_dir: workspace/data/train/images
  cal_cache_file: workspace/export/cal.bin
  cal_batch_size: 8
  cal_batches: 20               # 20 × 8 = 160 calibration images

  # Input shape
  input_dims: [3, 544, 960]     # C, H, W

  # Batch size for engine
  batch_size: 1                 # Jetson real-time: batch=1
  max_batch_size: 8

  # Jetson target (for DLA)
  enable_dla: False             # set True to target DLA
```

```bash
# Export to ONNX (intermediate)
tao model yolo_v4 export \
  -e specs/yolov4_export.yaml \
  -m workspace/results/yolov4_qat/weights/yolov4_resnet18_epoch_015.hdf5 \
  -o workspace/export/yolov4_resnet18.onnx

# Convert ONNX → TensorRT engine
tao deploy tao-converter \
  -k your_ngc_api_key \
  -d 3,544,960 \
  -o BatchedNMS \
  -e workspace/export/yolov4_int8.engine \
  -p Input,1x3x544x960,4x3x544x960,8x3x544x960 \
  -t int8 \
  -c workspace/export/cal.bin \
  workspace/export/yolov4_resnet18.onnx

# Alternative: use trtexec directly
trtexec \
  --onnx=workspace/export/yolov4_resnet18.onnx \
  --saveEngine=workspace/export/yolov4_int8.engine \
  --int8 \
  --calib=workspace/export/cal.bin \
  --workspace=2048 \
  --verbose
```

### Verify the Engine

```bash
# Run benchmark on the engine
trtexec \
  --loadEngine=workspace/export/yolov4_int8.engine \
  --batch=1 \
  --iterations=200 \
  --avgRuns=100

# Output:
# [I] mean: 21.33 ms  ← 47 FPS on workstation RTX 3080
# [I] max:  23.1 ms
# [I] GPU Compute: 20.2 ms
# [I] H2D Latency: 0.47 ms
# [I] D2H Latency: 0.63 ms
```

---

## 10. Deploying on Jetson Orin Nano

### Transfer Engine to Jetson

```bash
# TRT engines are NOT portable between GPUs — must build on Jetson
# Transfer the ONNX and calibration file, then build on device

# On workstation — copy files to Jetson
scp workspace/export/yolov4_resnet18.onnx jetson@192.168.1.100:~/models/
scp workspace/export/cal.bin jetson@192.168.1.100:~/models/

# On Jetson — build INT8 engine
ssh jetson@192.168.1.100
trtexec \
  --onnx=/home/jetson/models/yolov4_resnet18.onnx \
  --saveEngine=/home/jetson/models/yolov4_int8_jetson.engine \
  --int8 \
  --calib=/home/jetson/models/cal.bin \
  --workspace=2048
# This takes ~5–15 minutes on Jetson (engine building is slow)
```

### Python Inference on Jetson

```python
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import cv2

class TAOYOLOInference:
    """Runs TAO-exported YOLO TensorRT engine on Jetson."""

    CLASS_NAMES = ['forklift', 'person', 'pallet']
    INPUT_H, INPUT_W = 544, 960
    CONF_THRESHOLD = 0.4
    NMS_THRESHOLD = 0.5

    def __init__(self, engine_path: str):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.inputs, self.outputs, self.bindings, self.stream = [], [], [], cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize, normalize, NHWC→NCHW."""
        img = cv2.resize(frame, (self.INPUT_W, self.INPUT_H))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)   # HWC → CHW
        img = np.expand_dims(img, 0)   # add batch dim
        return np.ascontiguousarray(img)

    def infer(self, frame: np.ndarray) -> list[dict]:
        """Run inference, return list of detections."""
        # Preprocess
        inp = self.preprocess(frame)
        np.copyto(self.inputs[0]['host'], inp.ravel())

        # H2D
        cuda.memcpy_htod_async(
            self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

        # Execute
        self.context.execute_async_v2(
            bindings=self.bindings, stream_handle=self.stream.handle)

        # D2H
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()

        return self._parse_detections(self.outputs[0]['host'], frame.shape)

    def _parse_detections(
        self, raw: np.ndarray, orig_shape: tuple
    ) -> list[dict]:
        """Parse TRT BatchedNMS output.

        TAO YOLOv4 output after BatchedNMS plugin:
          [num_dets, boxes(4), scores(1), classes(1)] per detection
        """
        orig_h, orig_w = orig_shape[:2]
        sx = orig_w / self.INPUT_W
        sy = orig_h / self.INPUT_H

        # BatchedNMS output format: [1, keepTopK, 1, 4]
        # Reshape based on your export's output binding shape
        num_dets = int(raw[0])
        detections = []
        for i in range(num_dets):
            score = raw[1 + i]
            if score < self.CONF_THRESHOLD:
                continue
            cls_id = int(raw[1 + 100 + i])  # offset depends on keepTopK
            x1 = raw[1 + 200 + i * 4 + 0] * sx
            y1 = raw[1 + 200 + i * 4 + 1] * sy
            x2 = raw[1 + 200 + i * 4 + 2] * sx
            y2 = raw[1 + 200 + i * 4 + 3] * sy
            detections.append({
                'class': self.CLASS_NAMES[cls_id],
                'score': float(score),
                'box': [int(x1), int(y1), int(x2), int(y2)]
            })
        return detections


# Usage on Jetson
detector = TAOYOLOInference('/home/jetson/models/yolov4_int8_jetson.engine')
cap = cv2.VideoCapture(0)  # or GStreamer pipeline

while True:
    ret, frame = cap.read()
    if not ret:
        break

    dets = detector.infer(frame)
    for d in dets:
        x1, y1, x2, y2 = d['box']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{d['class']} {d['score']:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('TAO Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### Benchmark on Jetson

```bash
# On Jetson — set max performance mode first
sudo jetson_clocks --fan

# Benchmark
trtexec \
  --loadEngine=/home/jetson/models/yolov4_int8_jetson.engine \
  --batch=1 \
  --iterations=500 \
  --avgRuns=200 \
  --warmUp=100

# Expected results on Orin Nano 8GB:
# INT8 ResNet-18 + SSD (544×960):  ~42 FPS
# INT8 ResNet-18 + SSD (416×416):  ~68 FPS
# INT8 EfficientDet-EF0 (512×512): ~35 FPS
# FP16 ResNet-18 + SSD (544×960):  ~28 FPS
```

---

## 11. DeepStream Integration

TAO-exported models plug directly into DeepStream as `nvinfer` elements — no custom code needed.

### nvinfer Config File

```ini
# config_infer_primary_yolov4.txt
[property]
gpu-id=0
net-scale-factor=0.0039215697906911373   # 1/255
model-engine-file=/home/jetson/models/yolov4_int8_jetson.engine
int8-calib-file=/home/jetson/models/cal.bin
batch-size=1
process-mode=1                           # 1=primary detector
model-color-format=0                     # 0=RGB
labelfile-path=labels.txt                # one class name per line
gie-unique-id=1
output-blob-names=BatchedNMS
num-detected-classes=3
interval=0

[class-attrs-all]
threshold=0.4
roi-top-offset=0
roi-bottom-offset=0
detected-min-w=20
detected-min-h=20
detected-max-w=9999
detected-max-h=9999
```

```bash
# labels.txt
forklift
person
pallet
```

### DeepStream Pipeline YAML

```yaml
# deepstream_app_config.yaml
[application]
enable-perf-measurement=1
perf-measurement-interval-sec=2

[source0]
enable=1
type=4                    # 4=RTSP
uri=rtsp://192.168.1.50/stream
num-sources=1

[sink0]
enable=1
type=5                    # 5=RTSP output
sync=0
codec=1                   # H.264
bitrate=4000000
rtsp-port=8554
udp-port=5400

[osd]
enable=1
gpu-id=0

[streammux]
gpu-id=0
batch-size=1
batched-push-timeout=40000
width=960
height=544

[primary-gie]
enable=1
gpu-id=0
gie-unique-id=1
config-file=config_infer_primary_yolov4.txt
```

```bash
# Run the full pipeline
deepstream-app -c deepstream_app_config.yaml
```

---

## 12. TAO vs Manual Pipeline Comparison

| Aspect | TAO Toolkit | Manual PyTorch + TensorRT |
|--------|------------|--------------------------|
| **Dataset labeling format** | KITTI, COCO | Any (you write the loader) |
| **Model architecture** | Fixed list (~50 models) | Any model you can write |
| **Training code** | Zero (YAML config) | Full training loop |
| **Pruning** | One command | Custom pruning code |
| **INT8 calibration** | Built-in calibrator | Custom `IInt8Calibrator` |
| **TRT export** | `tao export` command | `torch.onnx.export` + `trtexec` |
| **DeepStream integration** | Drop-in nvinfer config | Custom Gst plugin |
| **Customization** | Low | Full control |
| **Time to first working model** | Hours | Days–weeks |
| **Custom layer support** | No (use standard layers) | Yes |

**Use TAO when:**
- You need a standard detection/classification model fast
- Your architecture is in the NGC catalog
- You want push-button optimization pipeline
- You're deploying with DeepStream

**Use manual pipeline when:**
- Custom architecture (attention heads, custom losses)
- Research experiments
- Need to debug gradient flow
- Architectures not in NGC catalog (BEVFusion, PointPillars, etc.)

---

## 13. Projects

### Project 1: Warehouse Object Detector

Train a 3-class detector (forklift, person, pallet) for a Jetson-powered warehouse camera.

```
Data collection:
  - Record 30-min warehouse video at 1080p
  - Sample every 5 seconds: ~360 frames
  - Label with CVAT (free, open source)
  - Export as KITTI format

TAO workflow:
  1. tao download pretrained_object_detection:resnet18
  2. Train 80 epochs (4 hours on RTX 3080)
  3. Evaluate: target mAP > 80%
  4. Prune 50%, retrain 40 epochs
  5. QAT 15 epochs
  6. Export INT8 engine
  7. Deploy on Jetson + DeepStream RTSP stream

Success metric: >40 FPS on Orin Nano, mAP > 78%
```

### Project 2: Defect Detection on Conveyor Belt

Binary classifier: defect vs no-defect on manufactured parts.

```
TAO classification workflow:
  1. tao download pretrained_classification:efficientnet_b0
  2. Collect 1,000 defect + 1,000 no-defect images under controlled lighting
  3. Train classifier (10 epochs — classification is fast)
  4. Export FP16 engine (classification needs less optimization than detection)
  5. Jetson reads camera at 120 FPS, classify every frame
  6. Trigger alarm on 3 consecutive defect predictions

Success metric: <0.5% false positive rate, >99% recall on defects
```

### Project 3: Multi-Camera People Counter

Use TAO ReID (Re-Identification) to track people across 4 cameras in a building.

```
Components:
  Camera 1–4 → DeepStream source → Primary detector (TAO YOLOv4 person)
             → Secondary ReID model (TAO ReID)
             → Tracker (NvDCF)
             → Redis person count aggregator
             → Grafana dashboard

TAO models:
  Primary: pretrained_object_detection:resnet18 (person only, 2 classes: person/background)
  Secondary: pretrained_re_identification:resnet50 (128-dim embedding)
```

---

## 14. Resources

### Official Documentation
- **TAO Toolkit User Guide** — docs.nvidia.com/tao/tao-toolkit/
- **TAO Toolkit API Reference** — docs.nvidia.com/tao/tao-toolkit/text/tao_toolkit_api/
- **NGC Model Catalog** — ngc.nvidia.com/catalog/models (filter: TAO)
- **TAO Toolkit GitHub** — github.com/NVIDIA/tao_pytorch_backend

### Tutorials
- **NVIDIA Developer Blog: TAO Toolkit Getting Started** — developer.nvidia.com/blog/training-like-a-pro-with-tao-toolkit/
- **Jetson AI Lab — TAO + DeepStream** — jetson-ai-lab.com
- **TAO Toolkit Quick Start Guide** — catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/resources/tao_toolkit_quick_start_guide

### Related
- **DeepStream SDK** — docs.nvidia.com/metropolis/deepstream/dev-guide/
- **trtexec documentation** — docs.nvidia.com/deeplearning/tensorrt/developer-guide/ (section: trtexec)
- **CVAT** (labeling tool) — github.com/opencv/cvat — recommended for KITTI export

### Benchmark Reference (Orin Nano 8GB, JetPack 6.x)

| Model | Task | Precision | Input | FPS |
|-------|------|-----------|-------|-----|
| ResNet-18 + SSD | Detection | INT8 | 544×960 | 42 |
| ResNet-18 + SSD | Detection | FP16 | 544×960 | 28 |
| EfficientNet-B0 | Classification | INT8 | 224×224 | 380 |
| EfficientDet-EF0 | Detection | INT8 | 512×512 | 35 |
| MobileNet-V2 + SSD | Detection | INT8 | 300×300 | 95 |

---

*Previous: [3. Edge AI Optimization](../Guide.md)*
*See also: [Edge AI Optimization main guide](../Guide.md) for quantization theory, TensorRT manual pipeline, and DLA usage*
