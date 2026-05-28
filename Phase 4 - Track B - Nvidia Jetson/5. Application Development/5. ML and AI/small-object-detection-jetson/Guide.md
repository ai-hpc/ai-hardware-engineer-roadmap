# Small Object Detection on Jetson — Project Guide

<div class="course-identity auto-course" style="--course-accent: #4f46e5; --course-accent-rgb: 79, 70, 229;" markdown="1">
<div class="course-identity__icon">SODO</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Jetson Track</p>
<p class="course-identity__title">Specialized course identity for Small Object Detection on Jetson — Project Guide.</p>
<p class="course-identity__meta">Artifact: Jetson integration demo · Measure: latency, memory, power, logs</p>
</div>
</div>


> **Goal:** Solve small-object detection for a real-time vision backend on NVIDIA Jetson Orin Nano using **VisDrone2019-DET** and established best practices. Deliver a trained model deployable via DeepStream + TensorRT with acceptable accuracy and latency.

**How to achieve the goal (overview):** Train a small-object–aware detector (e.g. YOLOv8 or a variant like SPD-YOLOv8 / LPAE-YOLOv8) on VisDrone2019-DET using higher input resolution (832–960), multi-scale fusion, optional tiling for very high-res frames, and tuned confidence/NMS. Export the model to TensorRT (FP16 or INT8) and integrate it as the primary detector in your existing DeepStream + GStreamer + FastAPI pipeline on Jetson Orin Nano. For 4K or high-resolution drone footage, use tiled inference (e.g. SAHI) with overlapping patches and merge detections so small targets retain enough pixels per tile while balancing latency and throughput on the edge device.

---


## 1. Project Overview

### Problem

- **Platform:** NVIDIA Jetson Orin Nano; stack: DeepStream, GStreamer, FastAPI.
- **Task:** Improve primary object detection and optional secondary classification; reduce false positives; handle **very small targets** (few pixels in frame).
- **Constraints:** Real-time pipeline, limited memory and compute; production code quality.

### Approach

| Phase | Content |
|-------|--------|
| **Data** | Use VisDrone2019-DET (drone-view, small objects, public benchmark). Download from official source, convert annotations to YOLO format. |
| **Model** | YOLOv8 (or improved variant) with small-object–oriented tweaks: multi-scale features, higher input resolution, anchor/loss tuning. |
| **Training** | Transfer learning from COCO-pretrained weights; fine-tune on VisDrone; augmentation for scale/lighting/motion. |
| **Deploy** | Export to ONNX → TensorRT (FP16/INT8); integrate as primary detector in DeepStream; tune confidence/NMS for small objects. |

### Demo output (small object detection)

The pipeline produces detections with bounding boxes and optional segmentation masks. The image below shows **tiled inference (SAHI)** on a high-angle VisDrone-style frame: a crowded outdoor scene at night with many small instances. The model detects **people** (red), **bicycles** (orange), and other classes across scale, including very small targets in the scene.

![Small object detection demo output — SAHI tiled inference on crowded scene](0000066_01097_d_0000002_sahi.png)

*Example: bounding boxes and segmentation masks on a dense, multi-scale scene; useful for validating small-object recall and tiling (SAHI) behavior.*

---

## 2. Dataset: VisDrone2019-DET

### Source and License

- **Repository:** [VisDrone/VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset) (GitHub).
- **Challenge / description:** [Vision Meets Drones (VisDrone)](https://aiskyeye.com/) — ICCV 2019.
- **Registration:** Some splits (e.g. test-dev) may require registration at [aiskyeye.com](http://www.aiskyeye.com/views/register).

### Splits and Size

| Split | Images | Size (approx.) | Use |
|-------|--------|----------------|------|
| **Train** | 6,471 | ~1.44 GB | Training + optional validation holdout |
| **Val** | 548 | ~0.77 GB | Validation / early stopping |
| **Test-dev** | 1,610 | ~0.56 GB | Evaluation (labels not public) |

- **Total instances:** 2.6+ million bounding boxes; many are small (drone perspective).

### Download

**Option A — Official (recommended)**  
- Follow [VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset) README and download links (often Google Drive or the challenge site after registration).  
- **VisDrone2019-DET-train:** e.g. training images + annotations.  
- **VisDrone2019-DET-val:** validation images + annotations.

**Option B — Alternative mirrors**  
- [dataset-ninja/vis-drone-2019-det](https://github.com/dataset-ninja/vis-drone-2019-det) (see `DOWNLOAD.md`) may list direct links (e.g. Google Drive) for train/val.

**Option C — Ultralytics (auto-download)**  
- If using Ultralytics YOLOv8, the [VisDrone dataset card](https://docs.ultralytics.com/datasets/detect/visdrone/) can drive automatic download when using the built-in `visdrone.yaml` (where supported).

### Directory Layout (raw VisDrone2019-DET)

After downloading, you typically have:

```
VisDrone2019-DET/
├── images/
│   ├── train/          # 6471 images
│   ├── val/            # 548 images
│   └── test-dev/       # 1610 images (optional)
└── annotations/
    ├── train/          # one .txt per image, same base name
    ├── val/
    └── test-dev/
```

Each annotation file corresponds 1:1 to an image (e.g. `0000001_00000_d_0000001.jpg` → `0000001_00000_d_0000001.txt`).

---

## 3. Annotation Format and Conversion to YOLO

### VisDrone Ground-Truth Format (per line)

One line per object; 8 comma-separated fields:

```
<x>,<y>,<width>,<height>,<score>,<object_category>,<truncation>,<occlusion>
```

| Field | Meaning |
|-------|--------|
| `x`, `y` | Top-left corner of bounding box (pixels). |
| `width`, `height` | Box size (pixels). |
| `score` | In ground truth: `1` = used in evaluation, `0` = ignored. |
| `object_category` | Class index (see below). |
| `truncation` | 0 = none, 1 = partial (1–50% outside frame). |
| `occlusion` | 0 = none, 1 = partial (1–50%), 2 = heavy (50–100%). |

### Object Categories (DET)

| Index | Class |
|-------|--------|
| 0 | ignored regions |
| 1 | pedestrian |
| 2 | people |
| 3 | bicycle |
| 4 | car |
| 5 | van |
| 6 | truck |
| 7 | tricycle |
| 8 | awning-tricycle |
| 9 | bus |
| 10 | motor |
| 11 | others |

For **detection training** we usually use **1–10** (10 classes). Index `0` and `11` are ignored in evaluation; you can skip them or map `others` to a single “other” class if desired.

### Conversion to YOLO Format

YOLO expects one `.txt` per image, each line: `class_id x_center y_center width height` (normalized 0–1 relative to image size).

- **Class:** Map VisDrone category to 0–9: `yolo_cls = visdrone_cls - 1` (so pedestrian=0, people=1, …, motor=9). Optionally merge pedestrian/people or drop “others” depending on task.
- **Box:** Convert from `(x_tl, y_tl, w, h)` in pixels to normalized `(x_center, y_center, w, h)`:
  - `x_center = (x + w/2) / img_width`
  - `y_center = (y + h/2) / img_height`
  - `w_norm = w / img_width`, `h_norm = h / img_height`
- **Filter:** Ignore rows with `score == 0` or category `0` / `11` if you do not want to train on them.

**Reference implementations:**

- Ultralytics: [yolov5/data/VisDrone.yaml](https://github.com/ultralytics/yolov5/blob/master/data/VisDrone.yaml) (legacy YOLOv5; conversion logic in dataset loader).
- [adityatandon/VisDrone2YOLO](https://github.com/adityatandon/VisDrone2YOLO): standalone conversion script and pre-converted labels.

### Resulting YOLO-Style Layout

After conversion, use a layout compatible with Ultralytics (e.g. YOLOv8):

```
VisDrone/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/   # .txt per image, same base name
│   └── val/
└── data.yaml   # dataset config (see below)
```

**Example `data.yaml`:**

```yaml
path: /path/to/VisDrone
train: images/train
val: images/val
# test: images/test-dev  # optional

nc: 10
names: ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
```

---

## 4. Best Methods for Small-Object Detection (Drone / UAV)

These are established improvements for small objects in aerial/drone imagery; use them to pick architecture and training settings.

### Model Family: YOLOv8

- **Default choice:** YOLOv8n/s/m (nano/small/medium) — good speed/accuracy tradeoff and easy export to TensorRT/DeepStream.
- **Why:** Strong baseline on VisDrone; well supported by Ultralytics (train, export, validation); TensorRT and DeepStream support for YOLO.

### Architectural / Training Improvements (summary)

| Technique | Role |
|-----------|------|
| **Multi-scale feature fusion** | Extra detection heads or FPN-style fusion for small objects; avoid losing small instances in deep strides. |
| **Higher input resolution** | e.g. 640→960 or 1280 for inference/training when latency allows; critical for very small targets. |
| **Small-object–specific layers** | Some variants remove the largest-object head and add an additional small-object head (e.g. LPAE-YOLOv8). |
| **SPD (space-to-depth)** | Replace stride-2 conv/pool with SPD modules to retain information for small objects (e.g. SPD-YOLOv8). |
| **Anchor / loss tuning** | Adjust anchor scales for small boxes; use losses that help small objects (e.g. Wise-IoU, EfficiCIoU, MPDIoU). |
| **Lightweight attention** | Lightweight attention (e.g. in LPAE-YOLOv8) for better feature weighting without blowing up latency. |
| **Tiling (slicing / tiled inference)** | Split high-resolution images into overlapping patches; run detection per tile and merge results so small objects keep enough pixels. |

The following subsections explain each technique and how to apply it to reach the project goal.

---

#### 1. Multi-Scale Feature Fusion — How and Why for Small Object Detection

**What It Is, in Depth:**  
In modern object detectors like YOLO, the input image is passed through a backbone network that **progressively reduces the image’s spatial resolution** via convolutional and pooling layers (called "downsampling," with strides like 8, 16, or 32). While this process helps extract higher-level features (like object types), it also means **tiny objects can be lost**—imagine a 12×12 pixel car being squished down to a single “cell” or vanishing entirely in deep layers.

**Multi-scale feature fusion** directly addresses this by **combining features at multiple resolutions**. Instead of only processing deep (low-res) features, it collects and merges both:
- **Shallow, high-resolution features:** Good for *precise spatial detail* (edges, corners, tiny blobs)
- **Deep, low-resolution features:** Good for *semantic meaning* (“this is a car”)

This fusion is usually done using structures called **FPN (Feature Pyramid Network)** or its variants (like PANet), which use:
- **Top-down pathways:** Upsample deep features and merge with earlier layer features through lateral (side) connections
- **Bottom-up pathways (optional):** Pass information back down, to refine high-res features with semantic info

**Why Is This Important for Small Objects?**
- Small objects (especially in drones or crowded scenes) might only take up a few pixels, so if you only use the deepest layers, tiny targets will disappear.
- By merging shallow (detailed) and deep (semantic) features, each detection head gets “the best of both worlds.” Small-object heads get fine detail, and large-object heads get semantic context.
- **Practically, multi-scale fusion means your model retains the sensitivity to tiny signals, not just large and easy targets!**

**Step-by-Step: How to Do This Technically**

1. **With YOLOv8 (Out-of-the-Box):**
   - **No code changes needed:** Ultralytics YOLOv8 already has a **PANet-style FPN**, meaning it fuses multi-scale features by default.
   - There are 3 detection heads (P3–P5), so the model can handle a range of sizes.
   - *What you should do:* Focus on your **dataset and data augmentations**—avoid augmentation settings (like excessive cropping or resizing) that might exclude small objects. Keep the small instances in your data!
   - When training, **set the `imgsz` parameter large enough** (see “Higher input resolution”) to make small objects visible on early feature maps.

2. **With Enhanced Architectures (For Even Better Results):**
   - For many aerial/drone datasets, the default YOLO heads may not be enough. Some advanced research adds:
     - An **extra detection head** at even higher resolution (P2, or 1/4 stride of input)
     - This extra head is dedicated to *very* small objects by connecting to shallower layers (closer to the input)
   - **How to implement:**
     - **Option 1:** Find a public YOLOv8 variant (like **LPAE-YOLOv8**) that already implements a P2 head and see their repo for usage.
     - **Option 2:** Custom code (advanced): Fork the [Ultralytics YOLOv8 repo](https://github.com/ultralytics/ultralytics), modify the model code to add a P2 feature map branch (often after the second block in the backbone), add a corresponding detection head, and update the anchor settings for small box sizes. You may also want to assign small-object ground truths specifically to this P2 output during loss assignment.
     - **Tip:** Initialize the weights for your new head randomly, but load COCO-pretrained weights for the rest. Fine-tune on your dataset.
     - **Reference:** See the paper *“Improved YOLOv8 for Small Object Detection in Aerial Images”* for network diagrams and ablation studies.

3. **Multi-Scale Training (Must-Do, Complements Architecture):**
   - Even if your model supports multi-scale detection, you want the model to be robust to object size variations.
   - **How:**  
     - Multi-scale training randomly resizes input images within a range (e.g. ±50% of base size). Ultralytics YOLO does this by default.
     - To focus more on small objects: **Set a larger base image size** (`imgsz`). This ensures that, even when training at the lower end of the scale, small objects *remain large enough* for the network to learn from.
     - **Command example:**  
       ```bash
       yolo detect train data=VisDrone/data.yaml model=yolov8s.pt imgsz=832  # or imgsz=960 for very small objects
       ```
     - **Best practice:** Watch memory usage! Larger image sizes increase VRAM needs—reduce batch size if you get OOM (“out of memory”) errors.

**Typical Workflow Recap / Checklist**

- [ ] Pick your YOLO variant (vanilla YOLOv8, or a small-object–enhanced variant)
- [ ] Adjust your training command to use a larger `imgsz` (e.g. 832 or 960)
- [ ] Verify your data loading/augmentation does *not* exclude/crop small objects
- [ ] For research or cutting-edge performance, consider modifying the model architecture to add P2 head (read papers / use public small-object YOLO variants)
- [ ] After training, always check performance *specifically* for small object categories — measure average precision for small boxes, not just overall mAP.

**Key Tools:**
- [Ultralytics YOLOv8 repo](https://github.com/ultralytics/ultralytics): for training, customizing models
- Papers/LPAE-YOLOv8 repos: for P2 head implementations
- Netron: visualize your trained ONNX or PyTorch model to confirm network structure, verify extra heads exist


---

#### 2. Higher input resolution

**What it is:**  
Input size is the height/width (in pixels) of the image fed to the network (e.g. 640×640). **Higher resolution** (e.g. 960×960 or 1280×1280) means more pixels per object, so a “tiny” object occupies more cells and gets a stronger feature response.

**Why it helps small objects:**  
- At 640×640, a 10×10 pixel object is ~0.15% of the image and may map to one or two grid cells.  
- At 1280×1280, the same object is ~0.06% of the image in absolute terms but has **4× more pixels** in the feature maps (stride-for-stride), so the detector has more signal to classify and regress.

**How to achieve it:**

- **Training:** Use a larger `imgsz` when training. Tradeoff: more VRAM and slower training; reduce batch size if needed.

  ```bash
  # Baseline
  yolo detect train data=VisDrone/data.yaml model=yolov8s.pt imgsz=640 batch=16 epochs=100

  # Better for small objects (expect ~1.5–2× VRAM, reduce batch)
  yolo detect train data=VisDrone/data.yaml model=yolov8s.pt imgsz=960 batch=8 epochs=100
  # or imgsz=832 batch=12 as a compromise
  ```

- **Inference:** Use the same (or slightly lower) resolution at inference so behaviour matches training. On Jetson, 960 or 1280 increases latency; try 832 first, then 960 if accuracy is still insufficient.

  ```bash
  yolo detect val model=best.pt data=data.yaml imgsz=960
  ```

- **Rule of thumb:** For “very small” targets (e.g. &lt;32×32 px in full image), prefer **832–960** for training and deployment when the latency budget allows; **640** is acceptable for larger small objects.

---

#### 3. Small-object–specific layers (extra head for small objects)

**What it is:**  
Standard YOLOv8 has three detection heads at 1/8, 1/16, and 1/32 of the input size. The 1/8 head is already the “smallest stride” and handles relatively small objects, but in drone data many targets are even smaller. **Small-object–specific** designs add an **additional head at higher resolution** (e.g. 1/4 stride) and sometimes **remove or downweight the largest-object head** (1/32) to balance compute and focus on small instances.

**Why it helps small objects:**  
- A dedicated high-resolution head increases the number of grid cells that can “see” tiny objects and regress boxes for them.  
- Shifting capacity from “very large object” head to “very small object” head matches the distribution of drone datasets (many small, few huge).

**How to achieve it:**

- **Use a published variant:** e.g. **LPAE-YOLOv8** adds a small-object detection layer and removes the large-object layer; it also uses a lightweight attention module (see below). Clone the repo, follow their training instructions, and export to ONNX/TensorRT as usual.
- **DIY (advanced):** Modify the YOLOv8 head in the Ultralytics codebase (or a fork): add a P2 (1/4) branch from the backbone, add a fourth detection head for that scale, and assign small anchors to it. Then train from a pretrained backbone (e.g. load COCO weights and init the new head randomly).

---

#### 4. SPD (space-to-depth)

**What it is:**  
**Space-to-depth** is a rearrangement of spatial dimensions into channels: instead of a stride-2 convolution or pooling that **discards** information, SPD **reorders** pixels so that a 2×2 (or 4×4) neighbourhood becomes extra channels. The effective spatial size is reduced (e.g. 2× smaller) but **no information is thrown away**; a following conv then mixes these channels.

**Why it helps small objects:**  
- Stride-2 convs and pooling **subsample** the feature map. Tiny objects (1–2 pixels in a feature map) can vanish in one stride.  
- SPD keeps all pixels in the “depth” dimension, so small structures are still present; the network can learn to use them. This is especially useful in the **early backbone** where spatial resolution is still high.

**How to achieve it:**

- **Use SPD-YOLOv8 (or similar):** Replace the first few stride-2 operations in the backbone with SPD + conv. Implementations are available in papers/repos such as “SPD-YOLOv8: small-size object detection model of UAV imagery”. Clone the model code, train on VisDrone, then export to ONNX/TensorRT (ensure the custom SPD op is supported or converted to a standard op sequence).
- **Concept (for implementation):**  
  - Input feature `H×W×C`.  
  - Split into 2×2 patches → reshape to `(H/2)×(W/2)×(4*C)`.  
  - Conv 4*C → C.  
  Result: same spatial downsampling as stride-2, but no information loss from pooling.

---

#### 5. Anchor / loss tuning

**What it is:**  
YOLOv8 is **anchor-free** (uses anchor-free regression heads). “Anchor tuning” here means: (1) ensuring the **effective receptive fields / scale ranges** of the heads match your object size distribution, and (2) using **loss functions** that treat small boxes more fairly (e.g. better gradient for tiny boxes, or scale-invariant metrics).

**Why it helps small objects:**  
- Standard IoU and L1 loss can be dominated by large objects; small boxes have small absolute errors and may get weak gradients.  
- **Wise-IoU**, **EfficiCIoU**, **MPDIoU** (and similar) add dynamic weighting or shape terms so that small and large boxes contribute more equally to the loss and get better regression behaviour.

**How to achieve it:**

- **Wise-IoU (WIoU):** Often used in improved-YOLOv8 papers. Replaces the default box loss with a version that down-weights “easy” (e.g. large, high-IoU) examples and focuses learning on hard ones (often small). Implement by forking Ultralytics and swapping the box loss in the head, or use a third-party YOLOv8 codebase that already includes WIoU.
- **EfficiCIoU / MPDIoU:** Alternative IoU variants that improve small-object regression (e.g. consider diagonal length, minimal point distance). Same idea: replace the box regression loss in the training loop with the new formulation.
- **Practical step without code change:** Increase **small-object presence** in the batch via **augmentation** (scale jitter, copy-paste of small instances) so the optimizer sees more small-box examples; this partially compensates for loss imbalance.

---

#### 6. Lightweight attention

**What it is:**  
**Attention** modules (e.g. channel attention, spatial attention, or both) reweight features so the network focuses on important regions and channels. **Lightweight** versions (e.g. squeeze-and-excitation, CBAM, or small MLPs) add limited parameters and compute so they are suitable for edge deployment.

**Why it helps small objects:**  
- In cluttered drone scenes, small objects compete with background and large objects for feature capacity. Attention lets the network **emphasize** channels and spatial locations that correspond to small targets.  
- Lightweight attention avoids the heavy cost of full self-attention while still improving feature weighting.

**How to achieve it:**

- **Use a variant that includes it:** e.g. **LPAE-YOLOv8** uses an “ACMConv” (adaptive channel modulation) style module. Train their model on VisDrone and export; ensure the custom layer is supported in ONNX/TensorRT (usually standard conv + sigmoid/mul is fine).
- **Add to vanilla YOLOv8 (advanced):** Insert a lightweight SE or CBAM block after the backbone or before the neck (e.g. in C2f). Retrain; expect a small latency increase. Prefer **channel-only** (SE) or very small kernel **spatial** attention to keep inference fast on Jetson.

---

#### 7. Tiling (slicing / tiled inference)

**What it is:**  
**Tiling** (also called **slicing** or **sliding-window inference**) divides high-resolution images into smaller, often **overlapping** patches (tiles). The detector runs on each tile at native model input size (e.g. 640×640), so small objects in the full image are seen at a larger effective scale inside the tile. At inference, detections from all tiles are **merged** and transformed back to the original image coordinates (with NMS across overlapping regions to remove duplicates).

**Why it helps small objects:**  
- In a single pass at 640×640, a 4K frame is heavily downsampled and tiny objects can shrink to a handful of pixels or less.  
- By tiling (e.g. 640×640 windows with overlap), each tile is at full model resolution; a small object that would be 2×2 pixels in a downsampled full image may be 10×10 or more inside one tile, making it detectable.  
- Overlap ensures objects near tile boundaries are not cut in half and missed; they appear fully in at least one tile.

**Key aspects:**

| Aspect | Recommendation |
|--------|----------------|
| **Overlapping tiles** | Use overlap (e.g. **25%** or 50%) so objects at tile edges are fully contained in at least one tile. Reduces boundary cut-off and missed detections. |
| **Inference flow** | Run the model on each tile → get boxes in tile coordinates → map boxes to full-image coordinates → run NMS over all detections to merge duplicates from overlapping tiles. |
| **Tools** | **SAHI** (Slicing Aided Hyper Inference) is a popular library for tiled inference with YOLO and other detectors. Custom scripts can also slice images, run inference, and merge. |
| **Performance** | Tiling improves detection accuracy on high-res imagery but **increases compute and memory**: many tiles per image mean many forward passes. On Jetson, balance tile size, overlap, and throughput. |

**When to use:**  
Tiling is especially useful for **aerial imagery**, **drone footage**, and **medical imaging**, where images are large (e.g. 1920×1080, 4K) and small details are critical. Use it when single-frame downsampling would make your targets too small for the model.

**How to achieve it:**

- **SAHI (Slicing Aided Hyper Inference):**  
  Install `sahi`, then use it to slice the image (or video frame), run your YOLO (or other) model on each slice, and merge results. Supports overlap ratio, slice size, and NMS after merge.

  ```bash
  pip install sahi
  ```

  Example pattern (concept): define slice dimensions and overlap, run `get_sliced_prediction()` (or equivalent) with your model and image; SAHI returns merged detections in original image coordinates. See the [demo output](#demo-output-small-object-detection) image above for an example of SAHI output on a crowded, multi-scale scene.

- **Custom pipeline:**  
  1. Split image into tiles (e.g. 640×640, stride = 640×(1 − overlap)).  
  2. Run detector on each tile; collect boxes.  
  3. Offset box coordinates by tile top-left.  
  4. Run NMS on the combined list (IoU threshold e.g. 0.5) to drop duplicates from overlapping tiles.

- **Training:**  
  You can also **train** on tiles (e.g. crop high-res images into overlapping patches and annotate or map labels to tiles). That way the model always sees “zoomed” content; at inference, use the same tiling strategy.

**Jetson note:**  
Tiling multiplies inference count per frame. For real-time video, use fewer/smaller tiles or lower overlap if needed, or reserve tiling for key frames / offline processing when accuracy matters more than latency.

---

### Suggested application order to achieve the goal

1. **Quick wins (no model change):**  
   **Higher input resolution** (train and infer at 832 or 960) + **multi-scale training** (default in Ultralytics) + **confidence/NMS tuning** (see §8). Validate mAP on VisDrone val; if small-class mAP is acceptable and latency on Jetson is fine, stop here.

2. **Next step (better small-object mAP):**  
   Try a **published small-object variant**: **SPD-YOLOv8** (SPD in backbone) or **LPAE-YOLOv8** (extra small-object head + lightweight attention). Train on VisDrone with the same data pipeline; compare mAP and inference speed vs vanilla YOLOv8s.

3. **If you need to reduce false positives without losing recall:**  
   Keep **anchor/loss tuning** in mind (e.g. Wise-IoU) when training a variant; tune **confidence and NMS** after deployment (see §8).

4. **Jetson deployment:**  
   Prefer **YOLOv8n or YOLOv8s** (or equivalent variant) + **FP16 or INT8** TensorRT. Use **832** as a compromise resolution if 960 is too slow; document the resolution vs accuracy vs latency tradeoff.

5. **Very high-res or 4K drone footage:**  
   Use **tiling** (e.g. **SAHI** or custom slicing): overlapping tiles (e.g. 25% overlap), run detector per tile, merge and NMS in full-image coordinates. Trade off tile count/overlap vs real-time throughput on Jetson.

### Data Augmentation (important for small objects)

- **Scale:** Multi-scale training (e.g. 0.5–1.5× or similar) so the model sees small objects at different resolutions.
- **Mosaic / MixUp:** Improves robustness; use with care to avoid over-blurring small instances.
- **Motion blur, brightness/contrast:** Simulate challenging drone conditions.
- **Copy-paste (optional):** Paste small instances onto other images to increase small-object density.

---

## 5. Training Pipeline

### Environment

- Python 3.8+; CUDA-capable GPU (training on Jetson is possible but slow; prefer a desktop GPU or cloud for training).
- Install: `ultralytics` (YOLOv8), and optionally PyTorch with CUDA.

```bash
pip install ultralytics
# or from source for latest
```

### Steps

1. **Prepare data:** Download VisDrone2019-DET train/val; convert annotations to YOLO format; create `data.yaml` as above.
2. **Train:** Transfer learning from COCO-pretrained YOLOv8:

   ```bash
   yolo detect train data=path/to/VisDrone/data.yaml model=yolov8s.pt epochs=100 imgsz=640 batch=16
   ```

   Adjust `imgsz` (e.g. 832 for more small-object signal), `batch`, and `epochs` as needed. For Jetson, `yolov8n.pt` or `yolov8s.pt` are typical.
3. **Validate:**  
   `yolo detect val model=runs/detect/train/weights/best.pt data=path/to/VisDrone/data.yaml`
4. **Tune:** If small-class mAP is low, increase resolution, add augmentations, or try a small-object–oriented variant (see [Resources](#10-resources)).

### Confidence and NMS (for deployment)

- **Confidence threshold:** Lower (e.g. 0.2–0.35) can improve recall on small objects at the cost of more false positives; tune on val set.
- **NMS IoU:** Slightly higher IoU (e.g. 0.5–0.6) can help merge duplicate boxes on small instances; validate with your metric.

---

## 6. Export and TensorRT Optimization

### Export to ONNX then TensorRT

```bash
# From best.pt
yolo export model=runs/detect/train/weights/best.pt format=onnx simplify=True
yolo export model=runs/detect/train/weights/best.pt format=engine device=0 half=True  # FP16 on Jetson
# INT8 (recommended on Jetson; provide calibration data if needed)
yolo export model=runs/detect/train/weights/best.pt format=engine device=0 int8=True data=path/to/VisDrone/data.yaml
```

Or use `trtexec` for more control (batch size, workspace, precision).

### Optimization Tips (Jetson Orin Nano)

- **FP16:** Default choice for speed vs accuracy.
- **INT8:** Best throughput; calibrate with a representative subset of VisDrone (or your deployment domain) to limit accuracy drop.
- **Input size:** Match training (e.g. 640 or 832); larger size improves small-object detection but increases latency.
- **DLA:** For supported layers, offloading to DLA can free GPU for other tasks; benchmark GPU vs DLA vs GPU+DLA.

Validate mAP (or your metric) after export: run the same validation script with the `.engine` model if your framework supports it.

---

## 7. DeepStream Integration on Jetson

### Role of the Model

- The VisDrone-trained model serves as the **primary detector** in the existing DeepStream + GStreamer + FastAPI pipeline.
- DeepStream’s `nvinfer` (primary GIE) loads the TensorRT engine; run inference on decoded frames; optionally run a secondary classifier (secondary GIE) on cropped detections.

### Configuration

- Use the same pattern as in the main [ML and AI — DeepStream](../Guide.md#13-deepstream-for-video-pipelines) guide:
  - **Engine:** Your generated `.engine` (from VisDrone-trained YOLOv8).
  - **Labels:** 10 classes (or however many you kept); label file path in config.
  - **Preprocessing:** Input size (e.g. 640 or 832), normalize to match training.
  - **Post-processing:** Confidence threshold and NMS (e.g. `nms-iou-threshold`, `cluster-mode`) tuned for small objects.

### Reducing False Positives

- Slightly raise confidence threshold and/or tighten NMS.
- Optional: add a lightweight secondary classifier (e.g. on cropped patches) to re-score or filter classes.
- Temporal consistency: use tracking (e.g. DeepStream tracker) and require detections to persist over several frames before exposing them to the API.

### Stability for Small / Fast-Moving Targets

- Associate detections across frames with the built-in tracker (e.g. IOU or NvDCF).
- Optional: temporal smoothing or voting over a short window before returning results to FastAPI.

---

## 8. Tuning Confidence and NMS

Post-processing has two main knobs: **confidence threshold** (which detections to keep) and **NMS IoU threshold** (how aggressively to merge overlapping boxes). Both affect recall, precision, and behaviour on small objects.

### What each parameter does

| Parameter | Effect |
|-----------|--------|
| **Confidence threshold** | Detections with score below this are dropped. **Lower** → more detections (higher recall, more false positives). **Higher** → fewer, more confident detections (higher precision, may miss small/ambiguous objects). |
| **NMS IoU threshold** | When two boxes overlap with IoU above this value, the lower-scoring one is removed. **Lower** IoU → more aggressive merging (fewer duplicates, can over-merge nearby small objects). **Higher** IoU → keep more overlapping boxes (better for dense small objects, but more duplicate detections). |

### Why small objects need different tuning

- **Small objects** often get lower raw confidence than large ones (fewer pixels, less context). A high confidence threshold (e.g. 0.5) can wipe out most small detections.
- **Dense small instances** (e.g. crowds, parked cars) can produce many overlapping boxes; NMS with default IoU (e.g. 0.45) may over-suppress. Slightly **higher** NMS IoU (e.g. 0.5–0.6) keeps more distinct small boxes.
- **False positives** on background or noise tend to have mid–low scores; raising confidence helps, but too high hurts small-object recall.

### Recommended tuning flow

1. **Fix NMS first (optional but useful).**  
   Use a typical confidence (e.g. 0.25) and sweep NMS IoU (e.g. 0.4, 0.45, 0.5, 0.55, 0.6) on the **validation set**. Compute mAP@0.5 (and mAP@0.5:0.95 if available). Pick the IoU that gives the best mAP or best tradeoff for your small classes.
2. **Sweep confidence.**  
   With NMS fixed, sweep confidence (e.g. 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5) and again measure mAP and, if needed, precision/recall per class.
3. **Choose operating point.**  
   - Prefer **recall** (e.g. downstream classifier or human review): lower confidence (0.2–0.3).  
   - Prefer **precision** (fewer false alarms): higher confidence (0.35–0.5).  
   - For small objects specifically, start with **conf ≈ 0.25, NMS IoU ≈ 0.5** and adjust from there.

### Where to set them

**YOLOv8 (validation / inference):**

```bash
# Validation with custom conf and IoU
yolo detect val model=best.pt data=data.yaml conf=0.25 iou=0.5

# Predict: pass at inference
yolo detect predict model=best.pt source=images/ conf=0.25 iou=0.5
```

In Python:

```python
from ultralytics import YOLO
model = YOLO("best.pt")
model.val(data="data.yaml", conf=0.25, iou=0.5)
# or predict
model.predict(source="images/", conf=0.25, iou=0.5)
```

**DeepStream (nvinfer):**  
In the config (e.g. `config_infer_primary_*.txt` or app config):

- **Confidence:** `threshold` or `conf-threshold` (depends on custom parser; often 0–1).
- **NMS:** `nms-iou-threshold` (or similar). Set to the same value you chose (e.g. 0.5).

If your pipeline uses a custom post-processor, pass the same `conf` and `iou` there so behaviour matches validation.

### Quick reference

| Goal | Confidence | NMS IoU |
|------|------------|---------|
| Maximize recall (small objects) | 0.20–0.30 | 0.50–0.60 |
| Balanced | 0.25–0.35 | 0.45–0.55 |
| Reduce false positives | 0.35–0.50 | 0.45–0.50 |

Always re-check mAP (and optional per-class precision/recall) on the val set after changing either parameter.

---

## 9. End-to-End Checklist

- [ ] Download VisDrone2019-DET train and val from [VisDrone/VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset) (or linked mirrors).
- [ ] Convert annotations to YOLO format; create `data.yaml` with correct `path`, `train`, `val`, `nc`, `names`.
- [ ] Train YOLOv8 (n/s/m) from COCO pretrained; use multi-scale and small-object–friendly augmentation.
- [ ] Validate mAP (overall and per-class); tune confidence/NMS for small-object recall vs false positives.
- [ ] Export to TensorRT (FP16 or INT8); verify accuracy on Jetson.
- [ ] Integrate engine into DeepStream primary GIE; set batch size, input size, and labels.
- [ ] Tune confidence and NMS in DeepStream config; add tracking and optional secondary classification if needed.
- [ ] Run full pipeline (video ingest → detection → optional classification → streaming/recording); measure latency and quality; document resolution vs speed tradeoffs.
- [ ] If using high-res or 4K input: consider **tiling** (e.g. SAHI) with overlapping tiles; merge detections and run NMS; tune tile size/overlap vs throughput on Jetson.

---

## 10. Resources

### Dataset

- [VisDrone/VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset) — official repo and links.
- [VisDrone dataset (Ultralytics)](https://docs.ultralytics.com/datasets/detect/visdrone/) — format and YOLO usage.
- [VisDrone2YOLO](https://github.com/adityatandon/VisDrone2YOLO) — conversion script and YOLO-format labels.
- [YOLOv5 VisDrone.yaml](https://github.com/ultralytics/yolov5/blob/master/data/VisDrone.yaml) — reference for paths and conversion.

### Small-Object Detection (Drone / UAV)

- Improved YOLOv8 for small object detection in aerial images (multi-scale features, Wise-IoU) — e.g. IEEE/Springer variants targeting VisDrone.
- **SPD-YOLOv8** — SPD-Conv, MPDIoU; improved mAP on VisDrone.
- **LPAE-YOLOv8** — lightweight; LSE-Head, small-object layer, adaptive attention; VisDrone benchmarks.
- **RLRD-YOLO** — improved YOLOv8 for UAV small-object detection.

### Tiling / Slicing (high-resolution and small objects)

- **SAHI (Slicing Aided Hyper Inference)** — [GitHub](https://github.com/obss/sahi): run detection on sliced/overlapping tiles and merge results; supports YOLO and other backends.
- **Microsoft Learn** — overlapping tiles (e.g. 25% overlap) so objects at boundaries are fully captured.
- **“How to Use SAHI Tiling Windows to Detect Small Objects”** (Nicolai Nielsen, YouTube) — practical tiled inference with YOLO.
- **“Improving Small Object Detection in 4K Drone Footage Using Tiling”** (Hailo Community) — tiling for aerial/drone and edge deployment.
- **CVF / IEEE** — tiling for small object detection on high-resolution images; tradeoff between accuracy and compute.

### Deployment

- Main guide: [Edge AI Optimization](../../Guide.md) — TensorRT, DeepStream, Jetson profiling.
- [NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) — pipeline and nvinfer config.
- [NVIDIA TAO Toolkit](../tao-toolkit/Guide.md) — optional: train/adapt with TAO and export to TensorRT/DeepStream.

---

*Back to [ML and AI — Projects](../Guide.md#14-projects).*
