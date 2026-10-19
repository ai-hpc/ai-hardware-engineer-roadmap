# Small-object detection: keep pixels, labels and coordinates consistent

<div class="course-identity auto-course" style="--course-accent: #4f46e5; --course-accent-rgb: 79, 70, 229;" markdown="1">
<div class="course-identity__icon">SODO</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Jetson Track</p>
<p class="course-identity__title">Specialized course identity for Small Object Detection on Jetson — Project Guide.</p>
<p class="course-identity__meta">Artifact: Jetson integration demo · Measure: latency, memory, power, logs</p>
</div>
</div>

**Updated 18 October 2026.** Learn why tiny objects disappear during preprocessing, build a tested coordinate transform, and evaluate a detector without confusing benchmark and deployment settings. Begin with the CPU exercise; training, TensorRT and DeepStream are separate hardware extensions.

Prerequisites: image coordinates, bounding boxes, precision/recall, and the [ML deployment workflow](../Guide.md). The [camera geometry lab](../../../../Phase%203%20-%20Artificial%20Intelligence/Track%20A%20-%20Hardware%20and%20Edge%20AI/4.%20Sensor%20Fusion/Camera-Time-Fusion-Lab/Guide.md) is useful when detections must become robot-frame measurements.

## 1. Compute the pixel budget

An 8-pixel-wide target in a 1920×1080 frame becomes about **2.67 pixels** wide when the full image is letterboxed to 640×640. The scale is `640/1920=1/3`, the resized height is 360, and top/bottom padding is 140 pixels each. A large model cannot reconstruct information that resizing removed.

A 640×640 crop resized to 640×640 keeps that target 8 pixels wide. Cropping may help recognition but removes global context, repeats work in overlaps and can cut objects at boundaries. The right experiment compares full-frame inference, higher input resolution and tiles on the same source frames.

## 2. Run the original CPU geometry exercise

From the repository root:

```bash
python3 'Phase 4 - Track B - Nvidia Jetson/5. Application Development/5. ML and AI/small-object-detection-jetson/labs/geometry.py'
python3 -m unittest discover -s \
  'Phase 4 - Track B - Nvidia Jetson/5. Application Development/5. ML and AI/small-object-detection-jetson/labs' -v
```

Expected output includes widths `2.6667` and `8.0`, top padding `140`, horizontal tile origins `[0,512,1024,1280]`, and vertical origins `[0,440]`. These 640-pixel tiles use at least 128 pixels of overlap; the final tile is shifted to cover the image edge. There are **8 tiles**, not a single inference per source frame.

Four actual standard-library tests cover full-frame round trips, odd-size integer rounding, crop offsets, clipped/ignored annotations and complete edge coverage. This verifies geometry; no detector accuracy or GPU throughput is implied.

## 3. Understand the dataset and preserve its protocol

VisDrone2019-DET has 6,471 training images, 548 validation images and 1,610 test-dev images in the common static-image setup. The official repository now explicitly says **test-dev bounding-box annotations are available**; test-challenge annotations are withheld. The full VisDrone benchmark's 2.6-million-box figure covers more than this static-image training split. [Official dataset repository](https://github.com/VisDrone/VisDrone-Dataset), [maintained format/integration card](https://docs.ultralytics.com/datasets/detect/visdrone/).

Use the upstream dataset terms and citation supplied with the files you obtain. The software converter's license is not automatically the dataset's license. The course's small geometry exercise downloads no imagery or external annotations.

Keep official partitions for benchmark comparisons. For your own drone footage, group frames by flight/sequence/site before splitting. Keep source-frame identity through tiling: tiles from one image must stay in the same split.

## 4. Convert labels without changing their meaning

A VisDrone ground-truth row contains:

```text
x, y, width, height, score, category, truncation, occlusion
```

For ground truth, score 0 marks an ignored annotation. Categories 1–10 are the ten evaluated foreground classes; 0 and 11 are special/ignored categories in the standard ten-class setup. The CPU converter records ignored rows separately instead of pretending they are ordinary background.

| VisDrone category | Foreground class | YOLO index |
|---:|---|---:|
| 1 | pedestrian | 0 |
| 2 | people | 1 |
| 3 | bicycle | 2 |
| 4 | car | 3 |
| 5 | van | 4 |
| 6 | truck | 5 |
| 7 | tricycle | 6 |
| 8 | awning-tricycle | 7 |
| 9 | bus | 8 |
| 10 | motor | 9 |

For a box `[x,y,w,h]` in image `W×H`, YOLO's normalized centre/size is `[(x+w/2)/W,(y+h/2)/H,w/W,h/H]`. Validate finite coordinates and positive size; clip to the image according to a recorded policy before normalization. For `[-5,10,10,10]` in 100×100, clipping gives `[0,10,5,10]`, so the normalized centre/size is `[.025,.15,.05,.1]`.

Dropping an ignored-region label from a YOLO text file does not implement an ignored-region loss mask. Objects there may become false background supervision. Preserve the original annotations, document the training treatment, and use the [official evaluation toolkit](https://github.com/VisDrone/VisDrone2018-DET-toolkit) when reporting the benchmark's metric. A convenient converted YOLO metric is not automatically identical to it.

## 5. Choose a baseline before changing architecture

YOLOv8 remains a useful named historical baseline; it is not the newest YOLO family. Its detector head is **anchor-free**. Advice to tune anchor sizes is appropriate for some older architectures, not a YOLOv8-specific optimization. [YOLOv8 architecture documentation](https://docs.ultralytics.com/models/yolov8/).

Use a fixed model/checkpoint and start with its supported training configuration. Compare one change at a time:

- Increase input resolution and report memory plus latency.
- Add or choose a supported low-stride feature head, then retrain and verify the output decoder.
- Train with crops that preserve enough object pixels and context.
- Use appropriate scale, blur and lighting augmentation while checking tiny-box survival.
- Compare tiling with full-frame inference before adding attention or an unverified architecture variant.

A stride-4 feature map gives an 8-pixel object roughly two cells across, but does not guarantee detection. Loss assignment, context, occlusion and annotation quality still matter. A space-to-depth operation rearranges local pixels into channels; subsequent layers can still discard information, and the change requires compatible training.

## 6. Map tile detections back correctly

Record the **actual** resized dimensions after integer rounding. Let `sx,sy` be the resulting axis scales, `px,py` the padding, and `ox,oy` the source crop origin. For a model-space coordinate `(u,v)`:

\[
x=(u-p_x)/s_x+o_x,\qquad y=(v-p_y)/s_y+o_y.
\]

Apply this to both corners, clip to the valid source image and discard zero-area boxes. Do not add the crop offset before removing padding, and do not apply it twice.

Merge detections in common full-image coordinates, using class-aware suppression/merging when classes are separate. Preserve confidence definitions. Intersection-over-union, intersection-over-smaller-area and weighted merging are different rules; choose and record one, especially for objects cut by tile boundaries. SAHI supports several postprocessing choices, so its defaults should not be silently described as plain NMS. [SAHI project](https://github.com/obss/sahi).

For 8 sequential tiles, a hypothetical 10 ms per-tile engine already costs roughly 80 ms before preprocessing/merging. Batched tiles may improve throughput but require more memory; neither method implies 100 source frames/s. Include optional whole-image inference in the work count if enabled.

## 7. Distinguish three different thresholds

1. **Confidence threshold:** removes low-score predictions at an operating point.
2. **NMS overlap threshold:** decides which overlapping predictions compete during postprocessing.
3. **Evaluation IoU threshold:** decides whether a prediction matches a ground-truth box.

Raising the NMS IoU threshold generally suppresses fewer overlapping boxes, which may help crowded scenes but retain duplicates. It does not make ground-truth matching more permissive. Use a sufficiently low score cutoff and adequate maximum detections when computing a precision-recall curve; a deployment cutoff such as 0.5 can discard predictions needed for AP.

Report `AP50`, `AP50:95`, size slices and per-class results separately. Dense VisDrone scenes may exceed a default `max_det`; record that cap and the evaluator. Choose deployment confidence using validation false positives per frame and recall, then evaluate the frozen setting on held-out frames.

## 8. Export the exact trained model

Follow the [current Ultralytics TensorRT export contract](https://docs.ultralytics.com/integrations/tensorrt/) for the installed version, if that is your model family. Use an explicit representative calibration dataset for INT8. Never let an unrelated default dataset stand in for small, crowded target scenes.

Model version, preprocessing, image size, dynamic shapes, NMS inclusion and output tensor schema must match between reference and runtime. An engine file does not include an arbitrary DeepStream parser. Determine whether boxes are decoded, whether NMS is already applied and which layout the output uses; compare one saved preprocessed tensor and its outputs before timing.

Use the [Jetson compatibility matrix](../../../Compatibility.md) and [named-tensor inference guide](../../../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Real-Time-Inference/Guide.md). Do not claim generic TensorRT 11 support on JetPack or DLA availability on Orin Nano. Target training/export/DeepStream runs have not been executed for this lesson; the executable evidence is the coordinate lab and shared CPU export lab.

## 9. Completion experiment and answers

Produce a before/after sheet with one source image, its letterboxed image, tile locations, transformed boxes, and error analysis. Record input size, thresholds, merging rule and timing scope. If no detector hardware is available, use synthetic boxes with the runnable geometry code and label the artifact accordingly.

1. **Why does an 8-pixel object shrink to 2.67 pixels?** The full-frame horizontal scale is one third; padding does not restore object detail.
2. **Why might an edge tile overlap by more than 128 pixels?** Shifting the last tile to the boundary prevents uncovered pixels.
3. **Can class IDs be merged and the result reported as the same benchmark?** No. It changes the task and metric interpretation.
4. **Does lower confidence always improve AP?** Not by itself. AP depends on ranking and matching; changing a high cutoff may recover missing predictions, but it does not improve the model's ordering.
5. **Why is a correct crop transform relevant to robotics?** The wrong full-image box can produce a wrong bearing or depth ROI even if the detector itself is numerically correct.
