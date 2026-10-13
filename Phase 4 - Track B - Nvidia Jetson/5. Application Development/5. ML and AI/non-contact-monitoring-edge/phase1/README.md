# Phase 1: Camera Calibration and Real-Time Object Detection with Depth

Pipeline for the non-contact monitoring project using **camera01** from the [Free-Viewpoint RGB-D Video Dataset](https://medialab.sjtu.edu.cn/post/free-viewpoint-rgb-d-video-dataset/) (SJTU).

## What it does

- **Camera calibration:** Loads intrinsics (and extrinsics) from `Camera Parameters/paras.txt` for camera01 (index 0).
- **Depth:** Converts grayscale depth frames to metric depth (meters) using the dataset formula.
- **Detection:** Runs face (Haar) and person (HOG) detection on the RGB stream via OpenCV (no extra model files).
- **Depth per ROI:** For each detection bounding box, computes median depth and overlays it on the image.

## Setup

```bash
cd "Phase 4 - Track B - Nvidia Jetson/5. Edge AI Optimization/non-contact-monitoring-edge"
pip install -r phase1/requirements.txt
```

Ensure the dataset is present:

- `Free-Viewpoint-RGB-D-Video-Dataset-main/camera01-rgb.mp4`
- `Free-Viewpoint-RGB-D-Video-Dataset-main/camera01-depth.mp4`
- `Free-Viewpoint-RGB-D-Video-Dataset-main/Camera Parameters/paras.txt`

## Run

From the `non-contact-monitoring-edge` directory:

```bash
python phase1/run_pipeline.py
```

Or from `phase1`:

```bash
cd phase1
python run_pipeline.py --dataset-dir ../Free-Viewpoint-RGB-D-Video-Dataset-main
```

### Options

| Option | Description |
|--------|-------------|
| `--dataset-dir PATH` | Root of the dataset (default: `../Free-Viewpoint-RGB-D-Video-Dataset-main`) |
| `--camera-index N` | Camera index in paras.txt (0 = camera01) |
| `--no-person` | Only run face detection (faster) |
| `--no-display` | No GUI (e.g. headless); still processes and prints progress |
| `--out PATH` | Write output video (e.g. `phase1_out.mp4`) |
| `--max-frames N` | Process at most N frames (0 = all) |

## Module overview

| Module | Role |
|--------|------|
| `calibration.py` | Parse `paras.txt`, expose K, R, t; world-to-image projection. |
| `depth_utils.py` | Grayscale → metric depth; median depth in a bounding box. |
| `detection.py` | Face (Haar) and person (HOG) detectors; composite that avoids duplicate person/face boxes. |
| `run_pipeline.py` | Main: load calibration, read RGB + depth, detect, compute depth per box, visualize. |

## Citation

When using the dataset, cite: Guo S, Zhou K, Hu J, et al. A new free viewpoint video dataset and DIBR benchmark. *Proceedings of the 13th ACM Multimedia Systems Conference*. 2022: 265–271.

## Practice and assessment

**Entry diagnostic (prerequisite check):** Define the observable signal, ground-truth timing, and limits of a proxy measurement. If this is unfamiliar, use the prerequisite material above or the [roadmap diagnostics](../../../../../Learning-Guide.md#diagnose-before-reading) before starting the lab.

1. **Baseline lab:** Run the supplied recorded-data pipeline and compare estimated signals with a reference.
2. **Guided variation:** Inject motion, missing frames, or illumination changes and identify invalid-result handling.
3. **Independent transfer:** repeat the method on a new input or configuration and explain where your original assumptions stop applying.

**Evidence to submit:** Error versus reference, usable-window rate, latency, and failure categories; do not infer clinical validity. Keep raw results and reproduction steps; distinguish measured results from estimates or work not run.

**Self-check:** Which observation would disprove your explanation of the result? Identify a competing explanation and the smallest experiment that separates them. A passing answer names a concrete predicted difference, not only another tool to try.

**Exit criteria:** reproduce the baseline, demonstrate the negative/variation check, and meet the [shared completion rubric](../../../../../Learning-Guide.md#completion-rubric). Use the [lesson cycle](../../../../../Learning-Guide.md#a-repeatable-lesson-cycle) for retrieval practice and spaced review.
