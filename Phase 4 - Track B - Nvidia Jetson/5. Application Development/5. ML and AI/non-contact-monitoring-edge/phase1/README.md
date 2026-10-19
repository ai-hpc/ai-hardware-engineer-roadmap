# CPU RGB/depth exercise and synthetic signal checks

[Project concepts](../Guide.md) · [Camera/time fusion lab](../../../../../Phase%203%20-%20Artificial%20Intelligence/Track%20A%20-%20Hardware%20and%20Edge%20AI/4.%20Sensor%20Fusion/Camera-Time-Fusion-Lab/Guide.md)

This folder loads calibration, decodes a documented disparity representation, runs OpenCV's classical face/person detectors, and displays or saves paired RGB/depth frames. It also provides a small spectral-peak reference. The tests create their own tiny synthetic videos; they require no external dataset, GPU or camera.

## 1. Use the tested environment

The executed CPU stack is Python 3.12, NumPy 2.3.5 and OpenCV headless 4.13.0.92. Reuse the exact hash-locked environment from the [camera lab](../../../../../Phase%203%20-%20Artificial%20Intelligence/Track%20A%20-%20Hardware%20and%20Edge%20AI/4.%20Sensor%20Fusion/Camera-Time-Fusion-Lab/Guide.md). Headless OpenCV supports the test/video-file path but not `imshow`. The local [requirements file](requirements.txt) pins an interactive desktop alternative; that GUI package was not used in the retained headless test run. Install only one OpenCV wheel variant in an environment.

From the repository root, with that environment's Python:

```bash
/tmp/course-perception-cpu-venv/bin/python -m unittest discover -s \
  'Phase 4 - Track B - Nvidia Jetson/5. Application Development/5. ML and AI/non-contact-monitoring-edge/phase1/labs/optional' -v
```

If you created a differently named virtual environment, substitute its interpreter. **Actual result on 18 October 2026: 10 tests passed.** They exercise byte/normalized depth, ROI clipping and nonfinite values, camera-centre projection, malformed calibration, the actual CPU detectors on a blank image, a headless CLI run on lossless FFV1 video, pair mismatches, a known tone and an alias.

The synthetic video fixture intentionally has no people or faces; zero detections validates a blank-input path, not detection recall. Its `camera02` pair and two calibration records catch the previous bug where selecting another camera still opened camera01 video.

## 2. Optional external-video path

Obtain the paired RGB/depth files and calibration from the [upstream dataset](https://github.com/sjtu-medialab/Free-Viewpoint-RGB-D-Video-Dataset), respecting the terms supplied with your copy. The course did not download or execute those full videos. Expected structure:

```text
my_dataset/
  Camera Parameters/paras.txt
  camera01-rgb.mp4
  camera01-depth.mp4
```

Run up to 30 frames without a display:

```bash
/tmp/course-perception-cpu-venv/bin/python \
  'Phase 4 - Track B - Nvidia Jetson/5. Application Development/5. ML and AI/non-contact-monitoring-edge/phase1/run_pipeline.py' \
  --dataset-dir /absolute/path/to/my_dataset --camera-index 0 \
  --no-display --no-person --max-frames 30 --report /tmp/rgb-depth-report.json
```

`--camera-index 0` selects the first calibration block and `camera01`; index 1 selects the second block and `camera02`. Confirm that this mapping matches your dataset copy. `--suffix avi` supports the lossless synthetic test videos. `--out /tmp/overlay.mp4` optionally writes an annotated visualization; it is not a new quantitative depth file.

Depth is labeled **dataset units** unless you supply a positive, externally verified `--units-to-meters` scale. For a separately documented centimetre dataset, that factor would be 0.01; do not infer that unit merely from the number 40. Pairing assumes corresponding frame indices. Matching FPS/size and equal stream lengths are checked, but capture synchronization remains an external assumption.

Haar detection does not return calibrated probabilities here, so its score is `None`. HOG returns an uncalibrated SVM detector score, which is preserved rather than clipped to [0,1]. Face and person boxes represent different semantic objects and are not suppressed against each other.

## 3. Run the known-tone exercise

From this `phase1` directory:

```python
import numpy as np
from signal_reference import peak_in_band

t = np.arange(400) / 20.0
y = 2.0 + np.sin(2*np.pi*1.2*t) + 0.1*np.sin(2*np.pi*4.0*t)
print(peak_in_band(t, y))
```

Expected peak: **1.2 Hz**, sample rate **20 Hz**, bin spacing **0.05 Hz**. Now use 200 samples at 10 Hz with an 8 Hz sine wave; the detected peak is **2 Hz**. Explain why no later bandpass filter can identify which source frequency produced those samples.

Change one timestamp by 20 ms and rerun. The function rejects the now-irregular sampling instead of treating it as uniform. Restore timestamps before comparing window lengths.

## 4. Completion artifact

Save the test output, one synthetic signal report, and a diagram showing `Xc=R(Xw-C)` versus OpenCV's `Xc=RXw+t`. Explain how to reject a point behind the camera and why a box's median depth can include background. Live RGB/thermal calibration, radiometric measurement and physiological validation remain separate work.
