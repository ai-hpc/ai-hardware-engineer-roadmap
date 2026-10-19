# RGB/depth and thermal preparation: geometry, timing and signal evidence

<div class="course-identity auto-course" style="--course-accent: #7e22ce; --course-accent-rgb: 126, 34, 206;" markdown="1">
<div class="course-identity__icon">NCMS</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Jetson Track</p>
<p class="course-identity__title">Specialized course identity for Non-Contact Multi-Sensor Monitoring on Edge — Project Guide.</p>
<p class="course-identity__meta">Artifact: Jetson integration demo · Measure: latency, memory, power, logs</p>
</div>
</div>

**Reviewed 18 October 2026.** This project teaches camera/depth alignment and temporal signal analysis. Its supplied code is an offline RGB/depth viewer plus small synthetic signal tests. It does not measure a physiological variable, provide a medical monitor, or demonstrate live thermal fusion.

Prerequisites: the [camera/time fusion lab](../../../../Phase%203%20-%20Artificial%20Intelligence/Track%20A%20-%20Hardware%20and%20Edge%20AI/4.%20Sensor%20Fusion/Camera-Time-Fusion-Lab/Guide.md), image coordinates and elementary frequency analysis. Begin with the [implemented CPU exercise](phase1/README.md); no external video download is required for its tests.

## 1. Define the measurement before choosing a filter

A thermal camera may supply radiometric measurements, raw sensor counts or a colorized display image. These have different meanings. Automatic gain, palette mapping, exposure changes and nonuniformity corrections can create temporal variations that are unrelated to scene temperature. Retain raw values, timestamp/clock metadata and the camera's calibration information.

A 0.8–3 Hz passband is a selected analysis band, not proof of the source of a signal. Motion, lighting, camera processing and sampling artifacts can produce energy there. This course validates a synthetic sinusoid and demonstrates an alias; it does not infer heart rate, blood flow or temperature from that band alone.

## 2. Map a point through the cameras

For a distortion-corrected RGB pixel `p=[u,v,1]` with **axial** depth `Z`, its RGB camera point is

\[
X_R=ZK_R^{-1}p.
\]

If the depth is radial distance along the ray instead, normalize the ray before multiplying by that distance. Distinguish these definitions in the source format.

Given a calibrated transform from RGB to thermal,

\[
X_T=R_{T\leftarrow R}X_R+t_{T\leftarrow R},\qquad
p_T\sim K_TX_T.
\]

Check positive thermal-camera depth, image bounds and occlusion. Apply the thermal lens model or use consistently rectified images. A 2-D homography works for its plane/pure-rotation assumptions; it is not a general depth-independent RGB-to-thermal mapping.

A rectangular RGB ROI need not remain a rectangle after projection. Background depth inside a box can project onto unrelated thermal pixels. Use a depth-valid object mask or explicitly state the approximation; the median depth of an entire face/person box is only a summary, not a calibrated surface reconstruction.

The [actual PnP/fusion lab](../../../../Phase%203%20-%20Artificial%20Intelligence/Track%20A%20-%20Hardware%20and%20Edge%20AI/4.%20Sensor%20Fusion/Camera-Time-Fusion-Lab/Guide.md) provides projected points, known camera geometry and timestamp tests. Use it to check transform directions before connecting a second sensor. [OpenCV calibration conventions](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html).

## 3. Treat timestamps as data

Same frame index and same nominal FPS do not prove synchronized sensors. Record exposure/capture timestamps, arrival timestamps, clock identity and estimated offset/drift. A frame available at processing time can still be too old for an ROI.

For offline paired videos, the supplied viewer assumes aligned frame indices and checks matching FPS, resolution and paired end-of-stream. That catches some file mismatches but cannot recover unrecorded capture offsets. For live RGB/thermal work, select or interpolate observations using a justified causal timing model and define a maximum age.

## 4. Understand the optional external RGB-D dataset

The [SJTU Free-Viewpoint RGB-D dataset repository](https://github.com/sjtu-medialab/Free-Viewpoint-RGB-D-Video-Dataset) supplies camera parameters and a disparity-to-depth expression. The inspected revision is `29dfe32d83094bd08deba0f4831d77ca459fcead`; its current citation is Guo et al., IEEE Transactions on Multimedia, 2024. It contains no thermal stream for this exercise.

The source formula is equivalent to

\[
Z(p)=\frac{32504}{(p/255)(32504/40-32504/150)+32504/150}.
\]

`p=0` maps to 150 and `p=255` to 40 in the dataset's depth units. The inspected README does not establish a metre conversion in that formula. The previous course's automatic centimetre interpretation was therefore removed. Supply a verified `--units-to-meters` scale only when a dataset-specific statement or independent calibration supports it; otherwise retain “dataset units”.

Do not infer byte versus normalized encoding from the current frame's maximum. A byte image containing only value 1 is a dark encoded image, not a normalized image at full scale. The fixed function requires the encoding explicitly. Compressed display video can also change grayscale values; use original depth data when quantitative precision matters.

In `paras.txt`, `world_position` is camera centre `C` and the projection is `Xc=R(Xw-C)`. OpenCV's translation in `Xc=RXw+t` is therefore `t=-RC`. Confusing those quantities moves the camera incorrectly. The first inspected block has `camera_id 0`; video filename numbering starts at `camera01`, so keep record index and file naming distinct.

The code loads calibration; it does not estimate calibration from video. It rejects incomplete records, duplicate IDs, nonfinite values and improper rotations. It rejects points at or behind the camera plane rather than hiding invalid division with a small epsilon.

## 5. Extract a synthetic frequency correctly

For uniformly sampled data at rate `fs`, frequency-bin spacing is `fs/N=1/T`. A 20-second record sampled at 20 Hz has 400 samples and 0.05 Hz bin spacing. A 1.2 Hz tone lies on a bin in this case. The supplied function removes the mean, applies a Hann window, computes a real FFT and finds the largest in-band peak.

This peak picker does not establish an amplitude-calibrated power spectral density, physiological identity or a confidence interval. A window changes amplitude normalization. Zero-padding can interpolate a plotted spectrum but does not add independent information or improve the fundamental record-length resolution.

An 8 Hz tone sampled at 10 Hz aliases to 2 Hz. The actual test demonstrates that it falls inside the chosen 0.8–3 Hz band. Post-sampling filtering cannot distinguish that alias from a real 2 Hz tone; anti-alias filtering and adequate acquisition rate must come before sampling.

The implementation rejects irregular timestamps, constants and a band touching/exceeding Nyquist. For real irregular data, use a documented resampling or nonuniform spectral method, retain gaps, and measure the effect. Averaging FPS silently is not a valid repair for arbitrary dropped frames.

## 6. Plan the thermal extension as a new experiment

Use a controlled target with a known temperature variation or reference measurement before a human subject. Record sensor response, sampling, missing frames, drift and motion. Estimate uncertainty and compare a null/control condition. If analyzing people, a separate validated protocol and reference measurement are required; an RGB-D replay cannot validate a thermal or physiological claim.

Eulerian magnification can make a temporal variation more visible, but it also amplifies noise and motion artifacts. It does not recover information below the sensor's usable resolution. Choose a raw-signal baseline before adding magnification or a neural model.

For edge deployment, measure capture-to-result latency, queue age, DSP window duration, memory and sustained thermal behavior. A 20-second analysis window is not a 20 ms measurement even if its FFT computes in 20 ms.

## 7. Completion exercise and answers

Run the [ten actual CPU tests](phase1/README.md), explain the byte-value-1 regression, and compare the known-tone/alias examples. Draw the two-camera transform and annotate every unit and timestamp. If using real data, retain provenance and scale evidence separately.

1. **Does grayscale zero always mean missing depth?** No. In this source mapping it means the far endpoint; any validity mask must come from the source convention.
2. **Does matching FPS prove synchronization?** No. It does not establish capture offset, drift or dropped-frame identity.
3. **Why does the viewer avoid a metre label by default?** The inspected source formula alone does not establish physical scale.
4. **Does a strong 1.2 Hz peak prove a biological source?** No. The measured quantity and reference protocol must establish interpretation.
