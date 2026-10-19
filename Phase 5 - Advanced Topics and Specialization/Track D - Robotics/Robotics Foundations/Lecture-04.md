# 4. Camera geometry and uncertainty

**Purpose:** convert a detection into a spatial estimate with meaningful units, confidence assumptions, and a timestamp. **Prerequisite:** [frames](Lecture-01.md). **Next:** [planning](Lecture-05.md).

## A box on an image is not yet a grasp point

A detector can locate pixels belonging to a cup. To move an arm, you also need depth or another geometric constraint, camera calibration, the camera-to-robot transform, and the time of the observation. The center of a bounding box may fall on background or an empty handle; it is not automatically the object's center of mass or a valid grasp pose.

For an ideal rectified pinhole camera, a point `(X,Y,Z)` in the optical frame projects to pixel `(u,v)`:

\[
u=f_xX/Z+c_x,\qquad v=f_yY/Z+c_y.
\]

The focal lengths `fx,fy` and principal point `cx,cy` are measured in pixels. `Z` is forward distance along the optical axis. Pixel `u` grows rightward, `v` downward. Lens distortion must be removed or included in a richer model. Calibration belongs to a particular camera mode and image geometry. [OpenCV camera model and calibration documentation](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html)

### Worked example: one pixel with measured depth

For `fx=fy=500 px`, `cx=320 px`, `cy=240 px`, and point `(0.1,0.05,1.0) m`, the pixel is `(370,265)`. Given that pixel and `Z=1.0 m`, invert the equations:

\[
X=(u-c_x)Z/f_x=0.1\;\mathrm m,\quad
Y=(v-c_y)Z/f_y=0.05\;\mathrm m.
\]

The pixel alone defines a ray: `(0.2,0.1,2.0)` projects to the same location. A range sensor may report Euclidean distance along the ray rather than optical-axis depth; those are different quantities away from the image center. A depth image may use millimetres, metres, or a scale factor. Inspect the encoding and sensor documentation before multiplying.

If you halve the image dimensions without cropping, scale `fx,fy,cx,cy` by one half too. If you crop 100 pixels from the left, subtract 100 from `cx` in the cropped coordinate system. Retaining the old intrinsics introduces a systematic position error that a more accurate detector cannot fix.

## Estimate uncertainty instead of hiding it

A **bias** is a systematic error; **variance** describes spread. A measurement with small variance can still have a large calibration bias. A neural confidence score is not automatically a position covariance.

Consider a one-dimensional position estimate `mu=1.0 m` with variance `P=0.04 m²`, and an independent sensor measurement `z=1.3 m` with noise variance `R=0.01 m²`. Under a scalar linear Gaussian model:

\[
K=P/(P+R)=0.8,\quad
\mu_{new}=\mu+K(z-\mu)=1.24\;\mathrm m,\quad
P_{new}=(1-K)P=0.008\;\mathrm{m}^2.
\]

The **innovation** is `z-mu=0.3 m`: disagreement between prediction and observation. `K` determines how much to trust that disagreement. A larger measurement variance makes the gain smaller. Standard deviation is the square root of variance; do not substitute `0.2 m` where the formula expects `0.04 m²`.

Between measurements, a simple model predicts motion and increases uncertainty. For a known constant velocity `v`, `mu_pred=mu+v*dt` and `P_pred=P+Q`, where `Q` is the process variance for that interval. Real state estimators track coupled position, velocity, orientation, and biases with covariance matrices. The scalar example teaches the update; it is not an EKF implementation.

Fusing the same measurement twice as if independent artificially reduces uncertainty. For example, wheel odometry and a derived pose may share the same encoder information. Missing this correlation makes a filter overconfident. A large innovation also deserves investigation: wrong frame, stale timestamp, wrong association, or a failed sensor may be more likely than a sudden real motion.

## From estimation to perception evaluation

Evaluate detections or poses on held-out scenes, including occlusion, lighting changes, reflective surfaces, blur, and missing depth. Split by scene or recording session when adjacent frames are correlated. A random frame split can let nearly identical views leak across training and evaluation.

Track both task quality and age: 2D detection accuracy, 3D position/orientation error where ground truth exists, invalid-depth rate, ID switches for tracking, and capture-to-use delay. If a filter improves smoothness but adds lag, show both. A tracker can look visually stable while following the wrong object.

## Practice and answer guidance

Run `python3 labs/foundations.py perception`.

**Guided variation:** hold the pixel fixed and double depth. The reconstructed X, Y, and Z all double. Increase `R` from `0.01` to `1 m²`; the gain falls from `0.8` to about `0.03846`, so the result stays much closer to the prior. Predict both before running.

**Independent task:** generate a set of points with positive depth, project them, and reconstruct them using the original depths. Then intentionally use intrinsics for an image of twice the size. Report the resulting error and show that restoring the correct intrinsics fixes it. Include zero, negative, and nonfinite depth cases; reject them rather than emitting a plausible grasp point.

**Self-check:** Can a monocular network output metric depth? It may, using learned priors or additional information, but a single ideal image does not uniquely determine scale. Does smaller posterior variance prove greater physical accuracy? No; the noise, independence, and calibration assumptions may be wrong. Can a detection captured 100 ms ago be treated as current because its confidence is high? No; confidence does not correct state age.

**Handoff:** produce a detection contract containing pixel/image geometry, depth convention, frame, capture timestamp, invalid-value behavior, and the meaning of any uncertainty field. Use it before adding a learned perception model.
