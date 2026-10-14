# Sensor Fusion

<div class="course-identity auto-course" style="--course-accent: #475569; --course-accent-rgb: 71, 85, 105;" markdown="1">
<div class="course-identity__icon">SF</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · AI Workloads</p>
<p class="course-identity__title">Specialized course identity for Sensor Fusion.</p>
<p class="course-identity__meta">Artifact: model or workload study · Measure: accuracy, latency, memory, throughput</p>
</div>
</div>


**Phase 3 — Artificial Intelligence.** Multi-sensor calibration, filtering, and fusion (Kalibr, BEVFusion, multi-object tracking). Complements **[Computer Vision](../3.%20Computer%20Vision/Guide.md)** and **[Neural Networks](../../1.%20Neural%20Networks/Guide.md)**. Sub-guides that mention Jetson, TensorRT, or ROS2 assume you are also following **Phase 4 Track B** ([Jetson](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Guide.md), [ROS2](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/5.%20Application%20Development/6.%20ROS2/Guide.md)).

**Hub:** [Phase 3 — Artificial Intelligence](../../Guide.md)

---

## Course sequence: one observation contract

Begin with two sensors and one coordinate transform. Add modalities only after you can explain a residual plot. This sequence replaces calendar-based lists of advanced topics with artifacts the robotics courses can consume.

| Step | Study | Build | Passing evidence |
|---|---|---|---|
| 1 | [Kalman filter series](kalman-filter/README.md) | A seeded constant-velocity estimator | State error, innovation, covariance, and behavior during a dropout |
| 2 | [Kalibr](Kalibr/Guide.md) | Camera/IMU calibration manifest | Transform direction, units, clock convention, and held-out residuals |
| 3 | [Multi-object tracking](multi-object-tracking/Guide.md) | Gated assignment with track birth/death | Identity switches and tracking metrics on an unchanged detector |
| 4 | [BEVFusion](BEVFusion/Guide.md) | Camera/LiDAR feature alignment | Same-split modality ablations and measured runtime memory |
| 5 | ROS 2 integration | Timestamped observation publisher | Replay, stale-message rejection, and frame-consistent outputs |

**Entry knowledge:** vectors, covariance, coordinate transforms, and basic Python. Learn linear filtering before EKF/UKF; a UKF is not universally more accurate than an EKF. Each approximates nonlinear uncertainty differently, so compare on the same known trajectory.

## Worked error budget

For a pinhole camera near its optical axis, a one-pixel error at depth `Z` corresponds approximately to `Z/f` metres, where `f` is focal length in pixels. At `Z=20 m` and `f=500 px`, that is `0.04 m`. This is geometry under stated assumptions, not a universal odometry drift rate.

A one-degree orientation error displaces a ray at 30 m by roughly `30*tan(1°)=0.524 m`. A 1 ms offset at constant 1 m/s is a 1 mm displacement; angular motion, acceleration, rolling shutter, and filter feedback can change the impact. Test them separately.

**Exercise:** replay a reference trajectory, inject a known 10 ms timestamp offset, and compare residuals with a one-degree frame error. Predict which residual changes with speed before running the experiment. Carry the timestamp/frame manifest into the VLA observation adapter.

## From a good estimate to a usable robot input

Every observation needs a frame ID, units, source timestamp, clock domain, validity interval, and covariance or other uncertainty description where meaningful. Distinguish time synchronization from spatial calibration. A low calibration residual on the fitted sequence does not establish accuracy on a new trajectory.

Use ROS 2 for the integrated robot path. Legacy ROS 1 calibration utilities may run offline in their own pinned environment; document bag conversion and verify timestamp preservation. Calibration outputs can then be consumed by a ROS 2 deployment without treating the calibration tool as a ROS 2 package.

**Advanced branches after this gate:** factor graphs/VIO, radar, event cameras, or multi-robot fusion. Select one to answer a task-specific failure seen in the baseline. These are extensions, not requirements before the first working VLA loop.


## Carry this course into the physical-AI system

**Start from:** Two timestamped sensors and a known transform convention.

**Build and challenge:** Work through estimation, calibration, association, then fusion. Inject time offset and spatial error separately so their residual patterns can be distinguished.

**Pass forward:** A frame/time/uncertainty contract that the robot observation pipeline can validate. Continue with [the connected course](../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20D%20-%20Robotics/Isaac%20Sim%20and%20Isaac%20Lab/README.md) and keep the artifact in the [Physical AI Design Studio](../../../Physical-AI-Design-Studio.md).
