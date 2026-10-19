# 21 — Integrate an estimator into a robot observation pipeline

**Before:** lessons 12–20 and the [camera/time lab](../Camera-Time-Fusion-Lab/Guide.md). **After:** define what an estimate means when delivered to a controller.

A pose value alone is incomplete. Its contract needs state timestamp, publication timestamp, clock domain, frame, units, covariance convention, validity/health state, and the transform/calibration revision. Distinguish a camera's exposure/capture time from message arrival. A clock offset and transport delay are different quantities: a corrected clock does not remove queueing.

**Worked deadline:** an image captured at 2.0 s arrives at 2.6 s. A robot moving at 0.5 m/s moved roughly 0.3 m during transport. A measurement at 2.0 s updates the historical state; a current output at 2.6 s needs propagation and appropriate process uncertainty. Publishing the old position with a new timestamp falsely claims freshness. If your retained history starts at 2.2 s, the measurement is outside that buffer: reject it explicitly or use a documented alternative estimator; do not clamp its timestamp to 2.2 s.

The CPU lab preserves these semantics with causal replay from its initial prior. It is deliberately simple and increasingly expensive as the log grows. A deployed fixed-lag estimator checkpoints old state and covariance, retains needed transitions/cross-correlations, handles clock resets, and records out-of-window drops. These requirements are independent of whether transport is ROS 2, shared memory, or a log file.

For GPS/IMU/camera fusion, check more than sensor rate. GPS position must be transformed into the declared local frame; camera optical axes differ from common robot body axes; accelerometer specific force needs gravity and orientation treatment; repeated correlated estimates must not be fused as independent measurements. Sensor extrinsics and their uncertainty may affect the residual model.

**Independent assignment:** run baseline, zero delay, high wheel bias, and 10% board scale error using fixed seeds. Save `report.json`, both CSV files, the dependency lock, and a paragraph predicting each failure before comparing results. Propose one change that addresses a cause and one tempting change that only hides it.

**Answer outline:** zero delay removes this timing discrepancy; wheel bias causes drift/overconfidence; wrong board scale can retain a tiny reprojection residual while corrupting position; adding sensors with the same wrong calibration does not restore truth.

**Pass:** explain one rejected observation using its original metadata; demonstrate no future-arrival leakage; state which synthetic results transfer to a real sensor pipeline and which require new calibration, measurement, and hardware testing.

**Navigation:** [Learning path](00-index.md) · [Executable CPU lab](../Camera-Time-Fusion-Lab/Guide.md).
