# A real ROS 2 loop on a CPU

Use two ROS 2 nodes to transport stamped points, obtain a transform from TF, and reject bad observations before a planner sees them. This exercise follows [lesson 6](Lecture-06.md); it needs ROS 2 Jazzy but no simulator, GPU, or robot.

**Tested here:** Ubuntu 24.04, Python 3.12.3, ROS 2 Jazzy, `rmw_fastrtps_cpp`, 2026-10-18. The exact package versions and observed output are saved in [the measured run](evidence/ros2-jazzy-local-run.json). The portable foundation tests remain independent of ROS installation.

## Run the loop

From the Robotics Foundations directory, in a terminal with ROS installed:

```bash
source /opt/ros/jazzy/setup.bash
python3 labs/ros2_integration_lab.py --domain-id 71 --output ros2-results.json
```

The script requires `rclpy`, `geometry_msgs`, `tf2_ros`, `tf2_geometry_msgs`, and the Fast DDS RMW. If these imports are missing, use the [official Jazzy installation documentation](https://docs.ros.org/en/jazzy/Installation.html) for your OS. A missing dependency is a setup failure, not evidence against the numerical model. Do not install Isaac Sim to run this lab.

The script limits discovery to localhost, uses the selected domain only inside its own process, creates a random namespace and frame names, and remaps both `/tf` and `/tf_static` into that namespace. It clears inherited remote-discovery/vendor-profile settings for the child process. It publishes points and transforms, with no robot command topic or actuator interface. Omit `--domain-id` to choose a random test domain from 20–79; the random namespace prevents topic collisions even if another local process happens to use the same domain.

A discovery or delivery phase times out after five seconds. Successful runs normally finish in a few seconds; there is no continuous background node to stop. All nodes and the executor are shut down on completion or a test failure.

## Trace what happens

The producer publishes a static sensor frame at `(1,2.2,0)` m, yaw `pi/2`, relative to the world frame. The consumer's TF listener receives it. The program waits for discovery and transform availability before sending each case; it does not mistake startup latency for a stale-sample experiment.

All input points are `(1,0,0)` m in their named frame. The consumer calculates age from the received capture timestamp and its local ROS clock. In this lab both clocks use the same host system time. It first checks age and frame, then applies the TF transform at the observation timestamp.

| Case | Injected condition | Expected outcome | Observed saved run |
|---|---|---|---|
| Fresh | Current timestamp, expected frame | Accept and transform to `(1,3.2,0)` m | Accepted; age about 0.615 ms |
| Stale | Timestamp moved 2 s backward | Reject: age exceeds 0.5 s | Rejected; age about 2.00036 s |
| Future | Timestamp moved 2 s forward | Reject: negative age | Rejected; age about −1.99973 s |
| Wrong frame | Unknown input frame name | Reject the frame contract | Rejected |

These four samples are **a correctness exercise**, not a latency benchmark. Small changes in callback age across machines or runs are expected. The synthetic age offsets intentionally exercise rejection logic; they are not measurements of network delay. Two nodes run in one process, so this does not characterize separate-process, wireless, distributed-clock, or real-time-controller behavior.

## Change one assumption

**Guided variation:** change the sensor translation or yaw and calculate the new point by hand. Update the expected point in the check only after writing that calculation. Keep the fresh/stale/future/frame outcomes intact.

**Independent transfer:** replace the static transform with a dynamic broadcaster in an isolated copy of the lab. Publish two different poses with known timestamps, then ask for a point at a capture time between them. Verify interpolation, and separately request a time outside the available history. Preserve the frame/age checks. This extension is an assignment; it has not been executed in the saved run.

**Self-check:** Why is `Time()` used while waiting for readiness, but the observation's timestamp used during conversion? The first asks whether any latest transform is available; the second needs the pose corresponding to the observation. Why can the static transform work for all four timestamps in principle? A static frame relation is time-independent, but freshness and frame validity remain application contracts. Why does a rejected point not prove that an actuator stopped? This program has no actuator; the controller watchdog and fallback require a separate integration test.

**Artifact:** save `ros2-results.json`, the source revision, the changed known-answer calculation, and a short account of what this test proves. Carry that into [Isaac ROS integration](../Isaac%20Sim%20and%20Isaac%20Lab/Lecture-08.md) when you add moving frames, simulation time, and a controller.
