# CARLA: match a sensor frame to a simulation tick

**Milestone:** simulator integration, optional after the [executed CPU lab](Lab-Guide.md). **Status:** script syntax and official APIs reviewed; **CARLA server execution was not available here**. No simulator run, collision result or driving-quality result is claimed.

## 1. Pin the simulator pair

The latest tagged CARLA release found on 18 October 2026 was [0.9.16](https://github.com/carla-simulator/carla/releases/tag/0.9.16), published September 2025. CARLA's separate [Unreal Engine 5 documentation](https://carla-ue5.readthedocs.io/) describes another development/install path; do not mix a UE5 server with this release's client assets. A moving documentation homepage is not a version pin.

Use the official 0.9.16 package/build instructions and its supported Python client for your platform. Installing the small Python `carla` package alone does not install the simulator server, maps, renderer or GPU driver. Record server/client versions, map, GPU/driver, physics settings and seed. The supplied script refuses mismatched versions.

## 2. Understand three different clocks

The wall-clock time spent waiting for an image is a performance observation. The image timestamp is simulation time. The frame number identifies a simulator update. Synchronous mode controls who advances the world; a fixed timestep controls how much simulation time advances per tick. Neither guarantees the wall-clock execution will meet a robot deadline.

Use one client as tick owner. Configure `synchronous_mode=True` and `fixed_delta_seconds=0.05`. Physics substepping must satisfy:

```text
fixed_delta_seconds <= max_substep_delta_time × max_substeps
0.05 <= 0.01 × 10
```

If a Traffic Manager participates, configure it consistently for synchronous operation and record its random seed. The supplied exercise does not create a Traffic Manager. CARLA documents further conditions for deterministic repetitions, including world reload and consistent actor initialization; setting a seed alone is insufficient. [Official synchrony and timestep guide](https://carla.readthedocs.io/en/0.9.16/adv_synchrony_timestep/).

## 3. Run the bounded sensor exercise on an unused local server

Start a CARLA 0.9.16 server dedicated to this exercise, with rendering available. From the repository root and an environment containing its client:

```bash
python3 "Phase 5 - Advanced Topics and Specialization/Track E - Autonomous Vehicles/1. Fundamentals/labs/carla_sync_lab.py" --ticks 40 --output /tmp/course-carla-frames.json
```

[The complete script](labs/carla_sync_lab.py) connects only to `127.0.0.1:2000`, rejects an already synchronous world, creates its own stationary vehicle and 320×240 RGB sensor, and checks 40 matched frames. It bounds the image queue and timeout, stops/destroys its actors, and restores the original world settings in `finally`. Use an unused server because changing world settings affects every client; the script cannot detect all asynchronous clients.

For each returned `world.tick()` frame, the script discards older images and requires the matching image frame. It compares image and world simulation timestamps, then checks `320×240×4 = 307,200` BGRA bytes. GPU cameras can deliver data later in wall time than the corresponding tick; reading “whatever image arrived last” can silently join different frames. A timeout is a failed integration test to diagnose, not permission to accept a stale image. [CARLA sensor and control API](https://carla.readthedocs.io/en/0.9.16/python_api/).

Expected successful output is `PASS: 40 matched frames`, with a JSON record. This is an expected outcome to reproduce, not a recorded local result.

## 4. Only then add a controller

CARLA uses Unreal coordinates: x forward, y right, z up, with rotations exposed in degrees. The CPU lab uses x forward, y left, and yaw in radians. Convert a planar state using `y_left=-y_right` and `yaw_left=-radians(yaw_degrees)`, and transform the target consistently.

`carla.VehicleControl.steer` is normalized to `[-1,1]`; it is not a road-wheel angle in radians. Throttle and brake are normalized commands, not acceleration in m/s². A scalar division by one “maximum steering angle” is at most a simplified initial mapping: wheel physics and the speed-dependent steering curve affect the simulated response. Identify that mapping in a separate low-speed simulator experiment before assessing a path controller.

**Guided fault:** change `sensor_tick` to 0.10 while still expecting a camera image every 0.05-s world tick. The matching-frame contract should fail on unsampled ticks. Repair the expected sensor schedule explicitly; increasing the timeout does not create a missing image.

**Independent transfer:** add a second sensor with a different sampling period. Log and join by frame/timestamp according to an explicit tolerance, preserve missingness, and inject a delayed callback. Report the number of matched, missing and rejected observations. Keep simulator timing, host latency and controller task outcomes separate.

## Handoff to openpilot

The reviewed current openpilot simulator bridge targets **MetaDrive**, as documented in its [pinned simulator README](https://github.com/commaai/openpilot/blob/b1e88c2357f07dfc4e0872370b64d97ad606391b/openpilot/tools/sim/README.md). This CARLA exercise is an independent simulator lesson; it does not claim CARLA is that checkout's built-in bridge. Continue with [source and replay contracts](../2.%20openpilot%20Reference%20Stack/Replay-Lab.md).
