# 6. Put the pieces into ROS 2

**Purpose:** turn the numerical lessons into a debuggable robot data flow. **Prerequisite:** [lessons 1–5](README.md#follow-the-chain). **Next:** [Advanced ROS 2](../Advanced%20Robot%20Operating%20System/Lecture-01.md).

ROS 2 (**Robot Operating System 2**) provides tools and libraries for independently
developed components to exchange typed messages. You still run it on a host
operating system such as Linux. It does not automatically make the components'
frames, units, timing, or failure behavior agree. That is your integration work.

## Start with one complete loop

For the mobile manipulator, draw this conceptual flow before choosing packages:

```text
camera + calibration -> perception -> object estimate -> task/route planner
                                                       |
encoder + IMU -> state estimate -> local controller <---+
                                      |
                             actuator or simulator
```

Transforms connect coordinate systems; timestamps connect observations to state history. The controller can run faster than perception. A learner's first integration should operate in simulation or a recorded-data harness, with an explicit way to reject invalid commands.

An **encoder** measures rotation or displacement. An **IMU** (inertial
measurement unit) commonly supplies angular-rate and acceleration measurements.
**TF** refers to ROS's coordinate-transform system, implemented by the `tf2`
libraries. These measurements and transforms help an estimator infer the state;
their names do not guarantee correct calibration or timing.

A **node** performs a coherent piece of work. A **topic** carries a stream, such as images or state. A **service** handles a short request/response, such as querying a configuration. An **action** represents a longer operation with feedback, cancellation, and a result, such as navigation to a pose. A successful action result is a task-level assertion; retain the measured state that supports it.

## Write the message contract before the adapter

| Edge | Example ROS representation | Meaning your contract must preserve |
|---|---|---|
| Camera → perception | `sensor_msgs/msg/Image`, `CameraInfo` | Capture time, frame, encoding, calibrated image geometry |
| Perception → target selection | `vision_msgs` detections or a documented pose message | Object identity, coordinates, depth validity, uncertainty meaning |
| Estimator → controller | `nav_msgs/msg/Odometry`, joint states, TF | Pose/velocity frames, timestamp, joint order and units |
| Planner → base controller | Package-specific action/path and velocity interface | Motion constraints, reference frame, period, timeout |
| Planner → arm controller | `FollowJointTrajectory` action or controller-supported trajectory interface | Joint names, positions/velocities, time-from-start, limits |

Check the installed package interface. For example, velocity commands may use `Twist` or `TwistStamped` depending on package/version/configuration; a timestamp-free message cannot by itself prove command freshness. Record the selected ROS distribution, middleware, package versions, and interface types with the artifact.

Here is an **exercise contract**, not a universal robot specification:

```yaml
observation:
  frame: camera_optical_frame
  position_units: metres
  timestamp: capture_time_in_sim_clock
  max_age_seconds: 0.10
action:
  semantics: joint_position_target
  joint_order: [shoulder, elbow]
  units: radians
  policy_period_seconds: 0.05
controller:
  period_seconds: 0.005
  policy_timeout_seconds: 0.15
  timeout_behavior: stop_accepting_new_targets_and_enter_defined_hold_mode
```

Decide what “hold” means for the plant. A wheeled base, gravity-loaded arm, and flying vehicle cannot use one identical physical fallback. Verify the chosen controller's behavior in its simulator before connecting that adapter to hardware.

## QoS is directional

**Quality of service (QoS)** describes requested and offered message-delivery
properties. A publisher sends messages; a subscriber asks to receive them.
Middleware is the software that transports messages between those endpoints.

For the reliability policy, a reliable publisher can serve a best-effort subscriber. A best-effort publisher cannot satisfy a subscriber requesting reliable delivery. Other policies must also be compatible. A reliable delivery setting does not promise fresh data or a bounded end-to-end control deadline. [ROS 2 Jazzy QoS documentation source](https://github.com/ros2/ros2_documentation/blob/jazzy/source/Concepts/Intermediate/About-Quality-of-Service-Settings.rst)

Keep queues bounded and measure observation age. If a 30 Hz camera feeds a serial inference worker that takes 100 ms per frame, processing every frame builds backlog. A latest-observation policy can reduce backlog, but it discards intermediate observations and must be evaluated for the task. Changing reliability does not remove the compute bottleneck.

A callback that runs long inference or waits for a network service can delay another callback in the same executor. More threads help only when callback groups, shared state, and available compute permit useful concurrency. They do not create real-time guarantees. [ROS 2 executor documentation source](https://github.com/ros2/ros2_documentation/blob/jazzy/source/Concepts/Intermediate/About-Executors.rst)

## Optional ROS exercise: inspect a known transform

For a complete runnable producer/consumer with measured evidence, use the [ROS 2 CPU integration lab](ROS2-Lab.md). The CLI exercise below is an alternative way to inspect the same geometry interactively.

**Upstream documented/source-checked, not executed here.** These commands target ROS 2 Jazzy with `tf2_ros` installed and the ROS environment sourced in every terminal. They use static test frames and do not command a robot. Jazzy is a named teaching baseline, not a claim that it is the newest release. Review the [upstream publisher flags](https://github.com/ros2/geometry2/blob/jazzy/tf2_ros/src/static_transform_broadcaster_program.cpp) if using another distribution.

In terminal 1, create the base pose from lesson 1:

```bash
ros2 run tf2_ros static_transform_publisher --x 1 --y 2 --z 0 --yaw 1.5707963267948966 --pitch 0 --roll 0 --frame-id lesson_world --child-frame-id lesson_base
```

In terminal 2, create the aligned planar sensor mount:

```bash
ros2 run tf2_ros static_transform_publisher --x 0.2 --y 0 --z 0 --yaw 0 --pitch 0 --roll 0 --frame-id lesson_base --child-frame-id lesson_sensor
```

In terminal 3, inspect their composition:

```bash
ros2 run tf2_ros tf2_echo lesson_world lesson_sensor
ros2 topic info /tf_static --verbose
```

The echo command runs until interrupted; stop it with Ctrl-C before running the next command. The expected sensor origin in world coordinates is `(1, 2.2, 0)` m, with yaw `pi/2`. The object one metre ahead of that sensor is `(1,3.2,0)` m. Distinguish the **frame origin** echoed by the tool from the **object point** transformed by your code. [Upstream tf2_echo implementation](https://github.com/ros2/geometry2/blob/jazzy/tf2_ros/src/tf2_echo.cpp)

Save the output, publisher/subscriber endpoints, and actual package versions. Stop both static publishers when finished. Static transforms are intentionally persistent in the broadcaster's transient-local publication; stopping a static broadcaster is not a valid simulation of a stale moving-base transform. For dynamic timestamp faults, use a dynamic broadcaster and explicit test timestamps in your later ROS lab.

## Debug one boundary at a time

| Injected fault in the harness/simulator | Detection | Expected integration response |
|---|---|---|
| Camera optical axes treated as body axes | Known basis-vector transform fails | Reject the adapter/configuration |
| Depth expressed in mm but labelled m | Known-distance fixture fails | Correct conversion before issuing targets |
| Observation older than 100 ms | Capture-to-use age check | Reject observation; apply defined fallback |
| Goal outside the arm's reach | IK/reachability check | Report invalid target; request a new plan |
| Doorway smaller than footprint | Collision/path check | Return no path or replan |
| Policy misses its 150 ms timeout | Independent controller watchdog | Enter the documented fallback |

A watchdog running inside the same blocked callback cannot reliably detect that callback's failure in time. Its scheduling and actuator-side timeout are part of the design. Measure the resulting command behavior; a log message alone does not prove that motion changed.

## Finish the integration exercise

**CPU route:** fill in the contract, connect each field to a tested function from the lab, and produce the six-row fault matrix above with your expected responses. Label middleware execution and controller timing **not run**.

**ROS route:** add the transform trace, actual topic types/QoS, and a simulator or recorded-input test of one stale-observation case. Keep the command path disconnected from physical hardware during this lesson. Record capture-to-use age, update intervals, and the response to rejection.

**Self-check:** Why does `ros2 topic hz` not establish end-to-end latency? It observes message arrival intervals at its subscriber, not necessarily capture time or actuator application. Why can a healthy ROS graph still produce a wrong motion? Types can agree while frames, scales, or action semantics disagree. Why is a VLA's valid tensor shape insufficient? The consuming controller needs semantic and timing agreement too.

**Handoff:** bring the completed contract and numerical tests to [Advanced ROS](../Advanced%20Robot%20Operating%20System/Lecture-01.md). Then validate reset, dynamics, sensors, and closed-loop behavior in [Isaac](../Isaac%20Sim%20and%20Isaac%20Lab/README.md) before treating a learned policy's success rate as meaningful.
