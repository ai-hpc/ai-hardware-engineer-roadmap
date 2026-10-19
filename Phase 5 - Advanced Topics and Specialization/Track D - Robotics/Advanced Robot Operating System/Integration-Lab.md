# Diagnose a ROS graph before tuning a controller

A camera publisher can be visible in the graph while delivering no images to a subscriber. A subscriber can receive every image while its output gets progressively older. These are different failures. This lab makes both observable before you add navigation or a learned policy.

Allow **60–90 minutes**. The queue exercise needs only Python 3.10+. The optional middleware exercise needs ROS 2 Jazzy, `rclpy`, `std_msgs`, and `rmw_fastrtps_cpp`; it was executed with Python 3.12.3 on Ubuntu 24.04. Run commands from this lesson directory. It uses a random namespace and localhost discovery and has no robot command endpoint.

**Version choice:** Lyrical Luth is the current ROS 2 long-term distribution, released May 22, 2026 with support scheduled through May 2031. Jazzy remains supported through May 2029 and is the explicitly executed baseline here. Choose a distribution compatible with your OS and robot dependencies; do not combine packages from different distributions just to obtain newer names. [Official release table](https://github.com/ros2/ros2_documentation/blob/rolling/source/Releases.rst).

## 1. Predict which reliability pairs communicate

A publisher **offers** reliability; a subscriber **requests** it. Predict the four rows before running:

| Offered by publisher | Requested by subscriber | Result |
| --- | --- | --- |
| Best effort | Best effort | Compatible |
| Best effort | Reliable | Incompatible |
| Reliable | Best effort | Compatible |
| Reliable | Reliable | Compatible |

Reliability is only one policy. Durability, deadline and liveliness can also prevent matching. `qos_check_compatible` checks the complete profiles it is given; system-default/unknown policies can produce a warning rather than a definite answer. This exercise explicitly sets automatic liveliness and volatile durability for the reliability cases. [Jazzy QoS documentation source](https://github.com/ros2/ros2_documentation/blob/jazzy/source/Concepts/Intermediate/About-Quality-of-Service-Settings.rst).

```bash
source /opt/ros/jazzy/setup.bash
python3 labs/qos_ros2.py --domain-id 79 --output my-qos-run.json
```

The program creates two nodes, waits for discovery and the incompatibility event, then sends ten numbered strings on each topic. The [recorded run](labs/qos-reference.json) delivered ten messages for each compatible pair, zero for best-effort→reliable, and observed one incompatible-QoS event for that pair. The expected warning names `RELIABILITY`; it is the deliberately introduced fault.

Receiving ten of ten messages on localhost does not prove loss-free behavior under congestion. Zero deliveries alone could mean many things; here the profile check and incompatibility event explain the zero. In your own graph, inspect both ends with `ros2 topic info /your_topic --verbose` before changing policies. A graph diagram does not show whether endpoints can exchange samples.

**Variation:** request transient-local durability from a volatile writer. Predict and inspect the incompatibility rather than silently changing both endpoints. Retain your modified source and the actual output.

## 2. Explain late subscribers

The same executable creates a reliable transient-local publisher with history depth one and publishes `before-subscription`. It then creates two late subscribers:

- The volatile subscriber receives no historical sample.
- The transient-local subscriber receives the retained sample.
- Both receive a new `after-subscription` sample.

These six total cases ran through real ROS middleware. The writer stays alive throughout the durability experiment. Transient-local history is not a persistent database and this test does not claim that history survives writer destruction. Static transforms and configuration topics often need late-joiner behavior; time-varying sensor frames usually need a different contract. Match the actual publisher rather than selecting durability by habit.

**Check:** will a reliable subscriber automatically recover messages published before it existed? No: reliability and durability address different questions. The durability policy, retained history and writer lifetime matter.

## 3. Predict when a correct connection becomes unusably old

Now separate delivery from processing. A sensor produces one sample every **20 ms**. One non-preemptive inference worker needs **50 ms** per sample. Its arrival rate is 50 samples/s while service capacity is 20 samples/s. An unbounded FIFO cannot keep up indefinitely.

```bash
python3 labs/backlog.py
python3 -m unittest discover -s labs -p 'test_*.py' -v
```

[backlog.py](labs/backlog.py) is an integer-time event model, **not a ROS benchmark**. It compares an unbounded FIFO with one active job plus one replaceable pending job. A new sample can replace a waiting sample; it cannot cancel inference already running. At equal event times, arrivals are enqueued before the worker selects its next job.

For just three samples captured at 0, 10 and 20 ms, each requiring 25 ms:

| Policy | Completed sample IDs | Completion times | Capture-to-complete ages |
| --- | --- | --- | --- |
| FIFO | 0, 1, 2 | 25, 50, 75 ms | 25, 40, 55 ms |
| Replace latest pending | 0, 2 | 25, 50 ms | 25, 30 ms |

For the default twenty-sample fixture, FIFO completes all twenty but the last output is **620 ms old**. Replacing pending samples completes nine, drops eleven, and has a maximum output age of **70 ms**. The dropped IDs are recorded explicitly in [the reference trace](labs/backlog-reference.json). This tradeoff can suit a latest-state display or tracker; it can destroy an application that requires every event, such as transaction accounting.

DDS `KEEP_LAST(depth=1)` controls a middleware history, not every queue in your application. A callback may copy messages into another unbounded queue. A GPU stream may have queued old inference work. A planner may retain an old result. Measure capture-to-use age across the whole path.

**Variation:** make inference take 100 ms while the application allows 80 ms. Even the freshest captured sample is too old when inference finishes. Dropping pending jobs cannot make service time shorter. Consider a cheaper model, a different rate, bounded prediction, or a revised application requirement, then test the resulting behavior.

## 4. Connect the result to executor design

Keep subscriptions short when practical: validate metadata, place or replace a bounded pending item, and return. A worker performs expensive computation and publishes a result carrying the **original capture timestamp**. The consumer still checks age and reset/session identity before use.

Adding a `MultiThreadedExecutor` does not automatically make a node's callbacks concurrent: callbacks in its default mutually-exclusive group remain serialized. Separate callback groups or a suitable reentrant group change scheduling opportunities; they also require explicit synchronization of shared state. Python CPU execution, blocking calls, driver behavior and GPU queues remain relevant. [ROS callback-group guide](https://github.com/ros2/ros2_documentation/blob/jazzy/source/How-To-Guides/Using-callback-groups.rst).

A QoS deadline event reports a missed expected interval; it does not itself stop a controller or expire a target already copied into application state. The [atomic policy-message lab](../Isaac%20Sim%20and%20Isaac%20Lab/Simulation-CPU-Lab.md#3-atomic-observations-and-simulator-owned-actions) implements explicit identity, age, at-most-once application and hold behavior.

## 5. Finish with an explained failure

Submit the QoS results if ROS is available, the queue trace, and a short diagnosis of one failure. The standard-library path has **seven tests** covering known timelines, event ties, drop accounting, faster/slower workers and invalid input. Mark the ROS part “not run” if its packages are unavailable; the model is still a useful separate milestone.

Answer these without running code first:

1. A best-effort image publisher and a reliable subscriber see one another but no callbacks run. Which policy changes could establish compatibility, and what application tradeoff must you consider?
2. All twenty images arrive, but the last result is 620 ms old. Why will switching from best effort to reliable not fix the worker capacity?
3. A late volatile subscriber sees the next configuration update but not the retained one. Why is this consistent with the contract?

**Answer guidance:** (1) A publisher capable of reliable delivery can offer it, or the subscriber can request best effort if loss is acceptable; first verify the full interface and other policies. (2) Reliability cannot raise the 20 jobs/s service capacity to the 50 samples/s arrival rate. (3) The subscriber did not request historical delivery; receiving future samples is a separate behavior.

Continue with [time-correct perception and tracking](../Advanced%20Perception%20and%20AI%20for%20Robotics/Tracking-Lab.md) or return to the [ROS survey](Lecture-01.md).
