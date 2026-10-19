# Simulation before the GPU: physics, scene composition, and timed actions

This lab gives you a practical entry into the Isaac course on a CPU. First test the mathematics and interfaces with the standard library. Then run actual standalone Newton physics and OpenUSD composition in a small optional environment. ROS 2 adds a real transport exercise. **Isaac Sim rendering, Isaac Lab training and physical robot actuation are separate milestones.**

Allow 2–3 hours for the full path, including reading and variations. The first path needs Python 3.10+; the optional library environment was executed with Python 3.12.3 on 18 October 2026. Run commands from this lesson directory.

## 1. Establish the portable contracts

```bash
python3 labs/robotics_contracts.py
python3 labs/simulation_semantics.py
python3 -m unittest discover -s labs -p 'test_*.py' -v
```

The suite has **27 tests**: eight frame/reset/rate/statistics checks, eleven atomic-message checks, and eight identification/paired-evaluation checks. Tests deliberately corrupt inputs; an all-valid happy path cannot show that rejection works.

For a timing example, 16 environments × 100 policy steps over two wall seconds gives 800 transitions/s. With decimation two and physics dt 1/120 s, each individual world advances 1.667 simulated seconds: its real-time factor is 0.8333, not 800. Keep the number of environments out of the individual world's clock.

## 2. Run actual Newton and OpenUSD on CPU

Install outside any Isaac Sim environment:

```bash
python3.12 -m venv .venv-simulation-cpu
.venv-simulation-cpu/bin/python -m pip install -r labs/requirements-native-cpu.txt
.venv-simulation-cpu/bin/python -m pip check
.venv-simulation-cpu/bin/python labs/newton_cpu.py --output my-newton-run.json
.venv-simulation-cpu/bin/python labs/usd_cpu.py --output my-usd-run.json
```

The pinned packages are Newton 1.5.2, Warp 1.16.0, NumPy 2.5.4 and usd-core 26.8. This reproduces the Newton/Warp pair studied by the Lab EA release, while remaining a separate standalone environment. It does not install Kit, Isaac Lab, a viewer or MuJoCo-Warp. Warp compiles CPU kernels on first use, so the first run can take tens of seconds. A missing NVIDIA driver message is expected on this CPU path; the script explicitly selects `cpu`. Dependencies passed `pip check` in the recorded environment.

### Predict the falling box

Start a 10 cm box at z=2 m with zero velocity and gravity −9.81 m/s², without a floor. Continuous motion after t seconds is:

\[
z(t)=2-\tfrac12 9.81t^2.
\]

For the ballistic semi-implicit Euler update used here, velocity is updated before position. After n steps of width h:

\[
v_n=-9.81nh,\qquad z_n=2-9.81h^2\frac{n(n+1)}2.
\]

At t=0.1 s, the continuous answer is 1.95095 m. Ten 0.01 s steps predict **1.946045 m**; twenty 0.005 s steps predict **1.9484975 m**. The discrete error halves when h halves. A solver matching its discrete rule is not the same as exact continuous motion.

[newton_cpu.py](labs/newton_cpu.py) runs those two cases in actual Newton XPBD, compares with the discrete and continuous predictions, then adds a floor. It runs one world and two replicas occupying the same coordinates. Separate world IDs should prevent cross-world contact. After two simulated seconds both replicated boxes rest near z=0.05 m, their half-height.

[Recorded Newton output](labs/newton-reference.json): z=1.946045041 and 1.948497415 m in the free-fall cases; both resting centers are 0.049965665 m; the replicated and single-world poses agree in this fixture. These results validate this simple CPU scene and solver configuration. They do not establish MuJoCo-Warp, PhysX, contact-rich robot, differentiable-physics or GPU behavior.

**Variation:** change the height or box mass and predict which part of free fall changes. Add drag or restitution only as a new model with a new prediction. Keep contact tests separate from free fall; matching one does not validate the other.

### Predict which USD opinion wins

[usd_cpu.py](labs/usd_cpu.py) creates all assets locally in a temporary directory. No hosted robot download is needed.

1. The referenced payload authors cube size 2.
2. The consuming scene authors size 3, which wins over the asset.
3. A session-layer opinion authors size 4, which wins in the current stage.
4. Saving and reopening the root layer returns size 3; flattening the composed stage retains size 4.
5. Loading the stage without payloads keeps the host prim. A locally authored descendant can remain too, even though the payload's `Cube` type is absent.
6. Changing `metersPerUnit` from 0.01 to 1 does not rewrite an authored translation of `(100,0,0)`.

Those six predictions are asserted with actual OpenUSD. [Recorded output](labs/usd-reference.json) keeps the package and script hashes. This makes two common asset mistakes visible: “unloaded” does not mean the whole prim subtree vanishes if local opinions exist, and changing unit metadata does not convert geometry or inertial values.

**Variation:** remove the local size opinion and predict the reopen result. Then change the asset size. Explain why the session override can hide both changes. Use [Lecture 02](Lecture-02.md) for the full composition context.

## 3. Atomic observations and simulator-owned actions

Suppose the policy reads joints from tick 12, a cube pose from tick 11, and a goal from the previous episode. All three messages can be individually well formed. The assembled observation is still wrong.

The [capstone protocol](labs/capstone_protocol.py) uses a complete observation with one episode, sequence, frame and timestamp. A digest binds the reply to that exact snapshot. The simulator checks shape, finite values, normalized limits, joint order, expiry and identity, then integrates a delta **once**:

\[
q^*_{k+1}=\operatorname{clip}(q^*_k+0.05a_k,q_{\min},q_{\max}).
\]

With target 0 and action 0.5, the next target is 0.025 rad. Receiving the same command twice must leave it at 0.025, rather than move it to 0.05. A policy reply that expires must not change the target or `last_action`. A new episode clears the old request and action history.

The controller timestamps requests using its monotonic clock. The policy echoes that value; only the controller computes age. This avoids subtracting unrelated monotonic clocks across machines. It does not remove transport latency. The digest detects association errors; it is not an authentication mechanism.

### Transport the contract through real ROS 2

If Jazzy and its `rclpy`, `std_msgs`, and Fast DDS packages are installed:

```bash
source /opt/ros/jazzy/setup.bash
python3 labs/ros2_capstone_replay.py --domain-id 77 --output my-ros-replay.json
```

The program restricts discovery to localhost, creates a random namespace, waits for discovery, and shuts down its two nodes when finished. It never connects to a robot or uses a hardware command interface. The policy is a scripted 0.5 action; state values are fixtures. This is a real middleware exercise with synthetic observations.

| Case | Expected and recorded result |
|---|---|
| Fresh matching observation | Accept; target becomes 0.025 rad |
| Duplicate response | Reject; do not integrate twice |
| Request timestamp shifted two seconds into the past | Reject as expired |
| Response from the preceding episode | Reject after reset |
| Policy combines a changed cube pose with the request | Reject digest mismatch |
| Boolean supplied as a numeric action | Reject type mismatch |

[Saved ROS output](labs/ros2-capstone-reference.json) records all six outcomes. The fresh round trip was about 1.36 ms in this run. Six cases in one process are **not** a latency benchmark or a 60 Hz deadline guarantee; the 500 ms expiry isolates correctness from discovery/startup noise. The [full capstone](Lecture-13.md#lab-13e-export-and-bridge-over-ros-2) supplies optional policy/Isaac adapters with a one-policy-period free-running deadline and an explicit hold behavior. Their simulator execution remains pending.

### Execute the policy adapter and lesson tensors on CPU

This optional extension needs a separate Python 3.12 environment with CPU PyTorch and NumPy. If you do not already have one, install it outside the Isaac and Newton environments; the CPU wheel download is substantially larger than the standard-library path:

```bash
python3.12 -m venv .venv-policy-cpu
.venv-policy-cpu/bin/python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
.venv-policy-cpu/bin/python -m pip install numpy==2.5.4
.venv-policy-cpu/bin/python -m pip check
.venv-policy-cpu/bin/python labs/reach_tensor_cpu.py --output my-reach-tensors.json
.venv-policy-cpu/bin/python labs/push_tensor_cpu.py --output my-push-tensors.json
source /opt/ros/jazzy/setup.bash
.venv-policy-cpu/bin/python labs/policy_adapter_smoke.py --domain-id 78 --output my-policy-adapter.json
```

The first two scripts extract the **actual Python class in Lectures 10 and 13**, then execute its methods using real PyTorch and explicitly fake scene objects. The [reach result](labs/reach-reference.json) checks six contracts, including soft joint limits, timeout at exactly 240 transitions, the 27-value observation and a known reward. The [push result](labs/push-reference.json) checks eight contracts, including per-world action delay, the 31-value observation, applied-action rate penalty, termination, partial resets, clipping and requesting a kinematics refresh. Neither script executes Isaac's constructor, asset loading, data views or physical forward kinematics.

For the push example, issue action 1 with a delay of two. The first two transitions apply zero; only the third applies 1 and advances the target by 0.05 rad. Both training and deployment observations must report the applied action. Reporting the new request during training but the accepted response in deployment changes the policy's input meaning. Resetting one environment must clear only its delay/history state.

[policy_adapter_smoke.py](labs/policy_adapter_smoke.py) creates a temporary known-weight 31→7 linear policy, starts the actual policy adapter in a separate process, and exchanges three observations through localhost ROS. Its first action component is `2 * observation[0] + 0.1`; inputs 0.2, 0.4 and 0.1 must produce 0.5, 0.9 and 0.3. The last case starts a new episode and checks target reset. The helper removes temporary artifacts and verifies that the child exits cleanly.

The [saved run](labs/policy-adapter-reference.json) passed all three cases. Its cold input-construction/forward call took about **30.31 ms**, compared with **0.580 ms** and **0.111 ms** for the next two calls. Round trips were about **179.78, 1.65 and 0.89 ms**. These three observations illustrate cold-start effects; they are not latency percentiles or a 60 Hz guarantee. The five-second diagnostic expiry is intentionally distinct from a control deadline. The actual capstone still needs a trained artifact, verified export parity, a full simulator run and a representative timing distribution.

PyTorch 2.14 emits a TorchScript deprecation warning. This exercise reads the export format used by the pinned Lab example; it does not recommend TorchScript for new general-purpose deployments. Migrating to `torch.export` requires a corresponding loader and parity tests. `torch.compile` alone is not a replacement artifact format.

## 4. Identify delay separately from a slow actuator

[simulation_semantics.py](labs/simulation_semantics.py) generates a scalar response:

\[
v_{k+1}=av_k+(1-a)u_{k-d},\quad a=e^{-\Delta t/\tau}.
\]

Here dt=0.02 s, the unknown delay d=2 steps (0.04 s), and the time constant τ=0.08 s. For each candidate integer delay, fit a by least squares using

\[
v_{k+1}-u_{k-d}=a(v_k-u_{k-d}).
\]

Only candidates with 0<a<1 represent the assumed stable first-order response. An unchanging zero input contains no information about τ; the implementation rejects this unidentifiable case.

The noisy training fixture recovers d=2 and τ≈0.0799867 s. Forcing d=0 instead gives τ≈0.988 s and much worse residual error: a long apparent time constant can conceal an omitted delay. Validate the selected model on a new sinusoidal input without refitting. [Recorded output](labs/semantics-reference.json) includes every candidate and held-out error. This simplified least-squares estimator can be biased by measurement noise in its regressors; it is an educational model, not a universal actuator-identification method.

**Variation:** fit on step inputs, validate on a faster signal, then introduce saturation in the validation response. Explain the residual pattern rather than claiming the original first-order model is still correct. Continue with [Lecture 20](Lecture-20.md).

## 5. Compare outcomes without misusing confidence intervals

Two policies can both succeed on half the tasks while failing on different halves. Equal totals and overlapping intervals do not establish parity. Pair outcomes by saved scenario ID, then count **regressions**: baseline succeeds and candidate fails.

The supplied calculation uses a one-sided 95% Wilson upper bound on the regression probability and a predeclared 1% margin. With zero regressions in 100 independent pairs, the upper bound is about **2.63%**, so the gate fails. With zero in 1,000 pairs, it is about **0.270%**, so this particular gate passes. The independence assumption and held-out scenario distribution matter; correlated repeats and checkpoint selection on test data invalidate a naive interpretation.

This gate addresses a stated task-outcome loss. It does not prove matching dynamics, action values, timing, safety, or performance on other distributions. Establish numerical policy parity and physical contracts separately, as described in [Lecture 17](Lecture-17.md) and [the capstone](Lecture-13.md).

Submit your raw output, versions, source hashes, a prediction made before each run, and one deliberate failure that your checks caught. Mark optional sections you did not run. Use the [validation record](labs/validation-2026-10-18.json) to see exactly which repository results were executed.

Primary references: [Newton 1.5.2 source](https://github.com/newton-physics/newton/tree/v1.5.2), [Warp documentation](https://nvidia.github.io/warp/), [OpenUSD stage API](https://openusd.org/release/api/class_usd_stage.html), [Isaac Lab EA environment source](https://github.com/isaac-sim/IsaacLab/blob/ae37b028ea415c91ea2bc32609efcd759ed2b974/source/isaaclab/isaaclab/envs/direct_rl_env.py), and [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/).
