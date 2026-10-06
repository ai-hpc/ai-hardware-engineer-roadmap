# Lecture 12: Performance, Sim-to-Real, and Scale-Out

## Overview

By Lecture 11 you can build a scene, train a policy and evaluate it. This lecture is about the three things that decide whether that work is useful outside your desk. **Performance** is knowing where every millisecond of a training iteration goes, so you optimize the right layer. **Scale-out** covers moving the same job into a container, onto a cloud GPU, or across several GPUs without changing its meaning. **Sim-to-real** is closing the gaps between the simulator and the robot that you can name and model: actuators, latency, friction, mass and noise. You randomize those deliberately instead of hoping for the best.

All three need the same discipline as Lecture 01: measure first, change one knob, measure again, and record the conditions so someone else gets the same number.

By the end you should be able to:

* break a training iteration into physics, rendering, environment Python, policy inference and PPO update time, using Nsight Systems, Tracy, and Isaac Lab's benchmark harness
* measure what headless operation, livestreaming and containers cost on your card
* choose a cloud GPU that can actually run Isaac Sim, and keep a cost ledger
* launch Isaac Lab 3.0 multi-GPU training and compute its scaling efficiency
* state exactly what Isaac Sim and Isaac Lab guarantee about determinism, and what they do not
* model actuators and latency, set up domain randomization with the `EventManager`, and run an ablation that shows it works

---

## 1. Why it matters: three budgets

| Budget | Question | Failure mode | Tool |
|---|---|---|---|
| **Time** | Where does one iteration go? | Tuning physics when Python resets dominate, or buying a bigger GPU for a CPU-bound loop | Nsight Systems, Tracy, `isaaclab benchmark` |
| **Memory and money** | What fits on 8 GB, and what does the cloud run cost? | OOM at 3 a.m., or a cloud bill for an idle streaming instance | `nvidia-smi dmon`, `torch.cuda` stats, a cost ledger |
| **Reality gap** | Which simulator errors will the policy exploit? | A policy that works in sim and fails on the robot (or in another simulator) for reasons nobody wrote down | Actuator models, sysid, domain randomization, latency injection |

The order matters. Do not randomize physics before you know your step time, because randomization changes it. Do not scale out before you know your single-GPU bottleneck: if that bottleneck is CPU-side Python, eight GPUs give you eight copies of it.

---

## 2. Mental model: where a training iteration goes

An on-policy iteration (Isaac Lab + RSL-RL PPO, [Deep RL Lecture 06](../Deep%20RL%20for%20Robot%20Learning/Lecture-06.md)) collects \(S\) policy steps from \(N\) environments, then updates the network. Each policy step runs \(d\) physics steps (the *decimation*):

$$
T_{\text{iter}} = S\,\big(d\,t_{\text{phys}} + t_{\text{render}} + t_{\text{env}} + t_{\text{policy}}\big) + t_{\text{update}}
$$

$$
\text{env-steps/s} = \frac{N\,S}{T_{\text{iter}}}
$$

* \(t_{\text{phys}}\) is the GPU solver (PhysX or Newton). It grows with bodies, contacts and solver iterations ([Lecture 04](Lecture-04.md)).
* \(t_{\text{render}}\) is zero for state-only tasks. For camera tasks it is often the largest term ([Lecture 07](Lecture-07.md)).
* \(t_{\text{env}}\) is your observation, reward, termination and reset code. It is Python launching many small GPU kernels, and it is the term most often underestimated.
* \(t_{\text{policy}}\) is the actor forward pass, which is negligible for small MLPs.
* \(t_{\text{update}}\) is PPO epochs over the rollout. It scales with \(N S\) and network size, not with physics.

Amdahl's law applies: if \(t_{\text{env}}\) is half of \(T_{\text{iter}}\), a solver twice as fast saves only a quarter. **Measure the split before choosing what to optimize.**

### 2.1 The profiling toolbox

| Tool | What it sees | What it misses | Use it for |
|---|---|---|---|
| Wall-clock timers (`time.perf_counter`) | Total time of a Python block | GPU work still in flight (call `torch.cuda.synchronize()` before stopping the clock) | Lecture 01-style harness numbers |
| `isaaclab benchmark runtime` / `training` | Env-step FPS, collection vs total FPS, learning curves; physics and render scope timings with `ISAACLAB_PHYSICS_PROFILE=1` / `ISAACLAB_RENDER_PROFILE=1` (read by `runtime` only, and only for tasks that set `benchmark_mode`) | Why a scope is slow | Repeatable A/B numbers in a JSON schema |
| **Nsight Systems** (`nsys`) + NVTX | CPU threads, Python functions annotated through Isaac Lab's `nsys_trace.json`, every CUDA kernel and gap | Inside Kit's C++ zones | "Is the GPU idle while Python runs?" |
| **Tracy** (`omni.kit.profiler.tracy`) | Kit's own zones: *App Update → Timeline Update → Physics Step → Rendering*, plus GPU zones | Isaac Lab's Python structure unless you add zones | Standalone Isaac Sim scripts, rendering and ROS 2 graphs |
| `torch.cuda.max_memory_allocated()` / `memory_reserved()` | The PyTorch caching allocator | PhysX, Warp, RTX and Kit allocations | Policy and rollout buffer memory |
| `nvidia-smi dmon -s pucmt -o T` | Per-second power, SM / memory / **encoder** utilization, clocks, framebuffer memory, PCIe throughput | Which process or kernel | Background log for every run |

Three pitfalls apply to every tool. **Warm up** before measuring: the first run compiles shaders and Warp kernels and fills caches. **Profilers add overhead**, so take timings with profiling off and use the profile only for proportions. **GPU work is asynchronous**: a CPU timer around a kernel launch measures the launch, not the kernel.

---

## 3. Headless, streaming, containers and the cloud

### 3.1 Headless vs livestream

Headless (`SimulationApp({"headless": True})`, or `--viz none` in Isaac Lab 3.0) renders only what sensors need. For pure throughput the 6.1 performance handbook adds three more levers:

* `"disable_viewport_updates": True` stops rendering a viewport nobody watches (headless only).
* `"renderer": "MinimalRendering"` is the cheapest RTX mode for training in the loop.
* Turning off Fabric output (`--/physics/fabricUpdateTransformations=false` and `--/physics/fabricUpdateVelocities=false`, always both) removes a CPU sync per physics fetch. Only do this in GPU tensor workloads with no cameras, no streaming and nothing reading Fabric.

**Livestreaming** keeps the host headless and sends the UI over WebRTC. You can launch it with `isaac-sim.streaming.sh`, `isaacsim isaacsim.exp.full.streaming --no-window`, `./runheadless.sh` in the container, or from a standalone script by enabling `omni.kit.livestream.app`. Isaac Lab exposes it as `--livestream 1|2` (public / private network). The facts that shape your setup:

* Only **one client per instance**, and only one streaming method at a time.
* There is **no authentication or encryption**, so use trusted networks only. On a cloud VM, firewall the ports to your own IP: TCP 49100 (signaling) and UDP 47998 (media).
* It **requires NVENC**, so an A100 cannot livestream.
* Encoding runs on the dedicated NVENC block, not the SMs. The streamed viewport is still rendered by RTX, though, and that rendering is the cost you measure in Lab 12b.

### 3.2 Containers

`nvcr.io/nvidia/isaac-sim:6.1.0` runs rootless as UID 1234 and needs the NVIDIA Container Toolkit, `-e ACCEPT_EULA=Y`, and **`--network=host` for WebRTC** (bridge networking with `-p` breaks the media stream). The official `docker run` mounts persistent host directories for the Kit cache, the shader **ComputeCache**, logs and config. Keep those mounts. Without them every container start recompiles shaders, and you will wrongly blame the container for slow startup. Step-time overhead from the container itself should be negligible. Check it once against a bare-metal run in Lab 12b.

### 3.3 Picking a cloud GPU

| GPU class | Isaac Sim rendering | Livestream | Notes |
|---|---|---|---|
| RTX 4060 8 GB (this course) | Runs, **below** the 16 GB minimum | Yes | State-only training, few small cameras |
| RTX 4080 / 5080 16 GB | Minimum / "good" tier | Yes | Workstation upgrade path |
| **L40S** 48 GB (data center) | Recommended data-center tier | Yes | The usual cloud choice |
| **RTX PRO 6000 Blackwell** (server) | "Best" tier, 48 GB+ | Yes | Big camera counts, VLA alongside the sim |
| A100 / H100 | **Not supported** (no RT cores) | No (A100 has no NVENC) | Do not rent these for Isaac Sim |

NVIDIA lists the cloud paths in the 6.1 *Cloud Deployment* page. **Isaac Launchable** (`isaac-sim/isaac-launchable`) is a Brev template that gives you VS Code plus a browser-streamed Isaac Sim UI. Check which versions its containers pin before using it: when checked, its README listed Isaac Sim 6.0.1 and Isaac Lab 3.0.0-beta2, not this course's 6.1 / 3.0-EA. **Isaac Automator** (`isaac-sim/IsaacAutomator`) deploys a remote "Isaac Workstation" to AWS, GCP, Azure or Alibaba Cloud with `./deploy-aws` and similar commands, plus stop/start and `./destroy`.

### 3.4 The cost ledger

Write the run cost down before you start it:

$$
C_{\text{run}} = h_{\text{GPU}}\,r_{\text{GPU}} + h_{\text{idle}}\,r_{\text{GPU}} + C_{\text{storage}} + C_{\text{egress}}
$$

\(h_{\text{idle}}\) is the line people forget: setup, shader warmup, debugging over a stream, and instances left running overnight. Compare it with the local alternative using the same metric your experiment needs, which is usually **wall-clock to target reward**, not env-steps/s. A cloud GPU that is several times faster but sits idle half the time may still lose. Take \(r_{\text{GPU}}\) from your provider's price page on the day you run, not from memory.

---

## 4. Scale-out across GPUs

**Isaac Sim (one process).** 6.1 added `active_cuda_gpus` to the `SimulationApp` config. It selects which GPUs render (by CUDA index, honoring `CUDA_VISIBLE_DEVICES`) and cannot be combined with `active_gpu`. `physics_gpu` selects the physics device. The handbook gives the rules: **GPU physics uses one GPU only**, extra GPUs speed up rendering of *multiple cameras* (roughly one GPU per camera, no more), and GPU count does not change scene load time. With uneven camera resolutions, `ViewportManager.optimize_render_products()` can rebalance render products across GPUs. On bare-metal Linux, NVIDIA recommends disabling the IOMMU for multi-GPU work.

**Isaac Lab 3.0 (one process per GPU).** Verify the task trains on one GPU, then launch it on all visible GPUs:

```bash
uv run isaaclab train          --rl_library rsl_rl --task Isaac-Cartpole --num_envs 4096
uv run isaaclab train_multigpu --rl_library rsl_rl --task Isaac-Cartpole --num_envs 4096 --num_gpus 2
uv run isaaclab train_multigpu ... --dry_run      # print the torchrun command, launch nothing
```

* `train_multigpu` adds `--distributed` and builds a `torchrun` command.
* `--num_envs` is **per GPU**, and each rank trains with the launch seed plus its rank.
* Multi-node runs add `--nnodes`, `--node_rank`, `--master_addr` and `--master_port`.
* It needs Linux and NCCL, and supports RSL-RL, RL-Games and skrl.
* It is marked **experimental** in 3.0.
* In 2.x code you will see `python -m torch.distributed.run ... train.py --distributed` instead.

Measure scaling with `isaaclab benchmark training_multigpu`, keep `--num_envs` constant per rank, and report:

$$
\eta(G) = \frac{\text{global throughput with } G \text{ GPUs}}{G \times \text{throughput with 1 GPU}}
$$

Also report wall-clock to target reward. More environments per update change the PPO batch, so faster env-steps/s does not guarantee faster learning.

---

## 5. Determinism and reproducibility

From the Isaac Lab 3.0 reproducibility page:

* **Guaranteed:** the same hardware and the same Isaac Sim/PhysX version give identical results for rigid bodies and articulations. Results **can differ across GPU models**, and PhysX makes no determinism promise for cloth or soft bodies.
* **Seeds:** the env seed (`env_cfg.seed`, set from `--seed`) goes through `configure_seed` to Python, NumPy and PyTorch. Under `train_multigpu`, each rank uses seed + rank.
* **`--deterministic`** requests reproducible RTX rendering, strict PyTorch determinism in the RL entrypoints, `PhysicsCfg.deterministic` (PhysX "enhanced determinism", Newton `run_to_run`) and Warp's global deterministic mode. It can cost speed and memory, and on OvPhysX it is best effort.
* **The trap:** runtime randomization of physics materials can reorder GPU work and break determinism. The docs advise doing it **at startup only**, which shapes the event modes in §6.3.

Fix the timestep, record everything that defines a run, and treat "same seed, different GPU" as a different experiment. Commit a `RUN_MANIFEST.json` with each result: git SHAs of your code and Isaac Lab, the Isaac Sim version, driver, GPU model, physics backend, `dt`, decimation, `num_envs`, seed, and the `--deterministic` flag.

---

## 6. Sim-to-real: closing the gaps you can name

### 6.1 Gap taxonomy

| Gap | Symptom on the robot | Model it with | Randomize it with |
|---|---|---|---|
| Actuator dynamics (saturation, gearing, bandwidth) | Sluggish or oscillating joints | Explicit actuators: `DCMotorCfg`, `RemotizedPDActuatorCfg`, `ActuatorNetMLPCfg` / `ActuatorNetLSTMCfg` | `randomize_actuator_gains`, `randomize_joint_parameters` |
| Latency (comms, inference, motor drivers) | Overshoot, limit cycles | `DelayedPDActuatorCfg` (`min_delay`/`max_delay` in physics steps) or an action-delay buffer | Sample the delay per episode |
| Mass, inertia, CoM | Wrong forces when lifting or pushing | `MassAPI` values from CAD or a scale | `randomize_rigid_body_mass`, `_inertia`, `_com` |
| Contact and friction | Slip, unexpected stick | Physics materials ([Lecture 04](Lecture-04.md)) | `randomize_rigid_body_material` |
| Joint friction and armature | Stiction, wrong low-speed behavior | Armature, friction parameters | `randomize_joint_parameters` |
| Sensor noise and bias | Jittery actions, drift | Noise models | `observation_noise_model` (`GaussianNoiseCfg`, `NoiseModelWithAdditiveBiasCfg`) |
| Perturbations | Falls, dropped objects | None; these are disturbances | `push_by_setting_velocity`, `apply_external_force_torque` |
| Visuals (camera policies) | Perception fails | Replicator materials and lighting ([Lecture 07](Lecture-07.md)) | `randomize_visual_texture_material`, `randomize_visual_color` |

### 6.2 Actuators and system identification

In Isaac Lab 3.0, **implicit** actuators (`ImplicitActuatorCfg`) hand the PD gains to the solver. That is cheap and stable, but the solver drive is an idealized motor. **Explicit** actuators compute effort in Isaac Lab, clip it, and submit it during `write_data_to_sim()`. That is how you model saturation, delay and learned motor behavior. Hwangbo et al.'s actuator network is the classic example of the learned case. Commands go through `robot.actuators.target_command.set_position_index(...)`. The older `set_joint_position_target_index` still works but is deprecated in 3.0.

Get parameters from measurements, not guesses. Isaac Sim 6.1's **System Identification** extension (`isaacsim.robot_setup.sysid`, UI in `isaacsim.robot_setup.sysid.ui`, beta) replays recorded commands, compares the simulated response with measured telemetry, and fits parameters. It reads CSV, ROS 2 bags, MCAP and LeRobot data. It can fit drive gains, actuator models, inertia, joint limits and **command delay**, with held-out validation chunks. The output is the nominal value you randomize *around*.

### 6.3 Domain randomization with the `EventManager`

Both workflows take an `events` config of `EventTermCfg` terms. Each term has a **mode**: `prestartup` / `startup` (once, at construction), `reset` (per episode, for the reset env IDs), or `interval` (every so many seconds of sim time). Direct tasks get this through `DirectRLEnvCfg.events`, plus `observation_noise_model` and `action_noise_model`. Rules of thumb:

* Randomize **materials at startup** (determinism and GPU-transfer caveat from §5), and spread them over buckets (`num_buckets`).
* Randomize **mass, gains and delay at startup or reset**, around identified nominal values. Use a range wide enough to cover your uncertainty, not "as wide as possible".
* **Latency is the cheapest high-value randomization** for manipulation and locomotion. Inject it in the action path.
* DR makes the task harder. Expect slower convergence and a more conservative policy, and budget training time for it. Tobin et al. (visual) and Peng et al. (dynamics) are the founding papers. [Deep RL Lecture 13](../Deep%20RL%20for%20Robot%20Learning/Lecture-13.md) covers how to evaluate whether DR helped without fooling yourself.

### 6.4 Digital twins: the opposite strategy

Domain randomization makes the policy robust to a world you have **not** modeled exactly. A **digital twin** goes the other way: model one specific real system (a work cell, a warehouse, a production line) as closely as you can and keep the model in sync with it. In Isaac Sim terms, a twin is a USD stage built from the facility's CAD and scans and composed with references and payloads ([Lecture 02](Lecture-02.md)). It holds robots whose gains and inertias were identified on the real units (§6.2), and it is often connected live to the real system over ROS 2 ([Lecture 08](Lecture-08.md)), so the simulation mirrors real states or replays logged ones. Twins are mostly used for testing and validation: run the robot software stack against the twin before a change reaches the floor, reproduce a field failure from logs, or generate site-specific synthetic data ([Lecture 07](Lecture-07.md)). In practice the two strategies combine. You train robust policies with randomization around the twin's identified parameters, then validate them on the twin itself before deploying.

---

## 7. The hardware view

* **Profiling costs memory too.** `nsys` with CUDA tracing and Tracy with GPU zones both add buffers and overhead. Profile a few iterations at the same `num_envs`, then take timings from a clean run.
* **Streaming moves the cost and does not remove it.** NVENC is separate silicon (watch the `enc` column in `dmon -s u`). But a streamed session renders a full viewport through RTX, which costs both frame time and VRAM next to your environments.
* **Each CUDA context costs VRAM.** A second process on the same GPU (a policy server, a second sim, a notebook with torch imported) takes its own fixed memory before it allocates anything. Count processes as well as tensors.
* **`torch.cuda` stats undercount.** PhysX, Warp and RTX allocate outside PyTorch. Use `nvidia-smi` for the total, PyTorch stats for your share, and the difference for "everything else".
* **DR is nearly free in VRAM.** Per-env parameters are a few floats, and startup randomization runs once. Its real cost is training time.
* **Multi-GPU duplicates fixed cost.** Every rank pays the Kit/renderer baseline on its own GPU. Scaling helps when per-GPU env memory, not fixed cost, is the limit.

> **8 GB budget.** Everything in this lecture fits on the 4060 except real multi-GPU work. Profile state-only tasks at the `num_envs` you actually train with (from Lecture 10), not the maximum. A profiled run with `nsys` can OOM where the unprofiled one fits, so drop `num_envs` for the capture and say so. Livestream only for debugging. Measure Lab 12b's streaming overhead and do not stream during training. The DR ablation (Lab 12c) uses Cartpole and is cheap. For the scaling table in Lab 12d, rent a single L40S or RTX PRO 6000 by the hour, run the same command, and destroy the instance. Never rent A100/H100 for anything that renders.

---

## 8. Build it

All scripts extend `isaac_bench/` from Lecture 01. Keep a `dmon` log running for every lab:

```bash
nvidia-smi dmon -s pucmt -d 1 -o T > isaac_bench/logs/dmon_$(date +%s).log &
```

### Lab 12a — Where does a training iteration go?

Take the Lecture 10 Cartpole and Lift runs at your chosen `num_envs`. First, clean numbers from Isaac Lab's benchmark harness:

```bash
uv run --extra isaacsim isaaclab benchmark training --rl_library rsl_rl \
  --task Isaac-Cartpole-Direct --num_envs 4096 --max_iterations 50 --warmup_steps 50 --seed 42 \
  --visualizer none --benchmark_formatter schema,summary --output_path isaac_bench/results/12a_cartpole \
  physics=isaacsim_physx
```

From the JSON, `runtime.collection_fps` (rollout only), `runtime.total_fps` (rollout plus update) and `runtime.environment_step_timing.environment_step_fps` (env only) give you \(t_{\text{update}}\) and the collection terms by difference. `ISAACLAB_PHYSICS_PROFILE=1` is read only by `benchmark runtime`, and `extra.physics_mean_ms` appears only if the task config sets a non-`None` `benchmark_mode` (the docs say the scopes stay disabled otherwise; the stock Cartpole and Lift configs do not set it). If it is absent, take the physics share from the nsys trace below. Note that shipped tasks pick a default physics preset; pass `physics=` explicitly so runs are comparable.

Then the trace. Install `nvtx` into the Isaac Lab environment and capture three iterations with the annotation file Isaac Lab ships:

```bash
nsys profile -t nvtx,cuda --python-functions-trace=scripts/benchmarks/nsys_trace.json -o isaac_bench/results/12a_lift \
  uv run --extra isaacsim isaaclab train --rl_library rsl_rl --task Isaac-Lift-Franka --num_envs 1024 --max_iterations 3 --viz none \
  physics=isaacsim_physx
nsys stats --report nvtx_sum,cuda_gpu_kern_sum isaac_bench/results/12a_lift.nsys-rep > isaac_bench/results/12a_lift_stats.txt
```

Open the `.nsys-rep` file, expand the thread domains and the **CUDA HW** row, and look for **gaps**: stretches where Python runs and the GPU is idle. Fill in the table below for both tasks, as percentages of \(T_{\text{iter}}\):

| Term | Cartpole | Lift | Evidence |
|---|---|---|---|
| Physics | | | `physics_mean_ms` × \(d\) × \(S\), or PhysX kernels in nsys |
| Env Python (obs/reward/reset) | | | NVTX ranges |
| Policy inference | | | NVTX / kernel summary |
| PPO update | | | `total_fps` vs `collection_fps` |
| Other (logging, idle GPU) | | | Gaps on the CUDA row |

Expected shape, not a promise: Cartpole has so little physics that fixed per-step overhead and the update dominate. Lift spends noticeably more in physics and resets. Write one sentence per task saying which term you would attack first.

### Lab 12b — The cost of watching

Add a `--stream` flag to `falling_cubes.py` from Lecture 01:

```python
# isaac_bench/falling_cubes.py — additions
parser.add_argument("--stream", action="store_true")
parser.add_argument("--minimal", action="store_true")
...
cfg = {"headless": args.headless or args.stream}
if args.stream:
    cfg["hide_ui"] = False                     # stream the full UI, as in the official livestream sample
if args.minimal:
    cfg["renderer"] = "MinimalRendering"
simulation_app = SimulationApp(cfg)
if args.stream:
    from isaacsim.core.experimental.utils.app import enable_extension
    enable_extension("omni.kit.livestream.app")   # connect the WebRTC Streaming Client to measure with a client attached
```

Run `--n 1024` four ways: headless, headless + `--minimal`, `--stream` with no client, and `--stream` with the WebRTC client connected. Record physics ms/step, frame ms, peak VRAM and `dmon` encoder utilization. Repeat the headless and streaming runs in the container (`./runheadless.sh`-style launch, `--network=host`) to check that container overhead is negligible. Optionally capture one Tracy session: launch with `--enable omni.kit.profiler.tracy` and add `"profiler_backend": ["tracy"]` to the config. Report the *Physics Step* vs *Rendering* zone split inside *App Update*.

### Lab 12c — Domain randomization ablation

Use the 3.0 direct Cartpole and add mass randomization, observation noise and an **action-delay buffer**. Then test both policies on parameters *outside* the training range.

```python
# isaac_bench/bench_tasks/cartpole_dr.py
import torch
from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelCfg
from isaaclab_tasks.core.cartpole.cartpole_direct_env import CartpoleEnv
from isaaclab_tasks.core.cartpole.cartpole_direct_env_cfg import CartpoleEnvCfg

def _mass_term(lo, hi):
    return EventTerm(func=mdp.randomize_rigid_body_mass, mode="startup",
                     params={"asset_cfg": SceneEntityCfg("cartpole", body_names=".*"),
                             "mass_distribution_params": (lo, hi), "operation": "scale"})

@configclass
class TrainDREvents:
    mass = _mass_term(0.7, 1.3)

@configclass
class ShiftEvents:
    mass = _mass_term(1.6, 1.6)                       # outside the training range

@configclass
class CartpoleNoDREnvCfg(CartpoleEnvCfg):
    min_action_delay: int = 0                         # policy steps
    max_action_delay: int = 0

@configclass
class CartpoleDREnvCfg(CartpoleNoDREnvCfg):
    max_action_delay = 2
    events: TrainDREvents = TrainDREvents()
    observation_noise_model = NoiseModelCfg(noise_cfg=GaussianNoiseCfg(std=0.01))

@configclass
class CartpoleShiftedEnvCfg(CartpoleNoDREnvCfg):
    min_action_delay = 3
    max_action_delay = 3
    events: ShiftEvents = ShiftEvents()

class CartpoleDREnv(CartpoleEnv):
    cfg: CartpoleNoDREnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._hist = torch.zeros(self.num_envs, cfg.max_action_delay + 1, 1, device=self.device)
        self._delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._hist = torch.roll(self._hist, shifts=1, dims=1)
        self._hist[:, 0] = actions.clamp(-1.0, 1.0)
        rows = torch.arange(self.num_envs, device=self.device)
        super()._pre_physics_step(self._hist[rows, self._delay])   # the action from `delay` steps ago

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        env_ids = self.cartpole._ALL_INDICES if env_ids is None else env_ids
        self._hist[env_ids] = 0.0
        self._delay[env_ids] = torch.randint(self.cfg.min_action_delay, self.cfg.max_action_delay + 1,
                                             (len(env_ids),), device=self.device)
```

Register the three configs as `Bench-Cartpole-{NoDR,DR,Shifted}-Direct` with `gym.register`, following the pattern in Isaac Lab's `core/cartpole/__init__.py`. Use `entry_point=".../cartpole_dr:CartpoleDREnv"`, set `env_cfg_entry_point`, and reuse `isaaclab_tasks.core.cartpole.agents.rsl_rl_ppo_cfg:CartpoleDirectPPORunnerCfg`. `isaaclab train` imports every package that declares an **`isaaclab.tasks` entry point**. The external-project template from `uv run isaaclab --new` ([Lecture 10](Lecture-10.md) §4.1) writes one into its `pyproject.toml`, so the easiest route is to put `bench_tasks` in such a project.

Train each config with three seeds (`uv run --extra isaacsim isaaclab train ...`, `--seed 1/2/3`, `physics=isaacsim_physx`), run `play --checkpoint latest` once per run to export `exported/policy.pt`, and evaluate with the shared evaluator:

```python
# isaac_bench/eval_policy.py — one process per (policy, task, seed); Kit cannot relaunch in-process
import argparse, json
from isaaclab.app import add_launcher_args, launch_simulation
parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True); parser.add_argument("--policy", required=True)
parser.add_argument("--seed", type=int, required=True); parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--out", default="isaac_bench/results/eval.jsonl")
add_launcher_args(parser)

import importlib.metadata, gymnasium as gym, torch
import isaaclab_tasks  # noqa: F401  registers Isaac-* tasks
for ep in importlib.metadata.entry_points(group="isaaclab.tasks"):
    ep.load()                                         # registers Bench-*/Capstone-* tasks, as `isaaclab train` does
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli
args, hydra_args = setup_preset_cli(parser)           # keeps physics=... overrides
env_cfg, _ = resolve_task_config(args.task, "", overrides=hydra_args)
env_cfg.scene.num_envs, env_cfg.seed = args.num_envs, args.seed
args.device = env_cfg.sim.device

with launch_simulation(env_cfg, args):
    env = gym.make(args.task, cfg=env_cfg); u = env.unwrapped
    policy = torch.jit.load(args.policy, map_location=u.device).eval()
    obs, _ = env.reset()
    done = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device); ok = torch.zeros_like(done)
    with torch.inference_mode():
        while not done.all():                         # first complete episode of every env
            obs, _, term, trunc, _ = env.step(policy(obs["policy"]))
            ended = (term | trunc) & ~done
            success = u.last_episode_success if hasattr(u, "last_episode_success") else (trunc & ~term)
            ok |= ended & success; done |= ended
    rec = dict(task=args.task, policy=args.policy, seed=args.seed, n=int(done.sum()), k=int(ok.sum()))
    open(args.out, "a").write(json.dumps(rec) + "\n"); env.close()
```

For Cartpole, "success" means surviving to the timeout. Evaluate every policy on `NoDR` (nominal) and `Shifted`, with eval seeds disjoint from training seeds. Report Wilson intervals (the `stats.py` from Deep RL Lecture 13). The expected shape is that both do well on nominal, and the no-DR policy degrades more on shifted mass plus delay. If your result disagrees, report it anyway; it is data.

### Lab 12d — 8 GB vs 48 GB (optional, cloud)

On one rented L40S or RTX PRO 6000, rerun the Lab 12a benchmark at your 4060's `num_envs` and at 4× and 16× that. Fill in `scaling.csv` (GPU, `num_envs`, env-steps/s, peak VRAM, wall-clock to target reward, $ spent). If you have two cloud GPUs, add one `train_multigpu` row and its \(\eta(2)\).

---

## 9. Use it in the real stack

* **Regression guard:** put a short `isaaclab benchmark runtime` run in CI for your task and fail if `environment_step_fps` drops by more than its run-to-run noise. Isaac Sim ships the same idea in `standalone_examples/benchmarks/` and documents KPIs on its *Isaac Sim Benchmarks* page.
* **Robot teams** treat sysid, DR ranges and latency models as versioned artifacts next to the URDF, not as task code.
* **Sim-to-sim before sim-to-real:** Isaac Lab 3.0's PhysX↔Newton transfer guide lists the contract that must match across backends: action and observation layout, timing, joint ordering and episode definition. Treat it as a cheap rehearsal for the real robot. The capstone ([Lecture 13](Lecture-13.md)) runs this rehearsal into Isaac Sim over ROS 2.

---

## 10. Measure it

| Metric | How | Why it matters |
|---|---|---|
| **Iteration time breakdown** (%) | Lab 12a table | Tells you what to optimize |
| **Collection vs total FPS** | `isaaclab benchmark training` | Separates simulation from learning cost |
| **GPU idle fraction** | Gaps on the nsys CUDA row | Size of the CPU-bound share |
| **Streaming overhead** (frame ms, VRAM, enc %) | Lab 12b | Cost of watching |
| **Scaling efficiency \(\eta(G)\)** | Lab 12d | Whether more GPUs pay off |
| **Success: nominal vs shifted** (Wilson CI) | Lab 12c | Whether DR bought robustness |
| **$ per converged run** | Cost ledger | Whether the cloud was worth it |

---

## 11. Ship it

Commit under `isaac_bench/`:

* `results/12a_*` — benchmark JSON, `.nsys-rep` (or its stats text if too large) and `BREAKDOWN.md` with the time table and your "attack first" sentence per task
* `falling_cubes.py` with `--stream` / `--minimal`, plus `results/12b_streaming.csv`
* `bench_tasks/` (DR Cartpole), `eval_policy.py`, `results/12c_eval.jsonl` and `DR_ABLATION.md` (table + Wilson CIs + one paragraph)
* `scaling.csv` and `COST_LEDGER.md` (even if every row says "local, $0")
* `RUN_MANIFEST.json` for every reported run

---

## Exit criteria

You can move on when you can:

* produce a time breakdown of a training iteration and defend which term to optimize first
* say what livestreaming costs on your card and why the A100 cannot do it
* pick a cloud GPU for Isaac Sim, launch a containerized run with the right flags, and price it
* launch `train_multigpu` and report \(\eta(G)\) alongside wall-clock to reward
* list what `--deterministic` does and the one randomization pattern that breaks determinism
* show, with confidence intervals, whether your domain randomization improved robustness

---

## Self-check

1. Lab 12a shows the CUDA HW row is busy only a minority of each iteration on Lift at `num_envs=1024`. A teammate proposes doubling PhysX position iterations "since we have GPU headroom". What does the trace actually suggest you do, and why would doubling `num_envs` likely raise env-steps/s?
2. You rent an H100 instance to train a camera-based Isaac Lab task and stream the viewport. List both reasons this fails, and propose the instance type you should have rented.
3. Two runs with the same seed on the same machine diverge after a few hundred iterations. The task randomizes friction in `interval` mode. What does the Isaac Lab reproducibility page say about this, and how do you change the event config?
4. `train_multigpu --num_gpus 4 --num_envs 4096` reaches the reward threshold in about the same wall-clock time as one GPU with 4,096 envs, even though env-steps/s scaled well. Give two explanations rooted in PPO, not in hardware.
5. Your DR-trained Cartpole succeeds on shifted parameters with a Wilson interval of [0.88, 0.95]. The no-DR policy's interval is [0.84, 0.93]. Can you claim DR helped? What would you change in the evaluation to find out?
6. `torch.cuda.max_memory_allocated()` reports a small number, but `nvidia-smi` shows the card nearly full during training. Account for the difference.

---

## References

* Isaac Sim 6.1 — [Performance Optimization Handbook](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/reference_material/sim_performance_optimization_handbook.html) · [Profiling with Tracy](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/utilities/debugging/profiling_performance.html) · [Benchmarks](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/reference_material/benchmarks.html)
* Isaac Sim 6.1 — [Livestream clients](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/manual_livestream_clients.html) · [Container installation](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_container.html) · [Cloud deployment](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_cloud.html) · [Requirements](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/requirements.html)
* Isaac Sim 6.1 — [System Identification extension](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/robot_setup/ext_isaacsim_robot_setup_sysid.html)
* Isaac Lab 3.0 — [Profiling with Nsight Systems](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/how-to/profile_with_nsys.html) · [Run benchmarks](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/developer-tools/benchmarking/run_benchmarks.html) · [Multi-GPU and multi-node training](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/features/multi_gpu.html)
* Isaac Lab 3.0 — [Reproducibility and determinism](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/features/reproducibility.html) · [Actuators](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/concepts/actuators.html) · [Transfer policies between PhysX and Newton](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/how-to/transfer_policies_between_physx_and_newton.html) · [Troubleshooting: slow simulation](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/refs/troubleshooting.html)
* Cloud tooling — [isaac-sim/isaac-launchable](https://github.com/isaac-sim/isaac-launchable) · [isaac-sim/IsaacAutomator](https://github.com/isaac-sim/IsaacAutomator)
* NVIDIA tools — [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) · [nvidia-smi documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
* Tobin et al., *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World* — [arXiv:1703.06907](https://arxiv.org/abs/1703.06907)
* Peng et al., *Sim-to-Real Transfer of Robotic Control with Dynamics Randomization* — [arXiv:1710.06537](https://arxiv.org/abs/1710.06537)
* Hwangbo et al., *Learning agile and dynamic motor skills for legged robots* (actuator networks) — [arXiv:1901.08652](https://arxiv.org/abs/1901.08652)
* Rudin et al., *Learning to Walk in Minutes Using Massively Parallel Deep RL* — [arXiv:2109.11978](https://arxiv.org/abs/2109.11978)

---

## Next in this special course

* Next: [Lecture 13 — Capstone: A Task, a Policy, and a Bridge](Lecture-13.md)
* Previous: [Lecture 11 — Imitation, Teleop, and Policy Evaluation](Lecture-11.md)
* Back: [Isaac Sim and Isaac Lab — Overview](README.md)
