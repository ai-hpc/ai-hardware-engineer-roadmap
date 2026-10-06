# Lecture 11: Imitation, Teleop, and Policy Evaluation

## Overview

RL ([Lecture 10](Lecture-10.md)) needs a reward. Many manipulation tasks are easier to *show* than to reward: stack three cubes in order, insert a peg, open a specific drawer. For those, the pipeline is teleoperate a handful of demonstrations, multiply them synthetically, train a behavior-cloning policy, and then evaluate it honestly. The same evaluation discipline applies to the large pretrained policies (GR00T, π0.5) that robotics teams now fine-tune rather than train, except those models do not fit on your GPU next to the simulator.

This lecture covers the Isaac Lab 3.0 tooling for each step: Isaac Teleop (now rebranded Isaac Capture), Isaac Lab Mimic, robomimic, Isaac Lab-Arena, RLinf and LEAPP. It treats the result as a systems problem: dataset bytes, VRAM for two programs, and the network round trip between a simulator and a remote policy.

By the end you should be able to:

* record, replay and inspect teleoperated demonstrations with the 3.0 `isaaclab teleop` commands
* explain what Mimic does to a demonstration (subtasks, object frames, task-space actions) and run annotate → generate within an 8 GB budget
* train a robomimic BC policy and report its success rate over seeds with a confidence interval
* describe how Isaac Lab-Arena composes environments and evaluates a remote policy, and where RLinf and LEAPP fit
* build a client-server evaluation loop and measure its round-trip latency and effective env-steps/s

---

## 1. Why it matters: demos are slow, evaluation is easy to fake

Three costs dominate imitation learning in simulation, and each one maps to a resource:

| Cost | Why | Resource | Tool |
|---|---|---|---|
| Human demonstrations | A person at a keyboard or headset, in real time, one environment | Human time | Isaac Teleop / Capture, `isaaclab teleop record` |
| Data volume | BC wants hundreds to thousands of demos with spatial variety | GPU throughput (parallel generation) and disk | Isaac Lab Mimic |
| Evaluation | Success rates from a few rollouts are noise; big policies need their own GPU | Sim throughput, a second GPU or the cloud, network | robomimic eval scripts, Isaac Lab-Arena |

The trap is the third row. A BC policy "succeeding 9 out of 10 times" on one seed is a statement with a very wide confidence interval ([Deep RL Lecture 13](../Deep%20RL%20for%20Robot%20Learning/Lecture-13.md) covers the statistics). Policies also vary a lot across training epochs. The 3.0 Mimic tutorial explicitly tells you to evaluate checkpoints from several epochs, not just the last.

---

## 2. Mental model: the imitation pipeline

```text
 keyboard / SpaceMouse / XR ──► isaaclab teleop record ──► datasets/dataset.hdf5          (10 human demos)
                                isaaclab teleop replay  ◄──┘  (sanity check)
 annotate_demos.py  (--auto heuristics, or manual B/N/S keys) ──► annotated_dataset.hdf5   (subtask boundaries)
 generate_dataset.py (Mimic, many envs in parallel)          ──► generated_dataset.hdf5    (hundreds-thousands)
 robomimic/train.py --algo bc                                ──► logs/robomimic/...  (.pth per epoch)
 robomimic/play.py --seed S --num_rollouts R                 ──► success rate per (checkpoint, seed)
```

### 2.1 Teleop in Isaac Lab 3.0

* **Packages and names.** 3.0 moved teleop to the `isaaclab_teleop` package. Its integration point is `IsaacTeleopDevice`, configured per task through `IsaacTeleopCfg` (fields include `pipeline_builder`, `xr_cfg`, `sim_device`, `retargeting_execution`). `isaaclab.devices.openxr` is deprecated. The 3.0 docs note that **Isaac Teleop has been rebranded Isaac Capture**. Isaac Lab pins the 1.4 release, whose import package is still `isaacteleop`, and upstream renamed it `isaaccapture` in 1.6. Expect both names in the wild.
* **Two device paths.** XR headsets (Apple Vision Pro, Meta Quest 3, Pico 4 Ultra) and gloves stream through **CloudXR** into a retargeting graph (`Se3AbsRetargeter`, `GripperRetargeter`, dex-hand retargeters). Legacy **keyboard** and **SpaceMouse** devices still work through the task's `teleop_devices` config, selected with `--teleop_device keyboard|spacemouse`. On an 8 GB desktop without a headset, the legacy path is the one you will use.
* **Commands.** `isaaclab teleop run|record|replay` wraps `teleop_se3_agent.py`, `record_demos.py` and `replay_demos.py`. Recording writes HDF5 (`--dataset_file`, `--num_demos`, `--step_hz`, `--num_success_steps`). A demo counts as successful only after the success term holds for `--num_success_steps` consecutive steps.
* **MCAP.** `record_demos.py --mcap_record_path` writes the live Isaac Capture session to MCAP. The 3.0 source labels it **debug-only, not a data-generation format**: no per-episode segmentation, no reset state. Use HDF5 for anything you train on.
* **Selectors are not saved.** The HDF5 metadata stores the task ID but not the `physics=`/`presets=` tokens. Pass the same selectors when you replay.

> **You will see this in older code.** Isaac Lab 2.3 recorded with `./isaaclab.sh -p scripts/tools/record_demos.py --task Isaac-Stack-Cube-Franka-IK-Rel-v0 --teleop_device keyboard`. In 3.0 the base task is `IsaacContrib-Stack-Cube-Franka-IK-Rel` (it moved to the contrib namespace). The **Mimic** variants registered in `isaaclab_mimic` at `release/3.0.0` still carry `-v0`, e.g. `Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0`. Copy IDs from `list_envs`, not from memory.

### 2.2 Isaac Lab Mimic

Mimic builds on the MimicGen approach (Mandlekar et al., 2023) to multiply a few human demos into many:

1. Each demo is split into **subtasks**. A subtask is a segment where the end-effector's motion is governed by one **reference object** ("grasp red cube", then "place on blue cube"). Annotation marks where each subtask ends, either by heuristics (`--auto`, where the task exposes subtask signals as an observation group) or by hand (keys `B` pause, `N` continue, `S` mark a boundary).
2. To generate a new demo in a new scene layout, Mimic takes each human segment, applies the rigid transform between the old and new pose of that subtask's object, and **linearly interpolates** between segments.
3. Each candidate is simulated, and only successes are written out. That is why many environments run in parallel, and why the success rate of generation is a throughput number.

Requirements fall straight out of step 2. Actions must be in **task space** (end-effector poses, e.g. the `-IK-Rel`/`-IK-Abs` variants). A Mimic env subclasses `ManagerBasedRLMimicEnv` and implements `get_robot_eef_pose`, `target_eef_pose_to_action`, `action_to_target_eef_pose`, `actions_to_gripper_actions`, `get_object_poses` and `get_subtask_term_signals`. The config adds `SubTaskConfig` entries (`object_ref`, `subtask_term_signal`, interpolation steps). Mimic is **Linux only**. Fewer subtasks mean less stitching and higher generation success, and the docs list jerky, paused or overlong human demos as the usual cause of failed generation.

**SkillGen and licensing.** SkillGen replaces Mimic's linear interpolation with cuRobo motion planning between skill segments (`generate_dataset.py --use_skillgen`). `isaaclab_mimic` itself is Apache-2.0, but cuRobo is under a proprietary NVIDIA license, and the 3.0 docs build it from source against CUDA 13.0. Check the license before you put SkillGen in a product pipeline.

### 2.3 robomimic

robomimic (Mandlekar et al., 2021) is the BC library Isaac Lab wires in: `scripts/imitation_learning/robomimic/train.py --task … --algo bc --dataset …` trains from the HDF5 directly, with optional `--epochs` to override the JSON config and `--normalize_training_actions`. If you normalize, `play.py` then needs `--norm_factor_min/--norm_factor_max`. The stack task ships `bc_rnn_low_dim.json` (state) and `bc_rnn_image_200.json` (visuomotor) agent configs. Models and logs land under `logs/robomimic/`. `play.py` takes `--checkpoint`, `--num_rollouts`, `--horizon` and `--seed`, and prints a success rate.

### 2.4 Evaluation that survives review

For \( k \) successes in \( n \) rollouts, report a Wilson interval, not \( k/n \) alone:

$$
\frac{\hat p + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}, \qquad \hat p = k/n,\; z = 1.96
$$

With \( n = 50 \) and \( \hat p = 0.5 \) the interval is roughly ±0.13 wide on each side, so two checkpoints at 46% and 54% are not distinguishable. Fix the seeds and evaluate every candidate checkpoint on the **same** seeds. Report the spread across seeds separately from the binomial interval within a seed. Physics resets are not perfectly reproducible (the Mimic docs warn that replayed demos can fail), so the seed fixes the *distribution* of initial states, not bit-exact trajectories.

---

## 3. Evaluating big policies: Arena, RLinf, LEAPP

### 3.1 Isaac Lab-Arena

Isaac Lab-Arena is a separate repo (`isaac-sim/IsaacLab-Arena`, Apache-2.0) for authoring benchmarks and evaluating policies at scale. Its README is explicit: **alpha, v0.3, "Not an Early Access or General Availability Release"**, and the APIs will break.

* **Composition.** An environment is a **scene** (layout, objects, fixtures), an **embodiment** (robot, observations, actions, sensors, controllers) and a **task**. `ArenaEnvBuilder` combines them into a standard `ManagerBasedRLEnvCfg`, so Arena environments run as ordinary Isaac Lab envs and can be registered for RL or data generation. Variations (lighting, backgrounds, camera parameters, object mass) turn one environment into a sensitivity sweep, and subtask predicates (grasp, lift, place) show *where* a policy fails.
* **Client-server policies.** GR00T, π0.5 (through openpi) or a custom policy runs behind a server. The simulator side is a thin client. For openpi, Arena's `Pi0RemotePolicy` speaks openpi's WebSocket protocol (msgpack-encoded NumPy). It requests an **action chunk** per environment and replays the first `open_loop_horizon` actions before asking again. In the v0.3 source the client loops over environments, one request per env, so the refill cost scales with `num_envs`.
* **Scale-out.** Experiments (several tasks × policies) run locally or through **OSMO**, with a server task and a policy-runner task per run.
* **Versions.** Arena's compatibility table lists Isaac Lab 3.0.0 with **Isaac Sim 6.0.0**, not the 6.1 pinned in this course. Its `uv sync` installs its own locked stack. Give it its own checkout and environment, and expect drift until Arena tracks 6.1.

### 3.2 RLinf in Isaac Lab 3.0

RLinf is the fifth RL backend in 3.0, for RL *post-training* of VLAs. The 3.0 integration targets **GR00T and OpenVLA**, with PPO / actor-critic / SAC and FSDP-sharded actor workers. It splits into an actor worker (FSDP update), a rollout worker (VLA inference) and an env worker (Isaac Lab):

```bash
uv run --no-sync isaaclab train --rl_library rlinf \
    --config_name isaaclab_ppo_gr00t_assemble_trocar --model_path /path/to/base_model
```

Setup is deliberately special: `uv sync --inexact --extra rlinf --extra video`, then version-conflicting packages installed with `--no-deps`, a pinned Isaac-GR00T commit, and `uv run --no-sync` so the lockfile does not undo them. It is Linux-only, and the docs recommend multiple GPUs. Each checkpoint can be several GB. The algorithmic side (PPO/GRPO for VLAs, why rollouts dominate) is [Deep RL Lecture 11](../Deep%20RL%20for%20Robot%20Learning/Lecture-11.md). The π0.5 capstone in [Deep RL Lecture 14](../Deep%20RL%20for%20Robot%20Learning/Lecture-14.md) uses RLinf on LIBERO, outside Isaac Lab.

### 3.3 LEAPP: exporting what you trained

LEAPP ("Lightweight Export Annotations for Policy Pipelines") packages a trained policy **with** its input/output semantics: the model (`.onnx` by default, or `.pt`), metadata including the policy rate, initial values for recurrent state or last action, and a graph diagram. Downstream code then does not have to re-implement observation ordering or action scaling by hand:

```bash
uv run --extra leapp isaaclab leapp export --rl_library rsl_rl --task Isaac-Reach-Franka physics=newton_mjwarp
uv run --extra leapp isaaclab leapp deploy --task Isaac-Reach-Franka \
    --pipeline <exported.yaml> --viz newton_gl physics=newton_mjwarp
```

Manager-based tasks trained with rsl_rl, rl_games, skrl or sb3 are supported directly. Direct-workflow tasks (like your Lecture 10 reach) need LEAPP annotations added to the env and export only through RSL-RL, and `LeappDeploymentEnv` does not run them. Use the same `physics=` for export as for training.

---

## 4. The hardware view: two programs, one small card

**A VLA and Isaac Sim do not share 8 GB.** In bf16 a model needs 2 bytes per parameter for weights alone, so every billion parameters is about 2 GB before activations, the vision encoder's buffers, or any KV cache. Multi-billion-parameter VLAs therefore take most or all of an 8 GB card by themselves. Isaac Sim's renderer needs its own fixed share (Lecture 01), and the visuomotor tasks render two cameras per env. On a 4060 the only workable layout is **client-server**: the simulator on the 4060, the policy on another GPU or a cloud instance. That is exactly the layout Arena and openpi are built around. (The Arena docs report an ~11 GB checkpoint download for the openpi server alone.)

**The latency budget.** Let the policy act at \( f_c \) Hz with action chunks of horizon \( H \), and let one server round trip cost \( \text{RTT} \) (serialize + network + queue + inference + deserialize). For a client that refills each env's chunk with its own request:

$$
t_{\text{per env-step}} \approx t_{\text{sim}} + \frac{N_{\text{env}} \cdot \text{RTT}}{H},
\qquad
\text{RTF} = \frac{1 / f_c}{t_{\text{per env-step}}}
$$

In simulation the world waits for the policy, so latency costs **throughput, not correctness**. Your eval just takes longer. On a real robot the world does not wait, and the same RTT becomes stale actions. Measure it in sim (Lab 11e) to know what you will face. Three levers: batch all envs into one request (removes the \( N_{\text{env}} \) factor), raise \( H \) (fewer requests, but more open-loop drift), or shrink the payload (encode images, send fewer cameras).

**Dataset bytes.** For \( D \) demos of \( T \) steps:

$$
S \approx D \cdot T \cdot \Big( 4\,(d_{\text{obs}} + d_{\text{act}}) + \sum_{\text{cams}} H_c W_c C_c\, b_c \Big)
$$

State demos are small: hundreds of floats per step. Pixels dominate. The stack visuomotor config renders two 200×200 cameras (RGB plus depth). Counting only RGB as uint8, that is 240 KB per step, so 1,000 demos of 300 steps would be about 72 GB uncompressed. That is arithmetic under stated assumptions, so check your real files with `h5ls -r` and `du -h`. It decides whether the dataset fits in RAM for the robomimic loader or streams from NVMe, and whether visuomotor generation is a 4060 job at all.

> **8 GB budget.** Teleop with one env and `--viz kit` fits, though the Kit GUI is the largest single consumer, so close everything else. Mimic **state-based** generation fits if you scale `--num_envs` far below the docs' values (those assume an RTX PRO 6000 Blackwell). Start at 4-8 envs, run headless (omit `--viz`), and grow while watching `nvidia-smi`. Skip **visuomotor** generation and visuomotor BC training on the 4060. Two 200×200 cameras per env plus the image dataset put them on a 48 GB L40S / RTX PRO 6000. robomimic state BC training is small and fits. Any VLA (GR00T, π0.5) and RLinf post-training belong on a separate GPU or the cloud. On the 4060 you run only the simulator side of the client-server loop. Arena itself needs Isaac Sim (6.0 per its table) and its own environment, so treat it like any Kit workload in the budget.

---

## 5. Build it

Work from your `release/3.0.0` checkout. Results go in `isaac_bench/il/` and `isaac_bench/remote/`. Log VRAM with `nvidia-smi` as in Lab 1a.

### Lab 11a — Teleoperate and record

```bash
mkdir -p datasets
uv run --extra teleop,isaacsim isaaclab teleop run --task IsaacContrib-Stack-Cube-Franka-IK-Rel \
    --viz kit --num_envs 1 --sensitivity 4 --teleop_device keyboard        # practice: W/S A/D Q/E, K = gripper
uv run --extra teleop,isaacsim isaaclab teleop record --task IsaacContrib-Stack-Cube-Franka-IK-Rel \
    --viz kit --dataset_file ./datasets/dataset.hdf5 --num_demos 10 --teleop_device keyboard
uv run --extra teleop,isaacsim isaaclab teleop replay --task IsaacContrib-Stack-Cube-Franka-IK-Rel \
    --viz kit --num_envs 1 --dataset_file ./datasets/dataset.hdf5
```

Use `--teleop_device spacemouse` if you have a 3Dconnexion device. Smoother input gives better demos. The 3.0 docs include the udev rule for `/dev/bus/usb` access. Record: demos attempted vs kept, minutes per kept demo, steps per demo, file size, and peak VRAM. If stacking by keyboard is too hard, record `Isaac-Reach-Franka` instead (`physics=isaacsim_physx presets=diffik`, per the teleop docs) and use NVIDIA's pre-recorded 10-demo stack dataset (linked from the Mimic tutorial) for Labs 11b-11d.

### Lab 11b — Mimic at 8 GB scale

```bash
uv run --extra isaacsim,mimic python scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
    --task Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0 --viz kit --auto \
    --input_file ./datasets/dataset.hdf5 --output_file ./datasets/annotated_dataset.hdf5
for N in 4 8 16; do                                    # headless; watch nvidia-smi between runs
  uv run --extra isaacsim,mimic python scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
      --num_envs $N --generation_num_trials 50 --max_num_failures 500 \
      --input_file ./datasets/annotated_dataset.hdf5 --output_file ./datasets/gen_N$N.hdf5
done
```

Time each run. Compute **generated demos per minute**, **generation success rate** (successes / attempts, as reported in the script's output), and peak VRAM against `--num_envs`. Pick the best `N` that fits and generate the training set (start with a few hundred demos and scale up if time allows; the docs cite 1,000 as working well for their BC-RNN).

### Lab 11c — Train BC with robomimic

```bash
uv run --extra mimic python -c "import robomimic"                 # resolves robomimic in the uv env
uv run --extra isaacsim,mimic python scripts/imitation_learning/robomimic/train.py \
    --task IsaacContrib-Stack-Cube-Franka-IK-Rel --algo bc --dataset ./datasets/generated_dataset.hdf5
```

Log the wall-clock time per epoch and peak VRAM. Keep every saved checkpoint.

### Lab 11d — Evaluate over seeds, with intervals

```python
# isaac_bench/il/eval_seeds.py  <task> <ckpt1.pth> [ckpt2.pth ...]  — robomimic play.py × seeds → CSV + Wilson CI
import math, re, subprocess, sys
task, ckpts, seeds, n = sys.argv[1], sys.argv[2:], [0, 1, 2], 50
def wilson(k, n, z=1.96):
    p = k / n; c = (p + z*z/(2*n)) / (1 + z*z/n); h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return c - h, c + h
print("checkpoint,seed,successes,n,ci_low,ci_high")
for ck in ckpts:
    for s in seeds:
        out = subprocess.run(["uv", "run", "--extra", "isaacsim,mimic", "python",
                              "scripts/imitation_learning/robomimic/play.py", "--task", task,
                              "--checkpoint", ck, "--num_rollouts", str(n), "--seed", str(s)],
                             capture_output=True, text=True).stdout
        k = int(re.search(r"Successful trials: (\d+)", out).group(1))
        lo, hi = wilson(k, n)
        print(f"{ck},{s},{k},{n},{lo:.3f},{hi:.3f}")
```

Run it on at least three checkpoints from different epochs. Pick the best one by its *lower* confidence bound, not its point estimate. If the task you use implements visual evaluation settings (`eval_type`), `robomimic/robust_eval.py --seeds …` sweeps lighting and textures on top of seeds.

### Lab 11e — Client-server evaluation skeleton

A stand-in policy server lets you measure the transport before any VLA is involved. The transport is plain `websockets` + `msgpack` (the same building blocks openpi's protocol uses). Install both with `uv run --with websockets --with msgpack …`.

```python
# isaac_bench/remote/policy_server.py — run on the policy machine (another GPU, a cloud box, or localhost)
import argparse, time
import msgpack, numpy as np
from websockets.sync.server import serve
pack = lambda a: {"shape": list(a.shape), "dtype": str(a.dtype), "data": a.tobytes()}
unpack = lambda d: np.frombuffer(d["data"], dtype=d["dtype"]).reshape(d["shape"])
ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--delay_ms", type=float, default=0.0)    # emulate model inference time
ap.add_argument("--chunk", type=int, default=1)            # actions per request (H)
args = ap.parse_args()

def handler(ws):
    for msg in ws:
        req = msgpack.unpackb(msg)
        b = unpack(req["obs"]).shape[0]
        time.sleep(args.delay_ms / 1e3)
        ws.send(msgpack.packb({"actions": pack(np.zeros((b, args.chunk, req["act_dim"]), np.float32))}))

with serve(handler, "0.0.0.0", args.port, max_size=None, compression=None) as server:
    server.serve_forever()
```

```python
# isaac_bench/remote/eval_client.py — Isaac Lab 3.0 env loop with a remote policy
import argparse, sys, time
import numpy as np
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli
ap = argparse.ArgumentParser()
ap.add_argument("--task", default="Isaac-Reach-Franka"); ap.add_argument("--num_envs", type=int, default=16)
ap.add_argument("--steps", type=int, default=300); ap.add_argument("--server", default="ws://localhost:8765")
ap.add_argument("--per_env", action="store_true")         # one request per env, like a chunk-replay client
ap.add_argument("--image_px", type=int, default=0)        # add a fake HxWx3 uint8 image per env to the payload
add_launcher_args(ap)
args, hydra_args = setup_preset_cli(ap)
sys.argv = [sys.argv[0]] + hydra_args                     # physics=/presets= reach Hydra

import gymnasium as gym, msgpack, torch
import isaaclab_tasks  # noqa: F401  (registers core tasks)
from websockets.sync.client import connect
pack = lambda a: {"shape": list(a.shape), "dtype": str(a.dtype), "data": a.tobytes()}
unpack = lambda d: np.frombuffer(d["data"], dtype=d["dtype"]).reshape(d["shape"])

env_cfg, _ = resolve_task_config(args.task, "")
env_cfg.scene.num_envs = args.num_envs
with launch_simulation(env_cfg, args):
    env = gym.make(args.task, cfg=env_cfg)
    obs, _ = env.reset()
    act_dim = env.action_space.shape[-1]
    ws = connect(args.server, max_size=None, compression=None)
    rtt, t_sim, t0_all = [], [], time.perf_counter()
    for _ in range(args.steps):
        o = obs["policy"].cpu().numpy().astype(np.float32)
        acts = []
        for b in ([o[i:i + 1] for i in range(len(o))] if args.per_env else [o]):
            req = {"obs": pack(b), "act_dim": act_dim}
            if args.image_px:
                req["img"] = pack(np.zeros((len(b), args.image_px, args.image_px, 3), np.uint8))
            t0 = time.perf_counter(); ws.send(msgpack.packb(req)); rep = msgpack.unpackb(ws.recv())
            rtt.append(time.perf_counter() - t0)
            acts.append(unpack(rep["actions"])[:, 0])
        a = torch.from_numpy(np.concatenate(acts)).to(env.unwrapped.device)
        t0 = time.perf_counter(); obs, *_ = env.step(a); torch.cuda.synchronize(); t_sim.append(time.perf_counter() - t0)
    wall = time.perf_counter() - t0_all
    r = np.array(rtt) * 1e3
    print(f"N={args.num_envs} per_env={args.per_env} px={args.image_px} RTT ms p50={np.median(r):.2f} "
          f"p99={np.percentile(r, 99):.2f} sim ms/step={1e3 * np.mean(t_sim):.2f} "
          f"env-steps/s={args.num_envs * args.steps / wall:.0f}")
    env.close()
```

The env-loop calls (`add_launcher_args`, `setup_preset_cli`, `resolve_task_config`, `launch_simulation`, `gym.make(task, cfg=…)`) follow the 3.0 `zero_agent`/`random_agent` entry point source. The default task runs kit-less on Newton. Sweep: `--num_envs ∈ {1, 16, 64}` × batched vs `--per_env` × `--image_px ∈ {0, 224}` × server `--delay_ms ∈ {0, 50}`. Do it on localhost first, then with the server on another machine on your LAN (or a cloud box). Check the measured env-steps/s against the §4 formula.

---

## 6. Use it in the real stack

* **Real teleop rigs** use the XR path: CloudXR + Isaac Capture retargeters, configured per task with `IsaacTeleopCfg`. The same HDF5 → Mimic → BC flow applies, and humanoid examples (GR-1, G1 locomanipulation) ship in `isaaclab_mimic`.
* **GR00T workflows:** the 3.0 tree has a locomanipulation SDG pipeline with a GR00T dataset converter (`scripts/imitation_learning/locomanipulation_sdg/`), and `isaac-sim/IsaacLabEvalTasks` benchmarks GR00T N1 in Isaac Lab.
* **Benchmarks on Arena:** published suites (Lightwheel RoboCasa/LIBERO tasks, RoboTwin 2.0) run as Arena environments. That is the route to evaluating the π0.5 you post-train in [Deep RL Lecture 14](../Deep%20RL%20for%20Robot%20Learning/Lecture-14.md). Parity between the served policy and its reference implementation is the subject of the [VLA Optimization and Action-Parity Harness](../VLA%20Optimization%20and%20Action-Parity%20Harness/README.md) course.
* **Deployment:** LEAPP exports feed `LeappDeploymentEnv` in sim and are the hand-off format toward ROS 2 ([Lecture 08](Lecture-08.md), [Lecture 13](Lecture-13.md)).

---

## 7. Measure it

| Metric | How | Why it matters |
|---|---|---|
| **Human cost** | Minutes per kept demo, keep rate (Lab 11a) | The scarce input Mimic multiplies |
| **Generation throughput** | Generated demos/min and success rate vs `--num_envs` (Lab 11b) | Sizes the dataset you can afford on 8 GB |
| **Dataset bytes** | `du -h`, `h5ls -r` vs the §4 formula | RAM vs disk streaming for training |
| **BC training cost** | s/epoch, peak VRAM (Lab 11c) | Whether retraining sweeps are practical |
| **Success with CI** | `eval_seeds.py`: per seed, Wilson interval, spread across seeds (Lab 11d) | The only number worth reporting |
| **Round-trip latency** | p50 / p99 RTT vs payload and placement (Lab 11e) | Real-robot budget and eval throughput |
| **Effective env-steps/s** | Lab 11e, batched vs per-env | Cost of a naive client |

---

## 8. Ship it

Commit to `isaac_bench/il/` and `isaac_bench/remote/`:

* `il/DATA.md`: devices used, demos attempted/kept, minutes per demo, file sizes, Mimic generation table (`num_envs`, demos/min, success rate, peak VRAM)
* `il/eval_seeds.py` and `il/eval_results.csv`: checkpoint × seed × successes with Wilson intervals, and the checkpoint you chose and why
* `remote/policy_server.py`, `remote/eval_client.py`, `remote/latency.csv` (N, batched/per-env, payload, placement, p50/p99 RTT, env-steps/s)
* `remote/LATENCY.md`: one paragraph applying the §4 formula to a policy acting at your target control rate with your measured LAN and cloud RTTs. Would it be usable on a real robot, and with what chunk horizon?

---

## Exit criteria

You can move on when you can:

* record, replay and validate a teleoperated dataset with the 3.0 commands, and say what MCAP is and is not for
* explain Mimic's subtask transform and its task-space requirement, and size `--num_envs` for generation on your card
* report a BC policy's success rate as per-seed Wilson intervals over fixed seeds across several checkpoints
* explain why a VLA evaluation on an 8 GB card must be client-server, and what Arena, RLinf and LEAPP each contribute
* predict, then measure, how batching, chunk horizon and payload size change round-trip cost and env-steps/s

---

## Self-check

1. Your teammate's annotated dataset fails Mimic generation almost every time. The demos have long pauses and six annotated subtasks for a two-object task. What two changes do you make before touching any Mimic parameter?
2. Checkpoint A scores 27/50 and checkpoint B 31/50 on seed 0. Your lead wants to ship B. What do you compute and run before agreeing, and what would change your answer?
3. You try to run π0.5 and Isaac Sim's visuomotor stack task on the same 8 GB 4060 and get OOM. Lay out the client-server alternative: what runs where, what goes over the wire each step, and which number you measure first.
4. With 64 envs, a per-env chunk-replay client, \( H = 8 \) and a 40 ms RTT, estimate the added time per env step and the effective slowdown. How would batching change it?
5. A recorded dataset replays fine with `physics=isaacsim_physx presets=diffik` but fails without those tokens. Why, and where should that information live?
6. A colleague wants to use the MCAP files from `--mcap_record_path` as Mimic input "because they are smaller". What do you tell them?

---

## References

* Isaac Lab 3.0 — [Isaac Lab Mimic: data generation and imitation learning](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/features/imitation-learning/teleop_imitation.html), [SkillGen](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/features/imitation-learning/skillgen.html)
* Isaac Lab 3.0 — [Isaac Capture (teleop framework)](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/features/isaac_teleop.html), [CloudXR teleoperation](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/how-to/cloudxr_teleoperation.html), [Isaac Capture on GitHub](https://github.com/NVIDIA/IsaacCapture)
* Isaac Lab 3.0 — [RL post-training for VLA models (RLinf)](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/experimental-features/rlinf_vla_posttraining.html), [Deploy policies with LEAPP](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/policy_deployment/05_leapp/exporting_policies_with_leapp.html), [LEAPP docs](https://nvidia-isaac.github.io/leapp/)
* Isaac Lab 3.0 — [migration guide](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/migration/migrating_to_isaaclab_3-0.html), [v3.0.0-EA release notes](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-EA)
* Isaac Lab-Arena — [GitHub (README, v0.3 alpha)](https://github.com/isaac-sim/IsaacLab-Arena), [docs 0.3.0](https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.0/index.html), [running openpi (π0/π0.5) client-server](https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.0/pages/quickstart/running_a_real_policy/openpi.html)
* robomimic — [docs](https://robomimic.github.io/); Mandlekar et al., "What Matters in Learning from Offline Human Demonstrations for Robot Manipulation," CoRL 2021 — [arXiv:2108.03298](https://arxiv.org/abs/2108.03298)
* Mandlekar et al., "MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations," CoRL 2023 — [arXiv:2310.17596](https://arxiv.org/abs/2310.17596)
* RLinf — [GitHub](https://github.com/RLinf/RLinf)

---

## Next in this special course

* Next: [Lecture 12 — Performance, Sim-to-Real, and Scale-Out](Lecture-12.md)
* Previous: [Lecture 10b — Worked Example: Spot Waypoint Navigation](Lecture-10b.md)
* Back: [Isaac Sim and Isaac Lab — Overview](README.md)
