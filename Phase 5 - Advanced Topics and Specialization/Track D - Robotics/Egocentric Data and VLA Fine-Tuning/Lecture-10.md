# Lecture 10 — Building the Fine-Tuning Dataset: Formats, Normalization and Mixtures

Put retargeted human actions and robot demos into one dataset your chosen stack can load, with normalization statistics that match what deployment loads and a mixture plan that keeps the robot data visible.

**Prerequisites:** [Lecture 04](Lecture-04.md) (the LeRobot format), [Lecture 07](Lecture-07.md) (retargeting), [Lecture 09](Lecture-09.md) (stack choice), [VLA Foundations Lecture 14](../VLA%20Foundations%20to%20Frontier%202026/Lecture-14.md) (schemas, splits, mixtures) and [Lecture 16](../VLA%20Foundations%20to%20Frontier%202026/Lecture-16.md) (normalization). **Effort:** reading 1.5 hours, lab 3 hours. **Hardware:** CPU; converting real data is disk- and decode-bound (Lecture 04), not GPU-bound. **Roles:** robot learning data engineer, post-training engineer.

**Misconception this lecture corrects:** "normalization statistics are a preprocessing detail; any reasonable stats file will do." The stats are part of the checkpoint. The policy's outputs are unnormalized with them, so a stats file from another dataset, another mixture or another dimension order sends actions on the wrong scale to the robot, and nothing raises an error.

## 1. One schema for two embodiments

| Field | Robot source | Human source (Lecture 07) | Must match |
|---|---|---|---|
| `action` | The recorded command, re-expressed as relative EEF plus gripper if it was recorded in joints | 7-D: translation, rotation vector, gripper opening in [0, 1] | Dimension order, units, frame, gripper convention |
| `observation.state` | Proprioception: EEF pose and gripper | The retargeted virtual EEF pose and opening: an estimate, not a measurement | The same layout |
| Images | Wrist and scene cameras | A head camera | Key names and resolution; how a stack treats a view that one source lacks is stack-specific, so check its docs |
| `task` | The instruction string | The same string for the same task (`meta/tasks.parquet` maps it to `task_index`) | Exact text |
| `timestamp`, fps | The control rate | Resampled to the same rate (`retarget.resample`) | 1/fps within LeRobot's 1e-4 s load-time tolerance |

There are two ways to combine sources. Put both in **one action space**, as GR00T N1.7 does with relative EEF actions for robots and humans. Or keep them as **separate embodiments**, as π0.5's human-to-robot work does. The two are not exclusive: GR00T N1.7's pretraining tags include human-specific embodiments such as `REAL_R1_PRO_SHARPA_HUMAN` alongside the shared action space. GR00T's `meta/modality.json` names slices of the concatenated state and action arrays and renames video keys, so that is where its layout is declared.

**Tested here** (from the repository root; `EGO_COURSE` as in the [Lab guide](Lab-Guide.md)):

```bash
PYTHONPATH="$EGO_COURSE/labs" python3 - <<'EOF'
from dataset_contract import validate_frame
vec = lambda n: {'dtype': 'float32', 'shape': [n]}
schema = {'observation.state': vec(7), 'action': vec(7), 'task': {'dtype': 'string', 'shape': [1]}}
frames = {'robot': ([0.45, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], [0.003, 0.0, -0.001, 0.0, 0.0, 0.01, 0.9]),
          'human, tracking lost': ([0.5, 0.1, 0.25, 0.0, 0.0, 0.0, 0.8], [0.004, 0.0, 0.0, 0.0, 0.0, 0.0, float('nan')]),
          'robot, joint-space': ([0.0] * 6, [0.0] * 6)}
for name, (state, action) in frames.items():
    print(name, validate_frame({'observation.state': state, 'action': action, 'task': 'put the cup in the bowl'}, schema))
EOF
```

```text
robot []
human, tracking lost ['action: non-finite or non-numeric value']
robot, joint-space ['observation.state: expected shape [7], got 6', 'action: expected shape [7], got 6']
```

**Negative cases:** a frame where hand tracking was lost carries a NaN gripper value; drop or mask it before writing, and count it in the data card. A robot recorded in joint space fails the 7-D schema, which forces the action-space decision from Lecture 09, step 4.

## 2. Where each stack keeps its statistics

| Stack | File | Contents | Computed by |
|---|---|---|---|
| LeRobot v0.6.1 | `meta/stats.json` | Per feature: min, max, mean, std, count, q01, q10, q50, q90, q99; quantiles from 5000-bin histograms | Per episode, then merged; rebuilt by `recompute_stats` |
| openpi `215abfb` | `assets/<config_name>/<repo_id>/norm_stats.json` | `state` and `actions` only: mean, std, q01, q99 | `scripts/compute_norm_stats.py`, over the data *after* the repack and data transforms |
| Isaac-GR00T N1.7 | `meta/stats.json` plus `meta/relative_stats.json` (per step, shape `(action_horizon, D)`) | Min-max by default; optional `mean_std_embedding_keys`; sin/cos for radian states | `gr00t/data/stats.py` per `--embodiment-tag`; rerun after changing `delta_indices` |
| OpenVLA-OFT `e4287e9` | `dataset_statistics.json` in the checkpoint directory | Includes q01 and q99 | `get_dataset_statistics` |

The LeRobot v3 doc describes `stats.json` as mean/std/min/max only, but the code also writes count and quantiles, so open the file rather than trusting the doc. Because openpi computes stats after its transforms, they describe the transformed actions: change a transform and you must recompute them. For conversion, see Lecture 04; GR00T also ships `scripts/lerobot_conversion/convert_v3_to_v2.py`, and OpenVLA-OFT needs RLDS plus registration in its three config files.

**Upstream documented, not tested here** (add your dataset arguments as each doc shows):

```bash
lerobot-edit-dataset --operation.type recompute_stats                       # LeRobot v0.6.1, using_dataset_tools.mdx
uv run scripts/compute_norm_stats.py --config-name pi05_libero              # openpi README
python gr00t/data/stats.py --dataset-path <p> --embodiment-tag <tag> [--modality-config-path …]   # Isaac-GR00T data_config.md
```

## 3. Merging, parity and normalization

Mean and standard deviation merge exactly from per-episode summaries with counts \( n_i \). Quantiles do not, so `aggregate_stats` averages them by count and sets `quantiles_approximate`:

$$
\mu = \frac{\sum_i n_i \mu_i}{\sum_i n_i}, \qquad
\sigma^2 = \frac{\sum_i n_i\,(\sigma_i^2 + \mu_i^2)}{\sum_i n_i} - \mu^2
$$

The lab's MEAN_STD mode is \( y = (x - \mu)/\sigma \). Its range modes are \( y = 2(x - lo)/(hi - lo) - 1 \), with \( lo, hi \) set to min and max (MIN_MAX) or q01 and q99 (QUANTILES). OpenVLA-OFT's `BOUNDS_Q99` also maps q01..q99 to [-1, 1]. The **synthetic fixture** below has three action dimensions (dx, dz, gripper) from five robot and five human episodes, with human steps three times larger.

**Tested here:**

```bash
PYTHONPATH="$EGO_COURSE/labs" python3 - <<'EOF'
import random
from dataset_contract import feature_stats, aggregate_stats, stats_parity, normalize, unnormalize
rng = random.Random(0)
def ep(step_std, n=300):  # [dx, dz] in metres per frame, gripper opening in [0, 1]
    return [[rng.gauss(0.0, step_std), rng.gauss(0.0, step_std), rng.random()] for _ in range(n)]
robot, human = [ep(0.003) for _ in range(5)], [ep(0.009) for _ in range(5)]
full = feature_stats([v for e in robot + human for v in e])
mixed = aggregate_stats([feature_stats(e) for e in robot + human])
print('std  full', [round(v, 6) for v in full['std']], ' merged', [round(v, 6) for v in mixed['std']])
print('q99  full', [round(v, 5) for v in full['q99']], ' merged', [round(v, 5) for v in mixed['q99']],
      ' approximate:', mixed['quantiles_approximate'])
rob = feature_stats([v for e in robot for v in e])
print('robot-only std', [round(v, 6) for v in rob['std']], ' parity issues vs mixed:', len(stats_parity(rob, mixed)))
swapped = {k: [v[2], v[0], v[1]] for k, v in rob.items() if isinstance(v, list)}
print('swapped dims:', stats_parity(rob, swapped, keys=('mean',))[0], '...')
x = [0.012, -0.004, 1.0]      # one human-derived step, 4x the robot's typical dx
for mode in ('MEAN_STD', 'MIN_MAX', 'QUANTILES'):
    y = normalize(x, rob, mode)
    ok = all(abs(a - b) < 1e-9 for a, b in zip(x, unnormalize(y, rob, mode)))
    print(f'{mode:9s}', [round(v, 2) for v in y], 'round trip:', ok)
del rob['q01'], rob['q99']    # a stats.json holding only mean/std/min/max
try:
    normalize(x, rob, 'QUANTILES')
except ValueError as e:
    print('ValueError:', e)
EOF
```

```text
std  full [0.006692, 0.006667, 0.286646]  merged [0.006692, 0.006667, 0.286646]
q99  full [0.01764, 0.01755, 0.99009]  merged [0.0128, 0.01279, 0.98846]  approximate: True
robot-only std [0.003, 0.003075, 0.283606]  parity issues vs mixed: 17
swapped dims: mean[0]: 1.0578491094338661e-05 vs 0.5055761854151599 ...
MEAN_STD  [4.0, -1.29, 1.74] round trip: True
MIN_MAX   [1.22, -0.21, 1.0] round trip: True
QUANTILES [1.76, -0.6, 1.02] round trip: True
ValueError: QUANTILES normalization mode requires q01 and q99 stats
```

* **Merging.** The merged std is exact. The averaged q99 (0.0128) is far below the true q99 of the union (0.01764): the tail comes from the human episodes, and the robot episodes pull the average down. When a mode uses quantiles, recompute them from the data.
* **Parity.** `stats_parity` catches both negative cases: a converter that reordered dimensions, and robot-only stats compared with stats over the mix (17 of the 18 mean, std, min, max, q01 and q99 entries differ). Run it on the stats the checkpoint trained with against the file deployment loads; anything but `[]` blocks the release. Its default keys include q01 and q99, because a quantile-normalized stack uses only those two.
* **Per-embodiment stats.** Shared stats over the mix scale robot actions by a std more than twice the robot's own (0.0067 vs 0.0030). Under robot stats, the human step lands 4 standard deviations out and outside [-1, 1] in both range modes. Either give the human source its own stats, or recheck the time scaling from Lecture 07. GR00T computes stats per embodiment tag; openpi stores them per config and repo ID and can reload pretraining stats for known robots (asset IDs such as `franka`, `ur5e`, `droid`). Whichever you choose, the robot at deployment loads exactly the stats its embodiment was trained with.
* **Missing quantiles.** LeRobot's π0.5 uses QUANTILES and fails on a `stats.json` without them: `ValueError: QUANTILES normalization mode requires q01 and q99 stats` (the lab raises the same message). The documented fixes are `lerobot-edit-dataset --operation.type recompute_stats` or overriding the mode to MEAN_STD. The override changes how the pretrained model sees actions, so evaluate it like any other change.
* **Round trips.** Every one above is exact, but only because the same stats are used in both directions.

## 4. The mixture plan

Lecture 08 showed that sampling by size drowns the robot set. Fix weights instead, and record what they imply. This plan uses the course budget (45,000 robot frames, and about 4 h or 432,000 frames of retarget-ready human data, an assumption from Lecture 01) with the batch and step counts of LeRobot's SmolVLA command (64 × 20,000).

**Tested here:**

```bash
PYTHONPATH="$EGO_COURSE/labs" python3 - <<'EOF'
from budget_eval import mixture_batches, epochs_seen
sources, weights = {'robot_task': 45_000, 'human_retargeted': 432_000}, {'robot_task': 0.4, 'human_retargeted': 0.6}
counts = mixture_batches(sources, weights, batch_size=64, num_batches=20_000, seed=0)
for name, n in sources.items():
    print(f'{name:17s} drawn {counts[name]:7d}  share {counts[name] / 1_280_000:.3f}  epochs {epochs_seen(n, counts[name]):.2f}')
EOF
```

```text
robot_task        drawn  512684  share 0.401  epochs 11.39
human_retargeted  drawn  767316  share 0.599  epochs 1.78
```

Each robot frame is seen about 11 times, so select checkpoints by closed-loop evaluation, not training loss (Lecture 11). The plan also names a **robot-only baseline** and records whether it holds steps or robot epochs fixed, which Lecture 12 needs for the paired comparison. The lab models the sampling you intend; confirm how your stack's trainer actually mixes sources (OpenVLA-OFT registers mixtures in `mixtures.py`).

## 5. Exercise

1. **Worked example → run Sections 1, 3 and 4.** Write the schema for the course task and record every output.
2. **Guided variation.** Swap two action dimensions in the human converter and show that `stats_parity` catches it. Then choose shared or per-embodiment stats for your mix, and justify the choice with the robot std ratio.
3. **Independent transfer.** Convert your Lecture 07 stream and your robot demos into your stack's format (CPU track: lab-format frames). Compute stats with the stack's tool, compare them with `feature_stats` on the same frames (expect small quantile differences, because LeRobot uses histograms), and write the mixture plan for two weights.

**Deliverable:** the converted dataset (or a lab-format sample of it) plus `dataset_build_record.md`: the schema, validation counts, the stats file path and SHA-256 hash, the parity result and the mixture plan. **Acceptance:** every written frame passes `validate_frame`, and dropped frames are counted with a reason; the stats file has every key the chosen mode needs (q01 and q99 for QUANTILES or BOUNDS_Q99); `stats_parity` between the training stats and the file deployment loads returns `[]`; the plan lists weights, steps × batch, epochs per source and the robot-only baseline.

## Failure clinic and self-check

**Failure clinic:** NaN gripper values from lost tracking are written into the dataset. Joint-space robot actions and EEF human actions share one `action` key. A transform changes, but the old `norm_stats.json` still ships with the checkpoint. π0.5 is trained on a `stats.json` without quantiles. Per-episode quantiles are averaged across embodiments. A converter reorders dimensions. GR00T's `relative_stats.json` goes stale after `delta_indices` changes. The mixture is sampled by size.

1. **Why is the merged std exact but the merged q99 not?** Means and second moments combine by counts. A quantile depends on the whole distribution, so per-episode quantiles cannot be combined exactly; recompute them from data.
2. **A human step normalizes to 4.0 under robot MEAN_STD stats. What does that tell you?** It sits far outside the robot's action distribution. Use per-embodiment stats, or recheck the time scaling and units of the retargeting.
3. **LeRobot's π0.5 raises the q01/q99 error. What are the two fixes?** Recompute the stats with `lerobot-edit-dataset --operation.type recompute_stats`, or override the mode to MEAN_STD and evaluate the effect, since QUANTILES is π0.5's default in both LeRobot and openpi.
4. **Robot-only and mixed stats differ in 17 entries. Which does the deployed robot load?** The stats its embodiment was trained with, whichever choice you made, verified by `stats_parity` returning `[]`.
5. **The plan shows the robot set repeated about 11 times. What do you watch for?** Overfitting to the robot demos. Select checkpoints by closed-loop success on held-out conditions, and compare with a lower robot weight.

[Previous: choosing a base model and stack](Lecture-09.md) · [Course index](README.md) · [Next: fine-tuning mechanics](Lecture-11.md)
