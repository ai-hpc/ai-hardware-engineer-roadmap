# Lecture 04 — The Data Pipeline: Decoding, Sharding and the LeRobot Format

Budget the disk and decode throughput a dataset needs, lay it out in the format your target stack actually reads, and validate frames and timestamps before a conversion reaches training.

**Prerequisites:** [Lecture 01, Section 5](Lecture-01.md) (hours to frames to bytes), [Lecture 03](Lecture-03.md) (clocks and timestamps), [VLA Foundations Lecture 14](../VLA%20Foundations%20to%20Frontier%202026/Lecture-14.md) (dataset schemas, splits and mixtures) and [Lecture 18](../VLA%20Foundations%20to%20Frontier%202026/Lecture-18.md) (training infrastructure). **Effort:** reading 1.5 hours, lab 1.5 hours. **Hardware:** CPU track for the lab; the optional decode measurement runs on whichever machine will feed your GPU. **Roles:** ML data platform engineer, robot learning data engineer.

**Misconception this lecture corrects:** "the data is in LeRobot format, so any LeRobot-based stack will read it." On the check date three different on-disk layouts are in use (LeRobot v3.0, v2.1 pinned by openpi, and GR00T's v2 plus `modality.json`), and a dataset that loads can still have frames at the wrong rate or of the wrong shape.

## 1. Storage: budget the parts you need

These are **documented sizes**, not measurements:

| Source | Documented size | Notes and worked arithmetic |
|---|---|---|
| Ego4D full-scale video | "~7 TB" (start-here docs) vs "5TB" (CLI docs) | **Conflict**: budget the larger. Annotations "~2 GB" |
| Ego-Exo4D | "default" 12,112.778 GB; "all" 38,449.753 GB | The `egoexo` CLI selects `--parts`, `--benchmarks` and `--views ego exo`; 448 px downscaled takes exist |
| EgoDex | "2.0 TB of storage on disk" (paper) | README zips: 5 × 300 GB + 16 GB + 200 GB = 1,716 GB |
| HoloAssist | Hand pose 219.24 GB, eye gaze 2.45 GB, compressed video 144.62 GB, AHAT depth 560.46 GB | All four: 926.77 GB. Hands plus video: 363.86 GB, 39% of the total |
| ADT | About 3.5 TB without MPS | Research only (Lecture 02) |
| UMI `cup_in_the_wild` | 17,989,829,476 bytes (about 18.0 GB) | One zarr zip |

Download only the streams your pipeline consumes: a HoloAssist project that needs hands and RGB skips 61% of the bytes by leaving out depth and gaze. Then add working space: conversion writes a second copy, so **assume** the converted dataset is about as large as the video you keep, and measure it after converting one shard.

**Upstream documented, not tested here** (run only after your Ego4D license is approved; credentials expire after 14 days):

```bash
pip install ego4d==1.7.3          # latest on PyPI at the check date; the CLI defaults to v2_1
ego4d --output_directory="~/ego4d_data" --datasets full_scale annotations
```

## 2. Decode: frames per second the trainer consumes

Lecture 01 showed why pipelines decode video on the fly. The decode rate you need is set by the trainer:

$$
\text{decoded frames/s} = \text{batch size} \times \text{steps/s} \times \text{cameras} \times \text{frames per camera per sample}
$$

**Worked arithmetic with stated assumptions:** batch 64, 2 optimizer steps/s, 2 cameras, 1 frame per camera gives 256 decoded frames/s. Over one hour of training that is 921,600 frames, the same as 8.5 hours of 30 fps video. If one dataloader worker sustains \( R \) frames/s on your machine (measure it; this course does not supply a number), you need at least \( \lceil 256 / R \rceil \) workers, each with its own decoder, CPU core and memory. GR00T's documented fine-tuning command sets `--dataloader-num-workers 4`; check your stack's equivalent setting at its pinned version.

Resolution is the other lever. One decoded 1404 × 1404 RGB frame (Ego-Exo4D's Aria ego view) is 5,913,648 bytes; a 224 × 224 model input is 150,528 bytes, 39.3× fewer. When the model input is small, start from the downscaled variants (Ego4D's 540ss videos, Ego-Exo4D's 448 px takes) and scale the intrinsics with them (Lecture 03). Random access also costs: a sampled frame is decoded from the preceding keyframe of its video file, so record your encoder's keyframe interval in the budget. On an RTX 4060 8 GB workstation the CPU cores doing this decoding, not the GPU, can set the pace. If GPU utilization is low while the dataloader waits, add workers, cut cameras or resolution, or pre-resize once.

## 3. The LeRobot v3.0 layout

LeRobot's `CODEBASE_VERSION` is v3.0, included in `lerobot >= 0.4.0`; the latest release on the check date is **v0.6.1**. Cite the v0.6.1 docs, because main is ahead of the release.

**Upstream documented, not tested here** (`lerobot-dataset-v3.mdx` and `datasets/utils.py` at v0.6.1):

```text
meta/info.json                                   schema, fps, codebase_version, path templates
meta/stats.json                                  global per-feature statistics
meta/tasks.parquet                               task string -> task_index
meta/episodes/chunk-XXX/file-YYY.parquet         one record per episode
data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet          frames of many episodes
videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4  video of many episodes
```

* **Many episodes per file.** v2.x wrote one file per episode. In that layout EgoDex's 338,000 demonstrations would be 338,000 Parquet files plus one MP4 per episode per camera. v3.0 bounds files by size instead: data files ≤ 100 MB, video files ≤ 200 MB (a dataset can override this; `svla_so101_pickplace` uses 500), and `DEFAULT_CHUNK_SIZE=1000` files per chunk.
* **Indexing.** Every frame row carries `timestamp`, `frame_index` (within the episode), `episode_index`, `index` (global) and `task_index`. Every episode row carries `dataset_from_index` / `dataset_to_index`, `data/chunk_index` and `data/file_index`, `videos/<key>/chunk_index|file_index` with from/to timestamps, and `meta/episodes/chunk_index|file_index`. An episode is a slice of a shared file, so a wrong index range silently mixes two episodes.
* **Schema example.** `lerobot/svla_so101_pickplace` declares `codebase_version: v3.0`, `robot_type: so100_follower`, `fps: 30`, `total_episodes: 50`, `total_frames: 11939`, `action` and `observation.state` as `float32 [6]`, and `observation.images.{up,side}` as `video [480, 640, 3]`.
* **Stats.** `stats.json` holds `min`, `max`, `mean`, `std`, `count` and quantiles `q01`–`q99` from 5,000-bin histograms, computed per episode and merged. The v3 doc lists only mean/std/min/max (a minor conflict with the code). Lecture 10 makes these a gate.

## 4. One dataset, three on-disk layouts

| Stack (pin) | What it reads | Consequence |
|---|---|---|
| LeRobot v0.6.1 | LeRobot v3.0 | Native |
| openpi `215abfb` | LeRobot, via a pinned lerobot commit `0cf8648` whose `CODEBASE_VERSION` is v2.1 | `physical-intelligence/libero` is v2.0 while `lerobot/libero` is v3.0. **[est]** A v3.0 dataset probably won't load without conversion or an upgrade (not tested) |
| Isaac-GR00T `n1.7-release` | "a flavor of the LeRobot v2 dataset format with an additional `meta/modality.json`": `meta/episodes.jsonl`, `meta/tasks.jsonl`, one Parquet/MP4 per episode | Convert with `scripts/lerobot_conversion/convert_v3_to_v2.py`; `modality.json` maps `{start, end}` slices of `observation.state` and `action` to named fields |
| OpenVLA-OFT `e4287e9` | RLDS | Register a new dataset in `configs.py`, `transforms.py` and `mixtures.py` |

Choose the stack (Lecture 09) before you choose the on-disk version. Keep one canonical copy and a conversion script per stack, and record the version each copy was written in.

**Upstream documented, not tested here** (LeRobot v0.6.1):

```bash
pip install lerobot==0.6.1
python -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id=<HF_USER/DATASET_ID>
lerobot-edit-dataset --operation.type recompute_stats   # add the dataset flags from using_dataset_tools.mdx
```

`lerobot-edit-dataset` can also delete episodes, split, merge, remove features, modify tasks, convert images to video, re-encode, and print `info` (`--operation.show_features true`). Writers must call `dataset.finalize()` before `push_to_hub()`, or the Parquet files are left incomplete. Inspect results with `lerobot-dataset-viz` or the web visualizer Space; for large ports, see `examples/port_datasets/port_droid.py` and `slurm_port_shards.py` (`porting_datasets_v3.mdx`).

## 5. Lab: validate before you convert everything

This course's source check found no standalone validation CLI (`lerobot-info` prints environment information only). Validation is built into the library: `validate_frame`, `validate_feature_dtype_and_shape` and `validate_episode_buffer` at write time; `check_version_compatibility` and a `tolerance_s=1e-4` timestamp check at load time, where consecutive frames must be 1/fps ± tolerance apart. [`labs/dataset_contract.py`](labs/dataset_contract.py) mirrors the frame and timestamp rules on plain Python data, so you can test a converter on one synthetic episode before writing a terabyte. `validate_frame` checks one frame (a scalar float is accepted for a shape-`[1]` feature such as `timestamp`, and a bool is rejected as a number); the timing rule needs the whole sequence, so `check_timestamps` runs on the timestamp list separately.

**Tested here** (from the repository root, `EGO_COURSE` as in the [Lab guide](Lab-Guide.md); a synthetic episode, where the bad frame and two of the four timestamp lists are negative cases):

```bash
PYTHONPATH="$EGO_COURSE/labs" python3 - <<'EOF'
from dataset_contract import validate_frame, check_timestamps
schema = {"observation.state": {"dtype": "float32", "shape": [7]},
          "action": {"dtype": "float32", "shape": [7]},
          "episode_index": {"dtype": "int64", "shape": [1]},
          "frame_index": {"dtype": "int64", "shape": [1]},
          "task": {"dtype": "string", "shape": [1]}}
fps = 30.0
episode = [{"observation.state": [0.0] * 7, "action": [0.01 * k] * 6 + [1.0], "episode_index": 0,
            "frame_index": k, "task": "put the cup in the bowl"} for k in range(90)]
print("good episode problems:", [p for f in episode for p in validate_frame(f, schema)])
bad = {"observation.state": [0.0] * 6, "action": [0.0] * 6 + [float("nan")], "episode_index": 0, "frame_idx": 12}
print("bad frame:", validate_frame(bad, schema))
ts = [k / fps for k in range(90)]
for name, t in (("exact k/30", ts), ("rounded to 1 ms", [round(x, 3) for x in ts]),
                ("one frame dropped", ts[:40] + ts[41:]), ("29.97 fps declared 30", [k / 29.97 for k in range(90)])):
    print(f"{name:22s} check_timestamps -> {check_timestamps(t, fps)}")
EOF
```

```text
good episode problems: []
bad frame: ["missing feature 'frame_index'", "missing feature 'task'", "unexpected feature 'frame_idx'", 'observation.state: expected shape [7], got 6', 'action: non-finite or non-numeric value']
exact k/30             check_timestamps -> True
rounded to 1 ms        check_timestamps -> False
one frame dropped      check_timestamps -> False
29.97 fps declared 30  check_timestamps -> True
```

* **Shape, NaN and names.** One bad frame produces five problems: a 6-D state against a 7-D schema, a NaN in the gripper channel, a renamed key and a missing task. Validate every frame a converter writes, not a sample.
* **Millisecond timestamps fail.** Rounded to 1 ms, 30 fps gaps become 0.033 or 0.034 s, at least 3.3e-4 s away from 1/30, more than the 1e-4 s tolerance. After resampling to a fixed rate, write `frame_index / fps` at full float precision.
* **Slow drift passes.** A 29.97 fps stream declared as 30 is off by 3.34e-5 s per gap, inside the tolerance, yet it falls a full frame behind after 999 frames (about 33 s). A consecutive-gap check cannot see drift. Add a duration check: \( |t_{N-1} - t_0 - (N-1)/\text{fps}| \) below one frame period.

## 6. Exercise

**Exercise / pseudocode** (CPU only):

1. **Worked example → your sources.** Repeat Sections 1 and 2 for the sources in your `dataset_selection.md`: documented sizes with their source, the streams you will download, working space, and the decode arithmetic with every assumption labeled.
2. **Guided variation.** Run the tested snippet, then break the episode in new ways: a string where a float belongs, an extra camera key, `fps` changed to 10 without changing the timestamps. Write the duration check from Section 5 (**exercise / pseudocode**) and show that it rejects the 29.97 fps list while accepting the exact one.
3. **Independent transfer.** Write the schema your target stack expects: v3.0 `info.json` features, or GR00T's `modality.json` slices for your state and action. Validate every frame and the timestamps of one converted episode of your own data. Optionally, measure \( R \) by timing how fast your training machine decodes one of your clips at the model's input resolution, and redo the worker count.

**Deliverable:** `pipeline_budget.md`, the storage and decode budget plus a validated layout: the storage table, the decode arithmetic, the target layout and version for each stack with its pin, and the validation output for one episode. **Acceptance:** every size names its source, and conflicts are shown with the larger value budgeted; every decode input is labeled measured or assumed; the good episode reports no problems; each injected fault is either caught or listed as not caught by the lab (drift), with the check that would catch it.

## Failure clinic and self-check

**Failure clinic:** a team downloads Ego-Exo4D "all" (about 38 TB) when it needed ego video and hand poses. A v3.0 dataset is handed to GR00T, which expects v2 plus `modality.json`. A writer skips `finalize()` and pushes incomplete Parquet files. Timestamps stored in milliseconds fail the load-time check. A 29.97 fps camera is declared 30 fps and the actions slide against the video. Full-resolution 1404 × 1404 frames are decoded for a 224 × 224 model and the GPU waits. Flags are copied from the LeRobot main-branch docs (such as SmolVLA's `--rename_map`), which are ahead of the v0.6.1 docs.

1. **Why did LeRobot v3.0 move to many episodes per file?** One file per episode means hundreds of thousands of small files at egocentric scale. v3.0 caps files by size (100 MB data, 200 MB video by default) and finds an episode through its index range.
2. **Your converter writes timestamps rounded to the millisecond at 30 fps. What happens?** Consecutive gaps differ from 1/30 s by at least 3.3e-4 s, beyond the 1e-4 s tolerance, so the load-time check rejects them. Write `frame_index / fps`.
3. **A 29.97 fps stream labeled 30 fps passes `check_timestamps`. Is it fine?** No. Each gap is inside the tolerance, but the error accumulates to a full frame in about 33 s. Check total duration as well as gaps.
4. **You plan to fine-tune with LeRobot SmolVLA, openpi π0 and GR00T N1.7. How many on-disk layouts do you need?** Three on the check date: v3.0 for LeRobot, v2.1 for openpi's pinned LeRobot, and v2 plus `modality.json` for GR00T. Keep one canonical copy and a conversion script for each.
5. **The trainer needs 256 frames/s. Suppose one worker sustains a hypothetical R = 40 frames/s. How many workers, and what else do you check?** At least \( \lceil 256/40 \rceil = 7 \), each needing a CPU core and memory. If the machine lacks the cores, lower the resolution, the camera count or the batch rate instead.

[Previous: capture geometry](Lecture-03.md) · [Course index](README.md) · [Next: auditing a dataset](Lecture-05.md)
