# Egocentric data and fine-tuning laboratory guide

[Course overview](README.md) · [Capstone](Lecture-13.md) · [Sources](Sources.md)

The CPU labs implement the checks that decide whether egocentric data and a fine-tuning dataset can be trusted: frames and camera models, dataset auditing, human-to-robot retargeting, schema and normalization contracts, training-memory arithmetic and paired evaluation statistics. They use only Python's standard library, download no data or weights, and never send robot commands. GPU work (hand-pose estimation, VLA fine-tuning) is described in the lectures and labeled **upstream documented, not tested here** until you run it.

## What is implemented and tested here

| File | Implemented behavior | What it does not establish |
|---|---|---|
| [ego_geometry.py](labs/ego_geometry.py) | Quaternions (wxyz), SE(3) composition and inverse, slerp, pinhole and Kannala–Brandt fisheye projection with Newton unprojection, the camera → device → world chain | Project Aria's exact `Fisheye624` model (15 parameters: more radial terms, tangential and thin-prism distortion); calibration estimation |
| [ego_audit.py](labs/ego_audit.py) | Frame-timing check with a tolerance, dropped/non-monotonic frames, duplicate clips by content hash, split leakage by participant or location, grouped splits, label long-tail statistics, Cohen's kappa | Visual near-duplicate detection, privacy review, label correctness |
| [retarget.py](labs/retarget.py) | Control-rate resampling, wrist → virtual gripper poses in a base frame, relative end-effector actions, thumb–index gripper mapping, workspace and speed feasibility report | Inverse kinematics, collision checking, contact physics, whether a policy trained on the result works |
| [dataset_contract.py](labs/dataset_contract.py) | Frame schema validation, per-feature statistics (min, max, mean, std, count, q01…q99), exact merge of count/min/max/mean/std across episodes, MEAN_STD / MIN_MAX / QUANTILES / IDENTITY normalization and inverse, train-vs-deploy stats parity (q01/q99 included) | Parquet/MP4 I/O; exact quantile merging (merged quantiles are count-weighted averages, flagged `quantiles_approximate`); LeRobot's histogram-based quantile estimates |
| [budget_eval.py](labs/budget_eval.py) | Weights + gradients + optimizer-state arithmetic, LoRA parameter counts, a necessary-condition fit check, weighted co-training sampling, Wilson intervals, exact McNemar test for paired episodes | Activation memory, measured peak VRAM, training dynamics |
| [test_geometry_retarget.py](labs/test_geometry_retarget.py), [test_data_pipeline.py](labs/test_data_pipeline.py) | 29 tests, including negative cases: wrong intrinsics, swapped calibration, leaking splits, missing quantiles, mismatched stats, unpaired evaluation | GPU performance or real-data quality |

**Local validation:** the suite was executed on October 26, 2026 with Python 3.12 (29 tests, all passing). It also runs in this roadmap's CI through `scripts/run_portable_checks.py`.

## Run from the repository root

These commands are **tested here**. The quoted path handles the spaces in the directory name.

```bash
EGO_COURSE='Phase 5 - Advanced Topics and Specialization/Track D - Robotics/Egocentric Data and VLA Fine-Tuning'
python3 -m unittest discover -s "$EGO_COURSE/labs" -p 'test_*.py' -v
```

To use a module interactively:

```bash
cd "$EGO_COURSE/labs"
python3 -c "from budget_eval import training_state_bytes; print(training_state_bytes(450e6, 450e6))"
```

## How the labs map to lectures

| Lecture | Lab module | Worked example → guided variation → transfer |
|---|---|---|
| [03](Lecture-03.md) | `ego_geometry` | Trace a point camera → world; break the intrinsics; apply the chain to your own calibration |
| [04](Lecture-04.md) | `dataset_contract` (`validate_frame`, `check_timestamps`) | Validate a synthetic episode; break its schema and frame timing; validate one converted episode of your own data |
| [05](Lecture-05.md) | `ego_audit` | Audit the synthetic episodes; fix a leaking split; audit a real metadata file |
| [06](Lecture-06.md) | `retarget` (`moving_average`) | Measure dropout and jitter on a synthetic hand track; write a gap-fill rule; report on a hand track you are licensed to use |
| [07](Lecture-07.md) | `retarget` | Retarget the synthetic reach; swap the calibration and watch the report; retarget real hand tracks |
| [08](Lecture-08.md) | `budget_eval` (mixtures) | Compare size-proportional and weighted mixtures; rerun with the retarget-ready hours; plan the mixture for your method |
| [09](Lecture-09.md) | `budget_eval` (memory) | Full fine-tuning state for the candidate checkpoints, as a necessary condition beside each documented requirement |
| [10](Lecture-10.md) | `dataset_contract`, `budget_eval` (mixtures) | Compute and merge stats; break parity; build the co-training mixture for your dataset |
| [11](Lecture-11.md) | `budget_eval` (memory) | Budget SmolVLA full vs expert-only; test LoRA settings; budget your chosen stack and compare to its documented requirement |
| [12](Lecture-12.md), [13](Lecture-13.md) | `budget_eval` (evaluation) | Interval and paired test on the fixture; change the episode count; analyze your own paired results |

## Rules for extending the labs

* Keep the CPU modules standard-library only, so they run in CI. Put framework code (PyTorch, JAX, LeRobot) in separate files whose tests skip with `unittest.skipUnless` when the framework is missing.
* Every new check needs a negative case that shows it catches the failure it is meant to catch.
* Never commit real dataset content. Many egocentric datasets forbid redistribution (see [Lecture 02](Lecture-02.md)). Commit metadata-shaped fixtures you generate yourself.
