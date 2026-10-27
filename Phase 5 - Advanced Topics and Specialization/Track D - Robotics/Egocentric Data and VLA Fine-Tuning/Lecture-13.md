# Lecture 13 — Capstone: An Egocentric-Augmented Fine-Tune

Fine-tune one policy twice, with and without egocentric human data, and ship a result that another engineer can reproduce and check.

**Prerequisites:** Lectures [02](Lecture-02.md) to [12](Lecture-12.md) of this course and the [Lab guide](Lab-Guide.md). **Effort:** about 6 hours for the CPU track; several days for a GPU track, including training and robot or simulator time (course estimates). **Hardware:** any track: CPU, 8 GB, or rented 24–80 GB (Section 3). **Roles:** robot learning data engineer, post-training engineer, embodied AI research engineer, ML data platform engineer.

**Misconception this lecture corrects:** "the capstone is the training run." Training is one milestone of eleven. Most of the grade is in the records that make the result checkable: the license audit, the data audit, retargeting feasibility, stats parity and a paired evaluation. An honest "inconclusive" with a sound protocol passes; an unpaired "2× better" does not.

## 1. The question

One task, one robot, one question: does adding retargeted egocentric human data to a fixed robot dataset change closed-loop success on a held-out condition named in advance? The answer is "helped", "hurt" or "inconclusive", with the numbers from Lecture 12. Check first that both halves of the data can share an action space. The retargeting lab produces 7-D actions: a relative end-effector step plus the gripper opening (Lecture 07). If your robot logs joint actions, convert them with your robot's kinematics, or choose a stack with a relative end-effector path (Lecture 09).

## 2. Milestones and gates

| # | Milestone | Lecture | Artifact | Gate before you move on |
|---|---|---|---|---|
| M1 | Source selection | 02 | `dataset_selection.md` | Every source has a license, a commercial or research-only verdict and access terms; MANO-derived labels are flagged |
| M2 | Geometry check | 03 | `geometry_chain.md` | A known point round-trips; intrinsics, frame names and time domain are checked |
| M3 | Budget and audit | 04, 05 | `pipeline_budget.md`, `data_card.md` (draft) | Storage and decode budget done; `split_leakage` empty for the held-out key (participant or location), with any other leaking key named, or its crossing episodes dropped and counted; duplicates resolved; timestamps pass |
| M4 | Hand quality | 06 | `hand_quality_report.md` | Dropout and jitter per clip; excluded clips listed with reasons |
| M5 | Retargeting | 07 | `retargeted_actions.jsonl`, `retarget_report.md` | `feasibility_report` is `ok`, or infeasible segments are cut and counted |
| M6 | Method and stack | 08, 09 | `method_comparison.md`, `stack_selection.md` | Code, weights and data licenses fit the intended use; format, normalization and action space match |
| M7 | Dataset and stats | 10 | Converted dataset, `dataset_build_record.md` (stats hash, mixture plan) | `validate_frame` clean; `stats_parity` empty; robot `epochs_seen` recorded |
| M8 | Memory budget | 11 | `finetune_record.md` | `fits()` is `True` and the documented requirement is at most your GPU's memory; selection rule dated |
| M9 | Two runs | 11 | Baseline (robot only) and augmented (robot plus human) run records | The runs differ only in their data |
| M10 | Paired evaluation | 12 | `paired_evaluation.md` | Same episode IDs; Wilson intervals and exact McNemar on the primary condition; failure taxonomy |
| M11 | Cards | 13 | Final `data_card.md` and `model_card.md` | Every number labeled measured, documented, estimated or synthetic |

## 3. Three tracks

| Track | Policy and runs | Evaluation | Done when |
|---|---|---|---|
| **CPU** | No training. Run every lab on your own metadata where you have it and on synthetic fixtures where you don't | The synthetic paired example in Section 4, labeled synthetic | All milestones recorded; M9 planned, with GPU fields marked "not run" |
| **8 GB** (RTX 4060) | ACT for both runs, inside the LeRobot guide's indicative "~2–6GB" at batch 8; it is not a Lecture 09 VLA stack, but it answers the data question. SmolVLA expert-only or LoRA is a **[est] stretch goal** you must verify | Paired, on a robot or in simulation ([Isaac Sim and Isaac Lab](../Isaac%20Sim%20and%20Isaac%20Lab/README.md)) | Measured peak memory recorded for both runs |
| **24–80 GB** rented | SmolVLA (24 GB), openpi π0 LoRA ("> 22.5 GB") or GR00T N1.7 default tuning ("40 GB+") | Paired, as above | Both runs on the same GPU type and pins |

## 4. The CPU track on synthetic fixtures

**Tested here** (from the repository root): the lab suite, then the capstone gates. Every number the gate script prints comes from a **synthetic fixture**; none is a result. **Negative case:** a split by episode order leaks every participant.

```bash
EGO_COURSE='Phase 5 - Advanced Topics and Specialization/Track D - Robotics/Egocentric Data and VLA Fine-Tuning'
python3 -m unittest discover -s "$EGO_COURSE/labs" -p 'test_*.py'
PYTHONPATH="$EGO_COURSE/labs" python3 - <<'EOF'
import random
from ego_audit import group_split, split_leakage
from dataset_contract import feature_stats, stats_parity
from budget_eval import training_state_bytes, fits, paired_counts, mcnemar_exact_p, wilson_interval
rng = random.Random(13)                                   # SYNTHETIC capstone fixture, not a result
eps = [{"episode_id": f"h{i:03d}", "participant": f"p{i % 8}"} for i in range(80)]
splits = group_split(eps, "participant", {"train": 0.75, "test": 0.25}, seed=0)
print("G1 participant leakage:", split_leakage(splits, "participant") or "none")
print("G1 negative, first-60 split leaks:", sorted(split_leakage({"train": eps[:60], "test": eps[60:]}, "participant")))
actions = [[rng.gauss(0, 0.02) for _ in range(6)] + [rng.random()] for _ in range(500)]
print("G2 stats parity issues:", len(stats_parity(feature_stats(actions), feature_stats(actions))))
state = training_state_bytes(450_046_176, 100_000_000)["total_gb"]   # expert-only, assumed count
print("G3 fits 8 GB (necessary only, 3 GB assumed activations):", fits(state, 8.0, activation_gb=3.0))
base = [rng.random() < 0.45 for _ in range(40)]                      # robot-only policy
aug = [(s and rng.random() > 0.2) or (not s and rng.random() < 0.4) for s in base]  # robot + human
b, c = paired_counts(aug, base)
lo, hi = wilson_interval(sum(aug), 40); blo, bhi = wilson_interval(sum(base), 40)
print(f"G4 augmented {sum(aug)}/40 [{lo:.2f}, {hi:.2f}]  baseline {sum(base)}/40 [{blo:.2f}, {bhi:.2f}]"
      f"  b={b} c={c}  p={mcnemar_exact_p(b, c):.4f}")
EOF
```

```text
.............................
----------------------------------------------------------------------
Ran 29 tests in 0.037s

OK
G1 participant leakage: none
G1 negative, first-60 split leaks: ['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7']
G2 stats parity issues: 0
G3 fits 8 GB (necessary only, 3 GB assumed activations): True
G4 augmented 26/40 [0.50, 0.78]  baseline 20/40 [0.35, 0.65]  b=11 c=5  p=0.2101
```

G1 passes for the grouped split and catches all eight participants in the order-based one. G2 compares the stats a checkpoint trained with against the stats its deployment loads, identical here by construction. G3 only fails to rule the configuration out; it uses Lecture 11's assumed 100M trainable parameters. In G4, 65% vs 50% looks like a 15-point gain, but 11 vs 5 discordant episodes give p = 0.21. The capstone answer for this fixture is "inconclusive at 40 paired episodes", and Lecture 12, Section 4 shows how many more disagreements a 2:1 split would need.

## 5. Grading rubric

| Area | Milestones | Points | Full marks when |
|---|---|---|---|
| Data rights | M1 | 10 | Every source has a license verdict and access terms; no dataset content is committed |
| Data quality | M2–M4 | 15 | Geometry verified; audit output proves a leakage-free split; hand-quality exclusions justified |
| Retargeting | M5 | 15 | Frames and calibration named; feasibility report attached; infeasible parts counted, not silently clipped |
| Dataset contract | M6–M7 | 15 | Format and normalization match the stack; stats parity empty; mixture weights justified |
| Training | M8–M9 | 15 | Budget labeled; the runs differ only in data; selection rule dated before training |
| Evaluation | M10 | 20 | Paired episodes; primary condition named in advance; intervals and exact test; taxonomy |
| Reporting | M11 | 10 | Cards complete; the conclusion matches the test |

**Automatic fail:** data used outside its license; an unpaired comparison presented as paired; a synthetic or estimated number presented as measured; a conclusion stronger than the paired test supports.

## 6. Reproducibility checklist

* **Pins:** stack version or commit (LeRobot v0.6.1, openpi `215abfb`, Isaac-GR00T `n1.7-release` or a commit such as `944fe70`, with docs read at that ref), `projectaria-tools==2.3.0` if you used Aria data, Python version and lockfile.
* **Checkpoint:** base model ID and its Hugging Face revision.
* **Dataset revision:** repo IDs and revisions of the robot and human parts, the conversion script's commit, and `codebase_version`.
* **Stats:** file path, SHA-256 hash (for example `sha256sum meta/stats.json`) and normalization mode.
* **Seeds:** split, mixture and training seeds.
* **Eval protocol:** conditions, primary condition, episode IDs with layouts and seeds, success criterion, run order, and the date it was fixed.
* **Hardware:** GPU model, measured peak memory and wall-clock for each run.

## 7. Card templates

**Exercise / pseudocode** — fill in every field; write "not run" or "synthetic" where it applies.

```markdown
# Data card: <dataset name and revision>
- Sources and licenses: <per source: license, commercial verdict, access terms>
- Contents: <hours, episodes, participants, locations; robot vs human share>
- Splits: <grouping key; split_leakage output>
- Quality: <timestamp report, duplicates, label distribution, kappa, hand-track dropouts>
- Retargeting: <T_wrist_eef, source of T_base_world, feasibility summary>
- Stats: <file, SHA-256, normalization mode>
- Known gaps and uses not allowed: <...>

# Model card: <name>
- Base model and stack: <checkpoint and revision; stack pin; weights license>
- Training data: <baseline vs augmented; mixture weights; link to data card>
- Training: <trainable set, effective batch, steps, seed, GPU, measured peak memory>
- Checkpoint selection: <rule and the date it was fixed>
- Evaluation: <primary condition; success with Wilson intervals; b, c, exact p; taxonomy>
- Conclusion: <helped / hurt / inconclusive, and on which condition>
- Limits: <conditions not tested; estimated or synthetic numbers, labeled>
```

## 8. Exercise

**Exercise / pseudocode** — CPU for steps 1–2; step 3 on your GPU track.

1. **Worked example →** run the CPU track end to end: the suite, the gate script and both cards filled from the fixtures, with every number labeled synthetic.
2. **Guided variation.** Replace one fixture with real input, such as your own episode metadata in the audit or your own hand tracks in retargeting, and rerun the gates. Record which gate failed first and why.
3. **Independent transfer.** On your GPU track, complete M1–M11 for your task: two runs, a paired evaluation and both cards.

**Deliverable:** a `capstone/` folder with the artifacts of all eleven milestones, including `data_card.md`, `model_card.md` and `paired_evaluation.md`. **Acceptance:** every gate is recorded as passed or failed with its output; the reproducibility checklist is complete; no automatic-fail item applies; another engineer can recompute your intervals and p-value from your episode table.

## Failure clinic and self-check

**Failure clinic:** dataset files were committed to the project repository, though many egocentric licenses forbid redistribution. The augmented run also got more steps or a different seed. The augmented checkpoint shipped with the baseline's stats file. Human actions were relative end-effector deltas while the robot data was absolute joints. The two policies ran on different layouts. The checkpoint was picked on the test episodes. The model card says "human data helps" after an inconclusive test. MANO-based hand labels ended up in a commercial project.

1. **Why is "inconclusive" a passing capstone answer?** The grade rewards a protocol that can detect an effect and an honest report of what it found. Inflating an inconclusive result is an automatic fail.
2. **What must match before robot and human data can be co-trained?** The action space and frames (Lectures 07, 10), the dataset schema of the chosen stack, and the normalization stats for each embodiment.
3. **Why does the checklist ask for the stats file's hash?** The same checkpoint fed different stats sends actions on a different scale, so the hash ties the result to the exact stats it was trained and evaluated with.
4. **On the CPU track, G3 printed `True`. What may you claim?** Only that lab arithmetic does not rule the configuration out. A fit is claimed only after a measured run.

[Previous: evaluating fine-tuned policies](Lecture-12.md) · [Course index](README.md) · [Lab guide](Lab-Guide.md)
