# Lecture 12 — Evaluating Fine-Tuned Policies and What Human Data Bought

Decide whether adding human data changed closed-loop success, using a paired design, confidence intervals, an exact paired test and a failure taxonomy.

**Prerequisites:** [Lecture 11](Lecture-11.md) (two runs and a checkpoint-selection rule), [VLA Foundations Lecture 19](../VLA%20Foundations%20to%20Frontier%202026/Lecture-19.md) (closed-loop evaluation and statistical evidence), [Deep RL for Robot Learning Lecture 13](../Deep%20RL%20for%20Robot%20Learning/Lecture-13.md) (evaluation rigor). **Effort:** reading 1.5 hours, exercise 2 hours. **Hardware:** CPU for the statistics; the episodes run on your robot or simulator, on whichever track trained the policies. **Roles:** robot evaluation engineer, post-training engineer, embodied AI research engineer.

**Misconception this lecture corrects:** "the policy trained with human data scored 60% and the baseline 54%, so human data helped." On 50 episodes that gap is inside the noise. The claim needs the same episodes for both policies, conditions chosen in advance, and a test on the episodes where the two policies disagree.

## 1. The comparison you are making

Compare two checkpoints from Lecture 11 that differ only in their training data. The **baseline** trains on robot demos; the **augmented** run trains on the same robot demos plus retargeted human data (the Lecture 10 mixture). Everything else is held equal: base checkpoint, stack pin, effective batch, steps, seed and the checkpoint-selection rule with its selection episodes. Each checkpoint ships its own stats file, parity-checked against what it was trained with (Lecture 10).

At a fixed step count, the augmented run sees each robot example fewer times (`epochs_seen`, Lecture 10). Decide whether you hold steps or robot epochs fixed, and write it down. Either choice is defensible; an unrecorded one is not.

## 2. Paired design on held-out conditions

An *episode* is a condition, an initial layout and a seed. Build the list once and run both policies on every entry, with the same object poses (a layout sheet or photo per episode) and a success criterion written before the first run. Alternate which policy goes first on each layout, so drift (lighting, battery, wear, a tired operator) hits both. If a person resets the scene or judges success, do not tell them which policy is running. Fix the episode count per condition in advance; do not stop because the result looks good.

| Condition | Example for "put the cup in the bowl" | What it answers |
|---|---|---|
| In-distribution | The cups and table from the robot demos | Did adding human data hurt the base task? A drop here is a regression |
| New object instances | Cups that appear in the human clips but not in the robot demos | Does the human data's object coverage transfer? |
| New scene | Another table, lighting or background | Visual robustness |
| Human-only variation | A variation only the human data covers | Transfer; Physical Intelligence's report measured scenarios of this kind |

Name one **primary** condition before running. If you test four conditions as equals and report the one under p = 0.05, you have made a multiple-comparison error; divide the threshold by the number of conditions (Bonferroni) instead.

## 3. Intervals, and the paired test on discordant episodes

For \(k\) successes in \(n\) episodes, \(\hat p = k/n\), and \(z = 1.96\) for 95%, the Wilson interval is

$$
\frac{\hat p + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}
$$

Episodes where both policies succeed, or both fail, say nothing about which is better. Count \(b\) (augmented succeeded, baseline failed) and \(c\) (the reverse). If the policies were equally good, each discordant episode would be a fair coin flip, so the exact two-sided McNemar test is

$$
p = \min\left(1,\; 2\sum_{i=0}^{\min(b,c)} \binom{b+c}{i}\, 2^{-(b+c)}\right)
$$

**Tested here** (from the repository root) on a **synthetic fixture**, not a result. **Negative cases:** 5 vs 2 discordant pairs is not significant, and unpaired lists raise.

```bash
EGO_COURSE='Phase 5 - Advanced Topics and Specialization/Track D - Robotics/Egocentric Data and VLA Fine-Tuning'
PYTHONPATH="$EGO_COURSE/labs" python3 - <<'EOF'
from budget_eval import wilson_interval, paired_counts, mcnemar_exact_p
def fixture(both, aug_only, base_only, neither):   # SYNTHETIC: one entry per paired episode
    aug  = [True] * both + [True] * aug_only + [False] * base_only + [False] * neither
    base = [True] * both + [False] * aug_only + [True] * base_only + [False] * neither
    return aug, base
for name, counts in [("fixture 1", (25, 5, 2, 18)), ("fixture 2", (21, 9, 1, 19))]:
    aug, base = fixture(*counts)
    b, c = paired_counts(aug, base)
    ia, ib = wilson_interval(sum(aug), len(aug)), wilson_interval(sum(base), len(base))
    print(f"{name}: augmented {sum(aug)}/{len(aug)} [{ia[0]:.3f}, {ia[1]:.3f}]  "
          f"baseline {sum(base)}/{len(base)} [{ib[0]:.3f}, {ib[1]:.3f}]  "
          f"discordant b={b} c={c}  McNemar p={mcnemar_exact_p(b, c):.4f}")
try:
    paired_counts(aug, base[:-1])                    # one baseline episode missing
except ValueError as e:
    print("ValueError:", e)
EOF
```

```text
fixture 1: augmented 30/50 [0.462, 0.724]  baseline 27/50 [0.404, 0.670]  discordant b=5 c=2  McNemar p=0.4531
fixture 2: augmented 30/50 [0.462, 0.724]  baseline 22/50 [0.312, 0.577]  discordant b=9 c=1  McNemar p=0.0215
ValueError: paired evaluation needs the same episodes for both policies
```

* **Fixture 1** is the lab test's fixture: 30/50 vs 27/50, with 5 vs 2 discordant episodes and p = 0.45. That is not evidence that human data helped. Report it as inconclusive.
* **Fixture 2:** the two intervals overlap (0.462–0.724 and 0.312–0.577), yet p = 0.0215, because 9 of the 10 disagreements favor the augmented policy. Comparing two independent intervals is the wrong test for paired data; pairing cancels the layout difficulty both policies share.
* An unpaired run, with each policy on its own layouts, cannot use this test, and `paired_counts` refuses lists of different lengths. It cannot catch two same-length lists in different orders: key both by episode ID and sort before calling it.

## 4. How many episodes you need

**Tested here** (from the repository root, with `EGO_COURSE` set as above): interval width against \(n\), and how many disagreements a result needs.

```bash
PYTHONPATH="$EGO_COURSE/labs" python3 - <<'EOF'
from budget_eval import wilson_interval, mcnemar_exact_p
for n in (10, 20, 50, 100, 200, 400):
    lo, hi = wilson_interval(round(0.6 * n), n)       # an observed 60% success rate
    print(f"n={n:3d}  60% -> [{lo:.3f}, {hi:.3f}]  width={hi - lo:.3f}")
print("all discordant pairs favour one policy:", {n: round(mcnemar_exact_p(n, 0), 4) for n in (4, 5, 6, 8)})
print("a 2:1 split of discordant pairs:      ", {f"{2*k}:{k}": round(mcnemar_exact_p(2 * k, k), 4) for k in (3, 5, 10, 20)})
EOF
```

```text
n= 10  60% -> [0.313, 0.832]  width=0.519
n= 20  60% -> [0.387, 0.781]  width=0.395
n= 50  60% -> [0.462, 0.724]  width=0.262
n=100  60% -> [0.502, 0.691]  width=0.189
n=200  60% -> [0.531, 0.665]  width=0.135
n=400  60% -> [0.551, 0.647]  width=0.096
all discordant pairs favour one policy: {4: 0.125, 5: 0.0625, 6: 0.0312, 8: 0.0078}
a 2:1 split of discordant pairs:       {'6:3': 0.5078, '10:5': 0.3018, '20:10': 0.0987, '40:20': 0.0135}
```

At 50 episodes, the interval around an observed 60% is 26 points wide. Halving the width takes about four times the episodes (50 → 200), and 10 or 20 episodes per condition cannot tell 60% from 40%. Even if every disagreement favors one policy, you need at least 6 discordant episodes for p < 0.05. At a 2:1 split, 30 discordant episodes are not enough (p = 0.0987) and 60 are (p = 0.0135). Plan the episode count from the disagreements you expect, not from the total.

## 5. Failure taxonomy

The success rate says how often; the taxonomy says what changed. Give every failed episode of both policies one primary code, judged from video by a labeler who does not know which policy ran.

| Code | Failure | Where to look if it changes between policies |
|---|---|---|
| PER | Goes for the wrong object or target | Object coverage of the human data (Lecture 05) |
| APP | Reach ends in the wrong pose | Frames and calibration of the retargeted data (Lectures 03, 07) |
| GRS | Grasp missed, slipped or closed early | Thumb–index gripper mapping and hand-track quality (Lectures 06, 07) |
| PLC | Dropped or misplaced on transport | Time scaling and speed of retargeted segments (Lecture 07) |
| LIM | Workspace, speed or safety stop | Retargeting feasibility report (Lecture 07) |
| STL | Hesitation, oscillation or timeout | Stats parity and normalization mode (Lecture 10) |
| TSK | Does a different task | Task strings and language labels in the mixture |

Then cross-tabulate the discordant episodes. For the \(b\) episodes, which baseline failure codes did the augmented policy fix? For the \(c\) episodes, which failure did the human data introduce?

## 6. Reading published claims

| Reported result | Source |
|---|---|
| "1 hour of additional hand data is significantly more valuable than 1 hour of additional robot data" | EgoMimic abstract ([arXiv 2410.24221](https://arxiv.org/abs/2410.24221)) |
| About 2× with π0.5 on 4 generalization scenarios that only human data covered | Physical Intelligence [blog](https://pi.website/research/human_to_robot) and [arXiv 2512.22414](https://arxiv.org/abs/2512.22414) |
| +44% absolute over human-augmented baselines | EgoBridge ([arXiv 2509.19626](https://arxiv.org/abs/2509.19626)) |
| 68.5% success in unseen environments | DexWild ([arXiv 2505.07813](https://arxiv.org/abs/2505.07813)) |

These are **reported results** under each paper's own tasks, hardware and protocol; none was reproduced in this course. The sources this course checked record the headline numbers, not the episode counts or intervals behind them. Before you lean on one, read the paper's evaluation section and ask: what was the baseline, were the conditions held out, how many episodes, were runs paired, and is there an interval? A hypothetical 2× from 20% to 40% means little at 10 episodes per scenario (Section 4) and much more at 100.

## 7. Exercise

**Exercise / pseudocode** — CPU.

1. **Worked example →** rerun Sections 3–4 and write, for each fixture, the one-sentence conclusion it supports.
2. **Guided variation.** Double every count in both fixtures and rerun. Then, in fixture 2, move two episodes from \(b\) to \(c\). Which conclusions change, and why does the total episode count matter less than the discordant split?
3. **Independent transfer.** Run baseline and augmented policies on a primary held-out condition and the in-distribution condition, and fill in the paired table and the taxonomy. **Reduced CPU alternative:** assume success rates for both policies, build a fixture, and work out how many episodes give about 60 discordant ones.

**Deliverable:** `paired_evaluation.md`, containing the protocol (conditions, the primary condition, episode IDs with layouts and seeds, success criterion, run order), per-condition success with Wilson intervals, \(b\), \(c\) and the exact p-value, failure-code counts for both policies, and a one-sentence conclusion: helped, hurt or inconclusive. **Acceptance:** both policies ran identical episode IDs; the protocol is dated before the first episode; p-values come from discordant pairs; synthetic numbers are labeled synthetic; the conclusion claims no more than the primary-condition test supports.

## Failure clinic and self-check

**Failure clinic:** the two policies ran on different days with different layouts, so the results are unpaired. Same-length result lists were passed in different orders and silently mis-paired. Only in-distribution conditions were tested, where human data is least expected to matter. Evaluation stopped early when the augmented policy led. Checkpoints were selected on the test episodes. 30/50 vs 27/50 was reported as a gain. Four conditions were tested and only the significant one reported. The operator knew which policy was running and reset the scene more carefully for one of them. The augmented checkpoint loaded the baseline's stats file.

1. **Why do episodes where both policies succeed, or both fail, not enter McNemar's test?** They carry no information about which policy is better. Only disagreements do.
2. **You see 5 vs 2 discordant episodes out of 50. What do you report?** p = 0.45: inconclusive. A 6-point difference is consistent with no effect. Plan more episodes on the primary condition.
3. **The intervals overlap but the paired p is 0.02. Which do you trust?** The paired test, because it uses the pairing. Overlap of independent intervals is a conservative visual check, not a test for paired data.
4. **Can you cite EgoMimic's hour-for-hour statement to justify replacing robot data with human data on your robot?** No. It is a reported result for their tasks and pipeline. Test it with your own paired evaluation.

[Previous: fine-tuning mechanics](Lecture-11.md) · [Course index](README.md) · [Next: the capstone](Lecture-13.md)
