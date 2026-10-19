# From recorded episodes to timed actions

[Course](README.md) · [Dataset lesson](Lecture-14.md) · [Asynchronous control lesson](Lecture-22.md)

An action tensor is useful only when its numbers, time intervals, and source observations remain meaningful. This supplied lab follows that complete path: read a small episode file, align observations causally, convert physical actions, fit training statistics, make masked chunks, and replay delayed results through an offline dispatcher.

**Prerequisites:** Python lists/dictionaries, vectors, and Lectures 06, 14, and 22. **Effort:** 15 minutes setup, 60–90 minutes guided work, 45 minutes independent extension. **Hardware:** any CPU with Python 3.10+; standard library only. **Roles/layers:** robotics data engineer and inference integration engineer, L3–L7. The result is a JSON trace with enough provenance to explain every accepted action. There is no model download, network request, ROS node, or actuator connection.

## 1. Run the supplied experiment

Run these **tested here** commands from the repository root:

```bash
VLA_COURSE='Phase 5 - Advanced Topics and Specialization/Track D - Robotics/VLA Foundations to Frontier 2026'
python3 -m unittest discover -s "$VLA_COURSE/labs" -p 'test_*.py'
python3 "$VLA_COURSE/labs/episode_pipeline.py" > /tmp/vla-episode-pipeline.json
cat /tmp/vla-episode-pipeline.json
```

The course suite includes the original grounding/contract tests and the new data/dispatch tests. The reference run on October 18, 2026 passed **31 tests**. The [recorded result](labs/episode-reference.json) and [validation manifest](labs/episode-validation.json) retain source hashes and execution scope. Floating-point reconstruction can produce `0.019999999999999997` for 0.02; compare physical values with an explicit tolerance, not string equality.

Open the [fixture](labs/episode-fixture.json), [implementation](labs/episode_pipeline.py), and [tests](labs/test_episode_pipeline.py) together. The fixture contains three synthetic episodes and tiny RGB markers. It is an original teaching schema, not an RLDS or LeRobot export. The markers are validated and associated with decision times; no visual representation is learned in this lab. Use [the grounding capstone](Lecture-31.md) for the separate learning experiment.

## 2. Read a record before reading an algorithm

Each episode declares a source action convention, collection session, original recording ID, train/test split, clock, instruction, camera records, and timed robot observations. A camera record has both capture and arrival time. A step records measured state separately from the command applied over the following interval.

| Episode | Recorded command | Source interval | Role |
|---|---|---:|---|
| `train-a` | Camera-frame displacement in millimeters | 20 ms | Training |
| `train-b` | Base-frame velocity in meters/second | 40 ms | Training |
| `test-c` | Camera-frame displacement in millimeters | 20 ms | Held out; intentionally larger motions |

The last step contains the final observation and `action: null`. It is not another zero action. A terminated episode and a truncated recording both have a final observation; their termination flags remain different in converted rows. We do not calculate rewards or RL targets here.

This boundary is consistent with [RLDS's documented final-step semantics](https://github.com/google-research/rlds/blob/b35dac3a6b73396b0cb8773095999c4b5d70947c/README.md#dataset-format): action/reward/discount at `is_last` are invalid. RLDS's repository is archived, so preserve the schema and exact reader dependency separately when working with existing data. [LeRobotDataset v3](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3) has its own metadata and file organization; a video or Parquet file boundary does not define an episode. These references inform the exercise; this program imports neither library.

## 3. Preserve motion when changing units, frames, and rate

The target convention is `base_xy_delta_m/v1`: a two-coordinate displacement in the base frame over each 20 ms action interval. For the fixture, camera x points along base y:

\[
R_{B\leftarrow C}=\begin{bmatrix}0&-1\\1&0\end{bmatrix}.
\]

A camera-frame delta of `[10, 0]` mm becomes `[0, 0.01]` m. Translation does not enter this calculation because a displacement has no frame-origin offset.

For `train-b`, a base velocity `[0, 0.5]` m/s held for 40 ms covers `[0, 0.02]` m. Two 20 ms target intervals therefore each receive `[0, 0.01]` m. Repeating the whole 40 ms displacement twice would incorrectly double the motion.

The converter treats velocity as constant within each source interval. It interprets interval deltas using the same explicitly declared assumption, then integrates overlap with each target interval:

\[
\Delta p_j=\sum_i v_i\;\operatorname{duration}
([t_i,t_{i+1})\cap[t_j,t_{j+1})).
\]

This preserves integrated motion under that model. It does not recover unrecorded contact events or the actual within-interval trajectory. Absolute poses, rotations, binary gripper commands, missing intervals, and partial final intervals need different rules; this two-dimensional converter rejects unsupported source conventions.

Predict `train-a`'s four base-y deltas before opening the output: `0.01, 0, 0.01, 0.01` m. Their sum is 0.03 m. The test also resamples that recording to two 40 ms intervals, obtaining `0.01, 0.02` m with the same total. The dispatch model remains fixed at 20 ms, so a normalizer fitted for a different cadence is rejected.

## 4. Align what the policy could actually know

At decision time \(t\), the image must satisfy both

\[
t_{capture}\leq t,\qquad t_{arrival}\leq t.
\]

Then apply the maximum age rule \(t-t_{capture}\leq25\) ms. Choosing the nearest capture time alone leaks observations that have not arrived yet.

| Decision | Tempting unusable frame | Correct selected frame | Age |
|---|---|---|---:|
| 20 ms | Captured at 15 ms, arrives at 25 ms | Frame 0, captured at 0 ms | 20 ms |
| 40 ms | Captured at 39 ms, arrives at 70 ms | Frame 2, captured at 35 ms | 5 ms |
| 60 ms | Captured at 55 ms, arrives at 90 ms | Frame 2, captured at 35 ms | 25 ms |

At 61 ms, frame 2 exceeds the age bound and no newer frame has arrived. The result is a missing observation, with an explicit validity flag. The fitter excludes such examples; a production data pipeline may apply a different documented missing-view policy.

Robot state uses the last available measurement; this fixture assumes state measurements arrive at their recorded time. At 20 ms in `train-b`, the next state measurement is at 40 ms, so interpolating between the two would expose future state. A real delayed state stream needs arrival metadata too. Command labels may legitimately describe future demonstrated motion; policy inputs may not use future measurements. Converted rows retain instruction, selected RGB markers, and frame identity. Distinguish these input and target directions of information flow.

Integer microseconds make exact start/expiry comparisons unambiguous in this exercise. They do not synchronize real cameras. Every record must still carry the same clock identity, and a reset starts a new epoch.

## 5. Fit only training data and mask the end of an episode

The eight valid training actions have five y-deltas of 0.01 m and three of zero. The population mean and standard deviation are

\[
\mu_y=0.00625\;\mathrm m,\qquad
\sigma_y=\sqrt{0.0000234375}\approx0.00484123\;\mathrm m.
\]

Every x-delta is zero. The fixture marks that coordinate constant and uses scale 1.0 to avoid division by zero. This is an explicit encoding convention, not a claim that the robot can never move in x. The held-out 0.1 m actions do not affect either statistic. One test changes those test actions dramatically and requires identical fitted statistics and training provenance.

The saved normalizer includes action schema, interval, training-row digest, and a digest of its own contents. A valid but different normalizer fails identity comparison; a modified normalizer with its old digest fails content verification. Hashing detects accidental artifact substitution, not a falsified provenance claim by an adversary.

Starting at `train-a`'s third action with horizon four yields two real targets and two padded copies: mask `[1, 1, 0, 0]`. The copies are storage placeholders. They contribute neither loss nor executable commands. For zero predictions on the two valid targets, mean squared error over valid coordinates is

\[
\frac{2(0^2+0.01^2)}{2\times2}=0.00005\;\mathrm{m}^2.
\]

The test inserts very large errors only in padded positions and requires the same result. It also rejects an all-zero mask because a mean with no valid targets is undefined.

Session IDs cannot appear in both splits. Reusing the same original recording ID under another episode name is rejected. These checks cannot discover an undocumented copy with a fabricated new provenance ID; real deduplication needs additional recording hashes or similarity evidence.

## 6. Replay the result instead of assuming it arrives on time

The offline executor receives normalized chunks, checks artifact identity and physical bounds, then retains only future starts within its 100 ms observation-age limit. Actions are scheduled at `observation_time + index * 20 ms`. A result arriving at 70 ms retains indices `[4, 5]`, scheduled at 80 and 100 ms. Later indices would be too old at their own start times.

The trace demonstrates these events:

1. At 70 ms, no action is due. The numerical fallback is a zero delta.
2. At 80 ms, request 1's index 4 executes, with observation age 80 ms.
3. A newer request observed at 60 ms arrives at 85 ms. Its future targets replace overlapping unexecuted targets; at 100 ms, request 2 supplies 0.02 m with observation age 40 ms.
4. An older reply arriving out of order is rejected. A malformed high request ID also cannot advance the accepted sequence or alter the queue.
5. Reset clears all queued actions and advances the epoch. A delayed old-epoch reply is rejected even when its numeric timestamps might otherwise look plausible.
6. A valid zero action in the new epoch executes with status `execute`. At a later buffer underrun, the same physical value has status `fallback`. Validity is metadata, not a test for nonzero values.

If dispatch jumps over a scheduled start, that action is counted as missed and discarded; the dispatcher does not hold or interpolate past commands. The replay at 60 ms skips the 40 ms command and reports one missed start. Duplicate dispatch ticks and backward clock movement without reset are rejected.

This is an exact-start bookkeeping model, not a real-time guarantee. It has no physical dynamics, collision checking, trajectory continuity test, or low-level controller. Zero displacement is only this fixture's chosen fallback; its appropriateness on a loaded physical robot is a separate controller question. It also does not implement the committed-prefix conditioning of [real-time action chunking flow policies](https://arxiv.org/abs/2506.07339).

## 7. Change one assumption, predict, then run

First change the first camera arrival from 25 to 19 ms in a copy of the fixture. Predict the selected frame at the 20 ms decision, then run `episode_pipeline.py --fixture /tmp/your-fixture.json`. The physical action should remain unchanged while its observation provenance changes.

Next change a test episode's session to `morning`. The program should stop on split leakage before reporting a clean experiment. Restore it, then change one source unit from `mm` to `m`; the declared source contract should reject it rather than guessing from magnitudes.

For independent work, add a binary gripper channel with a transition inside a 40 ms interval. Specify whether the event is applied at the interval start, end, or an explicit timestamp. Design a resampling test that would fail if a generic linear interpolator invented a half-open gripper category. Do not silently reuse the continuous displacement rule.

Finish by explaining one accepted action from original recording through frame selection, conversion, normalization, chunk identity, and dispatch. Submit the modified fixture, exact command, source hashes, output trace, and a short explanation of one rejected case. A correct report states that these tests validate a synthetic data/execution contract, not VLA task success.
