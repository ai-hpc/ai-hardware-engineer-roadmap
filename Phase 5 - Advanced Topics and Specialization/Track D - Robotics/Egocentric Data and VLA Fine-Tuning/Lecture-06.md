# Lecture 06 — Hands and Objects: 3D Hand Pose, Contact and Failure Modes

Know where 3D hand labels come from and how they fail during manipulation, and measure a hand track's quality before you turn it into robot actions.

**Prerequisites:** [Lecture 02](Lecture-02.md) (licenses, including MANO), [Lecture 03](Lecture-03.md) (device and world frames), [Lecture 05](Lecture-05.md) (auditing). **Effort:** reading 1.5 hours, exercise 2 hours, plus an optional GPU run. **Hardware:** CPU track for the exercise. The optional estimator run needs a GPU, and this course's sources document no memory figure for it, so measure it. **Roles:** robot learning data engineer, embodied AI research engineer.

**Misconception this lecture corrects:** "a 3D hand-pose label is ground truth for what the hand did." Most labels are estimates from a tracker or a network. They fail most at the moment manipulation depends on, the grasp, because the object hides the fingers. No label in this course's sources observes contact or force.

## 1. Where hand labels come from

| Kind | Source | What you get | License note (Lecture 02) |
|---|---|---|---|
| Device tracking | Aria MPS `hand_tracking/hand_tracking_results.csv` | 21 landmarks per hand (3D, meters, device frame), `T_Device_Wrist`, optional wrist and palm normals | Depends on the recording; the Aria datasets are mostly non-commercial |
| Device tracking | HoloAssist (HoloLens 2) | 26 joints per hand | CDLA-Permissive-2.0 |
| Device tracking | EgoDex (Apple Vision Pro, ARKit) | 25 joints per hand as SE(3) transforms in the ARKit origin frame; optional `confidences/*` | CC-by-NC-ND terms |
| Estimator | HaMeR (code MIT, HEAD `3a01849`) | `pred_mano_params`, `pred_keypoints_3d`, `pred_cam_t`, `focal_length`, `pred_vertices`, … from one image | Needs `MANO_RIGHT.pkl`; MANO is non-commercial |
| Estimator | WiLoR (HEAD `fcb9113`) | Same output fields as HaMeR; `--fast` runs half precision, "up to 1.6x faster" | Models CC BY-NC-ND 4.0; MANO and Ultralytics licenses also apply |
| Mocap ground truth | HOT3D | Hands from optical mocap in UmeTrack and MANO formats | Hand annotations CC BY-NC-SA; MANO-format hands also carry MANO's terms |
| Ground truth and automatic tiers | Ego-Exo4D EgoPose hands | 2 × 21 keypoints; ground truth on 458 takes (68K 3D poses), automatic on 976 takes (4.3M) | Signed agreement |

Use each kind for what it is good at. Ground truth (HOT3D's mocap hands, Ego-Exo4D's ground-truth takes) is for measuring the error of device tracking and estimators on similar footage. Device tracking gives metric hands at capture scale. Estimators label video that has no tracking, which is how large in-the-wild collections get hands: EgoScale's labels come from "off-the-shelf SLAM and hand-pose estimation", and its paper calls them noisy.

The skeletons differ (21, 25 or 26 joints), so never mix them in one feature without an explicit joint mapping. Retargeting to a parallel-jaw gripper (Lecture 07) needs only the wrist pose and the thumb and index tips: pick those from each skeleton and record the mapping.

## 2. Frames and scale

* **Device tracking is metric but local.** MPS landmarks are in the device frame. Chain them to the world with the closed-loop `T_world_device` of the same `graphUid` (Lecture 03): \( p_w = T_{wd}\, p_d \).
* **EgoDex poses live in a per-episode ARKit origin.** Positions are consistent within an episode and meaningless across episodes. Relative actions (Lecture 07) do not depend on the origin.
* **Monocular estimates are ambiguous in depth.** One image fixes the direction to the hand well and its distance poorly: a small hand near the camera and a large hand farther away project alike. The camera translation `pred_cam_t` and the `focal_length` used with it carry most of this depth ambiguity. Record the focal length, and check distance against SLAM, depth or device tracking where you have them.

## 3. Failure taxonomy

| Failure | When | Symptom in the track | Detect with |
|---|---|---|---|
| Occlusion by the object | During the grasp | Missing frames, or fingers collapsed to an average pose | Dropout rate inside vs outside the contact window |
| Hand leaves the field of view | Side reaches, hands low in the frame | Missing frames, or keypoints clamped at the image edge | Project the wrist with the camera model (Lecture 03) and test the image bounds |
| Motion blur | Fast reaches and releases | Noise or loss at peak speed | Dropouts and jitter that rise with speed |
| Left/right swap | Crossed hands, a hand entering the view | A one-frame jump of tens of centimeters | Per-frame displacement above a speed bound |
| Depth ambiguity | Monocular estimators | A smooth track at the wrong distance, scale drift | Comparison with SLAM, depth or device tracking |
| Jitter | Always | High-frequency noise | RMS second difference (Section 5) |

## 4. Contact, grasp state and confidence

Contact and grasp state are not observed. They are inferred from the thumb–index gap (the gripper mapping in Lecture 07), from object motion, from event labels such as Ego4D's Hands+Objects benchmarks (point-of-no-return localization, state change), or from datasets that also track objects (HOT3D ships object meshes and `dynamic_objects.csv`). All of these are proxies. Proximity is not force, a hand closed on a mug keeps a gap set by the mug, and fingers hidden by the mug are guesses.

Confidence fields help, within limits:

* EgoDex's `confidences/*` are optional. Check that each file has them before relying on them.
* The legacy Aria `WristAndPalmPose` record carries a per-hand confidence.
* Ego-Exo4D separates ground-truth hand poses from automatic ones. Keep that tier as a column and measure error on ground truth only.
* None of HaMeR's listed outputs is a confidence score, so build quality signals yourself (temporal consistency, agreement with device tracking).

A confidence threshold is a filter, not a fix. Check where the filtered frames sit: often in the grasp window, where you need them most.

## 5. Measuring a track: dropout, gaps, jumps and jitter

Jitter is the RMS of the second difference, which is zero for constant-velocity motion:

$$
a_k = p_{k-1} - 2p_k + p_{k+1}, \qquad \text{jitter} = \sqrt{\frac{1}{K}\sum_k \lVert a_k \rVert^2}
$$

With independent noise of standard deviation \( \sigma \) on each axis, the expected jitter is \( \sqrt{18}\,\sigma \), about 17 mm for \( \sigma = 4 \) mm. The block below builds a synthetic 3 s wrist reach at 30 fps, hides the grasp, swaps the hands for one frame and adds 4 mm jitter. It runs from the repository root with `EGO_COURSE` set as in the [Lab guide](Lab-Guide.md), and uses `moving_average` from [`labs/retarget.py`](labs/retarget.py).

**Tested here:**

```bash
PYTHONPATH="$EGO_COURSE/labs" python3 - <<'EOF'
import math, random, itertools
from retarget import moving_average

rng, fps = random.Random(0), 30.0
truth = [(0.4 + 0.15 * k / fps, 0.0, 0.2) for k in range(90)]      # 3 s wrist reach, meters
track = []
for k, p in enumerate(truth):
    if 40 <= k < 55:
        track.append(None)                                         # grasp: hand hidden by the object
    elif k == 70:
        track.append((p[0], p[1] + 0.25, p[2]))                    # one frame with the hands swapped
    else:
        track.append(tuple(c + rng.gauss(0.0, 0.004) for c in p))  # 4 mm per-axis jitter

missing = [p is None for p in track]
gaps = [len(list(g)) for m, g in itertools.groupby(missing) if m]
print("dropout", round(sum(missing) / 90, 3), "| in contact window 35-59:",
      round(sum(missing[35:60]) / 25, 3), "| longest gap", max(gaps) / fps, "s")
jumps = [k for k in range(1, 90) if track[k] and track[k - 1] and math.dist(track[k], track[k - 1]) > 0.1]
print("jumps > 0.1 m per frame at frames", jumps)

def jitter_mm(seg):     # RMS of p[i-1] - 2 p[i] + p[i+1]: zero for constant-velocity motion
    acc = [math.hypot(*(a - 2 * b + c for a, b, c in zip(*seg[i - 1:i + 2]))) for i in range(1, len(seg) - 1)]
    return round(1000 * math.sqrt(sum(x * x for x in acc) / len(acc)), 1)

def rms_err_mm(seg, ref):
    return round(1000 * math.sqrt(sum(math.dist(p, q) ** 2 for p, q in zip(seg, ref)) / len(seg)), 1)

for w in (1, 5, 9):                                                # smooth the first clean segment
    s = moving_average(track[:40], w)
    print(f"window {w}: jitter {jitter_mm(s)} mm, error vs truth {rms_err_mm(s, truth[:40])} mm")
EOF
```

```text
dropout 0.167 | in contact window 35-59: 0.6 | longest gap 0.5 s
jumps > 0.1 m per frame at frames [70, 71]
window 1: jitter 18.3 mm, error vs truth 7.2 mm
window 5: jitter 3.0 mm, error vs truth 3.4 mm
window 9: jitter 1.8 mm, error vs truth 3.8 mm
```

* **Negative case: occlusion at contact.** Overall dropout is 16.7%, which looks tolerable. Inside the contact window it is 60%, and the longest gap is 0.5 s. Report both rates, and do not interpolate across a gap that long through a grasp.
* **The swap** shows up as two jumps, into and out of the bad frame. The 0.1 m-per-frame bound (3 m/s at 30 fps) is an assumption; set yours from your task. Left in, this frame becomes a "too fast" action in Lecture 07.
* **Smoothing is a trade-off.** A 5-frame window cuts jitter from 18.3 to 3.0 mm and error from 7.2 to 3.4 mm. A 9-frame window lowers jitter further but raises error to 3.8 mm even on this straight, constant-speed line, because the window is truncated at the segment's ends and biases those positions inward, and this segment ends right at the grasp gap. Real reaches accelerate, so a wide window also lags and rounds them off. Use the smallest window that meets your jitter target, and measure error on a ground-truth source when you can.

## 6. Optional GPU run: HaMeR on your own clip

**Exercise / pseudocode** (steps from the HaMeR README at commit `3a01849`; the exact demo command is in the README and is not reproduced or tested here):

```text
1. git clone https://github.com/geopavlakos/hamer, then check out 3a01849f4148352e9260b69bf28b65d1671a4905.
2. Register on the MANO site and download MANO_RIGHT.pkl to the location the README names.
3. Run the README demo on frames from a clip you recorded yourself. It detects people with Detectron2
   (ViTDet by default; --body_detector regnety is the lighter option) and gets keypoints from ViTPose.
4. Save per frame and hand: pred_keypoints_3d, pred_cam_t, focal_length, frame time -> hand_track.jsonl
```

The CPU alternative is the synthetic track above. The MANO license allows non-commercial research only and prohibits training networks for commercial use, so these labels inherit those terms whatever the video's license. For commercial work, use device tracking under a license that permits it (HoloAssist's 26-joint hands, CDLA-Permissive-2.0) or ask for a MANO commercial license (ps-license@tue.mpg.de). This is a reading of the license text, not legal advice.

## 7. Exercise

**Exercise / pseudocode** (CPU only):

1. **Worked example → your track.** Run the tested-here block. Shorten the occlusion to frames 45–52, predict the overall and in-window dropout and the longest gap, and check your prediction.
2. **Guided variation.** Write a fill rule: interpolate gaps of at most 3 frames (0.1 s), mark longer ones as unusable, and drop or repair frames flagged as jumps. Report how many frames are filled, marked and dropped, and recompute jitter after filling.
3. **Independent transfer.** Take a hand track you are licensed to use (MPS output from your own Aria recording, an Ego-Exo4D ground-truth take, or your HaMeR output from Section 6), express the wrist in the world frame, and fill in the quality report.

**Deliverable:** `hand_quality_report.md` with the source and license (including whether MANO is involved), frame and units, dropout overall and inside contact windows, longest gap, jump count, jitter before and after smoothing with the window size, and the Section 3 taxonomy with a count and an example frame per row. **Acceptance:** every metric names its frame and units; contact-window dropout is reported separately from the overall rate; every taxonomy row has a count (zero allowed) and a detection rule; the license line says whether the labels can be used commercially.

## Failure clinic and self-check

**Failure clinic:** a team reports high detection coverage and misses that most of the misses sit inside grasps. A smoothing window wide enough to hide jitter shifts the grasp moment. A commercially licensed video set is relabeled with HaMeR, and the labels inherit MANO's non-commercial terms. EgoDex positions are compared across episodes that have different ARKit origins. Device-frame landmarks are used as if they were in the world frame. A one-frame left/right swap of 25 cm is a 7.5 m/s spike at 30 fps, which reaches the action stream unless something flags it.

1. **Why report dropout inside the contact window separately?** Occlusion is worst during the grasp, which is the part of the motion the policy most needs. An overall rate of 16.7% hid 60% at contact in the fixture.
2. **A monocular estimate is smooth. Why still distrust its distance?** One image constrains direction much better than depth, so a smooth track can sit at the wrong distance. Check it against SLAM, depth or device tracking.
3. **Can a confidence threshold fix occlusion?** No. It removes low-confidence frames, which tend to sit at contact, and so makes the grasp gap longer. It is a filter, not a repair.
4. **You want to ship a product trained on HaMeR labels of HoloAssist video. What is the problem?** HoloAssist's CDLA-Permissive-2.0 allows commercial use, but HaMeR needs MANO, whose license prohibits training networks for commercial use. Use HoloAssist's own device-tracked hands instead, or license MANO.
5. **Why can a wider moving average increase error?** At segment ends the truncated window biases positions, and on accelerating motion it lags. Error against ground truth, not jitter alone, decides the window.

[Previous: auditing a dataset](Lecture-05.md) · [Course index](README.md) · [Next: retargeting and feasibility](Lecture-07.md)
