# Lecture 05 — Auditing a Dataset: Statistics, Leakage and Label Quality

Check egocentric episode metadata for broken clocks, duplicate clips, leaking splits and weak labels before any of it reaches a training set, and write the findings into a data card.

**Prerequisites:** [Lecture 03](Lecture-03.md) (time domains), [Lecture 04](Lecture-04.md) (layout and the timestamp tolerance), [VLA Foundations Lecture 14](../VLA%20Foundations%20to%20Frontier%202026/Lecture-14.md) (dataset schemas, splits and mixtures). **Effort:** reading 1.5 hours, exercise 2–3 hours. **Hardware:** CPU track; the lab reads metadata only and decodes no video. **Roles:** robot learning data engineer, ML data platform engineer.

**Misconception this lecture corrects:** "a random 80/20 split of clips gives an honest test set." Clips from one person or one kitchen share hands, objects, lighting and habits. A random clip split puts the same people and rooms on both sides, so the test score measures how well the model remembers them, not how it generalizes.

## 1. What an audit checks

| Check | Failure it catches | Lab function (`labs/ego_audit.py`) |
|---|---|---|
| Timing and durations | Dropped frames, a clock that runs backwards (mixed host and device time), truncated episodes, totals that do not match the source | `timestamp_report`, `duration_stats` |
| Duplicates | Re-uploaded or re-exported clips that inflate counts and leak across splits | `find_duplicates` |
| Split leakage | The same participant or location in train and test | `split_leakage`, `group_split` |
| Label distribution | A long tail of rare classes; taxonomy classes with no examples | `label_distribution` |
| Annotator agreement | Labels too inconsistent to train or evaluate on | `cohens_kappa` |

The lab works on metadata dicts (episode id, participant, location, timestamps, label, content hash). It does not detect visual near-duplicates, review privacy or judge whether a label is correct ([Lab guide](Lab-Guide.md)); the data card lists those as open items. Every tested-here block below runs from the repository root, with `EGO_COURSE` set as in the Lab guide. The fixture is a copy of `make_episodes()` in [`labs/test_data_pipeline.py`](labs/test_data_pipeline.py): 60 synthetic episodes from 12 participants in 5 kitchens at 30 fps, with one re-uploaded clip. It contains no real data.

## 2. Clocks, durations and duplicates

**Tested here:**

```bash
PYTHONPATH="$EGO_COURSE/labs" python3 - <<'EOF'
import random
from ego_audit import timestamp_report, duration_stats, find_duplicates
rng = random.Random(1)                   # same fixture as make_episodes() in test_data_pipeline.py
verbs = ["pick"] * 20 + ["place"] * 12 + ["pour"] * 6 + ["open", "close", "wipe", "stir"]
eps = [{"episode_id": f"e{i:03d}", "participant": f"p{i % 12:02d}", "location": f"kitchen_{i % 5}",
        "timestamps_s": [k / 30.0 for k in range(rng.randint(60, 300))], "verb": rng.choice(verbs),
        "content_hash": f"h{i:03d}"} for i in range(60)]
eps[7]["content_hash"] = eps[3]["content_hash"]        # a re-uploaded clip
print({k: round(v, 3) for k, v in duration_stats(eps, 30.0).items()})
print(find_duplicates(eps))
ts = [k / 30.0 for k in range(100)]
ts[50] = ts[48]                                        # clock jumps back
del ts[70]                                             # one dropped frame
print(timestamp_report(ts, 30.0))
EOF
```

```text
{'count': 60, 'total_hours': 0.104, 'mean_s': 6.211, 'median_s': 6.233, 'min_s': 2.1, 'max_s': 10.0}
[['e003', 'e007']]
{'frames': 99, 'duration_s': 3.3, 'non_monotonic': 1, 'off_period': 3, 'estimated_dropped_frames': 3, 'ok': False}
```

* **Counts.** Compare your count and hours with what the source states, and record both with their source. Official numbers disagree: the UMI paper reports 1,400 cup-arrangement demonstrations, while the `lerobot/umi_cup_in_the_wild` mirror holds 1,447 episodes. Datasets also change between versions (Ego4D v2 removed erroneous videos and fixed VQ2D boxes; the `ego4d` CLI defaults to `v2_1`), so the card names the version you audited.
* **Duplicates.** `e003` and `e007` share a content hash, so one is a re-upload. Keep one and list both ids. An exact hash only catches byte-identical copies; re-encoded or trimmed copies need visual matching, which the lab does not do.
* **Negative case: the broken clock.** The report fails (`ok: False`). Only one frame was deleted, yet it estimates 3 drops: the backward jump at frame 50 makes the next gap three periods long, which reads as two more drops. Fix the clock domain first (single-device Aria data should use `DeviceTime`, Lecture 03), then count drops. The default `tolerance_s=1e-4` is the same tolerance as LeRobot's load-time timestamp check (Lecture 04).

## 3. Leakage: why random clip splits fail

Egocentric data is nested. Segments are cut from videos, videos come from sessions, and a session is one participant in one location. Neighboring segments of one video show nearly the same images. EPIC-KITCHENS-100, for example, has 37 participants across 45 kitchens, so a random segment split leaks both. Pick the axis the test set should measure (new people, new places, new objects) and split by that group key. Lecture 12 evaluates on these held-out conditions.

**Tested here:**

```bash
PYTHONPATH="$EGO_COURSE/labs" python3 - <<'EOF'
import random
from test_data_pipeline import make_episodes
from ego_audit import split_leakage, group_split
eps = make_episodes()
shuffled = eps[:]
random.Random(0).shuffle(shuffled)
naive = {"train": shuffled[:45], "test": shuffled[45:]}          # random clip split
print(len(split_leakage(naive, "participant")), "of 12 participants leak;",
      len(split_leakage(naive, "location")), "of 5 locations leak")
g = group_split(eps, "participant", {"train": 0.75, "test": 0.25}, seed=0)
print({k: len(v) for k, v in g.items()}, "participant leaks:", split_leakage(g, "participant"))
print("location leaks after a participant split:", len(split_leakage(g, "location")))
EOF
```

```text
8 of 12 participants leak; 5 of 5 locations leak
{'train': 45, 'test': 15} participant leaks: {}
location leaks after a participant split: 5
```

* **Negative case.** The random split leaks 8 of 12 participants and every kitchen. Any non-empty `split_leakage` result means the test set does not measure generalization along that key.
* `group_split` assigns whole participants to one side (45 / 15 episodes) and the participant check returns `{}`. All 5 kitchens still leak, though. In this fixture every participant cooked in every kitchen, so no split holds out both people and places without dropping episodes. State which axis the test measures; if you need both, drop the episodes that cross and record how many.

## 4. Labels: long tails and agreement

EPIC-KITCHENS-100 has 89,977 action segments over 97 verb and 300 noun classes, an average of about 928 segments per verb and 300 per noun. Averages hide the tail, and the tail decides whether a rare verb has enough examples to learn or to test.

**Tested here:**

```bash
PYTHONPATH="$EGO_COURSE/labs" python3 - <<'EOF'
from test_data_pipeline import make_episodes
from ego_audit import label_distribution, cohens_kappa
d = label_distribution(make_episodes(), "verb")
print({k: round(v, 3) if isinstance(v, float) else v for k, v in d.items()})
a = ["pick", "place", "pick", "pour", "pick", "place"]   # annotator A
b = ["pick", "place", "place", "pour", "pick", "pick"]   # annotator B, same six clips
print("raw agreement", round(sum(x == y for x, y in zip(a, b)) / len(a), 3),
      "kappa", round(cohens_kappa(a, b), 3))
EOF
```

```text
{'classes': 6, 'entropy_bits': 1.897, 'max_entropy_bits': 2.585, 'top10pct_share': 0.467, 'singletons': 2, 'counts': {'pick': 28, 'place': 17, 'pour': 9, 'stir': 4, 'open': 1, 'close': 1}}
raw agreement 0.667 kappa 0.455
```

* The fixture's taxonomy has 7 verbs but only 6 are observed: `wipe` never occurs. `label_distribution` counts only what it sees, so always compare against the official class list. Entropy is 1.897 bits against 2.585 for 6 equally frequent classes. The top class (`pick`, the top 10% of classes rounded up to one) holds 46.7%, and two classes have a single example. A test set says nothing about a class with one example.
* Cohen's kappa corrects raw agreement for chance, \( \kappa = (p_o - p_e)/(1 - p_e) \). Here \( p_o = 4/6 \) is the observed agreement and \( p_e = (3 \cdot 3 + 2 \cdot 2 + 1 \cdot 1)/36 = 14/36 \) is the agreement expected if each annotator labeled at random with their own class frequencies. \( \kappa = 0.455 \) means the annotators reached less than half of the agreement available above chance. Double-annotate a random sample of your clips and report kappa for each label field (verb, noun, mistake flag), not only the easiest one.

## 5. Privacy and consent

Egocentric video records bystanders, faces, homes, screens and voices. State what the sources document, and no more. The Ego4D paper's per-modality table (Table 1) lists 612 hours with unblurred faces, out of 3,670 hours of RGB video; do not assume anything about the rest beyond what the dataset documentation says. Ego4D's agreement forbids redistribution: the data "may not appear in or be visible in any program, dataset, or product". Many sources also carry audio (Ego4D lists 2,535 hours) and eye gaze (Ego4D 45 hours from 2 sites; HoloAssist and several Aria datasets record gaze too).

The lab checks none of this. The data card records what the source documents about de-identification, what your use adds (for example, sample frames in a published report), and who reviewed it. Consent and review for your own capture come from your organization's process. Never commit real frames, audio or metadata to the repository.

## 6. The data card

| Section | Contents | Evidence |
|---|---|---|
| Source and license | Dataset, version or revision, license, allowed use | Lecture 02 selection record |
| Counts | Episodes, hours, fps; counted vs stated, with the stating source | `duration_stats` |
| Timing and duplicates | Clips failing `timestamp_report` and the fix or exclusion; duplicate groups and the copy kept | `timestamp_report`, `find_duplicates` |
| Splits | Group key, fractions, seed, ids per split, and `split_leakage` for every key | `group_split`, `split_leakage` |
| Labels | Observed vs taxonomy classes, entropy, top share, singletons, kappa per field with sample size | `label_distribution`, `cohens_kappa` |
| Privacy | What the source documents, what your use adds, reviewer | Section 5 |
| Known gaps | Checks not run and why (near-duplicates, label correctness) | Lab guide |

## 7. Exercise

**Exercise / pseudocode** (CPU only):

1. **Worked example → your fixture.** Run the three tested-here blocks and confirm the outputs match. Then add a second duplicate and a second clock jump, predict the reports, and run them.
2. **Guided variation.** Split by location instead of participant and check both keys. Then build a split that holds out a set of participants *and* one kitchen, dropping the episodes that cross. Report how many you dropped and why this fixture forces it.
3. **Independent transfer.** Audit a real metadata file you are licensed to use, such as the EPIC-KITCHENS-100 annotation CSVs (non-commercial research only) or your own capture log. Map its columns to the episode dict, hash the files you downloaded for `content_hash`, and check leakage on every group key the source provides. Skip any check the metadata cannot support and say so. Do not commit the data.

**Deliverable:** `data_card.md` with the sections in Section 6, each audit output pasted verbatim, and the split file (episode ids per split) beside it. **Acceptance:** `split_leakage` returns `{}` for the key the test set is meant to hold out, and the card names any key that still leaks; every count names its source; every duplicate group is resolved; at least one label field has a kappa with its sample size, or the card says why not.

## Failure clinic and self-check

**Failure clinic:** a random clip split gives a high test score that disappears in a new kitchen. A mirror holds more episodes than the paper, and nobody asks why. Two copies of one clip land on opposite sides of the split. A clock mixing host and device time passes a spot check, then fails the timestamp report on some clips. A card reports kappa for the easiest label field only. A taxonomy class with no examples is silently missing from the label table.

1. **Why does a random clip split leak even when no clip appears twice?** Neighboring segments of one video, and all videos from one person or room, share appearance. The test set then holds people and rooms the model trained on, and `split_leakage` on the participant and location keys shows it.
2. **The timestamp report estimates 3 dropped frames, but only one is missing. Why?** A backward clock jump makes the following gap several periods long. Fix the time domain first, then count drops.
3. **You split by participant and the location check still leaks. Is the split wrong?** Not necessarily. It tests "new people", not "new places". If you need both, hold out both, drop the crossing episodes and record the cost.
4. **Two annotators agree on 67% of clips. Is that good?** Not on its own. With skewed class frequencies, chance agreement is high; a kappa of 0.455 says they reached less than half of the agreement available above chance. Kappa also misses agreement on a wrong label: it measures consistency, not correctness.

[Previous: the data pipeline](Lecture-04.md) · [Course index](README.md) · [Next: hands and objects](Lecture-06.md)
