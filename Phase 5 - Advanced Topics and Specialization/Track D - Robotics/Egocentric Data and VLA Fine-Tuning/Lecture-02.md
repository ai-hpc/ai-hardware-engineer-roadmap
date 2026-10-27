# Lecture 02 — The Egocentric Dataset Landscape and Its Licenses

Map the public egocentric and human-demonstration datasets by what they capture and how their hands are labeled, and decide which ones your project is allowed to use before you download anything.

**Prerequisites:** [Lecture 01](Lecture-01.md) (the data pyramid and the budget), [VLA Foundations Lecture 13](../VLA%20Foundations%20to%20Frontier%202026/Lecture-13.md) (how robot data is collected). **Effort:** reading 1.5 hours, exercise 2 hours. **Hardware:** CPU track; nothing is downloaded, and the exercise is a written audit. **Roles:** robot learning data engineer, ML data platform engineer, anyone who signs a data agreement on behalf of a team.

**Misconception this lecture corrects:** "if a dataset is public and its paper is open access, a company can train a product on it." Most egocentric datasets are non-commercial, several state no data license at all, and hand labels can carry stricter terms than the video they annotate.

**Not legal advice.** The license readings here are an engineer's reading of each license text as published on October 26, 2026 (see [Sources](Sources.md)). Have counsel review any commercial use.

## 1. Four questions for every dataset

1. **What was captured, and with which device?** Head-mounted camera, smart glasses, a mixed-reality headset, a hand-held gripper or a chest rig.
2. **How much, according to which official source?** Sources often disagree (Section 6).
3. **Where do the hand labels come from?** The capture device, motion capture, or a model such as HaMeR, and in which format (joint positions, SE(3) transforms, MANO parameters).
4. **What may you do with it?** Use it commercially, train commercial models, redistribute it, publish derivatives.

This lecture sorts every dataset into one of four classes:

| Class | Meaning |
|---|---|
| **Commercial OK** | The license text permits commercial use and model training, subject to its conditions (attribution, share-alike, listed purposes, no redistribution) |
| **Research only** | Non-commercial terms, such as CC BY-NC or a non-commercial agreement |
| **Unclear** | No data license is stated, or the governing text could not be read. Treat it as unusable until the owner grants terms in writing |
| **Not released** | There is no download |

## 2. General egocentric datasets

| Dataset | Capture | Scale (official source) | Hand labels | Access | Class |
|---|---|---|---|---|---|
| [Ego4D](https://ego4d-data.org/) | 7 head-mounted camera types (GoPro, Vuzix Blade, Pupil Labs and others) | 3,670 h, 923 participants, 74 locations, 9 countries (homepage); 3.85M narration sentences | No 3D hand pose among the listed annotations; the Hands+Objects benchmark labels state change, active objects and the point of no return | Signed agreement; approval about 48 h; AWS credentials valid 14 days; `ego4d` CLI | **Commercial OK** for the agreement's listed Purposes; no redistribution |
| [Ego-Exo4D](https://ego-exo4d-data.org/) | Project Aria glasses (ego) plus 4–5 GoPros (exo) | V2: 5,035 takes, 740 participants, 123 sites, 1,286.3 ego+exo hours, 221.26 ego-hours (site, changelog) | EgoPose hands: ground truth on 458 takes (68K 3D poses), automatic on 976 takes (4.3M); 2×21 keypoints | Separate signed agreement; about 48 h; `egoexo` CLI | **Unclear** until you read what you sign: the docs say it covers commercial use, but the agreement text is unverified |
| [EPIC-KITCHENS-100](https://epic-kitchens.github.io/) | Head-mounted GoPro Hero7 Black, Full HD | 100 h, 37 participants, 45 kitchens, 89,977 action segments, 97 verbs, 300 nouns | No 3D hands; VISOR masks (CC BY-NC 4.0) | No form: data.bris, Academic Torrents or `epic_downloader.py` | **Research only** (CC BY-NC 4.0); commercial licensing by email |
| [HOI4D](https://hoi4d.github.io/) | Helmet with a Kinect v2 and a RealSense D455 (RGB-D) | 2.4M RGB-D frames, 4,000 sequences, 800 object instances, 610 rooms | 3D hand pose as **MANO parameters** | Direct links | **Research only** (CC BY-NC 4.0, plus MANO terms) |
| [HoloAssist](https://holoassist.github.io/) | HoloLens 2: RGB, depth, head pose, gaze, IMU | 166 h, 350 instructor–performer pairs (abstract); 20 tasks, 16 objects | 26 joints per hand, tracked by the headset | Direct Azure downloads, no form | **Commercial OK** (CDLA-Permissive-2.0); include the agreement text if you share the data |

Two cautions on Ego4D. The public PDF is a 2022 draft, and whether the agreement signed today matches it is unverified. The draft lists training models to "detect, classify, recognise, retrieve, or understand … object, event, place, or activity" among its Purposes, and says derived models "may be used for academic, commercial, or noncommercial purposes", but robot control is not a named Purpose. A commercial robot project needs legal review, not just a signature.

## 3. Project Aria datasets

ADT, AEA, ASE and AEO share one Meta Dataset License Agreement. Its §1(a) allows only "non-commercial research in the fields of action recognition, scene understanding, human-object interaction, human understanding, audio-visual diarization, action forecasting, scene reconstruction, & camera localization & mapping", and adds that "you may not use the Dataset to develop any product that is … available for any commercial use". Robotics is not in that list, so even a research robot-learning project should read §1(a) closely or ask projectaria@meta.com. §3 forbids distributing derived works without permission.

| Dataset | Scale (official source) | Hands | Data license | Class |
|---|---|---|---|---|
| ADT | "200 sequences (~400 mins)" (site, paper) | No MPS hands; mocap ground truth for device and object poses and a 3D skeleton | Meta agreement | **Research only** |
| AEA | 143 recordings, 5 locations, 7.3 h (paper; the docs say over 7.5 h) | No hand tracking | Meta agreement | **Research only** |
| Nymeria | 300 h, 264 participants, 50 locations | No hand labels listed; full-body XSens mocap and two miniAria wristbands | CC BY-NC 4.0 | **Research only** |
| HOT3D | 833 min, 19 subjects, 33 objects; Aria and Quest 3 | Mocap ground truth in UmeTrack and **MANO** formats | Sequences CC BY-SA; **hand annotations CC BY-NC-SA** | Sequences **commercial OK** (share-alike); hand labels **research only** |
| Aria Gen 2 Pilot (A2PD) | 12 sequences of about 5 min, 4 participants | On-device 21-keypoint tracking and MPS hand tracking | CC BY-NC 4.0 | **Research only** |

Under research terms, these datasets are a good place to learn capture geometry (Lecture 03): most ship Aria MPS SLAM outputs with the raw recordings, and A2PD adds on-device and MPS hand tracking.

## 4. Datasets collected for robot learning

| Dataset | Capture | Scale (official source) | Hand or action labels | Class |
|---|---|---|---|---|
| [EgoDex](https://github.com/apple-aiml-research/ml-egodex) | Apple Vision Pro / ARKit | 829 h of 30 Hz 1080p video, 338,000 demonstrations, 194 tasks, "2.0 TB" (paper) | SE(3) transforms for head, upper body and 25 joints per hand; optional confidences | **Research only**, no derivatives ("CC-by-NC-ND terms", version not stated); ungated download |
| [EgoMimic data](https://huggingface.co/datasets/gatech/EgoMimic) | Project Aria, MPS hands and SLAM | Paper Table II, e.g. Object-in-Bowl 1,400 demos / 60 min | Robomimic-style HDF5, human and robot files | **Unclear**: no data license stated (code MIT) |
| [PH2D](https://huggingface.co/datasets/RogerQi/PH2D) | Vision Pro (ARKit), or Quest 3 / Vision Pro with a ZED Mini | "~27k demonstrations" (project site) | Head and hand poses (ARKit on Vision Pro) | **Commercial OK** per the HF card (`mit`); whether the upload holds every demo is unverified |
| [UMI](https://github.com/real-stanford/universal_manipulation_interface) `cup_in_the_wild` | Hand-held gripper; a GoPro (155° fisheye) is the only sensor | 1,400 demos, 30 locations (paper) | Gripper pose from ORB-SLAM3 plus gripper width (zarr) | **Unclear**: no data license from the authors |
| [DexCap-Data](https://huggingface.co/datasets/chenwangj/DexCap-Data) | Chest-mounted RealSense L515 plus 3× T265, Rokoko gloves | 30 min wiping plus 60 min in-the-wild packaging | Glove mocap, robomimic HDF5 | **Commercial OK** with attribution (CC BY 4.0) |
| [EgoScale](https://research.nvidia.com/labs/gear/egoscale/) | Mostly in-the-wild egocentric video; sources unverified | 20,854 h used for training (paper) | Off-the-shelf SLAM and hand-pose estimates, called noisy by the paper | **Not released** |
| [Egocentric-10K](https://huggingface.co/datasets/builddotai/Egocentric-10K) | Factory head cameras | 10,000 h | **None**: video only | **Commercial OK** per card (Apache-2.0, auto-gated) |

Other entries to know: DexWild (HF card MIT), HiFi-UMI-2K (CC BY 4.0, about 16 TB) and GigaHands (CC BY-NC 4.0, multi-view rather than egocentric). EgoVerse states no data license (**unclear**). VITRA-1M publishes only MIT-tagged metadata; the source videos (Ego4D, EPIC-KITCHENS, Ego-Exo4D, SSv2) stay under their own licenses.

## 5. Three license traps

**Trap 1: reading the wrong license file.** The license on an arXiv page covers the *paper*, not the data. A repository's LICENSE usually covers the *code*: the Ego4D repo is MIT, which does not cover the videos, and EgoMimic and UMI have MIT code with no stated data license. A Hugging Face tag can be set by whoever uploaded a mirror: `lerobot/umi_cup_in_the_wild` is tagged apache-2.0 by the uploader, while the UMI authors state no data license. Find the license the data owner publishes for the data.

**Trap 2: labels inherit the license of the tool that made them.** The [MANO license](https://mano.is.tue.mpg.de/license.html) is for "non-commercial scientific research purposes", "prohibits the use of the Software to train methods/algorithms/neural networks/etc. for commercial … use", and forbids distribution. HaMeR's code is MIT, but it needs `MANO_RIGHT.pkl` from the MANO site; WiLoR's models are CC BY-NC-ND 4.0 and also depend on MANO. Under this reading, HOI4D's hands, HOT3D's `mano_*` files and any labels you produce by running HaMeR or WiLoR over a commercial video set carry MANO's non-commercial terms. Labels produced by the capture device (HoloLens 2 joints, ARKit transforms, Aria MPS landmarks) are not MANO parameters, but they stay under their own dataset's license. OpenEgo, which consolidates six public datasets with MANO 21-joint labels, inherits each source's license as well.

**Trap 3: redistribution.** Ego4D data "may not appear in or be visible in any program, dataset, or product". The Aria agreement forbids distributing derived works without permission, MANO forbids distribution, and EgoDex is no-derivatives. HoloAssist's CDLA-Permissive-2.0 allows sharing but requires the agreement text to travel with the data. VITRA-1M shows the pattern to copy: it ships metadata and tells users to fetch the source videos themselves "due to license restrictions". This is why the course labs never commit real data and use metadata-shaped fixtures you generate yourself ([Lab guide](Lab-Guide.md)). Do the same in your repositories: commit dataset names, revisions, episode IDs and scripts, not frames.

## 6. Conflicting numbers: cite the number and its source

Official sources disagree on basic counts:

| Dataset | Values | Sources |
|---|---|---|
| Ego4D participants | 931 / 923 / 926 | paper / homepage / docs |
| Ego4D hours | 3,670 / "3,600 hrs (and counting)" / 3,000 | paper and homepage / docs / title |
| Ego-Exo4D | 1,422 h, >800 participants / 1,286.3 h, 740 | docs intro (arXiv v1) / site and V2 changelog |
| HOI4D participants | 9 / 4 | arXiv v1–v2 and CVPR paper / arXiv v3–v4 abstract |
| HoloAssist hours | 166 / 169 | abstract / website |
| ADT sequences | 200 / 236 | site and paper / docs and HF card |
| AEA hours | 7.3 / "Over 7.5 accumulated hours" | paper / docs |
| Nymeria sequences | 1,200 / 1,100 / 1,083 | paper and site / README / NymeriaPlus paper |
| HOT3D recordings | 425, listed as 198 Aria + 226 Quest 3 (which sums to 424) | project page |
| UMI cup demos | 1,400 / 1,447 | paper / LeRobot mirror episodes |
| EgoVerse hours | 1,362 / 4,003 | paper / egoverse.ai |

Some of these come from versions (the Ego-Exo4D docs intro still quotes arXiv v1); the rest are unexplained. Your record cannot resolve them, so it cites them: "923 participants (Ego4D homepage, checked 2026-10-26; the paper reports 931)". Never average two values or pick one silently. When a number drives a decision, such as hours in a storage budget, use the conservative value and say which one you used.

## 7. Worked example: a selection record

Task from Lecture 01: "put the cup in the bowl" with a parallel-jaw gripper. The same sources are judged for two uses:

| Source | Fit | Hand labels (provenance) | Research prototype | Commercial product | Reason |
|---|---|---|---|---|---|
| EPIC-KITCHENS-100 | Kitchen manipulation | None in 3D | In | **Out** | CC BY-NC 4.0 |
| HoloAssist | 20 tasks with 16 objects | Headset, 26 joints | In | In | CDLA-Permissive-2.0; ship the agreement text with any shared copy |
| Ego4D | Broad daily activity | None; you would estimate them | In, after signing | **Hold** for legal review | Robot control is not a named Purpose; the signed text may differ from the draft |
| EgoDex | 194 tabletop tasks | ARKit, 25 joints per hand | In | **Out** | NC-ND |
| Egocentric-10K | Factory work | None | In | In as video; **Out** once labeled with HaMeR + MANO | Apache-2.0 video, MANO-derived labels |
| EgoMimic data | Object-in-bowl task | Aria MPS | Hold: ask the authors | **Out** until a written grant | No data license stated |

Both negative cases in this lecture show up as a single cell. Picking EPIC-KITCHENS for a commercial product is caught by the license-class column. Labeling a commercial dataset with HaMeR + MANO is caught by the hand-label provenance column, which is why that column exists.

## 8. Exercise

**Exercise / pseudocode** — written, CPU only.

1. **Worked example → your task.** Copy the Section 7 record for your Lecture 01 task with at least five sources from Sections 2–4. For each, record the license name as stated, the URL of the data license (not the arXiv page or the code LICENSE), the check date, the class, the hand-label provenance, the access steps and the redistribution terms.
2. **Guided variation.** Flip the intended use (research to commercial, or the reverse) and mark every row that changes. Then assume every video-only source will be labeled with HaMeR, and mark the rows that change again.
3. **Independent transfer.** Pick a dataset that this lecture does not cover. Find its data license yourself, classify it, and list every scale number with its source and any conflict.

**Deliverable:** `dataset_selection.md`, the selection record with its license and access audit. **Acceptance:** every source has a license name, a data-license URL and a check date; every class is one of the four in Section 1, with a one-line reason; hand-label provenance is stated for every source that has labels; every scale number names its source and notes any conflict; the record says it is not legal advice.

## Failure clinic and self-check

**Failure clinic:** a startup prototype trains on EgoDex because the download needs no form; the record should have said NC-ND. A team quotes the CC BY license on an arXiv page as the data license. A project relies on a mirror's apache-2.0 tag while the authors state no license. HaMeR labels are added to an Apache-2.0 video set, and the policy trained on them goes into a product. A few EPIC-KITCHENS frames are committed to a public repo as test fixtures. A report states "Ego4D has 931 participants" with no source, and a reviewer finds 923 on the homepage.

1. **A dataset's arXiv page shows CC BY 4.0. Can you use the data commercially?** That page tells you nothing about the data. It licenses the paper. Find the license the data owner publishes for the data.
2. **You run HaMeR over Egocentric-10K (Apache-2.0) to train a commercial policy. What is the problem?** HaMeR needs MANO, whose license prohibits training networks for commercial use, so the labels carry non-commercial terms under this reading. Use device-tracked labels, a commercial MANO license or a different estimator whose terms you have checked.
3. **Why does VITRA-1M ship metadata only, and why do this course's labs use synthetic fixtures?** The source videos' licenses restrict redistribution. Publishing pointers and scripts keeps you inside those terms; publishing frames does not.
4. **Ego4D's paper, homepage and docs give 931, 923 and 926 participants. Which do you write?** Any one of them, with its source and date, plus a note that the others differ. Never an unsourced single number.
5. **The Ego4D draft allows commercial product development. Is a commercial robot policy covered?** Not clearly. Robot control is not a named Purpose and the signed agreement may differ from the draft. Mark it "hold for legal review" and remember that the data itself can never be redistributed.

[Previous: why egocentric data](Lecture-01.md) · [Course index](README.md) · [Next: capture geometry](Lecture-03.md)
