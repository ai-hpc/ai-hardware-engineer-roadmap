# VLA source and execution update — October 18, 2026

[Course](README.md) · [Original frontier register](Frontier-Register.md) · [Dataset-to-execution lab](Dataset-and-Execution-Lab.md)

This targeted update supplements the October 13 research snapshot. It checks current artifact interfaces, access, benchmark metadata, and executable course examples. It does not claim to review every paper published since that snapshot or reproduce a full VLA benchmark.

## What changed in the evidence

**Physical Intelligence API beta.** The October 13 announcement adds managed fine-tuning and hosted inference, with participation requested through a beta application. It is a service access route; it does not establish downloadable weights for the latest model or unrestricted enrollment. The provider says underlying models can change and describes network-connectivity difficulties in some deployments. Record service identity, timing, and availability separately from local openpi experiments. No API requests or data uploads were performed for this review. [Official announcement](https://www.pi.website/blog/api-beta)

**GR00T artifact details.** The inspected repository separates Apache-2.0 code from NVIDIA Open Model License weights in its license section. Its introductory wording remains broader; the pinned model card still says EA. The exact packaged terms and model configuration matter more than either label alone. [Pinned repository](https://github.com/NVIDIA/Isaac-GR00T/blob/dac54b313160c201c450f97df0fd74ebd379c892/README.md#license), [pinned card](https://huggingface.co/nvidia/GR00T-N1.7-3B/blob/2fc962b973bccdd5d8ce4f67cc63b264d6886495/README.md)

N1.7 source defaults are maximum state/action widths 132 and prediction horizon 40, with a Cosmos-Reason2-2B backbone and 16 action-transformer layers. These are configurable defaults, not the required physical dimensions of every embodiment. A selected checkpoint can override them. The deployment guide contains a schematic with older-looking layer counts, reinforcing the need to inspect loaded configuration. [Pinned model configuration](https://github.com/NVIDIA/Isaac-GR00T/blob/dac54b313160c201c450f97df0fd74ebd379c892/gr00t/configs/model/gr00t_n1d7.py), [deployment guide](https://github.com/NVIDIA/Isaac-GR00T/blob/dac54b313160c201c450f97df0fd74ebd379c892/scripts/deployment/README.md)

The current rollout option is `--execution-horizon`; it selects consumed actions rather than changing model prediction length. Its platform recipes name JetPack 7.2/CUDA 13.2 for Orin and Thor. Consult the roadmap's [Jetson compatibility record](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/Compatibility.md) for the separately released 7.2.1 stack; an upstream 7.2 recipe is not evidence that every dependency was tested against 7.2.1. No GR00T weights, TensorRT engines, or Jetson runtime were executed here. [Pinned recipe](https://github.com/NVIDIA/Isaac-GR00T/blob/dac54b313160c201c450f97df0fd74ebd379c892/README.md)

**RoboCasa365.** The live leaderboard now says updated October 16 and lists 19 models. Its comparison still uses 50 tasks, divided into 18 atomic-seen, 16 composite-seen, and 16 composite-unseen tasks. The old register's October 13 date describes its earlier inspection, not the current page. Preserve task counts, training restrictions, and submission details when comparing entries; the current ordering is not a general capability ranking. [Official leaderboard and protocol](https://robocasa.ai/leaderboard.html)

**Isaac integration.** Sim 6.2 GA is newer than the course's 6.1/Isaac Lab EA teaching baseline. The Lab EA tag and current release branch pin different dependencies. Follow the [Isaac compatibility record](../Isaac%20Sim%20and%20Isaac%20Lab/Compatibility.md), including its immutable documentation source links, before attempting the VLA adapter exercise.

## Inspect source conventions without loading a model

The reviewed openpi implementation interpolates from actions at time zero to noise at time one, predicts `noise - actions`, and integrates backward from one. Lecture 12 teaches the equally valid opposite convention. The new source-reading note makes that sign change explicit and traces prefix caching to the attention structure. Mixing these conventions would reverse generation. [Pinned implementation](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py)

The dataset lab uses RLDS's documented final-observation distinction and LeRobot v3's distinction between storage and episode organization as reading references. It deliberately implements a small standalone fixture, so it requires neither TensorFlow nor LeRobot installation. [RLDS schema](https://github.com/google-research/rlds/blob/b35dac3a6b73396b0cb8773095999c4b5d70947c/README.md), [pinned LeRobot v3 documentation](https://github.com/huggingface/lerobot/blob/146d8474932c8b566717eb2738ba459337ab71c4/docs/source/lerobot-dataset-v3.mdx)

| Reviewed repository | Immutable source commit |
|---|---|
| RLDS | `b35dac3a6b73396b0cb8773095999c4b5d70947c` |
| LeRobot | `146d8474932c8b566717eb2738ba459337ab71c4` |
| openpi | `215abfb217dbac7d5f1273282331b9b1866c0479` |
| Isaac-GR00T | `dac54b313160c201c450f97df0fd74ebd379c892` |

The [source manifest](labs/source-review-2026-10-18.json) records inspected file hashes. A source pin is not an environment lock or a claim that these packages were installed together. Model and dataset revisions still need separate pins for a full experiment.

## What actually ran

The [new CPU lab](Dataset-and-Execution-Lab.md) loads three supplied episodes, converts two action conventions at different sampling rates, selects observations using capture and arrival times, fits training-only normalization, masks terminal padding, and replays delayed/out-of-order/reset-epoch results. The course's 31 tests pass, including the original nine-parameter grounding learner. The new replay contains no physical dynamics or learned foundation model.

The lecture review also corrected a rate inference: a 90 ms p95 request latency does not establish average throughput of 11.1 requests/s. A reciprocal is only an interval calculation under explicit serial assumptions; real throughput needs request counts over measured time. This distinction matters when evaluating asynchronous VLA services.

For your own update, preserve the original snapshot and append dated evidence with its exact scope. A newly visible service, modified model card, changed checkpoint, and changed benchmark protocol are different events. Re-run the relevant contract before interpreting any of them as a deployment improvement.
