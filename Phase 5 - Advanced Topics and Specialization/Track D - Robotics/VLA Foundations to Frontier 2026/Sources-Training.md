# Training and Data Source Dossier — Lectures 09–18

**Evidence cutoff and verification date: October 13, 2026.** This dossier records primary sources used for the training/data block and the limits of the claims they support. It is a curated technical bibliography, not an assertion that every paper published before the cutoff has been reviewed.

Lecture explanations, toy examples, resource arithmetic, and lab designs are original teaching material. Numerical examples explicitly described as illustrative are not measurements. No cited benchmark is reproduced by merely completing a CPU contract exercise.

## Foundations and algorithms

| Primary source | Publication/revision context | Lecture and supported claim | Limits |
|---|---|---|---|
| [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html) | AISTATS 2011 | 09: sequential distribution shift and dataset aggregation using expert labels on learner-visited states | Theoretical results require their assumptions; they do not guarantee arbitrary robot safety or robustness |
| [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://roboticsproceedings.org/rss19/p016.html) | RSS 2023 | 10, 13: ALOHA/ACT, chunk prediction, CVAE training, temporal ensembling, integrated teleoperation | Lecture 10's age-based ensemble is explicitly a teaching variant; reproduce the original indexing convention from its code |
| [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) | Original RSS 2023 work; project also links the 2024 journal version | 11: conditional action diffusion, visual conditioning, receding-horizon execution | Different model variants and samplers are separate configurations; published task scores are not course measurements |
| [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) | 2020 | 11: Gaussian forward corruption, epsilon prediction, reverse process | General generative-model foundation; action masks, controller integration, and robot evaluation are additional engineering |
| [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) | 2020 preprint / ICLR 2021 | 11: alternative inference process and reduced sampling steps | Arbitrarily skipping DDPM iterations is not the documented DDIM method |
| [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) | 2022 preprint / ICLR 2023 | 12: regression of conditional vector fields and ODE generation | Does not establish that any selected solver budget preserves robotic task success |
| [LoRA](https://arxiv.org/abs/2106.09685) | 2021 preprint / ICLR 2022 | 16: low-rank trainable weight updates | Trainable-parameter reduction is not equal to total-memory reduction |
| [QLoRA](https://arxiv.org/abs/2305.14314) | 2023 | 16: quantized-base adaptation as a specific recipe | NF4 and NVFP4 are different representations; the paper does not validate VLA action parity |
| [Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169) | 2021 preprint / ICLR 2022 | 17: in-dataset value learning and advantage-weighted policy extraction | The lecture's generic weighted imitation example is not a full IQL implementation |
| [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) | 2017 | 17: policy probability ratios | A flow-regression MSE is not an exact action log probability for PPO |
| [ZeRO](https://arxiv.org/abs/1910.02054) | 2019 preprint | 18: progressively sharding optimizer states, gradients, and parameters | Actual memory/communication costs depend on runtime, partitioning, and hardware |

## Generalist policies and adaptation

| Primary source | Date/version verified | Lecture and supported claim | Limits |
|---|---|---|---|
| [Open X-Embodiment: Robotic Learning Datasets and RT-X Models](https://arxiv.org/abs/2310.08864) | 2023 original report | 15: pooled robot experience and cross-embodiment transfer research | A later selected data mixture need not equal the complete original collection; tensor shape does not standardize control meaning |
| [Octo](https://arxiv.org/abs/2405.12213) | 2024 original report | 15: flexible generalist policy interfaces and adaptation | Compare its specific observation/action setup and adaptation budget, not just model branding |
| [OpenVLA paper](https://arxiv.org/abs/2406.09246) and [project](https://openvla.github.io/) | 2024 original report; project consulted at cutoff | 15–16: open VLM-based action policy and adaptation experiments | Paper, moving repository, and checkpoint are separate artifacts requiring separate pins |
| [OpenVLA-OFT](https://openvla-oft.github.io/) | Project and primary paper links consulted at cutoff; 2025 work | 16: fine-tuning design can change action generation and throughput | An action-head change is not merely a choice of optimizer; published speedups need their measurement conditions |
| [π₀: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164v4) | First submitted October 31, 2024; v4 January 8, 2026 | 12: VLM plus action expert, flow action chunks, original architecture and inference setup | The original paper configuration is not every openpi configuration; control frequency is not necessarily policy invocation frequency |
| [Official openpi repository](https://github.com/Physical-Intelligence/openpi) | Moving repository consulted October 13, 2026 | 12, 16: implementation/configuration reference and model-specific adaptation instructions | This course does not assert that an unpinned default branch is reproducible or that every later PI model has released weights |
| [π₀.₅](https://arxiv.org/abs/2504.16054v1) | April 22, 2025 | 15: heterogeneous co-training and semantic prediction | Reported generalization must be interpreted with the paper's environments and data composition |
| [Knowledge Insulating Vision-Language-Action Models](https://arxiv.org/abs/2505.23705v1) | May 29, 2025 | 15: discrete supervision and a stop-gradient route for the continuous expert | This is not equivalent to freezing all backbone parameters; do not attribute it indiscriminately to every π-series configuration |
| [π₀.₇](https://arxiv.org/abs/2604.15483v2) | First submitted April 16, 2026; v2 April 24, 2026 | 15: diverse context conditioning, performance metadata/subgoal images, reported generalization experiments | Primary technical report; no course replication and no inference of public weights from publication alone |

## Data collection and storage

| Primary source | Date/version context | Lecture and supported claim | Limits |
|---|---|---|---|
| [DROID](https://arxiv.org/abs/2403.12945) and [project](https://droid-dataset.github.io/) | 2024 original report | 13: diverse distributed robot manipulation collection | Dataset-level diversity does not prove coverage of a chosen deployment task |
| [BridgeData V2](https://arxiv.org/abs/2308.12952) and [project](https://bridgedata-v2.github.io/) | 2023 original report | 13: reusable robot manipulation data with environmental diversity | A source collection still needs action/timestamp audits and license/provenance checks in downstream use |
| [Official RLDS repository](https://github.com/google-research/rlds) | Moving schema/documentation consulted at cutoff | 14: episode/step structure, `is_first`, `is_last`, `is_terminal`, action/reward alignment | Conversion must follow actual field semantics; terminal observations need not have valid action labels |
| [LeRobotDataset v3 documentation](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3) | v3 format documentation consulted at cutoff | 14: video/tabular storage, metadata, episodes spanning shared shards, temporal access | Dataset/library revisions must be pinned; a shard boundary is not an episode boundary |

## RL post-training frontier

| Primary preprint | Submission/revision verified | Lecture 17 claim | Evidence boundary |
|---|---|---|---|
| [π*₀.₆: a VLA That Learns From Experience](https://arxiv.org/abs/2511.14759v2) | First submitted November 18, 2025; v2 November 19, 2025 | RECAP combines value learning, advantage conditioning, and heterogeneous experience including corrections | Distinguish π₀.₆ base model from π*₀.₆; reported physical experiments are upstream results |
| [Potential-Guided Flow Matching for Vision-Language-Action Policy Improvement](https://arxiv.org/abs/2606.04968v1) | June 3, 2026 | ForesightFlow separates weighting of action and potential coordinates | Mentioned as a research direction; course does not reproduce performance or certify calibration |
| [Z-1: Efficient Reinforcement Learning for Vision-Language-Action Models](https://arxiv.org/abs/2606.31846v1) | June 30, 2026 | Flow-based VLA post-training with GRPO-style optimization in RoboCasa | Its likelihood/surrogate construction must be inspected before implementation; no universal ranking claimed |
| [TEMPO: Semantic-Action Decoupled RL Post-Training for Vision-Language-Action Models](https://arxiv.org/abs/2608.07314v1) | August 7, 2026 | Separate update rates for semantic projection and action expert with a frozen backbone | Primary preprint; reported gains require its task and evaluation context |

The verified late-2026 training examples above extend through August. This does not mean the field stopped in August or that the list is exhaustive through October 13. The course's frontier review adds broader model and systems evidence. New references should pass the same primary-source and cutoff checks before inclusion.

## Runtime documentation and verification status

- [PyTorch FSDP tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html): consulted for sharding and collective behavior. The page is moving documentation; Lecture 18 avoids prescribing a specific release or asserting that its example was executed here.
- [PyTorch AMP examples](https://docs.pytorch.org/docs/main/notes/amp_examples.html): consulted for accumulation, scaling, and optimizer sequencing. The `main` path is a development reference, not a version pin. Use documentation matching the installed runtime for an actual run.
- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/main/notes/randomness.html): consulted for nondeterminism and cross-environment limits. Seeds alone are not a reproducibility guarantee.

The advanced labs in Lectures 09–18 are **exercise / pseudocode** specifications. They define inputs, experiments, negative cases, stopping conditions, and acceptance criteria, but this block does not claim that full policy training, GPU sampling, distributed execution, or robot collection was performed during authoring. The [shared lab guide](Lab-Guide.md) records the separate runnable CPU scaffolding and its actual validation scope.

Direct Physical Intelligence blog URLs were intermittently inaccessible to the browsing tool. Claims therefore use accessible author-submitted arXiv records and official source repositories. Search-result summaries from third-party aggregators were used only to locate primary records, not as final technical evidence.

[Course index](README.md) · [Start training block](Lecture-09.md) · [Finish training block](Lecture-18.md)
