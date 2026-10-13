# Foundation sources and evidence boundaries

This dossier records primary references used for [Lectures 01–08](Lecture-01.md). It supports the course's **October 13, 2026 evidence cutoff**. A source's first publication date, a repository's current contents, and an independently reproduced result are different records.

The historical sequence intentionally ends at the foundations needed for later frontier lectures. It does not imply that FAST, OpenVLA, or any other historical example is the newest or best available model in October 2026. Model rankings require a specified task, evaluation protocol, and compared implementation revisions.

## Reading this record

- **Paper evidence:** establishes what authors proposed and reported under their experimental setup. It is not a local measurement.
- **Official implementation evidence:** establishes what maintainers document or expose at the inspected URL. A moving branch is not an immutable reproduction pin.
- **Original teaching material:** the numerical examples, synthetic fixtures, evaluation proposals, and CPU exercises in these lectures were constructed for this course. They are not paper results.
- **Local verification:** arithmetic and Markdown structure were checked during authoring. Upstream VLA training, checkpoint inference, physical robots, and Isaac execution were not run for these eight lectures.

When starting an upstream lab, record the actual paper version, code commit, checkpoint revision, dataset revision, processor/tokenizer revision, normalization statistics, environment lock, and robot/controller configuration. Do not substitute the date of this course for those pins.

## Historical and decision-making foundations

| Primary source | Date and role | Facts used | Scope and limitations |
|---|---|---|---|
| [Brooks: A Robust Layered Control System for a Mobile Robot](https://people.csail.mit.edu/brooks/papers/AIM-864.pdf) | MIT technical report, September 1985; associated journal publication 1986 | Lecture 02: reactive layered control as a historical design point | A control architecture, not a language-conditioned model; cited for the historical distinction |
| [ALVINN proceedings](https://proceedings.neurips.cc/paper/1988/hash/812b4ba287f5ee0bc9d43bbf5bbe87fb-Abstract.html) | NeurIPS 1988 conference volume | Lecture 02: early neural sensory-input-to-steering mapping | Conference-volume date differs from the publication convention below; no claim of modern semantic or cross-embodiment generality |
| [CMU publication catalog](https://www.ri.cmu.edu/pubs/page/2/?authors-first-last-and-separated=Dean+Pomerleau&publication-year=&wpv-publication-taxonomies=0&wpv_filter_submit=Search&wpv_post_search=&wpv_view_count=4514-TCPID4515) | Catalog lists ALVINN proceedings in 1989 | Lecture 02: explicit date distinction | Institutional bibliographic record, not a separate experiment |
| [Kaelbling, Littman, Cassandra: Planning and Acting in Partially Observable Stochastic Domains](https://www.cassandra.org/arc/papers/aij98.pdf) | 1998 journal article; earlier technical report exists | Lecture 01: partial observability, belief-state formulation | Formal model assumes specified transition/observation structure; a learned VLA need not compute an exact belief |
| [Ross, Gordon, Bagnell: A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html) | AISTATS 2011 | Lecture 02: DAgger and learner-induced state distributions | Guarantees depend on assumptions; the scalar lab is inspired by the idea and does not reproduce the paper |
| [Levine et al.: End-to-End Training of Deep Visuomotor Policies](https://www.jmlr.org/papers/v17/15-522.html) | JMLR 2016; earlier preprint exists | Lecture 02: guided policy search and learned image-to-torque policies | Historical manipulation setup; no broad-language or generalist-VLA claim |

## Language grounding and early large-model robotics

| Primary source | First arXiv submission | Facts used | Scope and limitations |
|---|---|---|---|
| [CLIP](https://arxiv.org/abs/2103.00020) | February 26, 2021 | Lectures 03 and 07: image-language supervision and representations | Representation learning; image-text similarity alone does not establish physical feasibility |
| [CLIPort](https://arxiv.org/abs/2109.12098) | September 24, 2021 | Lecture 03: complementary semantic and spatial pathways | Reported tabletop manipulation settings; not an unrestricted continuous-control interface |
| [SayCan](https://arxiv.org/abs/2204.01691) | April 4, 2022; revised August 16, 2022 | Lectures 01 and 03: language-informed selection grounded by available skills and feasibility | Planner/skill boundary; the course's scoring table is synthetic and its scores are not a paper benchmark |
| [RT-1 paper](https://arxiv.org/abs/2212.06817) and [official project](https://robotics-transformer1.github.io/) | December 13, 2022 | Lecture 04: visual/language conditioning, compressed visual tokens, discretized actions, scale/diversity study | Paper architecture and project description; no local execution or assumed complete reproduction bundle |
| [PaLM-E](https://arxiv.org/abs/2303.03378) | March 6, 2023 | Lecture 04: sensory embeddings in an embodied multimodal language model and manipulation-planning use | Grounded reasoning output must be distinguished from direct motor commands |
| [RT-2 paper](https://arxiv.org/abs/2307.15818) and [official project](https://robotics-transformer2.github.io/) | July 28, 2023 | Lectures 01 and 04: action tokens and vision-language transfer to robot control | Semantic transfer in evaluated settings; no claim that web knowledge alone supplies arbitrary motor skills |

## Open policies and data

| Primary source | Date and role | Facts used | Scope and limitations |
|---|---|---|---|
| [Open X-Embodiment / RT-X paper](https://arxiv.org/abs/2310.08864) | First submission October 13, 2023; revision history extends into 2025 | Lecture 05: multi-embodiment data and transfer question | Dataset counts/results can differ by version; the lecture avoids treating a moving paper as a fixed count ledger |
| [Official OXE repository](https://github.com/google-deepmind/open_x_embodiment) | Official dataset/project entry point | Lecture 05: dataset-access and implementation provenance | Constituent datasets retain their own access, schema, and licensing details |
| [Octo paper](https://arxiv.org/abs/2405.12213) | First submission May 20, 2024 | Lecture 05: 800,000-trajectory pretraining and adaptable generalist policy | A reported research configuration; does not prove zero-shot operation on arbitrary robots |
| [Official Octo implementation](https://github.com/octo-models/octo) | Moving official repository; select a commit for execution | Lecture 05: diffusion policy and observation/history masks | Versioned APIs and dependencies require inspection; repository examples are not locally validated here |
| [OpenVLA paper](https://arxiv.org/abs/2406.09246) | First submission June 13, 2024 | Lecture 05: seven-billion-parameter model, 970,000 demonstrations, fused visual features and Llama 2 backbone | Original reported design; the lecture does not generalize its benchmark comparisons beyond the evaluated tasks |
| [Official OpenVLA implementation](https://github.com/openvla/openvla) | Moving official repository; select a commit and checkpoint revision for execution | Lecture 05: dataset-specific action unnormalization and artifact dependencies | Code, base-model terms, checkpoints, processor and controller are separate artifacts; no universal plug-and-play guarantee |

## Geometry, encoders, and action codecs

| Primary source | Date and role | Facts used | Scope and limitations |
|---|---|---|---|
| [ROS REP-103 official source](https://github.com/ros-infrastructure/rep/blob/master/rep-0103.rst) | Created October 7, 2010; source contains later history | Lecture 06: units and coordinate conventions, including optical frames | Reference convention; actual devices and imported datasets may differ and must be adapted |
| [ROS REP-105 official source](https://github.com/ros-infrastructure/rep/blob/master/rep-0105.rst) | Created October 27, 2010 | Lecture 06: mobile platform frame semantics | Does not provide a camera calibration or guarantee localization accuracy |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | First submission June 12, 2017 | Lecture 07: scaled dot-product attention | VLA masks, shape calculations, and cache examples are course constructions |
| [SigLIP](https://arxiv.org/abs/2303.15343) | First submission March 27, 2023 | Lecture 07: pairwise sigmoid image-language objective | Representation objective, not a robot-success guarantee |
| [DINOv2](https://arxiv.org/abs/2304.07193) | First submission April 14, 2023 | Lecture 07: visual pretraining without language supervision | Generic feature results do not establish precise control performance |
| [Prismatic VLMs](https://arxiv.org/abs/2402.07865) | First submission February 12, 2024 | Lecture 07: systematic VLM design-space investigation | Robotics consequences require separate evaluation |
| [FAST paper, v1](https://arxiv.org/html/2501.09747v1) | January 16, 2025 | Lecture 08: frequency-space action encoding, coefficient quantization, BPE and FAST+ distinction | No local training-speed, compression-ratio, or robot-success reproduction; the CPU exercise omits BPE |
| [Official FAST tokenizer](https://huggingface.co/physical-intelligence/fast) | Official artifact entry point | Lecture 08: tokenizer implementation/artifact provenance | Tokenizer availability is not availability of an entire policy/controller system |
| [SciPy DCT documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html) | Versioned library reference; record the actual installed version if used | Lecture 08: explicit transform-normalization conventions | The required exercise can implement the formula directly; no SciPy installation is claimed |

The ROS rendered documentation endpoints returned a bot-challenge page during this research. The official source repository and raw source were inspected instead; no content was inferred from the challenge response.

## Checks performed during authoring

An independent Python 3 arithmetic check reproduced the examples for observation-age displacement, independent-step survival probability, pixel-to-millimeter scale, attention matrix element ratio, hidden-state storage, K/V-cache storage, scalar bin width, and the constant-signal orthonormal DCT coefficients. These are analytical/synthetic checks, not runtime benchmarks.

The eight pages were also checked for balanced inline/display math delimiters and code fences, absence of stray control characters, and the presence of prerequisite, hardware, effort, lab, and answer-guidance sections. The full course build and navigation are checked at the course integration level.

[Course](README.md) · [Start Lecture 01](Lecture-01.md) · [Continue after the foundations](Lecture-09.md)
