# Research ledger: VLA workloads and acceleration

**Evidence checked: 14 October 2026.** This ledger supports [Lecture 02](Lecture-02.md) and the accelerator design study. It records what the sources establish, what can be reproduced, and where evidence stops. A published result is an author-reported result until you reproduce it under a recorded configuration.

## Models: architecture is separate from access

| Workload | Primary evidence | What the course can use | Access and interpretation limits |
|---|---|---|---|
| SmolVLA | [Official model card](https://huggingface.co/lerobot/smolvla_base), [authors' architecture and asynchronous-inference article](https://huggingface.co/blog/smolvla), [paper](https://arxiv.org/abs/2506.01844) | Continuous action generation using flow matching; a compact open workload; an execution schedule that overlaps action playback with policy inference | The base checkpoint is intended for adaptation to a task/embodiment. Public weights and LeRobot code do not establish task success on an arbitrary robot. Asynchrony does not eliminate inference latency or observation age. |
| π₀, π₀-FAST, π₀.₅ | [Official openpi repository and checkpoint table](https://github.com/Physical-Intelligence/openpi) | Flow-based policies and an autoregressive action-token alternative; checkpoint, normalization, input/output adapter, and configuration form one workload | The inspected repository exposes these three families. Its π₀.₅ implementation supports the flow head; π₀-FAST is not in its PyTorch support list. Record the framework and revision rather than assuming all paths are interchangeable. |
| π₀.₇ | [Physical Intelligence announcement, 16 April 2026](https://www.pi.website/blog/pi07) | A research case about steering and generalization; questions for future workload requirements | No official π₀.₇ checkpoint was identified in the inspected openpi release table. An announcement is insufficient to make this a mandatory executable benchmark. This is an availability finding from the inspected sources, not proof that no other access exists. |
| GR00T N1.7 | [NVIDIA architecture article, 17 April 2026](https://huggingface.co/blog/nvidia/gr00t-n1-7), [official repository](https://github.com/NVIDIA/Isaac-GR00T), [base checkpoint](https://huggingface.co/nvidia/GR00T-N1.7-3B) | A roughly 3B model with an Action Cascade: Cosmos-Reason2-2B vision-language processing plus a diffusion-transformer action head; public implementation and task checkpoints | The current base identifier is `nvidia/GR00T-N1.7-3B`. The repository states that the Cosmos-Reason2-2B dependency requires approved Hugging Face access and authentication. Open code and a published checkpoint page do not imply unrestricted dependency access. |
| Multi-Scale Embodied Memory (MEM) | [Authors' paper, submitted 4 March 2026](https://arxiv.org/abs/2603.03596), [author-hosted paper](https://www.pi.website/download/Mem.pdf) | Short-horizon video history and long-horizon textual state motivate explicit temporal inputs, state lifetimes, and reset rules | A method/paper is available. An official runnable MEM checkpoint was not established in this review. A community reimplementation must not be represented as the authors' released model. |
| Gemini Robotics 2 / On-Device 2 | [DeepMind announcement, 30 July 2026](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/) | Frontier context for local execution, embodiment adaptation, and whole-body policies | The announcement describes early-access partnerships for the VLA and On-Device models. Gemini Robotics ER 2 availability is a separate reasoning-model offering. Do not assign an open-checkpoint reproduction lab for these restricted VLA models. |

The accessible lab route starts with a trace fixture, then one released policy. Add architectural diversity by comparing a flow head with an autoregressive head before increasing checkpoint size. None of the access checks above involved downloading model weights or executing robot policies.

## GR00T performance: preserve the measurement boundary

The [official hardware guide](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/hardware_recommendation.md) reports these **model replanning rates** for **N1.7, one camera, four denoising steps**:

| Platform named by source | PyTorch eager | TensorRT |
|---|---:|---:|
| Jetson AGX Thor | 8.9 Hz | 12.4 Hz |
| Orin, 64 GB | 2.9 Hz | 6.6 Hz |
| RTX Pro 6000 Blackwell | 12.8 Hz | 35.9 Hz |

These are vendor-reported figures, not results of this repository. They describe fresh action-chunk production, which differs from camera capture and action playback. Before using them as a comparison baseline, record the exact source revision, checkpoint, precision, power mode, thermal condition, execution horizon, and timing boundary. The displayed table alone does not establish equal task quality, energy consumption, or p99 latency.

## Accelerator and quantization research

### Cross-accelerator characterization — 27 April 2026

**Source:** [Zhou et al., arXiv:2604.24447v1](https://arxiv.org/html/2604.24447v1).

The paper studies stage-dependent bottlenecks and proposes DP-Cache and V-AEFusion. Its profiled cases motivate the hypothesis that a vision-language stage can have higher arithmetic intensity than an iterative action expert. The course does **not** generalize that classification to every model, precision, camera count, or accelerator. Operational intensity must count actual transferred bytes at a stated memory boundary; parameter counts alone omit activation traffic and cache behavior.

**Reproduction task:** identify a single evaluated configuration, reconstruct its execution graph, and compare serial execution against any proposed overlap. A schedule that uses an older observation changes the policy's input timing and needs behavioral evaluation. The paper is public; this review did not reproduce its reported speedups.

### EdgeVLA accelerator — ISLPED 2026

**Sources:** [official final program](https://www.islped.org/2026/final-program/), [author laboratory publication list](https://www.tianyuj.com/publications.html), [publication DOI](https://doi.org/10.1145/3816440.3818561).

The program and authors' list verify *EdgeVLA: A Heterogeneous Edge VLA Accelerator with Iteration-Aware Low-Rank Approximation for Embodied AI*, by Renati Tuerhong, Yiyang Sun, Mingxuan Li, and Tianyu Jia. The ACM full text could not be inspected through the available browser during this review. Consequently, process node, measured/simulated status, quantitative gains, RTL release, and silicon availability remain **unverified here**.

Use it as a prior-art lead, not a benchmark row with invented numbers. Do not confuse it with the separately authored [2025 EdgeVLA model paper](https://arxiv.org/abs/2507.14049); matching names do not establish matching implementations.

### QuantVLA — February–April 2026

**Sources:** [paper, arXiv:2602.20309v4, 6 April](https://arxiv.org/abs/2602.20309v4), [authors' project](https://quantvla.github.io/), [public code](https://github.com/AIoT-MLSys-Lab/QuantVLA).

The paper studies scale-calibrated post-training quantization, including the diffusion-transformer head. Its components include attention-temperature and output-interface calibration. The repository documents a GR00T W4A8 path and evaluation procedure. Quantization coverage is selective: inspect the actual layer list and arithmetic before labeling an entire policy “4-bit.” Memory reductions reported for quantized components are not reductions for the complete module or complete model.

**Reproduction task:** generate a layer-by-layer table containing weight storage, activation storage, accumulation dtype, scale granularity, and actual runtime kernel. Confirm which operations stay floating point. Public code was inspected; its hardware results were not rerun here.

### FoldQuantVLA — 21 September 2026

**Sources:** [paper, arXiv:2609.24433v1](https://arxiv.org/html/2609.24433v1), [authors' implementation](https://github.com/cair-vinuni/FoldQuantVLA).

This is the named work behind the pasted native-low-bit citation. It connects calibration and folded transforms to native integer projection execution and studies selective INT8 retention. It distinguishes floating-point TensorRT controls from the BF16 PyTorch reference. Vision processing and several non-projection operations remain floating point; “native W4A4” does not mean every policy operation uses four-bit arithmetic.

**Reproduction task:** compare the same exported graph with floating and low-bit projections; include transformation, scaling, and fallback costs in timing. The public repository contains integration paths and source kernels. We did not compile its plugins, inspect every kernel, or independently validate its success-rate and speedup claims.

### CHASE-VLA — 2 October 2026

**Source:** [Hyun et al., arXiv:2610.02666v1](https://arxiv.org/abs/2610.02666v1); the record states acceptance at ACCV 2026.

The method conditions action-expert activation scales on the previous action chunk and denoising-step groups. Its reported evaluations include π₀.₅ and GR00T N1.6. Those results do not establish N1.7 compatibility or universal W4A4 quality. Traffic/storage savings scoped to quantized action-expert layers must not be reported as whole-policy energy gains.

**Reproduction task:** capture calibration state, chunk availability, first-query initialization, and reset behavior. Check that the scale predictor uses information causally available when inference starts. The paper is available; code/checkpoint availability was not established in this review.

## How to maintain this ledger

For a new claim, add the primary source URL, publication/version date, inspection date, and precise scope. Classify the evidence as **announcement**, **paper**, **code inspected**, **weights accessible**, **run reproduced**, or **hardware measured**. Several labels can apply, but none implies the next.

Store source and model revisions in the experiment manifest. If an upstream URL tracks `main`, it is a discovery link rather than an immutable experimental dependency. If access is unavailable, preserve that fact and use the CPU trace lab instead of substituting invented measurements.

The course's toy arithmetic, trace schema, lab exercises, and design targets are teaching constructions. They are not extracted hardware traces or published model benchmarks.

[Course overview](README.md) · [Workload atlas](Lecture-02.md)
