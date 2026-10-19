# Primary sources for the 2026 VLA frontier

The [October 18 source update](Current-Update-2026-10-18.md) supplements the original snapshot with current access and pinned artifact details.

Read these alongside [Lectures 25–30](Lecture-25.md), using the dated [frontier register](Frontier-Register.md) to distinguish announcements, papers, and executable releases. **Checked 2026-10-13.** Sources support only the nearby claims in the lectures; the course's calculations and fixture designs are original teaching exercises, not measured results from these systems.

## Physical Intelligence — read architecture, then artifacts

1. [π0 paper and version history](https://arxiv.org/abs/2410.24164) — original architecture and dated submissions.
2. [π0 code/weight release announcement](https://www.pi.website/blog/openpi) — distinguishes publication from release.
3. [π0.5 paper](https://arxiv.org/abs/2504.16054) — heterogeneous supervision and generalization experiment framing.
4. [Knowledge-insulation paper](https://arxiv.org/abs/2505.23705) — the training-dynamics question behind continuous action experts.
5. [π*0.6 paper](https://arxiv.org/abs/2511.14759v2), [full text](https://arxiv.org/html/2511.14759v2), [MEM paper](https://arxiv.org/abs/2603.03596), and [official research index](https://www.pi.website/blog) — distinguish announcement dates from arXiv submission dates.
6. [π0.7 announcement](https://www.pi.website/blog/pi07), [paper/version record](https://arxiv.org/abs/2604.15483v2), and [full text](https://arxiv.org/html/2604.15483v2) — arXiv access succeeded after publisher-hosted PDF retrieval returned 403.
7. [openpi code and checkpoint inventory](https://github.com/Physical-Intelligence/openpi) — choose a reproducible baseline here; inspect [code terms](https://github.com/Physical-Intelligence/openpi/blob/main/LICENSE) and [Gemma terms](https://github.com/Physical-Intelligence/openpi/blob/main/LICENSE_GEMMA.txt) separately.

## NVIDIA — read version history before installation

1. [GR00T N1 paper](https://arxiv.org/abs/2503.14734) — historical architecture and evaluation.
2. [N1.7 early-access announcement](https://forums.developer.nvidia.com/t/early-access-isaac-gr00t-n1-7-open-reasoning-vla-model-for-humanoid-robotics/366916), [release tags](https://github.com/NVIDIA/Isaac-GR00T/releases), and [commit history](https://github.com/NVIDIA/Isaac-GR00T/commits/main/) — distinguish EA and GA events.
3. [N1.7 technical article](https://developer.nvidia.com/blog/develop-humanoid-robot-policies-end-to-end-with-nvidia-isaac-gr00t/) — dated general-release context.
4. [Isaac-GR00T repository](https://github.com/NVIDIA/Isaac-GR00T), [hardware guidance](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/hardware_recommendation.md), and [FAQ](https://github.com/NVIDIA/Isaac-GR00T/blob/main/FAQ.md) — pin a revision before applying recipes.
5. [N1.7 3B model card](https://huggingface.co/nvidia/GR00T-N1.7-3B) — observed version/license labeling differs from the current repository; preserve the discrepancy in a model manifest.

## Google DeepMind — identify which model is accessible

1. [Initial Robotics/ER announcement](https://deepmind.google/blog/gemini-robotics-brings-ai-into-the-physical-world/).
2. [On-Device announcement](https://deepmind.google/blog/gemini-robotics-on-device-brings-ai-to-local-robotic-devices/).
3. [Robotics 1.5 / ER 1.5 announcement](https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/).
4. [ER 1.6 announcement](https://deepmind.google/blog/gemini-robotics-er-1-6/).
5. [Robotics 2 family announcement](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/).
6. [ER 2 model card](https://deepmind.google/models/model-cards/gemini-robotics-er-2/) and [On-Device 2 model card](https://deepmind.google/models/model-cards/gemini-robotics-on-device-2/) — keep API access, partner access, and local weights separate.

## Open policies and representation learning

| Study | Paper/date evidence | Author artifacts to inspect |
|---|---|---|
| OpenVLA-OFT | [Paper](https://arxiv.org/abs/2502.19645) | [Code and evaluation guides](https://github.com/moojink/openvla-oft) |
| SmolVLA | [Announcement](https://huggingface.co/blog/smolvla) | [Base checkpoint](https://huggingface.co/lerobot/smolvla_base), [versioned LeRobot documentation](https://huggingface.co/docs/lerobot/v0.4.3/en/smolvla) |
| RDT | [Paper](https://arxiv.org/abs/2410.07864) | [Author repository](https://github.com/thu-ml/RoboticsDiffusionTransformer) |
| MolmoAct 2 | [Announcement and updates](https://allenai.org/blog/molmoact2), [paper](https://arxiv.org/abs/2605.02881) | [Code and model inventory](https://github.com/allenai/molmoact2), [model collection](https://huggingface.co/collections/allenai/molmoact2-models) |
| LingBot-VLA | [1.0 release history](https://github.com/Robbyant/lingbot-vla), [2.0 paper](https://arxiv.org/abs/2607.06403) | [2.0 code](https://github.com/Robbyant/lingbot-vla-v2), [2.0 6B card](https://huggingface.co/robbyant/lingbot-vla-v2-6b) |
| Qwen-VLA | [Paper and version history](https://arxiv.org/abs/2605.30280) | Executable artifact availability not established by this review |
| VLAct | [Paper v2, 2026-10-06](https://arxiv.org/abs/2608.27550v2) | [Code/model zoo](https://github.com/starVLA/VLAct); read artifact-versus-paper configuration notes |

## Reasoning, world models, and evaluation

1. [DreamZero paper](https://arxiv.org/abs/2602.15922) and [code/checkpoints](https://github.com/dreamzero0/dreamzero) — joint world/action prediction with an actual deployment path.
2. [Continuous Reasoning for VLA](https://arxiv.org/abs/2606.00229) — a dated research study of intermediate representations; code/weight availability was not established here.
3. [In-Context VLA](https://arxiv.org/abs/2608.05738) — grounded context and tool acquisition; paper verified, executable artifacts not established.
4. [VLAQuantBench](https://arxiv.org/abs/2609.25376) — a recent closed-loop quantization study; follow its linked artifacts when reproducing it.
5. [RoboCasa365 official leaderboard](https://robocasa.ai/leaderboard.html) — benchmark-specific October 13 snapshot; inspect protocol and individual artifact links before comparing entries.
6. [Figure Helix 02](https://www.figure.ai/news/helix-02) and [Helix 2.5](https://www.figure.ai/news/helix-2-5-zero-shot-30-home-generalization) — whole-body hierarchy and a September 2026 generalization study; public weights were not established.

## Reading deliverable

Choose one paper, one repository, and one model card for the same candidate. Write a three-column audit: **reported claim**, **available artifact**, and **your executable test**. Record missing information explicitly. The objective is to know what you can reproduce and what remains an author-reported result.
