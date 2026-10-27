# Lecture 11 — Fine-Tuning Mechanics on 8, 24 and 80 GB

Budget a fine-tuning run before you rent a GPU, run it with a pinned stack, and choose the checkpoint you ship by closed-loop results instead of training loss.

**Prerequisites:** [Lecture 09](Lecture-09.md) (stack choice), [Lecture 10](Lecture-10.md) (dataset and stats), [VLA Foundations Lecture 16](../VLA%20Foundations%20to%20Frontier%202026/Lecture-16.md) (fine-tuning and adapters) and [Lecture 18](../VLA%20Foundations%20to%20Frontier%202026/Lecture-18.md) (training infrastructure). **Effort:** reading 1.5 hours, exercise 2 hours on CPU, plus a training run if you have a GPU track. **Hardware:** CPU for the budget lab; 8 GB, 24 GB or 40–80 GB for the run, depending on the stack (Section 5). **Roles:** post-training engineer, ML infrastructure engineer.

**Misconception this lecture corrects:** "if the model's weights fit in GPU memory, the fine-tune fits." Training memory is mostly optimizer state for the parameters you train plus activations that grow with batch size. Even a budget that fits is only a necessary condition: a measured run decides.

## 1. What a fine-tuning run keeps in memory

| Part | Bytes | Scales with | Modeled by `labs/budget_eval.py`? |
|---|---|---|---|
| Weights (frozen and trainable) | 2 per parameter in bf16 | Total parameters | Yes |
| Gradients | 2 per trainable parameter (bf16) | Trainable parameters | Yes |
| fp32 master copy | 4 per trainable parameter | Trainable parameters | Yes |
| AdamW moments | 2 × 4 per trainable parameter | Trainable parameters | Yes |
| Activations kept for the backward pass | Depends on per-device batch, image tokens, action-chunk length and depth | Batch size | No: you pass an assumed `activation_gb` |
| CUDA context, allocator fragmentation, framework workspace | Stack-dependent | Stack | No: `fits()` reserves 1 GB by default |

$$
\text{state bytes} \approx 2\,N_{\text{total}} + (2 + 4 + 8)\,N_{\text{trainable}}
$$

A trainable parameter costs about 16 bytes and a frozen one 2 bytes. The lab's `GB` is \(10^9\) bytes. Its docstring warns that the unmodeled parts "often dominate".

## 2. Which parameters you train, and why "fits" is not enough

SmolVLA's base checkpoint `lerobot/smolvla_base` has 450,046,176 parameters (Hugging Face safetensors total) and defaults to `freeze_vision_encoder=True` and `train_expert_only=True`. The fact sheet behind this course does not give the action expert's parameter count, so the expert-only row below **assumes 100M trainable parameters**. The LoRA row **assumes 32 projections of 1024 × 1024 at rank 64**, a stand-in shape rather than SmolVLA's real layers; read the real shapes from your model's config. A LoRA adapter on a \(d_\text{in} \times d_\text{out}\) projection adds \(r(d_\text{in} + d_\text{out})\) parameters.

**Tested here** (from the repository root). **Negative case:** full fine-tuning on 8 GB.

```bash
EGO_COURSE='Phase 5 - Advanced Topics and Specialization/Track D - Robotics/Egocentric Data and VLA Fine-Tuning'
PYTHONPATH="$EGO_COURSE/labs" python3 - <<'EOF'
from budget_eval import training_state_bytes, lora_params, fits
P = 450_046_176                                # lerobot/smolvla_base safetensors total
EXPERT = 100_000_000                           # ASSUMPTION: expert-only trainable count
LORA = 32 * lora_params(1024, 1024, rank=64)   # ASSUMPTION: 32 projections of 1024x1024
print("LoRA adapter params:", LORA)
state = {}
for name, total, trainable in [("full", P, P), ("expert", P, EXPERT), ("lora", P + LORA, LORA)]:
    s = training_state_bytes(total, trainable)
    state[name] = s["total_gb"]
    print(f"{name:7s}", " ".join(f"{k}={v:.3f}" for k, v in s.items()))
for act in (0.0, 2.0, 4.0, 6.0):               # ASSUMED activation memory, GB
    print(f"activations={act:.0f} GB, fits 8 GB:", {n: fits(g, 8.0, act) for n, g in state.items()})
EOF
```

```text
LoRA adapter params: 4194304
full    weights_gb=0.900 grads_gb=0.900 master_gb=1.800 optimizer_gb=3.600 total_gb=7.201
expert  weights_gb=0.900 grads_gb=0.200 master_gb=0.400 optimizer_gb=0.800 total_gb=2.300
lora    weights_gb=0.908 grads_gb=0.008 master_gb=0.017 optimizer_gb=0.034 total_gb=0.967
activations=0 GB, fits 8 GB: {'full': False, 'expert': True, 'lora': True}
activations=2 GB, fits 8 GB: {'full': False, 'expert': True, 'lora': True}
activations=4 GB, fits 8 GB: {'full': False, 'expert': True, 'lora': True}
activations=6 GB, fits 8 GB: {'full': False, 'expert': False, 'lora': True}
```

* Full fine-tuning spends half its 7.2 GB on AdamW moments, and it fails on an 8 GB card even with zero activations: 7.2 GB of state plus the 1 GB reserve is 8.2 GB. Expert-only training cuts the state to about 2.3 GB, but passes only while activations stay under about 4.7 GB (8 − 2.3 − 1). With LoRA, the frozen bf16 weights are almost the whole state.
* Now compare a documented figure: the LeRobot v0.6.1 hardware guide gives SmolVLA fine-tuning an indicative peak of "~10–16GB" at batch 8. That is far above the assumed 2.3 GB of state, which suggests that at that batch most memory is not optimizer state. A `True` from `fits()` rules nothing in.

## 3. Arithmetic against documented requirements

**Tested here** (from the repository root, with `EGO_COURSE` set as above). Rows with 0 trainable parameters show only the frozen bf16 weights; the π0 rows use the LeRobot port's 3.50B count as a stand-in for openpi's checkpoint.

```bash
PYTHONPATH="$EGO_COURSE/labs" python3 - <<'EOF'
from budget_eval import training_state_bytes
for label, total, trainable in [("pi0 full (3.50B)", 3.50e9, 3.50e9), ("pi0 frozen weights", 3.50e9, 0),
                                ("pi0.5 full (3.62B)", 3.62e9, 3.62e9), ("OpenVLA-7B frozen weights", 7.54e9, 0),
                                ("GR00T N1.7 frozen weights", 3_144_016_000, 0)]:
    print(f"{label:26s} {training_state_bytes(total, trainable)['total_gb']:6.2f} GB")
EOF
```

```text
pi0 full (3.50B)            56.00 GB
pi0 frozen weights           7.00 GB
pi0.5 full (3.62B)          57.92 GB
OpenVLA-7B frozen weights   15.08 GB
GR00T N1.7 frozen weights    6.29 GB
```

| Configuration | Lab arithmetic (state only) | Documented requirement (pin) |
|---|---|---|
| π0 / π0.5 full | 56.00 / 57.92 GB | openpi: "> 70 GB", A100 80 GB / H100, single GPU (README @ `215abfb`). LeRobot π0.5 LIBERO: "Sized for a single 80 GB GPU" with gradient checkpointing, bf16, batch 64 (v0.6.1 `pi05.mdx`) |
| openpi π0 LoRA | 7.00 GB + adapters | "> 22.5 GB", RTX 4090 (README @ `215abfb`) |
| OpenVLA-OFT, LoRA rank 32 | 15.08 GB + adapters | 25.6 GB minimum at batch 1 (LIBERO, 1 image); 62.5 GB at batch 8 (LIBERO, 2 images + proprio) (OFT FAQ; `LIBERO.md` says "~62 GB") |
| GR00T N1.7 default (projector + diffusion head) | 6.29 GB + state for the projector and diffusion head (their parameter count is not in this course's sources) | "Minimum: 1 GPU with 40 GB+ VRAM", peak "under ~35 GB" (README @ `944fe70`) |

Every documented figure sits well above the arithmetic. The gap is most likely activations, overhead and whatever precision each stack really uses, which may not match the lab's bf16/fp32 defaults. This course did not measure it. Use arithmetic to rule configurations out, documented requirements to plan, and a measured run to commit.

## 4. The levers, and what each costs

* **Precision.** bf16 weights and gradients are the lab default. LeRobot's π0.5 example uses `--policy.dtype=bfloat16`. The openpi PyTorch port trains in full bf16 (default) or full fp32 and does not support mixed precision.
* **Gradient checkpointing** keeps fewer activations and recomputes them in the backward pass, trading compute for memory (`--policy.gradient_checkpointing=true` in the π0.5 example).
* **Batch size vs gradient accumulation.** The hardware guide's tip: memory is roughly linear in batch size, so drop the batch and accumulate gradients. The effective batch is per-device batch × accumulation steps × GPUs; batch 4 × 16 steps keeps SmolVLA's documented batch of 64 on one small GPU, at the cost of 16 forward/backward passes per optimizer step. Check the v0.6.1 config for the flag name. Record both numbers.
* **Freezing.** SmolVLA trains only the expert by default. LeRobot π0.5's `train_expert_only=true` / `freeze_vision_encoder=true` gives "less memory, at some cost in success rate". GR00T tunes the projector and diffusion head by default; `--tune-llm` or `--tune-visual` needs "80 GB+ per GPU".
* **LoRA.** LeRobot: install `lerobot[peft]` and add `--peft.method_type=LORA --peft.r=64 --peft.lora_alpha=64`, with a learning rate about 10× the full fine-tuning one (1e-3); for SmolVLA the default targets are `q_proj`/`v_proj` of the LM expert plus the state and action projections. openpi has `pi0_libero_low_mem_finetune` and `pi0_fast_libero_low_mem_finetune`, **no π0.5 LoRA config**, and its PyTorch port does not support LoRA. **GR00T documents no LoRA option.** OpenVLA-OFT uses `--lora_rank 32`.

## 5. Tracks and upstream commands

| Track | What to train | Documented basis (indicative where the guide says so) |
|---|---|---|
| **8 GB** (RTX 4060) | ACT (`--policy.type=act`, "Start with batch size 8"): inside the guide's figures. SmolVLA expert-only or LoRA at batch ≈ 2–4 with accumulation: **[est] stretch goal, verify empirically**. Otherwise rent | ACT peak "~2–6GB" at batch 8; no doc claims SmolVLA fine-tuning fits 8 GB; openpi inference alone needs "> 8 GB" |
| **24 GB** | SmolVLA at batch 8; openpi π0 / π0-FAST LoRA | SmolVLA "~10–16GB" at batch 8; openpi LoRA "> 22.5 GB" |
| **40–80 GB** | GR00T N1.7 default; π0.5 full; OpenVLA-OFT | "40 GB+"; "> 70 GB" / "single 80 GB GPU"; 25.6–73.5 GB |

To rent, LeRobot documents `--job.target=<flavor>` for HF Jobs. For several GPUs, openpi's figures "assume a single GPU" and FSDP is `--fsdp-devices <n>`; GR00T uses `uv run torchrun --nproc_per_node=8 …`. Training-time anchors from the hardware guide (5 epochs, about 45k frames): ACT on an RTX 4090/3090 at batch 8, ~30–60 min; SmolVLA on L4/A10G at batch 4, ~3–6 h; π0/π0.5 on A100 40 GB at batch 4, ~4–8 h.

**Upstream documented, not tested here.** Commands quoted from each stack's docs at the pins above. The SmolVLA doc says 20k steps take "~4 hrs on a single A100 GPU"; on smaller cards lower `--batch_size`, and use the v0.6.1 docs, not main, whose flags differ. openpi's `pi05_libero` is a full fine-tuning config (batch 256, peak lr 5e-5, 30k steps), so it belongs on the 80 GB track; on 24 GB, write a `TrainConfig` modeled on `pi0_libero_low_mem_finetune` instead.

```bash
# SmolVLA, LeRobot v0.6.1
cd lerobot && lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=${HF_USER}/mydataset --batch_size=64 --steps=20000 --output_dir=outputs/train/my_smolvla --job_name=my_smolvla_training --policy.device=cuda --wandb.enable=true
# openpi @ 215abfb: normalization stats, then training
uv run scripts/compute_norm_stats.py --config-name pi05_libero
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite
# Isaac-GR00T README @ 944fe70 (N1.7), default projector + diffusion-head tuning
CUDA_VISIBLE_DEVICES=0 uv run python gr00t/experiment/launch_finetune.py --base-model-path nvidia/GR00T-N1.7-3B --dataset-path demo_data/cube_to_bowl_5 --embodiment-tag NEW_EMBODIMENT --modality-config-path examples/SO100/so100_config.py --num-gpus 1 --output-dir /tmp/test_finetune --max-steps 2000 --global-batch-size 32 --dataloader-num-workers 4
```

## 6. Checkpoint selection and the run record

Training loss measures how well the policy imitates the demonstration actions, not whether the robot finishes the task. **Selection rule:** before training, fix the candidate checkpoints (for example every 5k steps and the last); evaluate each in closed loop on a small *selection* set of held-out-condition episodes with fixed seeds; pick the highest success rate, ties going to the earlier checkpoint; then report on a separate *test* set (Lecture 12). Selecting on the test episodes inflates the result. Use the same rule for the robot-only and the robot-plus-human runs, or the comparison measures the rule.

**Run record fields:** stack and pin (LeRobot v0.6.1, openpi `215abfb`, GR00T release `n1.7-release` or the commit you used, e.g. `944fe70`, and read the docs at that same ref); base checkpoint and its Hugging Face revision; dataset repo ID, revision and mixture weights; stats file and its hash, normalization mode; trainable set (full, expert-only, or LoRA rank, alpha and targets); precision and checkpointing; per-device batch, accumulation, GPU count and effective batch; learning rate, schedule, steps and seed; GPU model, predicted state (lab) and **measured** peak memory; wall-clock; checkpoints saved, the selection rule and its result.

## 7. Exercise

**Exercise / pseudocode** — CPU for steps 1–2; step 3 on your GPU track.

1. **Worked example →** rerun Sections 2–3. Write down which rows are lab arithmetic, which are documented, and which assumptions (the 100M expert, the LoRA shapes, the activation values) each depends on.
2. **Guided variation.** Change the LoRA rank to 16, 32 and 128 and the expert assumption to 50M and 200M. For each, find the largest `activation_gb` for which `fits()` stays `True` on 8 GB and on 24 GB.
3. **Independent transfer.** Budget the stack you chose in Lecture 09 and put the documented requirement next to it. On a GPU, run the documented command for a few hundred steps at your planned batch, record measured peak memory, and fill in the run record. **Reduced CPU alternative:** fill the record with planned values and mark every measured field "not run".

**Deliverable:** `finetune_record.md`, containing the memory budget table, the run record and the checkpoint-selection rule. **Acceptance:** every memory figure is labeled lab arithmetic, documented (with pin) or measured; every assumption is stated; the selection rule is dated before the first training step; no field is blank (unrun fields say "not run").

## Failure clinic and self-check

**Failure clinic:** `fits()` said `True` and the run ran out of memory at step 1, because activations were guessed too low. Full SmolVLA fine-tuning was attempted on an 8 GB card. The 80 GB example's `--batch_size=64` was copied onto a 24 GB GPU. Gradient accumulation changed between runs and nobody recorded the effective batch. A π0.5 LoRA config was expected in openpi, or a LoRA flag in GR00T; neither is documented. Flags came from LeRobot main instead of v0.6.1. LoRA was run at the full fine-tuning learning rate. The lowest-loss checkpoint was shipped and did worse in closed loop than an earlier one.

1. **Why does full SmolVLA fine-tuning fail `fits()` on 8 GB even with zero activations?** Its state is 7.2 GB (16 bytes × 450M) and the reserve is 1 GB, which totals more than 8 GB before any activation is stored.
2. **`fits()` returns `True` for expert-only training with 3 GB of activations. Can you promise the run fits?** No. It is a necessary condition. The indicative 10–16 GB at batch 8 in LeRobot's guide suggests activations and overhead dominate; only a measured run settles it.
3. **How do you keep an effective batch of 64 on an 8 GB card, and what do you record?** Something like per-device batch 4 with 16 accumulation steps. Record both, plus GPU count, so the effective batch and optimizer-step count can be reproduced.
4. **Why select checkpoints on held-out closed-loop episodes instead of validation loss?** Loss measures imitation of demonstration actions, not task success, and small action errors compound in closed loop. Selecting on separate episodes from the test set also keeps the final number unbiased.

[Previous: building the fine-tuning dataset](Lecture-10.md) · [Course index](README.md) · [Next: evaluating fine-tuned policies](Lecture-12.md)
