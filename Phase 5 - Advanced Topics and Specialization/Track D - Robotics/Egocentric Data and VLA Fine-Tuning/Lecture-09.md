# Lecture 09 — Choosing a Base Model and Fine-Tuning Stack

Pick a base checkpoint and the code that fine-tunes it by checking hardware, license, data format and action space, in that order, against pinned documentation.

**Prerequisites:** [Lecture 08](Lecture-08.md) (methods for human data), [VLA Foundations Lecture 16](../VLA%20Foundations%20to%20Frontier%202026/Lecture-16.md) (fine-tuning and normalization), [Lecture 28](../VLA%20Foundations%20to%20Frontier%202026/Lecture-28.md) (open VLA systems); optionally Lectures [25](../VLA%20Foundations%20to%20Frontier%202026/Lecture-25.md) (Physical Intelligence) and [26](../VLA%20Foundations%20to%20Frontier%202026/Lecture-26.md) (GR00T). **Effort:** reading 2 hours, exercise 2 hours. **Hardware:** CPU for the record and the arithmetic; the record targets one README track (8 GB, 24 GB or 40–80 GB). **Roles:** robot foundation-model (post-training) engineer, ML platform engineer.

**Misconception this lecture corrects:** "pick the model with the best published success rate, then work out the rest." What you can fine-tune is limited first by memory, then by the licenses of the code *and* the weights, then by the dataset format and normalization the stack expects, and then by whether its action space can absorb human-derived actions. Even two ports of one model differ: π0.5 in openpi reads LeRobot v2.1 data and writes `norm_stats.json`, while LeRobot's port reads v3.0 and needs quantiles in `meta/stats.json`.

## 1. The candidates and their pins

Pins are the state on 2026-10-26 (see [Sources](Sources.md)). License readings are an engineer's reading, not legal advice.

| Stack | Pin | Base checkpoint | Code license | Weights license |
|---|---|---|---|---|
| LeRobot: SmolVLA | Release **v0.6.1** (2026-08-03) | `lerobot/smolvla_base`, 450,046,176 params | Apache-2.0 | apache-2.0 (HF card) |
| LeRobot: π0 / π0.5 ports | v0.6.1 | `lerobot/pi0_base` (3.50B), `lerobot/pi05_base` (3.62B) | Apache-2.0 | **Conflict:** docs say "Apache 2.0 License"; HF cards are tagged `license: gemma`. Treat as Gemma-terms derivatives |
| openpi: π0, π0-FAST, π0.5 | HEAD `215abfb` (no tags or releases) | `gs://openpi-assets/checkpoints/{pi0_base, pi0_fast_base, pi05_base}` | Apache-2.0; the repo also ships `LICENSE_GEMMA.txt` | Read the Gemma Terms of Use before any commercial use |
| Isaac-GR00T N1.7 | `n1.7-release` (2026-04-18); HEAD `944fe70` | `nvidia/GR00T-N1.7-3B`, 3,144,016,000 params | Apache-2.0 | **Conflict:** README body says "fully commercially licensable under Apache 2.0"; its License section and the HF card say NVIDIA Open Model License, which allows commercial use. Use the latter |
| OpenVLA-OFT | HEAD `e4287e9` (no releases) | `openvla/openvla-7b`, 7.54B params | MIT | MIT |

Two access gates also count as license steps: LeRobot's π0.5 needs the gated `google/paligemma-3b-pt-224` tokenizer, and GR00T N1.7's backbone, Cosmos-Reason2-2B, is a gated model under the NVIDIA Open Model License. LeRobot v0.6.1 can also run GR00T N1.7 as a policy type; it removed N1.5 support (the docs say to pin `lerobot==0.5.1` to keep an N1.5 checkpoint).

**Upstream documented, not tested here** (LeRobot v0.6.1 `groot.mdx`):

```bash
lerobot-train --policy.type=groot --policy.base_model_path=nvidia/GR00T-N1.7-3B --policy.embodiment_tag=new_embodiment …
```

## 2. Format, normalization and action space

| Stack | Dataset format | Stats file | Default state/action normalization | Action space |
|---|---|---|---|---|
| LeRobot (SmolVLA, π0, π0.5) | LeRobot v3.0 | `meta/stats.json` | SmolVLA and π0: MEAN_STD; π0.5: QUANTILES (needs q01/q99) | Whatever the dataset's `action` feature holds, e.g. `float32 [6]` joints in `lerobot/svla_so101_pickplace`; SmolVLA `chunk_size=50` |
| openpi | LeRobot **v2.1** (pins LeRobot commit `0cf8648`; converters import the old `lerobot.common.datasets` path) | `assets/<config>/<repo_id>/norm_stats.json`, state and actions only | π0: mean/std; π0-FAST and π0.5: q01/q99 | Defined by your `…Inputs`/`…Outputs` transforms; π0.5 supports only its flow-matching action head |
| Isaac-GR00T N1.7 | LeRobot **v2** plus `meta/modality.json`; ships a v3→v2 converter | `meta/stats.json` plus `meta/relative_stats.json` | Min-max | Relative EEF, shared by robot and human data; `action_horizon` 40; custom robots use `NEW_EMBODIMENT` with `--modality-config-path` |
| OpenVLA-OFT | RLDS; a new dataset is registered in `configs.py`, `transforms.py` and `mixtures.py` | `dataset_statistics.json` in the checkpoint directory | `BOUNDS_Q99` (LIBERO, Bridge); `BOUNDS` (ALOHA) | Continuous, chunked, L1 regression; ALOHA uses absolute joint angles |

**[est]** A dataset written by current LeRobot (v3.0) probably will not load in openpi without conversion or an upgrade; this was not tested. Lecture 10 builds the dataset and its stats for whichever row you pick.

## 3. Documented memory per mode

| Stack | Inference | Adapter (LoRA) | Fine-tune, default settings | Full fine-tune |
|---|---|---|---|---|
| SmolVLA (LeRobot) | "~2GB" | `--peft` LoRA with documented default targets ([Lecture 11](Lecture-11.md)) | "~10–16GB" at batch 8 (hardware guide, "indicative") | Not documented |
| openpi π0 / π0-FAST | "> 8 GB" | "> 22.5 GB" | Not documented | "> 70 GB" |
| openpi π0.5 | "> 8 GB" | **No config** | Not documented | "> 70 GB" |
| LeRobot π0 / π0.5 | π0 "occupies 14GB" | Built-in default targets appear only in main's docs, after v0.6.1 | "~24–40GB … (24 GB tight at BS 1)" (hardware guide, "indicative") | π0.5 LIBERO "Sized for a single 80 GB GPU" |
| GR00T N1.7 | "16 GB+" | **None documented** | GPU "40 GB+"; peak "under ~35 GB" (projector and diffusion head) | `--tune-llm` / `--tune-visual`: "80 GB+" |
| OpenVLA-OFT | 15.9–18.0 GB | LoRA (`--lora_rank 32` in the documented command): minimum 25.6 GB (LIBERO, 1 image, batch 1), 62.5 GB at batch 8 (LIBERO, 2 images + proprio) | LoRA is the recipe | Not documented |

openpi states that its figures "assume a single GPU". As a cross-check, compute the floor that weights, gradients and AdamW state set for full fine-tuning (16 bytes per trainable parameter, Lecture 11).

**Tested here** (from the repository root; `EGO_COURSE` as in the [Lab guide](Lab-Guide.md)). Parameter counts are the documented totals from Section 1:

```bash
PYTHONPATH="$EGO_COURSE/labs" python3 - <<'EOF'
from budget_eval import training_state_bytes
models = {'lerobot/smolvla_base': 450_046_176, 'lerobot/pi05_base': 3.62e9,
          'nvidia/GR00T-N1.7-3B': 3_144_016_000, 'openvla/openvla-7b': 7.54e9}
for name, n in models.items():
    s = training_state_bytes(n, n)
    print(f"{name:22s} full-FT state {s['total_gb']:6.1f} GB  (bf16 weights alone {s['weights_gb']:5.1f} GB)")
EOF
```

```text
lerobot/smolvla_base   full-FT state    7.2 GB  (bf16 weights alone   0.9 GB)
lerobot/pi05_base      full-FT state   57.9 GB  (bf16 weights alone   7.2 GB)
nvidia/GR00T-N1.7-3B   full-FT state   50.3 GB  (bf16 weights alone   6.3 GB)
openvla/openvla-7b     full-FT state  120.6 GB  (bf16 weights alone  15.1 GB)
```

The arithmetic is consistent with the documents. π0.5's state alone is 57.9 GB before activations, so its full fine-tune is an 80 GB job. OpenVLA's 120.6 GB is consistent with a single-GPU recipe built on LoRA, and GR00T's 50.3 GB with a default that trains only the projector and diffusion head. This is a necessary condition, never proof that a run fits.

## 4. Does the stack have a path for human-derived actions?

Lecture 07 produced a 7-D action from human hands: a 6-D relative EEF step (translation, rotation vector) plus the gripper opening. Ask whether the stack can learn that from both sources:

* **GR00T N1.7:** its action space is relative EEF shared by robot and human data, and it is "pretrained on 20K hours of EgoScale human video". This is the closest match. You describe your layout in `modality.json`.
* **π0.5 (openpi base checkpoints):** Physical Intelligence reports human-to-robot transfer with π0.5, treating human video as an embodiment with 3D hand-position actions. This course's sources record no released openpi config for that recipe; check the repo at your pin before planning on one.
* **SmolVLA, the LeRobot π0 ports, OpenVLA-OFT:** no documented human path. The action is whatever your dataset defines. Co-training still works if robot and human actions mean the same thing (Lecture 10), but there is no documented human-data pretraining to build on.

## 5. The decision procedure

**Exercise / pseudocode:**

```text
for each candidate (stack, checkpoint, mode):
  1 hardware : documented requirement for that mode fits your track?      no -> drop, or rent a bigger track
  2 license  : code, weights, gated dependencies and every dataset in the
               mix allow your use (research or commercial)?              no -> drop
  3 format   : can you produce its dataset version (v3.0 / v2.1 /
               v2 + modality.json / RLDS) and the stats keys its
               normalization needs?                                       no -> budget a converter first
  4 actions  : same action meaning for robot and human data? documented
               human or relative-EEF path?                                no -> robot-only, or re-express actions
  5 pin      : record release or full SHA, checkpoint revision, doc tag
```

## 6. Three traps

* **"π0.5 with LoRA in openpi."** openpi's documented LoRA configs are `pi0_libero_low_mem_finetune` and `pi0_fast_libero_low_mem_finetune`; its config list has no π0.5 LoRA config, and its PyTorch port supports no LoRA at all. π0.5 in openpi means full fine-tuning (`pi05_libero`: batch 256, peak lr 5e-5, 30k steps, EMA 0.999) at "> 70 GB". On 24 GB, the openpi LoRA route is π0 or π0-FAST.
* **"GR00T with LoRA to fit 24 GB."** No LoRA option is documented. The default (`tune_llm=False`, `tune_visual=False`) is what keeps the peak under ~35 GB, on a GPU of 40 GB or more.
* **Reading LeRobot's main-branch docs.** Main is ahead of v0.6.1 (its `pyproject.toml` already says the unreleased 0.6.2). SmolVLA's `--rename_map` and the π0/π0.5 default LoRA targets are documented only on main, so a command copied from main may not exist in the release you installed. Read the docs at the `v0.6.1` tag.

Environments clash too: openpi pins `transformers==4.53.2`, `torch==2.7.1` and Python ≥3.11, while GR00T N1.7 uses transformers 4.57.3. Plan one environment per stack.

## 7. Exercise

1. **Worked example → the course task on the 8 GB track.** Step 1 drops openpi (inference "> 8 GB"), GR00T ("16 GB+") and OpenVLA-OFT (15.9 GB or more). SmolVLA inference ("~2GB") fits, but fine-tuning is documented at "~10–16GB" at batch 8, so it is a stretch goal at small batch (**[est]**, Lecture 11), with a rented 24 GB card as the fallback. Step 2: Apache-2.0 code and weights. Step 3: LeRobot v3.0, MEAN_STD, so `meta/stats.json` needs mean and std. Step 4: no human path, so the robot dataset must record the same 7-D action (6-D relative EEF plus gripper) as the human stream. Step 5: pin v0.6.1 and `lerobot/smolvla_base` at its revision.
2. **Guided variation.** Same task, but for a commercial product with a rented 80 GB GPU. Rerun steps 1–4 for GR00T N1.7 and for π0.5 in openpi. Record both sides of the GR00T weights-license conflict, the Gemma terms for π0.5, and which human datasets from your Lecture 02 record are still allowed.
3. **Independent transfer.** Run the procedure for your own task, hardware track and data.

**Deliverable:** `stack_selection.md`: the hardware track with the documented requirement for your mode (quoted, with source); licenses for code, weights and gated dependencies, with any conflict giving both values; the dataset format and conversion path; the normalization mode and the stats keys it needs; the action-space match; exact pins; and each rejected candidate with the step that rejected it. **Acceptance:** every requirement is quoted from a pinned document; at least two candidates are rejected at a named step; pins are a release tag or full SHA plus a checkpoint name; LoRA is claimed only where a config or flag is documented; the lab arithmetic is labeled as a necessary condition.

## Failure clinic and self-check

**Failure clinic:** a plan budgets π0.5 LoRA in openpi, then finds no config. A team buys a 24 GB card for GR00T expecting a LoRA mode. A command copied from LeRobot main fails on v0.6.1. The weights license is assumed to equal the code license (Apache-2.0 code, Gemma-terms weights). A v3.0 dataset is handed to openpi's pinned v2.1 loader. π0.5 is trained on a `stats.json` without q01/q99 (Lecture 10). openpi and GR00T are installed into one environment.

1. **Why is π0.5 LoRA in openpi on a 24 GB card not an option?** openpi has no π0.5 LoRA config and its PyTorch port has no LoRA. π0.5 there means full fine-tuning at "> 70 GB"; on 24 GB, use π0 or π0-FAST LoRA ("> 22.5 GB").
2. **The repo is Apache-2.0. Can you ship the weights in a product?** Not on that basis. Check the weights' own terms: Gemma terms for π0/π0.5, the NVIDIA Open Model License for GR00T N1.7, plus gated dependencies.
3. **Your robot data is 6-D joint positions and your human data is 7-D relative EEF. Which step fails, and what are the options?** Step 4. Re-express the robot actions in the same relative EEF space; treat human data as a separate embodiment (π0.5's human-to-robot work does, and GR00T N1 treated latent-action data as a "LAPA" embodiment) on a stack that supports it; or train robot-only.
4. **The lab says π0.5's full fine-tuning state is 57.9 GB (with the LeRobot port's 3.62B count), but openpi documents "> 70 GB". Why the gap?** The lab models weights, gradients and optimizer state only. Activations, the CUDA context and fragmentation are not modeled.
5. **Why cite LeRobot v0.6.1 and not main?** Main has unreleased changes (flags, default LoRA targets, policies), so a v0.6.1 install may lack what main's docs describe. The record must be reproducible from a release.

[Previous: learning from video without robot actions](Lecture-08.md) · [Course index](README.md) · [Next: building the fine-tuning dataset](Lecture-10.md)
