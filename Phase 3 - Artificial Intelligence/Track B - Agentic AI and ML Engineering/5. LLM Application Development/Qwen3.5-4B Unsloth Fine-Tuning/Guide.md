# Qwen3.5-4B-Base Fine-Tuning with Unsloth

<div class="course-identity auto-course" style="--course-accent: #be123c; --course-accent-rgb: 190, 18, 60;" markdown="1">
<div class="course-identity__icon">Q4BF</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · AI Workloads</p>
<p class="course-identity__title">Specialized course identity for Qwen3.5-4B-Base Fine-Tuning with Unsloth.</p>
<p class="course-identity__meta">Artifact: model or workload study · Measure: accuracy, latency, memory, throughput</p>
</div>
</div>


**Parent:** [Module 5B - LLM Application Development](../Guide.md) / [Phase 3 - Artificial Intelligence](../../../Guide.md)

> *Fine-tune `Qwen/Qwen3.5-4B-Base` with Unsloth, then measure the training, export, and inference costs that matter to an AI hardware engineer.*

**Layer mapping:** **L1** (Application & Framework) feeding **L2/L3** (compiler/runtime) and **L5/L6** (memory, precision, accelerator design).

**Prerequisites:** PyTorch basics, tokenizer/chat-template fluency, LoRA/PEFT concepts, and one CUDA GPU with enough memory for 16-bit LoRA.

**Role targets:** AI Engineer / LLM Fine-Tuning Engineer / ML Platform Engineer / AI Inference Engineer

**Course output:** a reproducible LoRA adapter, an evaluation report, an exported inference artifact, and a short hardware note explaining VRAM, throughput, and deployment tradeoffs.

---

## Course Currency Note

Model IDs, Unsloth support, and memory requirements change quickly. Checked 2026-10-18: [Unsloth documents](https://unsloth.ai/docs/models/qwen3.5/fine-tune) `Qwen/Qwen3.5-4B-Base` as part of its Qwen3.5 support and recommends 16-bit LoRA rather than 4-bit QLoRA for this family. The [official base-model metadata](https://huggingface.co/Qwen/Qwen3.5-4B-Base/tree/1001bb4d826a52d1f399e183466143f4da7b741b) was inspected at revision `1001bb4d826a52d1f399e183466143f4da7b741b`: architecture `Qwen3_5ForConditionalGeneration`, model type `qwen3_5`, and a chat template in tokenizer configuration. This is metadata evidence, not a local training result. The GPU recipe below remains unexecuted in the course check; package pinning and a short smoke run are required on the target GPU. The [CPU PyTorch lab](../../../2.%20Deep%20Learning%20Frameworks/PyTorch/CPU-Robot-Stop-Lab/Guide.md) is the executed prerequisite.

---

## Why This Matters for AI Hardware

Fine-tuning is not just a model-quality exercise. It changes the deployment artifact your hardware must serve:

- **Adapter vs merged weights:** unmerged LoRA adds extra low-rank matmuls at inference; merged weights remove that overhead but create a new model artifact.
- **Precision choices:** 16-bit LoRA, post-training quantization, and GGUF/AWQ/GPTQ exports all change memory traffic and kernel choices.
- **Context length:** SFT sequence length drives training activation memory, and evaluation context length drives KV-cache memory.
- **Data shape:** short instruction examples, long tool traces, and multimodal examples stress very different parts of the stack.
- **Serving target:** a fine-tuned 4B model can be trained on a workstation/cloud GPU, then quantized and profiled on Jetson or a custom runtime.

The hardware-relevant question is not "did loss go down?" It is: **what artifact did you create, how big is it, what precision does it use, and how fast does it run on the target hardware?**

---

## What You Will Build

| Artifact | Minimum expectation |
|----------|---------------------|
| Dataset | JSONL train/validation split with a documented chat template |
| Training run | Unsloth LoRA SFT run on `Qwen/Qwen3.5-4B-Base` |
| Adapter | Saved LoRA adapter with config, tokenizer, and training metadata |
| Evaluation | Base-vs-adapter comparison on held-out prompts |
| Export | Merged 16-bit model or quantized deployment artifact |
| Benchmark | VRAM, train tokens/s, eval tokens/s, adapter size, and output-quality notes |

---

## Hardware Budget

Use this course to learn the memory envelope before you start optimizing kernels.

| Target | Use it for | Notes |
|--------|------------|-------|
| 12-16 GB CUDA GPU | small LoRA SFT at modest sequence length | Use BF16 on Ampere or newer; use FP16 when BF16 is unavailable |
| 24 GB CUDA GPU | longer sequence length, larger batch, safer eval | Good local development target |
| Cloud L40S/A100/H100 | sweeps over LoRA rank, sequence length, and batch | Use when you want clean throughput data |
| Jetson Orin Nano / NX | post-training inference profiling only | Do not treat Jetson as the primary training target |

Keep the first run boring: sequence length 2048, LoRA rank 16, small clean dataset, and one evaluation script. Expand only after the pipeline is reproducible.

---

## 1. Define the Fine-Tuning Job

Start with a precise objective. Good objectives are narrow enough that the base model can be evaluated directly:

- domain Q&A over a hardware manual
- terminal command explanation for CUDA/Jetson workflows
- structured bug triage from logs
- tool-call argument generation from natural language
- short hardware-design tutoring with cited source snippets

Avoid turning the first run into a general assistant. A base model needs the training data to teach both behavior and format, so vague data creates vague failures.

### Dataset Schema

Use JSONL with explicit roles:

```json
{"system":"You are a concise AI hardware engineering assistant.","user":"Explain why Qwen decode is memory-bandwidth bound at batch 1.","assistant":"At batch 1, each generated token streams large weight matrices for GEMV while doing little arithmetic reuse..."}
```

Keep validation examples out of optimizer updates, and keep a third untouched test split out of model selection. The trainer may read validation data for evaluation:

```text
data/qwen35_train.jsonl
data/qwen35_valid.jsonl
data/qwen35_test.jsonl
```

### Chat Template

For supervised fine-tuning, the exact serialized text matters. Use the tokenizer's chat template when available instead of inventing one:

```python
def format_example(example, tokenizer):
    messages = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["user"]},
        {"role": "assistant", "content": example["assistant"]},
    ]
    return {
        "text": tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    }
```

Failure mode to watch for: loss falls, but generation never stops or repeats role tags. That usually means an EOS/chat-template mismatch, not a hardware problem.

---

## 2. Environment Setup

Use a fresh virtual environment. Qwen3.5 support may require current `transformers` and Unsloth packages.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --upgrade unsloth unsloth_zoo
pip install --upgrade datasets trl accelerate peft bitsandbytes torchvision pillow
```

Qwen3.5 support currently expects the current Unsloth stack and Transformers v5. If package resolution installs an older Transformers build, follow Unsloth's current install guidance before debugging training code.

Record the actual versions:

```bash
python - <<'PY'
import torch, transformers, trl, peft
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("trl", trl.__version__)
print("peft", peft.__version__)
PY
```

Put this output in your final report. Fine-tuning results without package versions are hard to reproduce.

---

## 3. Train with Unsloth LoRA

Use 16-bit LoRA first. Do not start with 4-bit QLoRA unless you have confirmed that your exact Qwen3.5 target and Unsloth version support it well enough for your quality bar.

```python
from unsloth import FastLanguageModel  # import before patched training libraries
import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen3.5-4B-Base"
MODEL_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
# Downloads the complete pinned model; allow multi-GB storage.
model_path = snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION)
MAX_SEQ_LENGTH = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    load_in_4bit=False,
    load_in_16bit=True,
    full_finetuning=False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    max_seq_length=MAX_SEQ_LENGTH,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

raw = load_dataset(
    "json",
    data_files={
        "train": "data/qwen35_train.jsonl",
        "validation": "data/qwen35_valid.jsonl",
    },
)

def to_text(example):
    messages = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["user"]},
        {"role": "assistant", "content": example["assistant"]},
    ]
    return {
        "text": tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    }

dataset = raw.map(to_text, remove_columns=raw["train"].column_names)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    args=SFTConfig(
        output_dir="runs/qwen35-4b-unsloth-lora",
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        eos_token="<|im_end|>",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_steps=10,  # reduce for a smoke run shorter than ten optimizer updates
        num_train_epochs=1,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=3407,
        data_seed=3407,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        report_to="none",
    ),
)

trainer.train()
trainer.save_model("artifacts/qwen35-4b-unsloth-lora")
tokenizer.save_pretrained("artifacts/qwen35-4b-unsloth-lora")
```

**Trainer API boundary:** the example uses current [TRL names](https://huggingface.co/docs/trl/sft_trainer): `processing_class`, with `dataset_text_field` and `max_length` inside `SFTConfig`. Some Unsloth snapshots patch older aliases; record the installed signatures instead of mixing snippets from different releases. Evaluation is explicitly enabled at each epoch; `eval_steps` by itself would not enable it.

**Token/loss contract:** this simple `text` dataset trains on the entire serialized sequence, including prompt tokens. It does not claim assistant-only loss. Inspect decoded inputs and labels before training; for assistant-only SFT use a supported conversational template/mask and verify prompt labels are `-100`. The inspected base tokenizer's default EOS is `<|endoftext|>`, while assistant turns end in `<|im_end|>`; configure the training/inference stop token consistently and test one turn without repeated role tags. Reject or account for examples whose answer is removed by truncation.

For a larger run, sweep one variable at a time:

| Variable | Starting value | Sweep |
|----------|----------------|-------|
| LoRA rank | 16 | 8, 16, 32 |
| Sequence length | 2048 | 1024, 2048, 4096 |
| Effective batch | 8 | 8, 16, 32 |
| Learning rate | 2e-4 | 1e-4, 2e-4, 5e-5 |

Do not compare runs unless dataset split, seed, sequence length, and eval prompts are fixed.

---

## 4. Evaluate Before You Export

After model selection is finished, run the same untouched test prompts against:

1. `Qwen/Qwen3.5-4B-Base`
2. base model + LoRA adapter
3. merged model, if you merge
4. quantized artifact, if you quantize

Measure both quality and systems behavior:

| Metric | Why it matters |
|--------|----------------|
| Validation loss | Fast sanity check, not final proof |
| Exact-format pass rate | Catches chat/template/schema regressions |
| Human or LLM-judge win rate | Measures task improvement |
| Hallucination / unsupported-claim rate | Important for hardware docs and manuals |
| Peak VRAM during train | Determines feasible local hardware |
| Train tokens/s | Captures loader, checkpointing, and GPU efficiency |
| Inference tok/s and TTFT | Determines serving cost |
| Adapter size | Determines OTA/update feasibility |

Minimal eval table:

| Model artifact | Pass rate | Win rate vs base | TTFT | Decode tok/s | Notes |
|----------------|-----------|------------------|------|--------------|-------|
| Base | | | | | |
| LoRA | | | | | |
| Merged | | | | | |
| Quantized | | | | | |

---

## 5. Merge, Export, and Deploy

Save the adapter first. Merge only after the adapter passes evaluation.

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="artifacts/qwen35-4b-unsloth-lora",
    max_seq_length=2048,
    load_in_4bit=False,
)

model.save_pretrained_merged(
    "artifacts/qwen35-4b-merged-16bit",
    tokenizer,
    save_method="merged_16bit",
)
```

For deployment, choose the artifact that matches the runtime:

| Runtime path | Artifact | What to check |
|--------------|----------|---------------|
| Transformers / PEFT | base + LoRA adapter | adapter load latency and extra matmul overhead |
| vLLM / TensorRT-LLM | merged model | version support for Qwen3.5, tokenizer, RoPE, QKV bias, and supported architecture |
| llama.cpp / GGUF | quantized GGUF | post-quant quality, prompt template, and tok/s |
| Jetson edge demo | small quantized artifact | RAM pressure, thermals, and sustained decode speed |

If you quantize after fine-tuning, evaluate again. A fine-tune that improves BF16 quality can still regress after aggressive quantization.

---

## 6. Hardware Note Template

Every student should finish with a one-page hardware note:

```markdown
# Qwen3.5-4B Unsloth Fine-Tuning Hardware Note

## Training setup
- GPU:
- VRAM:
- CUDA / PyTorch / Unsloth:
- Sequence length:
- LoRA rank:
- Effective batch:

## Results
- Peak train VRAM:
- Train tokens/s:
- Final train loss:
- Validation loss:
- Adapter size:
- Merged model size:

## Inference
- Runtime:
- Precision / quantization:
- Prompt length:
- TTFT:
- Decode tok/s:
- Peak memory:

## Interpretation
- What changed versus the base model?
- Did the adapter create measurable inference overhead?
- What precision is the best deployment compromise?
- What would this imply for SRAM, memory bandwidth, and kernel fusion on a custom accelerator?
```

The last four questions are the bridge from "I fine-tuned a model" to "I understand what this workload costs in hardware."

---

## Common Failure Modes

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Training OOMs immediately | sequence length or batch too high | lower sequence length, use gradient checkpointing, reduce batch |
| Loss falls but eval is worse | bad data, memorization, wrong eval template | inspect examples, deduplicate, fix template |
| Model repeats role tags | EOS/chat-template mismatch | verify tokenizer template and assistant termination |
| Exported model differs from adapter | merge/export path bug | compare base+adapter vs merged before quantization |
| Quantized artifact loses task skill | quantization too aggressive | try higher-bit quant or exempt sensitive tensors |
| Runtime crashes on Qwen weights | architecture support gap | verify QKV bias, RoPE layout, tokenizer, and model config support |

---

## Capstone

Fine-tune `Qwen/Qwen3.5-4B-Base` on a small hardware-engineering instruction dataset, then deploy the result through two inference paths:

1. base + LoRA adapter in Transformers
2. merged or quantized artifact in an inference runtime

Deliver:

- `data_card.md` describing source, filters, split, and template
- `train_config.yaml` or equivalent script arguments
- saved LoRA adapter
- base-vs-finetuned evaluation report
- inference benchmark table
- hardware note connecting the results to memory, precision, and serving cost

---

## Resources

| Resource | Why it matters |
|----------|----------------|
| [Qwen/Qwen3.5-4B-Base model card](https://huggingface.co/Qwen/Qwen3.5-4B-Base) | Source model, config, license, and usage notes |
| [Unsloth Qwen3.5 fine-tuning guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune) | Current Unsloth support, recommended precision, and training examples |
| [TRL SFTTrainer documentation](https://huggingface.co/docs/trl/sft_trainer) | Trainer API used for supervised fine-tuning |
| [PEFT LoRA documentation](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora) | Adapter mechanics and tunable parameters |
| [Phase 5 - Qwen Inference Optimization](../../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20C%20-%20Edge%20AI/Qwen%20Inference%20Optimization/README.md) | Companion inference course for profiling the exported model |

---

## Next

-> [Phase 5 - Qwen Inference Optimization](../../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20C%20-%20Edge%20AI/Qwen%20Inference%20Optimization/README.md) once the fine-tuned artifact is ready to benchmark.

**Study method:** use the [shared lesson cycle and completion rubric](../../../../Learning-Guide.md) with this course's existing labs. Predict first, test one variation, retain a failure case, and revisit the self-check after two days and one week.


## Carry this course into the physical-AI system

**Start from:** A pinned base model, tokenizer/template, and instruction dataset.

**Build and challenge:** Compare the unchanged base, adapter, and exported artifact on the same split. Record which graph components are trained; text instruction tuning alone does not create an action policy.

**Pass forward:** An adapter/export report and a list of missing action data, head, and embodiment contracts before VLA use. Continue with [the connected course](../../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20D%20-%20Robotics/VLA%20Foundations%20to%20Frontier%202026/README.md) and keep the artifact in the [Physical AI Design Studio](../../../../Physical-AI-Design-Studio.md).
