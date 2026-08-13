# Lecture 02 — Quantization and Format Conversion: Getting Gemma 4 to the Right Bits

**Collection:** [Gemma 4 Edge Deployment](README.md) | **Previous:** [← Lecture 01](Lecture-01.md) | **Next:** [Lecture 03 →](Lecture-03.md)

---

Gemma 4's architecture is designed for edge deployment, but the raw checkpoint is still BF16 — 8.6 GB for the 4B model. To fit the runtime targets from Lecture 01 (2.2 GB INT4 for 4B, 6.4 GB for 12B), you need to get from BF16 to the right quantized format **without unacceptable accuracy degradation**. This lecture covers the full quantization chain: why Gemma 4 quantizes more cleanly than most models, the specific algorithms (GPTQ, AWQ, K-quants), calibration strategies, and the format conversion step from a quantized checkpoint to the runtime format your deployment stack consumes.

---

## Learning objectives

1. Explain why Gemma 4's **QK-norm and GeGLU** architecture reduces quantization error vs models without them.
2. Choose between **GPTQ, AWQ, and K-quants (GGUF)** for a given target runtime and accuracy requirement.
3. Build a calibration dataset appropriate for Gemma 4's 128K-context window and run a calibration pass.
4. Convert a quantized Gemma 4 checkpoint to **GGUF** (llama.cpp), **TFLite FlatBuffer / LiteRT format**, and understand the ExecuTorch `.pte` and TRT engine paths.
5. Interpret quantization error metrics (`perplexity Δ`, `kv_cache_max_quant_error`, accuracy on downstream tasks) and set acceptance thresholds for edge deployment.

---

## 1. Why Gemma 4 quantizes cleanly

Before choosing an algorithm, understand why Gemma 4 is a good quantization target.

### 1.1 The two pathologies that kill quantization

Most INT4/INT8 quantization degradation comes from two phenomena:

```text
Pathology 1: ACTIVATION OUTLIERS
  In some layers, a small fraction of activation values are 10–100× larger than
  the rest (often channel-specific). A per-tensor scale sized to the outlier
  means every normal value loses ~7 bits of effective precision.

  Example (without protection): activation range = [-0.1, 47.3]
    per-tensor scale = 47.3 / 127 ≈ 0.37 (for INT8)
    value 0.1 quantizes to round(0.1 / 0.37) = round(0.27) = 0 → pure noise

Pathology 2: ATTENTION LOGIT OVERFLOW
  At long sequence lengths, Q @ K^T can produce very large logits. If Q or K
  values grow with context (as in pre-norm attention without QK-norm), the
  logit distribution shifts — and a fixed per-tensor scale chosen during
  calibration on short sequences is wrong at deployment time on long sequences.
```

**Gemma 4's answers:**

```text
QK-norm (for Pathology 2):
  Q = rms_norm(Q, g_q)   → Q values ∈ [-g_q_max, +g_q_max], bounded and predictable
  K = rms_norm(K, g_k)   → same for K
  The attention logit scale is then determined by g_q × g_k / sqrt(head_dim),
  which is constant across context lengths. Calibration on 512 tokens is valid
  for 128K tokens — no distribution shift.

GeGLU gating (for Pathology 1, partial):
  out = GELU(W_gate(x)) * W_up(x)
  The GELU gate saturates large positive/negative values to near-zero/near-one.
  This soft clipping reduces extreme activation values in the gate path,
  partially suppressing outliers in the FFN activation tensor.
```

The result: Gemma 4 at Q4_K_M (GGUF 4-bit with K-quants) typically loses **< 0.3 perplexity points** on standard benchmarks, compared to 0.5–1.5 for models without these architectural protections. For INT8, the degradation is often immeasurable.

### 1.2 What still goes wrong

Despite the protections, two quantization failure modes remain:

1. **Embedding table quantization**: The 256K × 2560 embedding is large (1.3 GB BF16). INT4 quantization of embeddings is lossy because each embedding vector is short (2560 values) — per-row quantization is viable but per-channel (which would be ideal) requires wide embeddings. Most runtimes keep the embedding in INT8 or BF16 even when the rest is INT4.

2. **First and last layers**: Layer 0 input projection and the final lm_head projection always have higher sensitivity. Most quantization pipelines skip INT4 for these and keep them in INT8 or BF16. Budget ~50–100 MB for these layers at higher precision.

---

## 2. GPTQ — for TensorRT-LLM and vLLM paths

**GPTQ** (Frantar et al., 2022) is the standard for weight-only INT4 quantization for transformer inference engines. It quantizes each weight row independently using second-order Hessian information from calibration data.

```python
# GPTQ for Gemma 4 4B using AutoGPTQ:
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# Load Gemma 4 4B from HuggingFace:
model_name = "google/gemma-4-4b-it"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", torch_dtype="bfloat16")

# GPTQ config: 4-bit, group_size 128 (recommended for Gemma 4):
quant_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,         # 128 rows share one scale — balance precision vs overhead
    desc_act=False,         # disable act-order reordering (not needed with QK-norm)
    damp_percent=0.1,       # Hessian damping for numerical stability
    static_groups=False,
)

# Calibration dataset: use diverse multilingual data matching 128K context capability
# Keep calibration sequences SHORT (512–2048 tokens) to match most deployment scenarios:
def get_calibration_data(tokenizer, n_samples=128, seq_len=2048):
    from datasets import load_dataset
    data = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    samples = []
    for sample in data:
        tokens = tokenizer(sample["text"], max_length=seq_len, truncation=True,
                           return_tensors="pt")
        samples.append(tokens["input_ids"])
        if len(samples) >= n_samples: break
    return samples

calib_data = get_calibration_data(tokenizer, n_samples=128, seq_len=2048)

# Run GPTQ quantization (requires 1× A100 or 4× A10 for 4B, more for 12B+):
quantized_model = AutoGPTQForCausalLM.from_pretrained(
    model_name, quant_config, calib_data=calib_data
)
quantized_model.save_quantized("gemma4-4b-gptq-int4")
```

**Calibration notes for Gemma 4:**

- **Do NOT calibrate at 128K context** unless your deployment is always 128K. Calibrating at the max context biases the scales toward the rare long-context distribution. Calibrate at your typical deployment length (512–2048 tokens for chat).
- Use **diverse data**: Gemma 4's 256K vocab covers 100+ languages. If your deployment is monolingual, calibrate on that language to get the best per-language accuracy. If multilingual, use C4-multilingual or mC4.
- **128 samples × 2048 tokens** is the standard minimum. More samples reduce variance but exhibit diminishing returns past 512.

**GPTQ output:** a `model.safetensors` with INT4 packed weights (2 FP16 values share one 32-bit word when `group_size=128`). Consumed by TensorRT-LLM and vLLM natively.

---

## 3. AWQ — for MLC-LLM and the activation-aware path

**AWQ** (Lin et al., 2023) is an alternative to GPTQ that protects **salient channels** (those with high activation magnitude) by scaling them before quantization:

```python
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained("google/gemma-4-4b-it")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-4b-it")

# AWQ needs activation statistics — collect with calibration:
model.quantize(
    tokenizer,
    quant_config={
        "zero_point": True,     # use zero_point offset (better for asymmetric data)
        "q_group_size": 128,    # same group size as GPTQ
        "w_bit": 4,             # INT4 weights
        "version": "GEMM",      # optimized for GEMM kernels vs "GEMV" for small batch
    }
)
model.save_quantized("gemma4-4b-awq-int4", safetensors=True)
```

**GPTQ vs AWQ for Gemma 4:**

| Criterion | GPTQ | AWQ |
|-----------|------|-----|
| Accuracy at INT4 | Slightly better (uses Hessian) | Slightly worse but fast calibration |
| Calibration time (4B) | ~15 min on A100 | ~5 min on A100 |
| Runtime support | TRT-LLM, vLLM, llama.cpp (via GGUF convert) | MLC-LLM native, vLLM, llama.cpp |
| Activation-aware channel scaling | No | **Yes** (key advantage for outlier models) |
| Best for Gemma 4? | **Yes** (QK-norm suppresses outliers, Hessian wins) | Good when outliers exist; less needed for Gemma 4 |

For Gemma 4, **GPTQ is preferred** because QK-norm already suppresses the outlier channels that AWQ's activation-aware scaling targets. Without outliers, GPTQ's second-order information gives cleaner quantization.

---

## 4. K-quants (GGUF) — for llama.cpp on Jetson

**K-quants** are llama.cpp's mixed-precision quantization format, shipped in the **GGUF** file format. They quantize different layers at different bit widths depending on their sensitivity.

### 4.1 GGUF K-quant formats for Gemma 4

| Format | Avg bits | Strategy | Size (4B) | Perplexity Δ |
|--------|----------|----------|-----------|--------------|
| Q2_K | 2.63 | Ultra-aggressive | ~1.6 GB | ~3.5 pp |
| Q4_K_S | 4.37 | 4-bit, small block | ~2.5 GB | ~0.4 pp |
| **Q4_K_M** | **4.85** | **4-bit, medium (recommended)** | **~2.8 GB** | **~0.2 pp** |
| Q5_K_M | 5.68 | 5-bit, medium | ~3.3 GB | ~0.1 pp |
| Q6_K | 6.56 | 6-bit blocks | ~3.8 GB | ~0.05 pp |
| Q8_0 | 8.5 | 8-bit, fast | ~4.6 GB | ~0.01 pp |

**Q4_K_M is the standard recommendation for Gemma 4 on Jetson Orin** — it fits all model sizes comfortably while losing < 0.2 perplexity points on standard benchmarks.

### 4.2 Converting Gemma 4 to GGUF

```bash
# Step 1: Clone llama.cpp (ensure Gemma 4 support — check tag ≥ b3500):
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp

# Step 2: Install conversion deps:
pip install -r requirements.txt

# Step 3: Convert HuggingFace Gemma 4 to GGUF (F16 first):
python convert_hf_to_gguf.py \
    /path/to/gemma-4-4b-it \
    --outfile gemma4-4b-f16.gguf \
    --outtype f16

# Step 4: Quantize to Q4_K_M (this is fast, runs on CPU, no GPU needed):
./build/bin/llama-quantize gemma4-4b-f16.gguf gemma4-4b-Q4_K_M.gguf Q4_K_M

# Step 5: Verify the result:
./build/bin/llama-perplexity -m gemma4-4b-Q4_K_M.gguf \
    -f wikitext-2-raw/wiki.test.raw --ctx 2048
```

**Gemma 4-specific note:** Ensure the `convert_hf_to_gguf.py` script has a Gemma 4 tokenizer handler. Gemma 4's `256K` vocabulary and the `model_type: "gemma4"` identifier in `config.json` must be recognized. If using an older llama.cpp, check that `models/gemma4` architecture support is present.

### 4.3 Imatrix quantization — better accuracy at the same bit count

`imatrix` (importance matrix) is llama.cpp's calibration-aware quantization:

```bash
# Build an importance matrix (calibration pass, ~30 min on CPU for 4B):
./build/bin/llama-imatrix \
    -m gemma4-4b-f16.gguf \
    -f calibration_data.txt \    # 512 × 2048-token text samples
    -o gemma4-4b.imatrix \
    --ctx 2048 -b 512

# Quantize with imatrix (lower perplexity than standard K-quant at same bits):
./build/bin/llama-quantize \
    --imatrix gemma4-4b.imatrix \
    gemma4-4b-f16.gguf \
    gemma4-4b-Q4_K_M_imatrix.gguf \
    Q4_K_M

# Result: Q4_K_M with imatrix typically ≈ Q5_K_M accuracy at Q4_K_M size
```

---

## 5. LiteRT format (Google AI Edge) — the PDL-native path

**LiteRT** (Google AI Edge, formerly TensorFlow Lite) is Google's on-device inference runtime. For Gemma 4 specifically, Google provides a direct export path via the **AI Edge Torch** library:

```python
# pip install ai-edge-torch

import ai_edge_torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load Gemma 4 4B:
model_id = "google/gemma-4-4b-it"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
model.eval()

# Export to LiteRT FlatBuffer (.tflite):
sample_input = torch.ones((1, 128), dtype=torch.long)  # (batch, seq_len)
edge_model = ai_edge_torch.convert(
    model,
    sample_args=(sample_input,),
    quant_config=ai_edge_torch.quantize.pt2e_quantizer.PT2EQuantizerConfig(
        is_per_channel=True,            # per-channel INT8 for weights
        is_symmetric=True,
    )
)
edge_model.export("gemma4-4b-int8.tflite")
```

**For INT4 with LiteRT (AI Edge Gemma-specific path):**

Google provides pre-quantized LiteRT variants for Gemma 4 via Kaggle Models. These are INT4/INT8 mixed-precision FlatBuffers with the embedding table preserved in INT8:

```bash
# Download via kaggle CLI (requires Kaggle API key + Gemma 4 terms acceptance):
kaggle models instances versions download \
    google/gemma/tfLite/gemma-4-4b-it-int4/1 \
    --untar
# → gemma4-4b-it-int4.tflite (~2.4 GB)
```

**LiteRT vs GGUF on Jetson:**

| Criterion | LiteRT (.tflite) | GGUF (llama.cpp) |
|-----------|-----------------|-----------------|
| Optimized for Android/embedded | **Yes (primary target)** | No (desktop/server-first) |
| CUDA backend on Jetson | Limited (via GPU delegate) | **Yes (CUDA, full)** |
| Google AI Edge ecosystem | **Yes (first-class)** | No |
| Kernel optimization for Jetson | Basic | **Full CUDA kernels** |
| Best for Jetson CUDA | No | **Yes** |
| Best for Jetson DLA / ARM | **Yes** | No |

**Recommendation:** For Jetson with CUDA backend (Orin/Thor), use **GGUF + llama.cpp** or **TRT-LLM** for peak throughput. Use **LiteRT** when deploying to Jetson's DLA accelerator, ARM CPU, or cross-deploying the same model to Android/iOS.

---

## 6. ExecuTorch `.pte` format — Meta's edge path for Gemma 4

ExecuTorch (Meta, 2024) exports PyTorch models to a portable `.pte` format for edge devices. Google added Gemma support:

```python
# export_gemma4_et.py  (requires executorch nightly)
import torch
from executorch.exir import to_edge
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-4b-it", torch_dtype=torch.float32
)
model.eval()

# Export with XNNPACK backend (optimized for ARM NEON / Jetson CPU fallback):
example_inputs = (torch.ones(1, 64, dtype=torch.long),)
edge_program = to_edge(
    torch.export.export(model, example_inputs),
    compile_config=torch.backends.xnnpack.XnnpackPartitioner()
)
et_program = edge_program.to_executorch()

with open("gemma4-4b.pte", "wb") as f:
    f.write(et_program.buffer)
```

ExecuTorch `.pte` is most useful when you need to run **the same model on ARM CPU + XNNPACK** across Android, iOS, and embedded Linux without separate compile steps. On Jetson with CUDA, TRT-LLM or llama.cpp will outperform ExecuTorch significantly.

---

## 7. Format selection guide

```text
DECISION TREE for Gemma 4 format choice on Jetson:

  Target: Jetson with CUDA (Orin / Thor)
  └─ Need peak throughput? → TensorRT-LLM engine (build from GPTQ safetensors)
  └─ Need cross-platform + portability? → GGUF Q4_K_M (llama.cpp)
  └─ Need Google AI Edge ecosystem integration? → LiteRT .tflite (via AI Edge)
  └─ Need TVM compilation with MLC-LLM? → AWQ safetensors → mlc_llm convert

  Target: Jetson DLA / ARM CPU only
  └─ Google AI Edge first choice → LiteRT .tflite (GPU delegate or CPU)
  └─ XNNPACK on ARM → ExecuTorch .pte

  Target: Jetson + Android/iOS same model
  └─ LiteRT .tflite (runs everywhere in the Google AI Edge ecosystem)

  Target: low-latency single-GPU cloud (Orin Orin server cluster)
  └─ GPTQ INT4 + TensorRT-LLM
```

---

## 8. Accuracy acceptance criteria

Before deploying a quantized Gemma 4 model, set explicit acceptance thresholds:

```text
Metric          │ INT8       │ Q4_K_M     │ Q4_K_M imatrix │ Q2_K
────────────────┼────────────┼────────────┼────────────────┼──────────
Perplexity Δ   │ < 0.1 pp   │ < 0.25 pp  │ < 0.15 pp      │ < 4 pp
MMLU accuracy Δ│ < 0.3%     │ < 1.5%     │ < 0.8%         │ < 5%
GSM8K accuracy Δ│ < 0.5%    │ < 2.0%     │ < 1.0%         │ unacceptable
HumanEval Δ    │ < 0.5%     │ < 2.0%     │ < 1.0%         │ unacceptable
```

Run these benchmarks against your deployment model before shipping:

```bash
# lm-evaluation-harness (fast perplexity + task eval):
lm_eval --model gguf --model_args pretrained=gemma4-4b-Q4_K_M.gguf \
        --tasks mmlu,gsm8k,hellaswag \
        --num_fewshot 5 --batch_size 4

# If MMLU degrades > 1.5% from BF16 baseline: try Q5_K_M or imatrix.
# If latency is acceptable with Q5_K_M: use that instead of Q4_K_M.
```

---

## Key takeaways

- Gemma 4 **quantizes cleanly** because QK-norm removes attention-logit outliers (the main culprit for quantization degradation at long sequences) and GeGLU partially suppresses FFN activation outliers.
- **Q4_K_M** (GGUF) is the standard recommendation for llama.cpp on Jetson: < 0.2 pp degradation, ~45% weight compression vs BF16. Add **imatrix** calibration to recover ~0.05 pp at no size cost.
- **GPTQ** (group_size=128, 128 calibration samples) is the best algorithm for Gemma 4 when feeding TensorRT-LLM or vLLM — second-order Hessian wins when outliers are already suppressed.
- **LiteRT / Google AI Edge** is the first-class path when deploying to DLA, ARM, or cross-platform (Jetson → Android → embedded Linux from one `.tflite`). For CUDA-on-Jetson, GGUF or TRT-LLM wins.
- Keep the **embedding table at INT8** even in INT4 deployments — the 256K vocab makes INT4 embedding rows too short for low-error quantization.
- Set explicit **accuracy acceptance thresholds** (Δ perplexity, MMLU, GSM8K) and measure before deploying — a quantized model that fails your downstream task is not a deployment.

---

## References

- GPTQ paper (Frantar et al., 2022) — arXiv:2210.17323
- AWQ paper (Lin et al., 2023) — arXiv:2306.00978
- llama.cpp GGUF format and K-quants documentation — [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- AI Edge Torch (Google's export path for Gemma → LiteRT) — [github.com/google-ai-edge/ai-edge-torch](https://github.com/google-ai-edge/ai-edge-torch)
- Google AI Edge LiteRT documentation — [ai.google.dev/edge/litert](https://ai.google.dev/edge/litert)
- ExecuTorch (Meta) — [github.com/pytorch/executorch](https://github.com/pytorch/executorch)
- Kaggle Models — Gemma 4 LiteRT INT4 variants — [kaggle.com/models/google/gemma](https://www.kaggle.com/models/google/gemma)

---

## Current as of 2026-06

llama.cpp tag b3500+ for Gemma 4 GGUF support; AI Edge Torch 0.3+; LiteRT 2.16+; AutoGPTQ 0.7+; AutoAWQ 0.2+. GPTQ group_size=128 recommended for Gemma 4 4B/12B; group_size=64 for 27B (finer granularity at larger scale). Always verify llama.cpp Gemma 4 tokenizer support before running — the 256K vocab requires explicit model type handling in `convert_hf_to_gguf.py`.

---

*Previous: [← Lecture 01](Lecture-01.md) · Up: [Gemma 4 Edge Deployment](README.md) · Next: [Lecture 03 — The PDL Runtime Stack](Lecture-03.md)*
