# Lecture 06 — Multi-Token Prediction with Gemma 4 E2B: The Official MTP Drafter

**Collection:** [Gemma 4 Edge Deployment](README.md) | **Previous:** [← Lecture 05](Lecture-05.md)

---

Lecture 04 covered speculative decoding with a Gemma 4 1B external draft and EAGLE-3-style self-draft heads. Google ships a purpose-built solution for the same problem: **Gemma 4 E2B**, a 4-layer, extremely lightweight MTP (Multi-Token Prediction) drafter designed to pair with any full Gemma 4 target. Where the 1B external draft requires 500 MB of extra memory and EAGLE-3 heads require 2–4 hours of training, E2B is the official drop-in drafter with published weights, framework support in vLLM, LiteRT-LM, and llama.cpp — and up to 3× speedup on text generation benchmarks.

This lecture explains what MTP is, how Gemma 4 E2B implements it, how it differs from standard speculative decoding, and how to deploy it on every major inference stack including Jetson edge targets.

---

## Learning objectives

1. Explain the difference between **standard speculative decoding** (separate draft model) and **MTP** (shared-state parallel token prediction).
2. Describe the Gemma 4 **E2B MTP drafter architecture** — why 4 layers, what it shares with the target, and why it converges to a high acceptance rate.
3. Configure and run MTP speculative decoding with Gemma 4 E2B + 4B/12B/27B target on **vLLM**, **LiteRT-LM**, and **llama.cpp**.
4. Compare the three speculative-decode approaches in this course (1B external, EAGLE-3 heads, E2B MTP) across memory, setup cost, speedup, and framework support.
5. Select the right approach for your deployment target (Orin, Thor, cloud GPU) and task.

---

## 1. Multi-Token Prediction — what it is and how it differs

### 1.1 Standard speculative decoding (Lecture 04 recap)

In standard spec-decode, a **separate draft model** (e.g., Gemma 4 1B) autoregressively generates K token proposals, and the target model verifies them in one parallel forward pass:

```text
Standard spec-decode:
  draft(context) → [x₁, x₂, x₃, x₄]      separate model, 4 steps
  target.verify(context + [x₁,x₂,x₃,x₄])  one parallel pass
  accept prefix, reject first mismatch

Problem: the draft model has DIFFERENT weights, different representations.
  Its token proposals can diverge from target in distribution → low τ on hard tasks.
  Memory cost: full second model weights (500 MB for 1B).
```

### 1.2 Multi-Token Prediction (MTP)

MTP predicts multiple future tokens **from the same model's internal representations**, using auxiliary heads or lightweight transformer modules attached to the target:

```text
MTP approach:
  target.forward(context) → hidden states h₀, h₁, ..., hₙ
  MTP head predicts x_{t+1}, x_{t+2}, ... from h_n (or from partial h)
  Target verifies proposed tokens — but the proposals came from TARGET's own features

Why this converges better:
  The MTP drafter sees the target's internal state, not a separately-learned state.
  Token proposals are in distribution with the target by construction.
  → higher acceptance length τ than an external draft model of equivalent size
```

### 1.3 DeepSeek MTP vs Gemma E2B MTP

MTP was popularized at scale by DeepSeek-V3 (Dec 2024), which trained MTP heads jointly during pretraining. Gemma 4 E2B takes a **post-hoc drafter** approach: a separate lightweight transformer trained to predict future tokens given the target's hidden state, then used at inference time. The effect is the same — target-aware proposals — but E2B doesn't require joint pretraining of the full target.

```text
DeepSeek MTP:  integrated into target during pretraining; heads share target weights
Gemma E2B MTP: separate 4-layer transformer trained post-hoc; reads target features
EAGLE-3:       separate head trained post-hoc; similar to E2B but smaller (1–2 layers)

E2B advantage over EAGLE-3: deeper (4 layers) → better multi-hop reasoning for
  subsequent token slots (x_{t+2}, x_{t+3}) which require more context propagation
E2B advantage over 1B external: operates from target's features → higher τ
E2B advantage vs joint MTP (DeepSeek-style): no target retraining required
```

---

## 2. Gemma 4 E2B — architecture

### 2.1 What "E2B" means

**E2B = Edge 2B** (2-billion parameter class, designed for edge co-deployment). The model is structured as:

```text
gemma-4-E2B-it-assistant (MTP drafter):
  Layers:        4 transformer decoder layers (vs 26 for 4B, 46 for 12B)
  Hidden dim:    matches target Gemma 4 (e.g., 2048 for 4B, 3072 for 12B)
                 required to accept the target's hidden states as input
  Heads:         same head structure as target (GQA 2:1, head_dim=256, QK-norm)
  Vocab:         same 256K tokenizer as all Gemma 4 models
  Parameters:    ~2B (4 × full target-width transformer layers)
  Size on disk:  ~4 GB BF16 / ~1 GB INT4

CRITICAL: the hidden_dim matches the target.
  E2B reads the target's LAST hidden state as its input (instead of generating
  its own from scratch). The 4 transformer layers then propagate that state
  to predict x_{t+1}, x_{t+2}, ... without needing to encode the full context.
```

### 2.2 Forward pass during MTP speculative decode

```text
For each decode step:

1. TARGET forward pass:
   target.forward(context) → last_hidden_state h  (shape: [1, 1, hidden_dim])
                           → logit_t → sample x_t (current token)

2. E2B DRAFT pass (parallel or immediately after):
   e2b.forward(h, x_t) → logit_{t+1} → draft token x̂_{t+1}
   e2b.forward(h', x̂_{t+1}) → logit_{t+2} → draft token x̂_{t+2}
   ... up to K=4 draft steps (typical: K=3–5)

3. TARGET VERIFY pass:
   target.verify(context + x_t + x̂_{t+1} + x̂_{t+2} + ... + x̂_{t+K})
   → one parallel forward → accept/reject per draft token

4. Result: accept longest prefix, emit accepted tokens (typically τ=3–4)
```

The E2B forward pass is extremely cheap because:
- It skips context encoding entirely (reads target's h directly)
- 4 layers vs target's 26–46 layers → ~10% of target's compute
- Operates on a single token position at a time (not full sequence)

### 2.3 Memory footprint on Jetson

```text
Deployment             Size         Notes
─────────────────────  ───────────  ─────────────────────────────────────────
E2B INT4 (GGUF)        ~1.0 GB      quantize like any Gemma model
E2B BF16               ~4.0 GB      not recommended for edge
Gemma 4 4B INT4        ~2.2 GB      target model
─────────────────────  ───────────  ─────────────────────────────────────────
TOTAL (4B + E2B INT4)  ~3.2 GB      fits Orin comfortably
TOTAL (12B + E2B INT4) ~7.4 GB      fits Orin 64 GB
TOTAL (27B + E2B INT4) ~15.4 GB     fits Thor 128 GB; tight on Orin (18 GB OK)
```

Compared to Lecture 04's 1B external draft (500 MB INT4), E2B costs 2× the memory but achieves 20–40% higher acceptance length due to target-state awareness.

---

## 3. Framework support and configuration

### 3.1 vLLM (best throughput for multi-user serving)

vLLM 0.6+ has native MTP speculative decoding support:

```python
from vllm import LLM, SamplingParams

# Initialize target + E2B drafter:
llm = LLM(
    model="google/gemma-4-4b-it",
    speculative_model="google/gemma-4-E2B-it-assistant",  # E2B MTP drafter
    num_speculative_tokens=4,              # K=4 draft tokens per step
    speculative_draft_tensor_parallel_size=1,
    # E2B is small enough to not need tensor parallelism
    dtype="bfloat16",
    gpu_memory_utilization=0.9,
    max_model_len=4096,
)

sampling_params = SamplingParams(
    temperature=0.0,   # greedy for maximum acceptance rate
    max_tokens=512,
)

outputs = llm.generate(
    ["Explain how Gemma 4 speculative decoding works."],
    sampling_params
)
print(outputs[0].outputs[0].text)

# Benchmark: measure tokens/s with and without E2B
import time

prompts = ["Write a detailed technical explanation. " * 10] * 8

t0 = time.perf_counter()
outputs = llm.generate(prompts, SamplingParams(max_tokens=200, temperature=0.0))
elapsed = time.perf_counter() - t0
total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
print(f"Throughput: {total_tokens / elapsed:.1f} tok/s")
# Expected on A100: ~380 tok/s (vs ~220 tok/s without E2B): ~1.7×
# Expected on Orin (1 request): ~90 tok/s (vs ~55 tok/s): ~1.6×
```

**vLLM speculative decode statistics:**

```python
# vLLM logs acceptance rate — look for:
# "Speculative decode draft acceptance rate: 0.78"  → τ ≈ 4 × 0.78 = 3.1
# Enable verbose logging:
import logging
logging.getLogger("vllm.spec_decode").setLevel(logging.DEBUG)
```

### 3.2 LiteRT-LM (recommended for edge / Jetson)

LiteRT-LM (formerly MediaPipe LLM Inference) is Google's recommended path for Gemma 4 E2B on mobile and edge. It is the only runtime with first-party E2B MTP support for Android, Linux ARM64, and Jetson:

```python
# Install LiteRT-LM (Google AI Edge):
# pip install litert-lm  (requires JetPack 6+ on Jetson)

from litert_lm import LlmInference, LlmInferenceOptions

options = LlmInferenceOptions(
    model_path="/data/gemma4-4b-it.task",           # LiteRT task file (text)
    # Speculative drafter via E2B:
    speculative_decoding_drafter_model_path="/data/gemma4-e2b-it.task",
    num_speculative_tokens=4,                        # K
    # Target runtime:
    num_threads=4,
    accelerator="gpu",                               # Jetson iGPU
    max_tokens=512,
)

inference = LlmInference.create_from_options(options)

# Streaming generation with E2B MTP:
response = ""
for token in inference.generate_response_async("What is speculative decoding?"):
    response += token
    print(token, end="", flush=True)
print()

# Benchmark:
import time
t0 = time.perf_counter()
_ = inference.generate_response("Write a 200-word explanation of transformers.")
elapsed = time.perf_counter() - t0
print(f"Generation time: {elapsed:.2f} s")
```

**LiteRT-LM model conversion for E2B:**

```bash
# Convert E2B drafter to LiteRT task format:
pip install ai-edge-torch ai-edge-litert

python3 - <<'EOF'
import ai_edge_torch
from transformers import AutoModelForCausalLM
import torch

# Convert E2B to LiteRT task file:
drafter = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-E2B-it-assistant",
    torch_dtype=torch.bfloat16
)
drafter.eval()

sample_input = torch.zeros(1, 1, dtype=torch.long)
edge_drafter = ai_edge_torch.convert(drafter, sample_args=(sample_input,))
edge_drafter.export("gemma4-e2b-it.tflite")
EOF

# Package as LiteRT task file:
# (Use the AI Edge Model Maker or the task_assembler CLI from LiteRT SDK)
litert-task-assembler \
    --model_file gemma4-e2b-it.tflite \
    --output gemma4-e2b-it.task
```

**Throughput comparison on Jetson Orin (Gemma 4 4B, 4K context):**

| Runtime | No spec-decode | With E2B MTP (K=4) | Speedup |
|---------|---------------|-------------------|---------|
| LiteRT-LM CPU | 8 tok/s | 12 tok/s | 1.5× |
| LiteRT-LM GPU delegate | 20 tok/s | 32 tok/s | 1.6× |
| llama.cpp CUDA (sm_87) | 55 tok/s | 90 tok/s | 1.6× |
| vLLM (on Orin iGPU via ROCm/custom) | — | — | varies |

### 3.3 llama.cpp (GGUF path)

llama.cpp community support for Gemma 4 E2B MTP via the `--model-draft` flag and GGUF conversion:

```bash
# Step 1: Convert E2B to GGUF
git clone https://github.com/ggml-org/llama.cpp
pip install transformers torch

python3 llama.cpp/convert_hf_to_gguf.py \
    google/gemma-4-E2B-it-assistant \
    --outfile gemma4-e2b-Q4_K_M.gguf \
    --outtype q4_k_m

# Step 2: Run with MTP speculative decoding
# E2B is treated as a draft model in llama.cpp's spec-decode interface:
./build/bin/llama-cli \
    --model       gemma4-4b-it-Q4_K_M.gguf \
    --model-draft gemma4-e2b-Q4_K_M.gguf \
    --n-gpu-layers      9999 \
    --n-gpu-layers-draft 9999 \
    --draft-max   5 \              # K=5 draft tokens (E2B is fast, can push to 5–6)
    --draft-min   1 \
    --draft-p-min 0.3 \            # prune low-probability E2B drafts early
    --ctx-size    4096 \
    -p "Explain the physics of black holes in detail."

# Benchmark comparing E2B vs 1B draft vs no spec-decode:
for draft in "" "gemma4-1b-Q4_K_M.gguf" "gemma4-e2b-Q4_K_M.gguf"; do
    flags="--model gemma4-4b-it-Q4_K_M.gguf --n-gpu-layers 9999 -n 200 -p 128"
    if [ -n "$draft" ]; then
        flags="$flags --model-draft $draft --n-gpu-layers-draft 9999 --draft-max 5"
    fi
    echo "Draft: ${draft:-none}"
    ./build/bin/llama-bench $flags 2>&1 | grep "tg"
done

# Expected on Orin:
#   No draft:     tg = 55 tok/s
#   1B draft:     tg = 80 tok/s (τ ≈ 2.8)
#   E2B draft:    tg = 90 tok/s (τ ≈ 3.3)   ← E2B wins by ~12%
```

**Note on llama.cpp and E2B:** as of mid-2026, E2B GGUF conversion and draft support is community-maintained. Check llama.cpp releases and the official Google model card for the latest GGUF-compatible E2B weights.

### 3.4 MLX (Apple Silicon, for completeness)

```bash
# mlx-lm supports Gemma 4 E2B via the speculative_drafting flag:
pip install mlx-lm

python3 -m mlx_lm.generate \
    --model google/gemma-4-4b-it \
    --draft-model google/gemma-4-E2B-it-assistant \
    --num-draft-tokens 4 \
    --prompt "Describe speculative decoding." \
    --max-tokens 200
# Expected on M4 Max (128 GB): ~85 tok/s vs ~55 tok/s without E2B
```

---

## 4. Acceptance length analysis for E2B vs alternatives

The acceptance length τ is the primary performance lever. Here is a calibrated comparison across tasks:

| Task | τ (1B external) | τ (EAGLE-3 heads) | τ (E2B MTP) |
|------|----------------|-------------------|-------------|
| Code generation | 3.5–4.5 | 3.8–5.0 | 4.0–5.5 |
| Technical Q&A | 3.0–4.0 | 3.5–4.5 | 3.8–5.0 |
| Creative writing | 2.5–3.5 | 3.0–4.0 | 3.2–4.3 |
| Math / CoT reasoning | 2.0–3.0 | 2.5–3.5 | 2.8–3.8 |
| Repetitive / templated | 4.5–6.0 | 5.0–6.5 | 5.0–7.0 |
| Instruction following | 3.0–4.0 | 3.5–4.5 | 3.8–5.0 |

E2B consistently achieves 0.3–0.5 higher τ than the 1B external draft across task categories, because E2B reads the target's own hidden state — its proposals are already aligned with the target distribution.

**When E2B underperforms 1B external draft:**

In pathological cases (very short outputs, highly random sampling temperature > 0.9), E2B's 4-layer architecture sometimes overshoots the distribution compared to the 1B model's simpler predictions. Use temperature ≤ 0.7 for best E2B acceptance rates.

---

## 5. Head-to-head comparison: three speculative decode approaches

This course has covered three approaches. Here is the decision matrix:

| Criterion | 1B External Draft | EAGLE-3 Heads | E2B MTP |
|-----------|------------------|---------------|---------|
| Extra memory | 500 MB INT4 | 150 MB | 1,000 MB INT4 |
| Setup effort | Zero (download) | 2–4h training | Zero (download) |
| Acceptance length τ | 2.5–4.0 | 3.5–5.0 | 3.8–5.5 |
| vLLM support | ✓ (--speculative) | Partial | ✓ (native) |
| LiteRT-LM support | ✗ | ✗ | ✓ (first-party) |
| llama.cpp support | ✓ (--model-draft) | ✗ | ✓ (--model-draft) |
| Greedy-decode parity | ✓ guaranteed | ✓ guaranteed | ✓ guaranteed |
| Output quality change | None (lossless) | None (lossless) | None (lossless) |
| Best on Jetson (power) | Medium | Best | Good |
| Google-official | ✗ | ✗ | ✓ |

**Decision guide:**

```text
Use 1B external draft when:
  → Fastest to deploy, no training, already have 1B weights on device
  → llama.cpp or vLLM, memory is tight, E2B's 1 GB is too expensive

Use EAGLE-3 heads when:
  → Memory-constrained (Orin Nano, 8–16 GB) and you can spare training time
  → Target is 4B/12B and you want the lowest memory overhead per acceptance gain

Use E2B MTP when:
  → LiteRT-LM / Google AI Edge ecosystem (only option with first-party support)
  → vLLM multi-user serving where τ improvements compound across many requests
  → You want the highest acceptance rate without any training investment
  → Production deployment where Google-official weights reduce maintenance risk
```

---

## 6. Deployment recipe: E2B + Gemma 4 4B on Jetson Orin

Complete deployment from scratch:

```bash
# Prerequisites: JetPack 6.0+, CUDA sm_87, Python 3.10+
# Estimated time: 20 minutes

# 1. Download models:
pip install huggingface-hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('google/gemma-4-4b-it', local_dir='./gemma4-4b')
snapshot_download('google/gemma-4-E2B-it-assistant', local_dir='./gemma4-e2b')
"

# 2. Build llama.cpp for Orin:
git clone https://github.com/ggml-org/llama.cpp
cmake -B llama.cpp/build -S llama.cpp -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="87"
cmake --build llama.cpp/build --config Release -j$(nproc)

# 3. Convert both models to GGUF INT4:
python3 llama.cpp/convert_hf_to_gguf.py ./gemma4-4b  --outfile gemma4-4b-Q4_K_M.gguf  --outtype q4_k_m
python3 llama.cpp/convert_hf_to_gguf.py ./gemma4-e2b --outfile gemma4-e2b-Q4_K_M.gguf --outtype q4_k_m

# 4. Quick sanity check:
./llama.cpp/build/bin/llama-cli \
    --model gemma4-4b-Q4_K_M.gguf \
    --model-draft gemma4-e2b-Q4_K_M.gguf \
    --n-gpu-layers 9999 --n-gpu-layers-draft 9999 \
    --draft-max 4 --ctx-size 2048 \
    -p "What is speculative decoding?" -n 100 2>&1 | tail -5

# 5. Benchmark (compare baseline vs E2B):
echo "=== Baseline (no spec decode) ===" && \
./llama.cpp/build/bin/llama-bench \
    --model gemma4-4b-Q4_K_M.gguf --n-gpu-layers 9999 -n 200 -p 128

echo "=== E2B MTP spec decode ===" && \
./llama.cpp/build/bin/llama-bench \
    --model gemma4-4b-Q4_K_M.gguf \
    --model-draft gemma4-e2b-Q4_K_M.gguf \
    --n-gpu-layers 9999 --n-gpu-layers-draft 9999 \
    --draft-max 4 -n 200 -p 128

# Expected:
#   Baseline:  tg = 55 tok/s
#   E2B MTP:   tg = 85–95 tok/s  (~1.6×)

# 6. Run as server (production mode):
./llama.cpp/build/bin/llama-server \
    --model      gemma4-4b-Q4_K_M.gguf \
    --model-draft gemma4-e2b-Q4_K_M.gguf \
    --n-gpu-layers 9999 --n-gpu-layers-draft 9999 \
    --draft-max 4 \
    --ctx-size 4096 \
    --kv-cache-type q8_0 \         # INT8 KV saves memory
    --parallel 2 \                  # 2 concurrent requests
    --port 8080 &

# Test the server:
curl http://localhost:8080/v1/completions -s \
    -H "Content-Type: application/json" \
    -d '{"model":"gemma4-4b","prompt":"Describe MTP speculative decoding.","max_tokens":100}' \
    | python3 -m json.tool | grep -E "text|usage"
```

---

## Key takeaways

- **Gemma 4 E2B** is a 4-layer, ~2B-parameter MTP drafter that reads the target model's last hidden state as input — it never re-encodes context, just predicts the next few tokens from the target's own features.
- **MTP achieves higher acceptance rates than external drafts** (τ=3.8–5.5 vs 2.5–4.0 for 1B) because E2B's proposals are anchored to the target's internal state, eliminating cross-model distribution mismatch.
- **LiteRT-LM is the first-party path for edge/Jetson** — Google ships E2B MTP support natively in the AI Edge SDK, making it the lowest-friction deployment option on Jetson Orin/Thor and Android.
- **vLLM's `speculative_model` flag** wires E2B MTP in one parameter for multi-user serving; expected speedup ~1.6–1.8× on GPU.
- **llama.cpp `--model-draft`** uses E2B as a conventional draft model — this works and gives the E2B acceptance rate benefit even though llama.cpp doesn't natively distinguish MTP from standard spec-decode internals.
- **Memory cost is ~1 GB INT4** — twice the 1B external draft, but E2B is Google-official and requires zero training, which makes it the right default for production unless memory is critically tight.
- All three approaches (1B external, EAGLE-3 heads, E2B MTP) are **lossless for greedy decode** — the output distribution is provably identical to the target running alone.

---

## References

- Gemma 4 E2B model card — `google/gemma-4-E2B-it-assistant` on HuggingFace
- "Multi-Token Prediction" (Gloeckle et al., Meta, 2024) — arXiv:2404.19737
- DeepSeek-V3 MTP training integration (2024) — arXiv:2412.19437
- LiteRT-LM MTP speculative decode docs — [ai.google.dev/edge/litert-lm](https://ai.google.dev/edge/litert-lm)
- vLLM speculative decoding (`speculative_model`) — vllm.readthedocs.io/en/latest/features/spec_decode.html
- llama.cpp speculative decode (--model-draft) — [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- *Gemma 4 Edge Deployment — Lecture 04* — standard spec-decode, EAGLE-3, acceptance math — [Lecture-04.md](Lecture-04.md)

---

## Current as of 2026-06

Gemma 4 E2B MTP drafter available at `google/gemma-4-E2B-it-assistant` on HuggingFace. LiteRT-LM MTP support: requires litert-lm ≥ 0.2.0 and AI Edge SDK ≥ June 2026 build. vLLM MTP support: requires vllm ≥ 0.6.0 with `speculative_model="google/gemma-4-E2B-it-assistant"`. llama.cpp GGUF conversion: check latest llama.cpp release for Gemma 4 E2B architecture support (may need updated convert_hf_to_gguf.py). Google-published benchmark: up to 3× inference speedup measured on internal benchmarks (text, English-only; your Jetson results will vary by task).

---

*Previous: [← Lecture 05 — Physical AI and Multimodal Gemma 4](Lecture-05.md) · Up: [Gemma 4 Edge Deployment](README.md)*
