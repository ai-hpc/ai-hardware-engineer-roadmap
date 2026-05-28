# LLM Optimization on Jetson — From Cloud Techniques to Edge Reality

<div class="course-identity auto-course" style="--course-accent: #7c3aed; --course-accent-rgb: 124, 58, 237;" markdown="1">
<div class="course-identity__icon">LOOJ</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Jetson Track</p>
<p class="course-identity__title">Specialized course identity for LLM Optimization on Jetson — From Cloud Techniques to Edge Reality.</p>
<p class="course-identity__meta">Artifact: Jetson integration demo · Measure: latency, memory, power, logs</p>
</div>
</div>


**Parent:** [ML and AI](../Guide.md)

> **Goal:** Take the optimization techniques used by cloud LLM platforms (vLLM, TensorRT-LLM, RunInfra, etc.) and adapt them to the extreme constraints of Jetson Orin Nano Super 8 GB — 102 GB/s bandwidth, 67 TOPS GPU + ~10 TOPS DLA ≈ 77 TOPS total, shared memory, 7–25W power.

---

## Pre-Flight: System Check Before Starting

Before any LLM work, verify your Jetson's software stack. **JetPack version determines which CUDA, TensorRT, and cuDNN versions you have** — and which LLM tools are compatible.

### Quick System Audit (copy-paste this)

```bash
#!/bin/bash
echo "═══════════════════════════════════════════════"
echo "  Jetson System Audit for LLM Deployment"
echo "═══════════════════════════════════════════════"

echo ""
echo "▸ JetPack / L4T version:"
cat /etc/nv_tegra_release 2>/dev/null || echo "  (not found — check dpkg)"
dpkg-query --show nvidia-l4t-core 2>/dev/null | awk '{print "  L4T:", $2}'

echo ""
echo "▸ CUDA version:"
nvcc --version 2>/dev/null | grep release || echo "  nvcc not found"

echo ""
echo "▸ TensorRT version:"
dpkg -l | grep tensorrt | head -1 | awk '{print " ", $3}'

echo ""
echo "▸ cuDNN version:"
dpkg -l | grep cudnn | head -1 | awk '{print " ", $3}'

echo ""
echo "▸ Python version:"
python3 --version

echo ""
echo "▸ Total RAM:"
free -m | awk '/Mem:/ {print "  " $2 " MB total"}'
echo "▸ Free RAM:"
free -m | awk '/Mem:/ {print "  " $7 " MB available"}'

echo ""
echo "▸ CMA allocation:"
grep Cma /proc/meminfo | awk '{print "  " $0}'

echo ""
echo "▸ GPU info:"
cat /sys/devices/17000000.ga10b/devfreq/17000000.ga10b/cur_freq 2>/dev/null \
    | awk '{print "  GPU freq: " $1/1000000 " MHz"}'

echo ""
echo "▸ Power mode:"
sudo nvpmodel -q 2>/dev/null | head -2 | sed 's/^/  /'

echo ""
echo "▸ Disk space:"
df -h / | tail -1 | awk '{print "  Root: " $4 " free of " $2}'
df -h /dev/nvme0n1p1 2>/dev/null | tail -1 | awk '{print "  NVMe: " $4 " free of " $2}'

echo ""
echo "▸ Thermal:"
cat /sys/devices/virtual/thermal/thermal_zone*/temp 2>/dev/null | head -3 \
    | awk '{print "  Zone: " $1/1000 "°C"}'

echo "═══════════════════════════════════════════════"
```

**Example output (Orin Nano Super, JetPack 6.1):**

```
═══════════════════════════════════════════════
  Jetson System Audit for LLM Deployment
═══════════════════════════════════════════════

▸ JetPack / L4T version:
  # R36 (release), REVISION: 4.0
  L4T: 36.4.0-20241031080721

▸ CUDA version:
  Cuda compilation tools, release 12.6, V12.6.77

▸ TensorRT version:
  10.3.0.30-1+cuda12.6

▸ cuDNN version:
  9.3.0.75-1+cuda12.6

▸ Python version:
  Python 3.10.12

▸ Total RAM:
  7633 MB total
▸ Free RAM:
  5814 MB available

▸ CMA allocation:
  CmaTotal:      786432 kB
  CmaFree:       654321 kB

▸ GPU info:
  GPU freq: 624 MHz

▸ Power mode:
  NV Power Mode: MAXN
  Power Mode: 25W

▸ Disk space:
  Root: 42G free of 100G

▸ Thermal:
  Zone: 38.5°C

═══════════════════════════════════════════════
```

### JetPack → CUDA → TensorRT Compatibility Matrix

This matrix determines which LLM tools work on your Jetson:

| JetPack | L4T | CUDA | TensorRT | cuDNN | Python | llama.cpp | Ollama | TRT-LLM |
|---------|-----|------|----------|-------|--------|-----------|--------|---------|
| **6.1** | R36.4 | **12.6** | **10.3** | 9.3 | 3.10 | Yes | Yes | Yes (0.15+) |
| **6.0** | R36.3 | **12.2** | **8.6** | 8.9 | 3.10 | Yes | Yes | Yes (0.9+) |
| 5.1.3 | R35.5 | 11.4 | 8.5 | 8.6 | 3.8 | Yes | Yes | Limited |
| 5.1.1 | R35.3 | 11.4 | 8.5 | 8.6 | 3.8 | Yes | Older | No |
| 5.0.2 | R35.1 | 11.4 | 8.4 | 8.4 | 3.8 | Yes | No | No |

> **Recommendation:** Use **JetPack 6.1** (R36.4) for LLM work. It has CUDA 12.6, TensorRT 10.3, and full TensorRT-LLM support. If you're on JetPack 5.x, consider upgrading — the CUDA 12.x ecosystem (llama.cpp, Ollama, PyTorch 2.x) is significantly better.

### Pre-Flight Checklist

```
Before deploying any LLM on Jetson:

□ JetPack version
  □ JetPack 6.0+ for TensorRT-LLM
  □ JetPack 5.1+ minimum for llama.cpp / Ollama

□ Available memory
  □ Run: free -m → note "available" column
  □ Expect ~5.5–6 GB free on stock Orin Nano Super 8 GB
  □ If < 5 GB: disable GUI (sudo systemctl set-default multi-user.target)
  □ If < 4 GB: reduce CMA, disable unnecessary services

□ Storage
  □ NVMe recommended (models are 1–5 GB each)
  □ SD card works but slower model loading
  □ At least 20 GB free for models + build cache

□ Power mode
  □ sudo nvpmodel -m 0  (MAXN = 25W for Orin Nano Super)
  □ sudo jetson_clocks   (lock to max frequency)

□ Thermal
  □ Active cooling attached (fan or heatsink with fan)
  □ Ambient temperature < 35°C for sustained workloads
  □ Monitor: tegrastats --interval 1000
```

### Disable GUI to Free ~500 MB RAM

For headless LLM deployment, disable the desktop environment:

```bash
# Switch to text-only mode (saves ~500 MB RAM)
sudo systemctl set-default multi-user.target
sudo reboot

# To re-enable GUI later:
sudo systemctl set-default graphical.target
sudo reboot

# Verify RAM freed:
free -m   # "available" should increase by ~500 MB
```

This single change can be the difference between a 3B model fitting comfortably and running out of memory.

---

## 0. The Optimization Stack — Cloud vs Jetson

Cloud platforms like RunInfra, Together AI, and Fireworks AI deploy LLMs using a layered optimization stack. Every technique has a Jetson equivalent — but the priorities are reversed because Jetson is severely memory-bandwidth-bound rather than compute-bound.

```
Cloud GPU (H100 80GB, 3,350 GB/s, 989 TFLOPS):
  ┌─────────────────────────────────────────────┐
  │ 1. Quantization (FP8, AWQ 4-bit)           │ ← saves VRAM, improves throughput
  │ 2. FlashAttention-2                         │ ← saves SRAM, fuses memory ops
  │ 3. Fused Kernels (RMSNorm, rotary, SwiGLU) │ ← fewer kernel launches
  │ 4. PagedAttention (vLLM)                    │ ← KV cache memory efficiency
  │ 5. Speculative Decoding                     │ ← higher tokens/sec
  │ 6. Batching (continuous batching)           │ ← amortize compute over requests
  │ 7. Tensor Parallelism (multi-GPU)          │ ← scale beyond 1 GPU
  └─────────────────────────────────────────────┘

Jetson Orin Nano Super 8GB (102 GB/s, 67 TOPS GPU + ~10 TOPS DLA ≈ 77 TOPS):
  ┌─────────────────────────────────────────────┐
  │ 1. Quantization (INT4/INT8) ★★★★★          │ ← MANDATORY: model must fit in 5 GB
  │ 2. Model Selection ★★★★★                    │ ← choose models that fit (≤3B params)
  │ 3. KV Cache Management ★★★★                │ ← memory is the #1 constraint
  │ 4. FlashAttention / Fused Ops ★★★★         │ ← reduce bandwidth pressure
  │ 5. TensorRT-LLM Engine ★★★                 │ ← compiled, optimized execution
  │ 6. Speculative Decoding ★★★                │ ← higher tokens/sec within power budget
  │ 7. Batching ★★                              │ ← limited by memory, not compute
  └─────────────────────────────────────────────┘
  ★ = importance on Jetson (more ★ = more critical)
```

---

## 1. Quantization — The Most Important Optimization

On cloud GPUs, quantization is optional (saves cost). On Jetson, **quantization is mandatory** — without it, nothing fits.

### 1.1 Why Quantization Matters More on Jetson

```
Model: Llama 3.2 3B parameters

FP16:  3B × 2 bytes = 6.0 GB   ← won't fit (only ~5 GB free after OS/CMA)
INT8:  3B × 1 byte  = 3.0 GB   ← fits, but tight
INT4:  3B × 0.5 byte = 1.5 GB  ← fits comfortably, room for KV cache

Model: Phi-3 Mini 3.8B parameters

FP16:  3.8B × 2 bytes = 7.6 GB  ← impossible on 8 GB
INT4:  3.8B × 0.5 byte = 1.9 GB ← fits with room for context
```

### 1.2 Quantization Methods Ranked for Jetson

| Method | Bits | Quality loss | Speed on Jetson | When to use |
|--------|------|-------------|-----------------|-------------|
| **AWQ (Activation-Aware Weight)** | 4-bit | Very low | Fast (INT4 GEMM) | Best quality/size for Jetson |
| **GPTQ** | 4-bit | Low | Fast | Alternative to AWQ, well-supported |
| **INT8 PTQ (TensorRT)** | 8-bit | Minimal | Fastest | If model fits at INT8 |
| **FP8 (E4M3)** | 8-bit | Minimal | Fast (Ampere+) | When you need FP-like quality |
| **GGUF (llama.cpp)** | 2–8 bit | Mixed-precision | Good (CPU+GPU) | Easy deployment, any model |
| **SqueezeLLM** | 3-4 bit | Low | Moderate | Extreme compression |

### 1.3 AWQ Quantization (Recommended for Jetson)

AWQ preserves quality by protecting salient weight channels — the 1% of weights that matter most for output quality.

```python
# Quantize on your workstation (not on Jetson — too slow)
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_name = "microsoft/Phi-3-mini-4k-instruct"
quant_path = "phi3-mini-awq-int4"

model = AutoAWQForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# AWQ calibration (needs ~128 samples)
model.quantize(
    tokenizer,
    quant_config={
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,           # 4-bit weights
        "version": "GEMM"     # optimized GEMM kernel
    }
)
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

### 1.4 GGUF with llama.cpp (Easiest Path)

llama.cpp runs on Jetson with CUDA support and handles mixed-precision quantization:

```bash
# On Jetson: install llama.cpp with CUDA
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)

# Download a pre-quantized GGUF model
# (Llama 3.2 3B in Q4_K_M = ~2 GB, good quality/size balance)
wget https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/\
Llama-3.2-3B-Instruct-Q4_K_M.gguf

# Run inference
./build/bin/llama-cli \
    -m Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    -ngl 99 \           # offload all layers to GPU
    -c 2048 \           # context length
    -p "Explain how Jetson unified memory works:"
```

**GGUF quantization levels for Jetson:**

| Quantization | Bits | Size (3B model) | Quality | Recommended? |
|-------------|------|-----------------|---------|-------------|
| Q2_K | 2.6 | ~1.0 GB | Poor | Only if nothing else fits |
| Q3_K_M | 3.4 | ~1.3 GB | Acceptable | Memory-critical deployments |
| **Q4_K_M** | **4.5** | **~1.8 GB** | **Good** | **Best balance for Jetson** |
| Q5_K_M | 5.5 | ~2.2 GB | Very good | If memory allows |
| Q6_K | 6.6 | ~2.5 GB | Excellent | Best quality that fits |
| Q8_0 | 8.0 | ~3.0 GB | Near-FP16 | Only for small models |

---

## 2. Model Selection — The Complete Landscape

The most important "optimization" is choosing the right model. A well-quantized small model beats a poorly-fitting large model every time.

### 2.1 Full Model Catalog (Q4_K_M Quantization, Orin Nano Super 8 GB)

All values at Q4_K_M — a 4-bit mixed-precision quantization that reduces memory by ~75% with minimal quality loss.

**Tier 1 — Runs comfortably (< 3 GB, room for long context + KV cache)**

| Model | Params | Q4_K_M size | Context | Gen. tok/s (est.) | Best for |
|-------|--------|-------------|---------|-------------------|----------|
| **TinyLlama 1.1B** | 1.1B | 0.6 GB | 2K | ~65 | Ultra-lightweight, draft model for speculative decoding |
| **Llama 3.2 1B** | 1.3B | 0.7 GB | 128K | ~55 | Lightweight chat, classification, tool calling |
| **StableLM 2 1.6B** | 1.6B | 0.9 GB | 4K | ~45 | Compact, fast edge assistant |
| **Gemma 3 1B** | 1B | 0.6 GB | 32K | ~60 | Multilingual, Google ecosystem |
| **Gemma 2 2B** | 2.6B | 1.5 GB | 8K | ~35 | Multilingual, general purpose |
| **Qwen 3 1.7B** | 1.7B | 1.0 GB | 32K | ~50 | Chinese + English, thinking mode |
| **SmolLM2 1.7B** | 1.7B | 1.0 GB | 8K | ~50 | Hugging Face, compact, well-trained |
| **Llama 3.2 3B** | 3.2B | 1.8 GB | 128K | ~25 | General chat, summarization, tool use |
| **Qwen 2.5 3B** | 3B | 1.7 GB | 32K | ~28 | Bilingual general purpose |

**Tier 2 — Fits but tight (3–4.5 GB, shorter context recommended)**

| Model | Params | Q4_K_M size | Context | Gen. tok/s (est.) | Best for |
|-------|--------|-------------|---------|-------------------|----------|
| **Phi-4 Mini** | 3.8B | 2.3 GB | 4K/128K | ~20 | Reasoning, math, code |
| **Phi-3 Mini** | 3.8B | 2.2 GB | 4K/128K | ~20 | Reasoning, code |
| **Qwen 3 4B** | 4B | 2.4 GB | 32K | ~18 | Thinking + non-thinking modes |
| **Gemma 3 4B** | 4B | 2.5 GB | 128K | ~17 | Vision + language (multimodal) |
| **Llama 3.3 8B** | 8B | 4.6 GB | 128K | ~10 | General purpose (needs short ctx) |
| **Mistral 7B v0.3** | 7B | 4.1 GB | 32K | ~11 | Chat, function calling |
| **Qwen 3 8B** | 8B | 4.9 GB | 32K | ~9 | Bilingual, thinking mode |

**Tier 3 — Barely fits (4.5+ GB, heavily constrained)**

| Model | Params | Q4_K_M size | Max ctx | Gen. tok/s (est.) | Notes |
|-------|--------|-------------|---------|-------------------|-------|
| **Llama 3.1 8B** | 8B | 4.7 GB | 512–1K | ~9 | Functional but cramped |
| **Mistral Nemo 12B** | 12B | 7.0 GB | 256 | ~5 | Needs Q3_K or Q2_K to fit |
| **Phi-4 14B** | 14B | 8.4 GB | — | Won't fit | Use Orin NX 16 GB |

> **Rule of thumb:** Model Q4_K_M size + 2.5 GB (OS/CMA/CUDA) + KV cache must be < 8 GB. For Tier 2 models, cap context at 1–2K tokens. For Tier 3, consider Q3_K_M or Q2_K quantization.

### 2.2 Models That Need Larger Jetson Hardware

For reference — what runs on bigger Jetson modules:

| Model | Params | Q4_K_M | Min Jetson module | Notes |
|-------|--------|--------|-------------------|-------|
| Phi-4 14B | 14B | 8.4 GB | **Orin NX 16 GB** | Good reasoning |
| Qwen 2.5 Coder 14B | 14B | 8.6 GB | **Orin NX 16 GB** | Code generation |
| Mistral Small 24B | 24B | 14 GB | **AGX Orin 32 GB** | Strong general purpose |
| DeepSeek-R1 Distill 32B | 32B | 19 GB | **AGX Orin 32 GB** | Reasoning chains |
| Llama 3.3 70B | 70B | 42 GB | **AGX Orin 64 GB** | Near-GPT-4 quality |
| Qwen 3 235B (MoE) | 235B | ~60 GB | **Multi-GPU / cloud** | 22B active params |
| DeepSeek V3 671B (MoE) | 671B | ~380 GB | **Multi-GPU cluster** | 37B active params |

### 2.3 Latest Models Worth Watching (2025–2026)

The edge LLM landscape moves fast. Models to evaluate as they release:

| Model family | Why it matters for Jetson |
|-------------|--------------------------|
| **Qwen 3 (0.6B–235B)** | Thinking + non-thinking modes, great small variants (1.7B, 4B) |
| **Gemma 3 (1B–27B)** | Multimodal (vision+text), good 1B and 4B for edge |
| **Phi-4 Mini (3.8B)** | Microsoft's strong reasoning at small size |
| **Llama 4 Scout/Maverick** | MoE architecture — active params may fit edge |
| **SmolLM2 (135M–1.7B)** | Hugging Face's ultra-compact series |
| **Step 3.5 Flash (196B MoE, 11B active)** | StepFun's open-source MoE reasoning model — only 11B active per token (fits Jetson!), 262K context, speed-optimized |
| **MiMo-V2-Pro (1T+ MoE)** | Xiaomi's flagship — agentic scenarios, OpenClaw-compatible, 1M context, approaches Opus-4.6 quality. Too large for Jetson directly, but **distilled/smaller MiMo variants** are edge targets |
| **MiMo (series, 1.5B–7B)** | Xiaomi's edge-optimized models — MiMo-7B has strong math/code benchmarks, 1.5B fits Jetson easily |
| **DeepSeek-R1 Distill (1.5B–70B)** | Distilled reasoning — 1.5B variant fits easily |
| **Nemotron Nano (series)** | NVIDIA's own edge-optimized models |

#### MoE Models — Why Active Parameters Matter for Jetson

MoE (Mixture of Experts) models activate only a fraction of their total parameters per token. This changes the memory math:

```
Dense model (Llama 3.2 3B):
  Total params = Active params = 3B
  Q4_K_M size: 1.8 GB  (must load ALL weights per token)
  Memory bandwidth per token: 1.8 GB

MoE model (Step 3.5 Flash 196B, 11B active):
  Total params: 196B → Q4_K_M: ~110 GB  (won't fit in 8 GB!)
  BUT active params per token: 11B → needs ~6.5 GB of the 110 GB

  Challenge: even though only 11B are active, the FULL 110 GB model
  must be in memory because different tokens may route to different experts.
  → MoE models need the full weight set loaded, not just active params.

  For Jetson: MoE only helps if the TOTAL model fits.
  Step 3.5 Flash (110 GB) → won't fit on 8 GB Jetson
  A hypothetical MoE with 16B total / 4B active → would fit and be fast!
```

**MoE models that could work on Jetson (if available in small total size):**

| Model | Total params | Active params | Q4_K_M total | Fits 8 GB? |
|-------|-------------|---------------|-------------|------------|
| Mixtral 8x0.5B (hypothetical) | 4B | 0.5B | ~2.4 GB | Yes |
| Llama 4 Scout small variant | TBD | TBD | TBD | Watch for small MoE releases |
| Step Flash distilled | TBD | TBD | TBD | If StepFun releases smaller variant |

The real edge MoE opportunity: models with **<8B total parameters** and **<2B active** — giving dense-model memory footprint with large-model routing quality. This is an active research area.

### 2.4 Choosing the Right Model — Decision Flowchart

```
What's your use case?
│
├── Simple classification / extraction / tool calling
│   └── Llama 3.2 1B or Qwen 3 1.7B  (< 1 GB, ~50+ tok/s)
│
├── General chat / assistant
│   └── Llama 3.2 3B or Gemma 2 2B  (1.5–1.8 GB, ~25–35 tok/s)
│
├── Reasoning / math / code
│   └── Phi-4 Mini 3.8B or Qwen 3 4B  (2.3–2.5 GB, ~18–20 tok/s)
│
├── Multimodal (image + text)
│   └── Gemma 3 4B  (2.5 GB, supports image input)
│
├── Bilingual (Chinese + English)
│   └── Qwen 3 4B or Qwen 2.5 3B  (1.7–2.4 GB)
│
├── Code generation
│   └── Phi-4 Mini or Qwen 2.5 Coder 3B  (if available at 3B)
│
├── Maximum quality (willing to accept slower speed)
│   └── Llama 3.3 8B Q4_K_M with 1K context  (4.6 GB, ~10 tok/s)
│
└── Need long context (8K+ tokens)
    └── Llama 3.2 3B (128K native) or Gemma 3 1B (32K)
        Cap actual context to fit KV cache budget
```

### 2.5 Ollama on Jetson — Easiest Deployment

[Ollama](https://ollama.com) runs natively on Jetson with CUDA support:

```bash
# Install Ollama on Jetson
curl -fsSL https://ollama.com/install.sh | sh

# Pull and run a model (automatically selects Q4_K_M)
ollama pull llama3.2:3b
ollama run llama3.2:3b

# Or for smaller/faster:
ollama pull qwen3:1.7b
ollama run qwen3:1.7b

# Serve as API (compatible with OpenAI format)
ollama serve &
curl http://localhost:11434/v1/chat/completions \
  -d '{"model": "llama3.2:3b", "messages": [{"role": "user", "content": "Hello"}]}'
```

**Ollama advantages on Jetson:**
- Automatic CUDA detection and GPU offloading
- Built-in model management (pull, delete, list)
- OpenAI-compatible API endpoint
- Handles quantization automatically
- One-command install, no Python dependencies

```bash
# Check GPU utilization while running
tegrastats --interval 1000

# List downloaded models and sizes
ollama list

# Remove a model to free space
ollama rm llama3.2:3b
```

### 2.6 Offline Model Transfer — Jetson Without Internet

Production Jetson devices often have no internet access (air-gapped, DNS issues, factory floor). Transfer models from an internet-connected machine via USB or LAN.

**Step 1 — Download on your workstation/VM:**

```bash
# On internet-connected machine (workstation, cloud VM, etc.)
cd /tmp

# Download GGUF model from Hugging Face
wget "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/\
Llama-3.2-3B-Instruct-Q4_K_M.gguf"

# Or for Nemotron (NVIDIA's edge model):
wget "https://huggingface.co/bartowski/Nemotron-Mini-4B-Instruct-GGUF/resolve/main/\
Nemotron-Mini-4B-Instruct-Q4_K_M.gguf"
```

**Step 2 — Transfer to Jetson via USB or LAN:**

```bash
# Method A: USB device-mode (Jetson acts as USB Ethernet at 192.168.55.1)
scp Llama-3.2-3B-Instruct-Q4_K_M.gguf user@192.168.55.1:/opt/models/

# Method B: Ethernet/WiFi LAN (replace with Jetson's IP)
scp Llama-3.2-3B-Instruct-Q4_K_M.gguf user@192.168.1.100:/opt/models/

# Method C: USB flash drive (if no network)
# Mount USB drive on workstation, copy model, unmount, plug into Jetson
cp Llama-3.2-3B-Instruct-Q4_K_M.gguf /media/usb_drive/
# On Jetson:
sudo mount /dev/sda1 /mnt
cp /mnt/Llama-3.2-3B-Instruct-Q4_K_M.gguf /opt/models/
```

**Step 3 — Verify on Jetson:**

```bash
# Check file exists and size is correct
ls -lh /opt/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
# Expected: ~1.8 GB for 3B Q4_K_M

# Verify integrity (optional — compare SHA256 with HuggingFace page)
sha256sum /opt/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf

# Check disk space
df -h /opt/models/
```

**Step 4 — Run inference:**

```bash
# With llama.cpp:
./llama-cli -m /opt/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf -ngl 99 -c 2048

# With Ollama (import local GGUF):
# Create a Modelfile
echo 'FROM /opt/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf' > Modelfile
ollama create my-llama -f Modelfile
ollama run my-llama
```

**USB device-mode setup (if not working):**

```bash
# Check if USB device-mode service is running
systemctl status nv-l4t-usb-device-mode

# Enable if disabled
sudo systemctl enable --now nv-l4t-usb-device-mode

# Verify Jetson has USB IP
ip addr show usb0    # should show 192.168.55.1

# On workstation, verify connectivity
ping 192.168.55.1
```

**Transfer speed reference:**

| Method | Speed | Time for 2 GB model |
|--------|-------|---------------------|
| USB 2.0 device-mode | ~30 MB/s | ~67 sec |
| USB 3.0 flash drive | ~100 MB/s | ~20 sec |
| Gigabit Ethernet | ~110 MB/s | ~18 sec |
| WiFi (802.11ac) | ~40 MB/s | ~50 sec |

---

## 3. KV Cache Management — The Hidden Memory Consumer

During autoregressive generation, the KV cache stores key/value tensors for every token in the context. This can consume more memory than the model itself.

### 3.1 KV Cache Size Calculation

```
KV cache size = 2 × num_layers × num_kv_heads × head_dim × context_length × bytes_per_element

Llama 3.2 3B (INT8 KV cache, 2048 context):
  = 2 × 26 layers × 8 kv_heads × 128 head_dim × 2048 tokens × 1 byte
  = 2 × 26 × 8 × 128 × 2048
  = ~109 MB

Same model at 8192 context:
  = ~435 MB  ← significant on 8 GB!

Same model at 128K context:
  = ~6.8 GB  ← impossible, exceeds total free memory
```

### 3.2 KV Cache Optimization Techniques

| Technique | Memory saving | Quality impact | Jetson support |
|-----------|-------------|----------------|----------------|
| **INT8 KV cache** | 2× vs FP16 | Minimal | llama.cpp, TensorRT-LLM |
| **INT4 KV cache** | 4× vs FP16 | Small | llama.cpp (experimental) |
| **GQA (Grouped Query Attention)** | 4–8× vs MHA | None (model-level) | Built into modern models |
| **Sliding window attention** | Bounded | Loses long context | Mistral, some models |
| **KV cache eviction** | Bounded | Loses old context | Custom implementation |
| **PagedAttention (vLLM)** | No fragmentation waste | None | Not on Jetson (vLLM = server) |

**GQA is the most important:** Llama 3.2 uses GQA with 8 KV heads (vs 32 query heads). This means the KV cache is 4× smaller than traditional MHA. Always prefer GQA models on Jetson.

### 3.3 Context Length Budget

```
Orin Nano 8GB Memory Budget for LLM:

Total DRAM:                    8.0 GB
  - Firmware carveouts:       -0.4 GB
  - OS + kernel:              -0.5 GB
  - CMA:                      -0.5 GB (reduced for LLM workload)
  - CUDA runtime:             -0.3 GB
  ────────────────────────────────────
  Available for LLM:           6.3 GB

Model (Llama 3.2 3B INT4):   -1.5 GB
KV cache (INT8, ctx=2048):   -0.1 GB
Activation memory:            -0.2 GB
  ────────────────────────────────────
  Remaining:                   4.5 GB  ← room for longer context or larger model

Model (Phi-3 Mini INT4):     -1.9 GB
KV cache (INT8, ctx=4096):   -0.3 GB
Activation memory:            -0.3 GB
  ────────────────────────────────────
  Remaining:                   3.8 GB  ← still comfortable
```

---

## 4. FlashAttention and Fused Kernels

### 4.1 Why Fused Kernels Matter on Jetson

On H100, fused kernels save SRAM bandwidth and improve Tensor Core utilization. On Jetson, they save **DRAM bandwidth** — the #1 bottleneck.

```
Unfused attention (naive):
  Q × K^T → write attention scores to DRAM → read back → softmax → write →
  read back → multiply by V → write output

  Total DRAM traffic: ~4× the minimum

FlashAttention-2 (fused):
  Q × K^T → softmax → × V  (all in SRAM/registers, ONE read + ONE write)

  Total DRAM traffic: ~1× the minimum → 4× less bandwidth used
```

On Jetson Orin Nano Super's 102 GB/s bandwidth, this is the difference between 20 tokens/sec and 50 tokens/sec.

### 4.2 FlashAttention on Jetson

```bash
# llama.cpp automatically uses FlashAttention when available
./llama-cli -m model.gguf -ngl 99 -fa  # -fa enables FlashAttention

# TensorRT-LLM compiles FlashAttention into the engine
trtllm-build --model_dir ./model --output_dir ./engine \
    --use_fused_mlp \
    --use_flash_attn \
    --max_batch_size 1 \
    --max_input_len 2048 \
    --max_seq_len 2560
```

### 4.3 Other Fused Operations

Each fused operation eliminates a DRAM round-trip:

| Fused operation | What it combines | Bandwidth saved |
|----------------|-----------------|-----------------|
| **Fused RMSNorm** | Norm + scale in one kernel | ~2× less traffic |
| **Fused SwiGLU** | Gate + activation + multiply | ~3× less traffic |
| **Fused Rotary Embedding** | Position encoding + Q/K projection | ~2× less traffic |
| **Fused Add + Norm** | Residual connection + layer norm | ~2× less traffic |

TensorRT-LLM enables these automatically. llama.cpp has many fused CUDA kernels built in.

---

## 5. TensorRT-LLM on Jetson

TensorRT-LLM is NVIDIA's optimized inference engine for LLMs. It compiles the model into a TensorRT engine with fused kernels, quantization, and Tensor Core usage.

### 5.1 Build a TensorRT-LLM Engine for Jetson

```bash
# Install TensorRT-LLM (JetPack 6.x includes TensorRT, add LLM extension)
pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com

# Convert Hugging Face model to TensorRT-LLM checkpoint
python convert_checkpoint.py \
    --model_dir ./Llama-3.2-3B-Instruct \
    --output_dir ./checkpoint \
    --dtype float16 \
    --tp_size 1          # single GPU on Jetson

# Build optimized engine
trtllm-build \
    --checkpoint_dir ./checkpoint \
    --output_dir ./engine \
    --gemm_plugin float16 \
    --max_batch_size 1 \
    --max_input_len 1024 \
    --max_seq_len 2048 \
    --max_num_tokens 2048 \
    --use_fused_mlp enable \
    --use_flash_attn enable \
    --strongly_typed

# Run inference
python run.py \
    --engine_dir ./engine \
    --tokenizer_dir ./Llama-3.2-3B-Instruct \
    --max_output_len 256 \
    --input_text "How does Jetson unified memory work?"
```

### 5.2 INT4 AWQ Engine for Minimum Memory

```bash
# Build with INT4 AWQ quantization
trtllm-build \
    --checkpoint_dir ./checkpoint-awq \
    --output_dir ./engine-int4 \
    --gemm_plugin auto \
    --max_batch_size 1 \
    --max_input_len 1024 \
    --max_seq_len 2048 \
    --use_fused_mlp enable \
    --weight_only_precision int4_awq
```

### 5.3 TensorRT-LLM vs llama.cpp on Jetson

| | TensorRT-LLM | llama.cpp |
|---|-------------|-----------|
| **Setup complexity** | High (build engine) | Low (download GGUF, run) |
| **Performance** | Best (compiled kernels) | Very good (hand-tuned CUDA) |
| **Quantization** | FP16, INT8, INT4 AWQ/GPTQ | Q2–Q8 mixed precision (GGUF) |
| **Flexibility** | Fixed engine (rebuild for changes) | Dynamic (change at runtime) |
| **Memory efficiency** | Excellent (preallocated) | Good (dynamic allocation) |
| **Model support** | Major models (Llama, Phi, Mistral, etc.) | Almost everything on HF |
| **Best for** | Production deployment | Prototyping + production |

**Recommendation:** Start with **llama.cpp** for prototyping (5-minute setup). Switch to **TensorRT-LLM** for production when you need maximum tokens/sec.

---

## 6. Speculative Decoding — Free Speed

Speculative decoding uses a small **draft model** to guess N tokens, then the large **target model** verifies all N in one forward pass. If the guess is correct, you get N tokens for the price of 1.

```
Without speculative decoding:
  Target model: generate token 1 → token 2 → token 3 → token 4
  Time: 4 forward passes × 50ms = 200ms

With speculative decoding (draft model guesses 4 tokens):
  Draft model:  generate 4 candidate tokens (fast, ~5ms total)
  Target model: verify all 4 in ONE forward pass (~55ms)
  If 3/4 accepted: 3 tokens in 60ms instead of 150ms → 2.5× faster

Speedup: 1.5–3× depending on acceptance rate
```

### 6.1 On Jetson

```bash
# llama.cpp supports speculative decoding
./llama-speculative \
    -m Llama-3.2-3B-Q4_K_M.gguf \       # target model (3B)
    -md TinyLlama-1.1B-Q4_K_M.gguf \    # draft model (1.1B)
    -ngl 99 \
    --draft 8 \                           # speculate 8 tokens ahead
    -p "Write a comprehensive guide to..."
```

**Memory budget for speculative decoding:**
- Target: Llama 3.2 3B INT4 = 1.5 GB
- Draft: TinyLlama 1.1B INT4 = 0.6 GB
- Total: 2.1 GB — fits easily on 8 GB

**When NOT to use on Jetson:** if the draft model pushes you over the memory budget, speculative decoding hurts more than it helps.

---

## 7. Runtime Optimizations

### 7.1 Power Mode Selection

```bash
# Check available power modes
sudo nvpmodel -q --verbose

# Set to maximum performance (15W on Orin Nano, 25W on Orin NX)
sudo nvpmodel -m 0
sudo jetson_clocks    # lock GPU/CPU at max frequency

# Or set power-efficient mode (7W)
sudo nvpmodel -m 1   # fewer CPU cores, lower GPU clock
```

Higher power mode = higher clock = more tokens/sec. But thermal design must support it.

### 7.2 GPU Frequency and Memory Clock

```bash
# Check current clocks
tegrastats --interval 500

# Lock GPU to max clock (prevents dynamic frequency scaling during inference)
sudo jetson_clocks

# Check GPU clock
cat /sys/devices/17000000.ga10b/devfreq/17000000.ga10b/cur_freq
```

Dynamic frequency scaling adds latency jitter. For consistent inference latency, lock clocks.

### 7.3 NUMA-Aware Allocation (AGX Orin)

On AGX Orin with larger memory, ensure CUDA allocations use the optimal memory controller:

```bash
# Pin process to specific CPU cores close to memory controller
taskset -c 0-5 ./llama-cli -m model.gguf -ngl 99
```

### 7.4 Swap / zram for Emergency Overflow

If a model barely doesn't fit, zram (compressed RAM swap) can help:

```bash
# Enable 4 GB zram (compressed in-memory swap)
sudo zramctl --find --size 4G --algorithm zstd
sudo mkswap /dev/zram0
sudo swapon /dev/zram0 -p 5

# Now models slightly over RAM can run (with performance penalty)
```

zram compresses pages in memory — ~2:1 ratio for model weights. A 6 GB model on 5.5 GB available might work via zram, but with 30–50% speed penalty due to compression/decompression overhead.

---

## 8. Kernel-Level Optimization — Where the Real Gains Are

Sections 1–7 cover model-level and system-level optimizations. This section goes deeper — into the **GPU kernels themselves**. This is where cloud platforms like RightNow Forge and RunInfra achieve their 3–7× speedups over baseline inference.

### 8.1 The GPU Utilization Problem

Most AI inference wastes 80%+ of available GPU cycles:

```
Typical unoptimized LLM inference on GPU:

SM·00░·····█░░░░·····█░░░░·····█░     █ = compute
SM·01····██░░░·····██░░░·····██░░     ░ = memory I/O (waiting for DRAM)
SM·02···█░░░░·····█░░░░·····█░░░░     · = idle (nothing scheduled)
SM·03·██░░░·····██░░░·····██░░░··
  ~16% SM utilization

After kernel optimization:

SM·00██·█████░·█████░████████████     Same hardware, 5× more useful work
SM·01░░████████████████·██████·██
SM·02██████████··█████░██████░███
SM·03███·█████░░████████████████·
  ~88% SM utilization
```

**Why this happens:** Default PyTorch/ONNX kernels are generic — they work on any GPU but optimize for none. Each operation (attention, norm, quantized GEMM) launches a separate kernel, reads from DRAM, computes, writes back. The GPU spends most of its time waiting for memory.

**On Jetson this matters even more:** 102 GB/s shared bandwidth (vs 3,350 GB/s on H100) means the GPU is frequently starved for data. Kernel optimization directly determines tokens/sec.

### 8.2 The Three Levels of Kernel Optimization

```
Level 1: Operator Fusion (easiest, biggest win)
  Combine multiple operations into one kernel → fewer DRAM round-trips
  Example: RMSNorm + residual add + SwiGLU → one kernel, one read, one write

Level 2: Hardware-Specific Tuning (moderate difficulty)
  Tune tile sizes, thread block dimensions, shared memory usage for YOUR specific GPU
  Example: Orin Nano Ampere SM has different optimal tile size than H100 Hopper SM

Level 3: Custom Kernel Generation (hardest, maximum performance)
  Write or generate Triton/CUDA kernels specifically for your model + GPU + precision
  Example: INT4 dequantize-fused-GEMM kernel for Ampere with 128-thread blocks
```

### 8.3 Profiling — Find the Bottleneck First

**Never optimize blind.** Profile to find which kernels consume the most time:

```bash
# On Jetson: profile with Nsight Systems
nsys profile --trace=cuda,nvtx -o llm_profile ./llama-cli -m model.gguf -ngl 99 -p "test"

# Analyze the trace
nsys stats llm_profile.nsys-rep

# Example output (typical LLM breakdown):
# Kernel                          Time%    Time       Calls
# ─────────────────────────────────────────────────────────
# attention_fwd                   41.2%    18.4ms     26      ← #1 bottleneck
# quantized_gemm_w4a16           22.8%    10.2ms     78
# rmsnorm_kernel                  14.1%     6.3ms     52
# silu_mul_kernel                  8.3%     3.7ms     26
# rotary_embedding                 5.1%     2.3ms     52
# others                           8.5%     3.8ms     ...
```

**On Jetson, attention dominates even more** than on server GPUs because the memory-bandwidth cost of loading Q, K, V from DRAM is proportionally higher.

### 8.4 Triton Kernels — Writing Custom Optimized Ops

Triton is NVIDIA's Python-based GPU kernel language. It's much easier than raw CUDA and generates near-optimal code.

**Example: Fused RMSNorm + Residual Add (common LLM bottleneck):**

```python
import triton
import triton.language as tl

@triton.jit
def fused_rmsnorm_residual_kernel(
    X_ptr, Residual_ptr, Weight_ptr, Out_ptr,
    N: tl.constexpr, eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """RMSNorm(X + Residual) * Weight — one kernel, one DRAM read, one write."""
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load X and Residual (one DRAM read each)
    x = tl.load(X_ptr + row * N + offsets, mask=mask, other=0.0)
    r = tl.load(Residual_ptr + row * N + offsets, mask=mask, other=0.0)

    # Fused: add residual + compute RMS norm + scale by weight
    h = x + r                                          # residual add
    mean_sq = tl.sum(h * h, axis=0) / N               # variance
    rrms = 1.0 / tl.sqrt(mean_sq + eps)               # reciprocal RMS
    w = tl.load(Weight_ptr + offsets, mask=mask, other=1.0)
    out = h * rrms * w                                 # normalize + scale

    # One DRAM write
    tl.store(Out_ptr + row * N + offsets, out, mask=mask)
```

**Without fusion:** 3 separate kernels (residual add, RMSNorm, weight multiply) = 6 DRAM accesses.
**With fusion:** 1 kernel = 2 DRAM accesses. **3× less memory traffic.**

### 8.5 Autokernel — Automated Kernel Generation

[RightNow Autokernel](https://github.com/RightNow-AI/autokernel) automates the process of generating optimized Triton/CUDA kernels for specific GPU hardware:

```bash
# Install autokernel
pip install autokernel

# Generate optimized kernels for your model + GPU
autokernel optimize \
    --model "Llama-3.2-3B" \
    --gpu "orin-nano" \
    --precision "int4" \
    --output ./optimized_kernels/
```

**What autokernel does:**
1. **Profiles** your model to find the slowest kernels (attention, GEMM, norm)
2. **Generates** Triton kernel variants with different tile sizes, thread configurations
3. **Benchmarks** all variants on your specific GPU
4. **Selects** the fastest configuration
5. **Verifies** numerical correctness against reference implementation

**The key insight:** optimal kernel parameters differ dramatically between GPUs:

| Parameter | H100 (Hopper) | A100 (Ampere) | Orin Nano (Ampere) |
|-----------|--------------|---------------|-------------------|
| Best GEMM tile | 256×128 | 128×128 | **64×64** (smaller SMs) |
| Thread block | 256 threads | 256 threads | **128 threads** (fewer warps) |
| Shared mem usage | 164 KB | 164 KB | **48 KB** (less per SM) |
| Optimal batch | 64+ | 32+ | **1–4** (memory limited) |

A kernel tuned for H100 can be **2–3× slower** on Orin Nano than a kernel tuned for Orin Nano specifically.

### 8.6 RightNow Forge — Enterprise Kernel Optimization

[RightNow Forge](https://www.rightnowai.co/forge) is the enterprise platform that automates the full kernel optimization pipeline:

```
Input:  model = "Llama-3.2-3B"
        gpu = "Jetson Orin Nano"
        baseline = "llama.cpp default"

Forge pipeline:
  1. Profile all kernels on target GPU
  2. Identify bottlenecks:
     ▲ attention       41% of total → generate FlashAttention variant for Ampere
     ▲ quantized GEMM  23% of total → generate INT4 dequant-fused GEMM
     ▲ rmsnorm         14% of total → generate fused RMSNorm+residual
  3. Compile optimized Triton kernels for Orin Nano SM
  4. Verify correctness (bit-accurate vs reference)
  5. Output: drop-in replacement kernels

Result:
  TTFT (Time to First Token): 320ms → 42ms (7.6× faster)
  Throughput: 15 tok/s → 45 tok/s (3× faster)
  SM utilization: 16% → 72%
```

### 8.7 Manual Kernel Optimization Checklist

If you're writing your own optimized kernels (Phase 5F territory), here's the priority order for Jetson:

```
Priority 1: Reduce DRAM traffic (Jetson's #1 bottleneck)
  □ Fuse consecutive elementwise ops (norm + add + activation)
  □ Fuse dequantize into GEMM (don't write dequantized weights to DRAM)
  □ Use FlashAttention (fuse Q×K + softmax + ×V into one kernel)
  □ Compute in registers/shared memory, write final result once

Priority 2: Maximize Tensor Core utilization
  □ Use WMMA/MMA instructions for matrix multiply (not scalar CUDA cores)
  □ Pad matrices to multiples of 16 for Tensor Core alignment
  □ Keep data in FP16/INT8 format that Tensor Cores consume directly

Priority 3: Tune for Orin Nano's specific SM
  □ Smaller tile sizes (64×64 vs 128×128 on server GPUs)
  □ Fewer threads per block (128 vs 256 — fewer warps available)
  □ Account for 48 KB shared memory limit per SM
  □ Fewer SMs (16 on Orin Nano vs 132 on H100) — fewer blocks in flight

Priority 4: Minimize kernel launch overhead
  □ Fuse small kernels into larger ones
  □ Use CUDA graphs to batch kernel launches
  □ Pre-allocate all buffers (no cudaMalloc during inference)
```

### 8.8 CUDA Graphs — Eliminate Launch Overhead

Each CUDA kernel launch has ~5–10 µs overhead. An LLM forward pass with 100+ kernel launches wastes ~1 ms just on launch overhead. CUDA graphs capture the entire sequence and replay it in one call:

```cpp
// Capture the inference graph once
cudaGraph_t graph;
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// All kernel launches are recorded, not executed
attention_kernel<<<grid, block, 0, stream>>>(q, k, v, out);
rmsnorm_kernel<<<grid, block, 0, stream>>>(out, norm_out);
ffn_kernel<<<grid, block, 0, stream>>>(norm_out, ffn_out);
// ... all layers ...

cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

// Replay the entire forward pass with ONE launch
for (int token = 0; token < max_tokens; token++) {
    update_input_pointers(token);  // update KV cache pointers
    cudaGraphLaunch(instance, stream);
    cudaStreamSynchronize(stream);
}
// Launch overhead: ~5µs total instead of ~1ms
```

TensorRT-LLM uses CUDA graphs internally. llama.cpp has experimental CUDA graph support.

---

## 9. Complete Optimization Checklist

```
Before deployment — run through this checklist:

□ Model Selection
  □ Model fits in INT4 with room for KV cache
  □ GQA-based model preferred (smaller KV cache)
  □ Context length budgeted against available memory

□ Quantization
  □ AWQ or GPTQ 4-bit for best quality/size
  □ GGUF Q4_K_M for llama.cpp deployment
  □ Calibration data representative of production input

□ KV Cache
  □ INT8 KV cache enabled
  □ Maximum context length capped to fit memory
  □ GQA model chosen to minimize KV memory

□ Inference Engine
  □ llama.cpp with -ngl 99 (full GPU offload)
  □ Or TensorRT-LLM engine compiled for target batch/context
  □ FlashAttention enabled

□ Kernel Optimization
  □ Profile with nsys to find top 3 bottleneck kernels
  □ Fused ops enabled (RMSNorm+residual, SwiGLU, rotary)
  □ CUDA graphs enabled for decode loop (reduce launch overhead)
  □ Tile sizes appropriate for Orin Nano SM (64×64, not 256×128)
  □ Consider autokernel/Forge for automated kernel tuning

□ System Configuration
  □ nvpmodel set to appropriate power mode
  □ jetson_clocks to lock frequencies
  □ CMA reduced (LLM doesn't need large CMA)
  □ Unnecessary services disabled (GUI, bluetooth)

□ Profiling
  □ tegrastats monitored during inference
  □ Tokens/sec measured at steady state
  □ Memory usage verified (no slow growth / leak)
  □ Thermal verified (no throttling under sustained load)
```

---

## 10. Benchmark Reference

Expected performance on Orin Nano Super 8 GB (25W mode, Q4_K_M, llama.cpp):

| Model | Params | GGUF size | Prompt eval | Generation | Context |
|-------|--------|-----------|-------------|-----------|---------|
| TinyLlama 1.1B | 1.1B | 0.6 GB | ~200 tok/s | ~65 tok/s | 2048 |
| Llama 3.2 1B | 1.3B | 0.7 GB | ~170 tok/s | ~55 tok/s | 2048 |
| Gemma 2 2B | 2.6B | 1.5 GB | ~85 tok/s | ~35 tok/s | 2048 |
| Llama 3.2 3B | 3.2B | 1.8 GB | ~65 tok/s | ~25 tok/s | 2048 |
| Phi-3 Mini 3.8B | 3.8B | 2.2 GB | ~50 tok/s | ~20 tok/s | 2048 |

> These are estimates for Orin Nano Super at 25W. The ~1.7× improvement over the original Orin Nano comes from 102 GB/s bandwidth (vs 68 GB/s) and higher clock speeds. Actual performance depends on power mode, thermal design, context length, and prompt content. Always benchmark your specific configuration.

---

## 11. Projects

| # | Project | What you learn |
|---|---------|---------------|
| 1 | **llama.cpp on Jetson** | Download Llama 3.2 3B Q4_K_M, build llama.cpp with CUDA, measure tokens/sec at different context lengths |
| 2 | **Quantization comparison** | Run same model at Q2_K, Q4_K_M, Q6_K, Q8_0. Measure tokens/sec, memory, and output quality (perplexity) |
| 3 | **TensorRT-LLM engine** | Build a TRT-LLM engine for Phi-3 Mini INT4. Compare latency with llama.cpp on same prompts |
| 4 | **Speculative decoding** | Set up TinyLlama as draft for Llama 3.2 3B. Measure acceptance rate and speedup |
| 5 | **Memory budget audit** | Run tegrastats during inference. Map every MB: OS, CMA, model, KV cache, activations. Verify against Section 3.3 |
| 6 | **Power vs performance** | Benchmark same model at 7W, 15W, 25W (if Orin NX). Plot tokens/sec vs power. Calculate tokens/joule |
| 7 | **Context length scaling** | Measure tokens/sec at context 512, 1024, 2048, 4096. Plot. Identify where KV cache pressure causes degradation |
| 8 | **Production chatbot** | Build a REST API serving Llama 3.2 3B on Jetson using llama.cpp server mode. Measure P50/P95 latency under concurrent requests |
| 9 | **Nsight profile analysis** | Profile llama.cpp inference with `nsys`. Identify top 3 kernels by time. Calculate SM utilization. Find memory-bound vs compute-bound kernels |
| 10 | **Fused RMSNorm Triton kernel** | Write the fused RMSNorm+residual kernel from Section 8.4. Benchmark against unfused PyTorch version. Measure DRAM traffic reduction |
| 11 | **Autokernel on Jetson** | Use autokernel to generate optimized kernels for a small model on Orin Nano. Compare throughput before/after. Document which kernels changed |
| 12 | **CUDA graphs** | Wrap the decode loop of a simple transformer in a CUDA graph. Measure kernel launch overhead before/after. Target: <10 µs total launch per token |

---

## 12. Resources

| Resource | What it covers |
|----------|---------------|
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | Best open-source LLM inference engine for edge |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | NVIDIA's optimized LLM engine |
| [NVIDIA Jetson AI Lab](https://www.jetson-ai-lab.com/) | Pre-built containers and tutorials for LLMs on Jetson |
| [Jetson Generative AI Playground](https://developer.nvidia.com/embedded/generative-ai) | NVIDIA's LLM deployment guides for Jetson |
| [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) | AWQ quantization library |
| [FlashAttention-2 paper](https://arxiv.org/abs/2307.08691) | Algorithm behind fused attention |
| [Speculative Decoding paper](https://arxiv.org/abs/2302.01318) | Original speculative decoding paper |
| [vLLM paper (PagedAttention)](https://arxiv.org/abs/2309.06180) | KV cache memory management (server reference) |
| [Orin Nano Memory Architecture](../../../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Memory-Architecture/Guide.md) | Unified memory deep dive (this roadmap) |
| [RightNow Autokernel](https://github.com/RightNow-AI/autokernel) | Open-source automated GPU kernel optimization |
| [RightNow Forge](https://www.rightnowai.co/forge) | Enterprise kernel optimization platform (profile → generate → verify) |
| [Triton Language](https://triton-lang.org/) | Python-based GPU kernel language (easier than CUDA) |
| [CUDA Graphs Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-graphs) | Capture and replay kernel sequences for reduced launch overhead |
| [RunInfra](https://www.runinfra.com/) | Cloud LLM optimization platform (reference for techniques) |

---

## Next

→ Back to [ML and AI hub](../Guide.md)
