# Lecture 05 — Physical AI and Multimodal Gemma 4: SigLIP, VLM Pipeline, and Robot Perception on Jetson

**Collection:** [Gemma 4 Edge Deployment](README.md) | **Previous:** [← Lecture 04](Lecture-04.md)

---

The previous four lectures covered Gemma 4 as a text model — architecture, quantization, runtimes, speculative decode. This lecture closes the loop: what happens when you attach a camera and deploy Gemma 4 multimodal on a robot? Gemma 4's **vision variants** (4B-IT-VL, 12B-IT-VL, 27B-IT-VL) use a SigLIP-400M vision encoder to produce visual tokens, which are prefixed into the Gemma text decoder. The combined model is a Vision-Language Model (VLM) capable of answering spatial questions about what it sees, describing scenes, and driving agentic actions — tasks that define **physical AI** on Jetson.

This lecture covers the SigLIP-400M encoder architecture, the VLM pipeline's latency budget, deployment of Gemma 4-VL on Jetson Orin, quantization for the vision stack, and real robot perception use cases.

---

## Learning objectives

1. Describe the **SigLIP-400M encoder** architecture and how it encodes visual tokens for Gemma 4's decoder.
2. Compute the **end-to-end VLM latency budget** (image encode → prefill → decode) for Gemma 4 4B-VL on Jetson Orin.
3. Deploy Gemma 4-VL on Jetson using Ollama (vision support), llama.cpp mmproj, or HuggingFace Transformers.
4. Explain the **quantization challenges specific to vision encoders** and why you treat the vision encoder and text decoder as separate quantization targets.
5. Design a **robot perception pipeline** (camera → VLM → action) with a measured TTFT latency target for closed-loop control.

---

## 1. SigLIP-400M — the vision encoder for Gemma 4

### 1.1 Architecture overview

SigLIP (Sigmoid Loss for Language-Image Pre-training, Zhai et al., 2023) is Google's successor to CLIP for vision-language alignment:

```text
CLIP vs SigLIP:
  CLIP objective:     softmax contrastive loss (global normalization across batch)
                      numerically stable but requires large batch sizes (>4096)
  SigLIP objective:   sigmoid binary loss per image-text pair (local, per-pair)
                      stable at smaller batches, scales better, stronger at retrieval

SigLIP-400M for Gemma 4:
  Architecture:  ViT-L/14 variant (Vision Transformer Large with 14×14 patches)
  Parameters:    ~400M (400 million)
  Input:         images resized to 224×224 (baseline) or 448×448 (high-res variant)
  Patch size:    14×14 pixels → 256 patches at 224px, 1024 patches at 448px
  Embedding dim: 1024
  Transformer:   24 layers, 16 attention heads, FFN dim 4096
  Output:        256 or 1024 visual tokens (each 1024-dim)
  Projection:    linear proj 1024 → Gemma 4 hidden_dim (e.g., 2048 for 4B)
```

### 1.2 Image tokenization pipeline

```python
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import PIL.Image

# Gemma 4-VL uses the PaliGemma/SigLIP processor for image tokenization:
processor = AutoProcessor.from_pretrained("google/gemma-4-4b-it")

# Load an image (e.g., from Jetson camera feed):
image = PIL.Image.open("/dev/video0")  # or file path

# Processor tokenizes both image and text:
inputs = processor(
    images=image,
    text="<image>\nDescribe what the robot arm is holding.",
    return_tensors="pt"
)

# inputs["pixel_values"]: (1, 3, 224, 224) — resized + normalized image
# inputs["input_ids"]:    (<img> token × 256) + text tokens

# SigLIP encodes pixel_values → 256 visual tokens of dim 1024
# Linear projection → Gemma 4 hidden_dim tokens
# These 256 tokens are PREPENDED to the text token sequence
# Gemma 4 decoder processes all (256 + text_tokens) as the prefill
```

### 1.3 Visual token count and prefill cost

The image resolution determines how many visual tokens are prepended:

```text
224×224 (standard):  256 visual tokens  → prefill adds 256 tokens
448×448 (high-res):  1024 visual tokens → prefill adds 1024 tokens

PREFILL COST on Orin (Gemma 4 4B INT4, 204 GB/s):
  Standard (256 tokens): prefill throughput ~500 tok/s (compute-bound)
                         latency: 256 / 500 = ~0.5 s prefill
                         (faster than text prefill because attention is local
                          for sliding-window layers)
  High-res (1024 tokens): latency: ~2.0 s prefill

  PLUS SigLIP encoding:  224×224 → ~80 ms on GPU (FP16 ViT-L/14)
                         448×448 → ~200 ms on GPU

Total TTFT (time-to-first-token):
  Standard:  80 ms (SigLIP) + 500 ms (prefill) + 16 ms (first decode) = ~600 ms
  High-res:  200 ms (SigLIP) + 2000 ms (prefill) + 16 ms = ~2.2 s
```

For **real-time robot perception** (target ≤ 500 ms TTFT), you must use standard 224×224 resolution. High-res is for offline analysis or less time-sensitive tasks.

---

## 2. Deployment paths for Gemma 4-VL on Jetson

### 2.1 Ollama (recommended for fast bring-up)

Ollama v0.4+ supports Gemma 4 vision models natively:

```bash
# Pull and run Gemma 4 4B vision model:
ollama pull gemma4:4b-instruct-vision-q4_K_M  # ~2.2 GB

# Interactive vision query:
ollama run gemma4:4b-instruct-vision-q4_K_M

# Programmatic with image:
curl http://localhost:11434/api/generate -d '{
  "model": "gemma4:4b-instruct-vision-q4_K_M",
  "prompt": "What objects are on the table?",
  "images": ["'$(base64 -w0 /tmp/camera_frame.jpg)'"],
  "stream": false
}'

# Expected throughput on Orin (INT4, SigLIP 224px):
# encode:  ~80 ms
# prefill: ~500 ms
# decode:  ~55 tok/s
# TTFT:    ~600 ms for a typical scene description prompt
```

### 2.2 llama.cpp with mmproj (multimodal projection)

llama.cpp handles Gemma 4-VL via its mmproj (multimodal projection) system:

```bash
# You need TWO files:
# 1. The main GGUF (text model weights):
#    gemma4-4b-it-q4_k_m.gguf
# 2. The vision projection GGUF (SigLIP + projection):
#    mmproj-gemma4-4b-f16.gguf  (keep FP16 — quantizing mmproj loses accuracy)

# Interactive VLM CLI:
./build/bin/llama-cli \
    --model      gemma4-4b-it-q4_k_m.gguf \
    --mmproj     mmproj-gemma4-4b-f16.gguf \
    --n-gpu-layers 9999 \
    --image      /tmp/robot_view.jpg \
    -p "Describe what the gripper is touching."

# Server mode for API access:
./build/bin/llama-server \
    --model      gemma4-4b-it-q4_k_m.gguf \
    --mmproj     mmproj-gemma4-4b-f16.gguf \
    --n-gpu-layers 9999 \
    --port 8080 \
    --ctx-size 4096

# curl the server with an image:
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "gemma4-4b",
      "messages": [{
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'$(base64 -w0 /tmp/frame.jpg)'"}},
          {"type": "text", "text": "What object is in front of the robot?"}
        ]
      }]
    }'
```

### 2.3 HuggingFace Transformers (full precision, good for prototyping)

```python
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch, PIL.Image

# Gemma 4 VL models use the Gemma3 family in HF:
model_id = "google/gemma-4-4b-it"

processor = AutoProcessor.from_pretrained(model_id)
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

image = PIL.Image.open("/tmp/jetson_camera.jpg").convert("RGB")

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Is the robot arm's gripper open or closed?"},
    ]
}]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device)

with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)

print(processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

---

## 3. Quantizing the vision stack

The Gemma 4-VL deployment has two quantization targets with different sensitivities:

### 3.1 Text decoder (same rules as Lecture 02)

```text
Gemma 4 text decoder: Q4_K_M GGUF, AWQ INT4, or GPTQ INT4
  Sensitivity: moderate — QK-norm + GQA make it robust to INT4
  Target: 2.2 GB for 4B (vs 8.2 GB BF16)
  Accuracy loss: ≤0.3 MMLU points at Q4_K_M (see Lecture 02)
```

### 3.2 SigLIP vision encoder — keep at FP16

The vision encoder is more sensitive to quantization than the text decoder:

```text
WHY FP16 FOR SIGLIP:
  1. The encoder produces 256 visual tokens that REPLACE the image for the decoder.
     Quantization errors in the visual tokens compound across all 256 token positions.
  2. Image patches are continuous-valued — no discrete tokenization to absorb errors.
     Unlike text (limited vocab = natural quantization), pixels vary continuously.
  3. SigLIP uses sigmoid loss training — the feature space is calibrated for FP32/FP16
     precision; INT8 introduces systematic bias that degrades spatial reasoning.
  4. The mmproj (projection layer) is tiny (~80 MB FP16 for 4B).
     Saving to INT8 saves ~40 MB — not worth the accuracy degradation.

RULE: Always keep the mmproj/SigLIP encoder in FP16.
      Only quantize the text decoder (the large component).

EXCEPTION: INT8 for SigLIP is acceptable if you measure VQA accuracy drop < 1%
           on your specific task. Use llm-compressor or quanto for dynamic INT8.
```

### 3.3 Memory layout for 4B-VL on Orin

```text
Component                   FP16 Size   INT4 Size   Notes
─────────────────────────── ─────────── ─────────── ───────────────────────────
SigLIP-400M encoder         800 MB      N/A         Keep FP16
mmproj projection           80 MB       N/A         Keep FP16
Gemma 4 4B text decoder     8,200 MB    2,200 MB    Quantize to INT4
KV cache (4096 ctx, FP16)   ~200 MB     ~100 MB     Use INT8 KV (–kv-cache-type q8_0)
─────────────────────────── ─────────── ─────────── ───────────────────────────
TOTAL (text INT4 + vis FP16) 1,280 MB   3,480 MB    ← fits on Orin 64 GB
TOTAL (text BF16 + vis FP16) 9,280 MB                ← still fits, slower
```

---

## 4. Robot perception pipeline on Jetson

### 4.1 Physical AI use cases for Gemma 4-VL

Gemma 4's combination of 128K context, bounded KV, and multimodal capability makes it the first edge VLM suitable for:

**Object recognition and spatial reasoning:**

```text
"What objects are within reach of the gripper?" → closed-loop pick-and-place
"Is the bin empty?" → warehouse inventory
"Which cable is the red one?" → assembly robot guidance
```

**Scene description for autonomous navigation:**

```text
"Describe the obstacles in front of the robot." → path planning assist
"Is the path to the dock clear?" → docking AI
"What is the surface condition of the floor?" → traction-aware navigation
```

**Human-robot interaction:**

```text
"What is the person pointing at?" → instruction following
"Is the person wearing safety equipment?" → compliance monitoring
"What task is the human performing?" → collaborative manipulation
```

### 4.2 End-to-end latency budget

For a robot operating at 2 Hz (one action decision per 500 ms):

```text
BUDGET: 500 ms total
  Image capture:     10 ms   (Jetson MIPI CSI pipeline, 1080p30)
  Image resize:      5 ms    (cudaResize to 224×224)
  SigLIP encode:     80 ms   (FP16, GPU)
  Prefill:           400 ms  (256 visual tokens + 50 text tokens = 306 tokens)
                             (Gemma 4 4B INT4, ~750 tok/s prefill on Orin)
  First decode:      18 ms   (1 token at ~55 tok/s)
  TOTAL TTFT:        513 ms  ← slightly over budget

OPTIMIZATION TO FIT 500 ms:
  1. Use INT8 KV cache: saves KV memory, more GPU cache available → prefill faster
  2. Reduce prompt length: 50 → 20 text tokens → saves ~30 ms
  3. Pipeline: start prefill while previous decode is completing (temporal overlap)
  4. Result: 80 + 330 + 18 = 428 ms ← fits in 500 ms budget with headroom
```

### 4.3 Python pipeline for closed-loop perception

```python
import threading, queue, time
import torch, PIL.Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

# Load model once at startup:
model_id = "google/gemma-4-4b-it"
processor = AutoProcessor.from_pretrained(model_id)
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="cuda"
)

frame_queue = queue.Queue(maxsize=2)
result_queue = queue.Queue()

def perception_worker():
    while True:
        frame = frame_queue.get()  # (PIL.Image, timestamp)
        if frame is None: break
        image, t_capture = frame

        t0 = time.perf_counter()
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What object is nearest to the gripper?"},
            ]
        }]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device)

        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=50, do_sample=False
            )

        answer = processor.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        result_queue.put({"answer": answer, "latency_ms": latency_ms, "t_capture": t_capture})

# Start worker:
t = threading.Thread(target=perception_worker, daemon=True)
t.start()

# Main loop (simulated camera at 10 Hz, VLM at 2 Hz):
for frame_num in range(100):
    if frame_num % 5 == 0:  # every 5th frame → 2 Hz VLM rate
        img = PIL.Image.open(f"/tmp/frame_{frame_num:04d}.jpg").convert("RGB")
        if not frame_queue.full():
            frame_queue.put((img, time.perf_counter()))

    if not result_queue.empty():
        result = result_queue.get()
        print(f"Perception: '{result['answer']}' | latency: {result['latency_ms']:.0f} ms")

    time.sleep(0.1)  # 10 Hz main loop

frame_queue.put(None)  # shutdown
t.join()
```

### 4.4 Jetson camera integration (GStreamer + CUDA)

```python
import cv2, PIL.Image
import numpy as np

# GStreamer pipeline for Jetson CSI camera (tested on Orin):
gst_pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=224, height=224, format=BGRx ! "  # resize in hardware
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "appsink"
)

cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    raise RuntimeError("Failed to open CSI camera with GStreamer pipeline.")

def get_frame_for_vlm():
    ret, frame = cap.read()
    if not ret: return None
    # Convert BGR numpy → RGB PIL:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return PIL.Image.fromarray(frame_rgb)  # already 224×224 from GStreamer

# SigLIP processes PIL images — GStreamer handles resize in hardware (free)
```

The GStreamer pipeline resizes from 1080p to 224×224 in the ISP/video converter hardware, eliminating the ~5 ms software resize from the latency budget.

---

## 5. Multimodal quantization with LiteRT

For production deployments that require Google AI Edge certification or DLA acceleration, LiteRT exports include both the text and vision components:

```python
import ai_edge_torch
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
import torch

model_id = "google/gemma-4-4b-it"
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16
)
model.eval()

processor = AutoProcessor.from_pretrained(model_id)

# Sample inputs for tracing:
sample_image = torch.zeros(1, 3, 224, 224, dtype=torch.bfloat16)  # dummy image
sample_text = processor.tokenizer("Hello", return_tensors="pt")["input_ids"]

# Export SEPARATE models for vision and text components:
# (LiteRT handles them in two graphs, stitched at the projection layer)

# Vision encoder export:
vision_encoder = model.model.vision_tower  # SigLIP ViT-L
edge_vision = ai_edge_torch.convert(
    vision_encoder.eval(),
    sample_args=(sample_image,),
)
edge_vision.export("gemma4-4b-siglip-fp16.tflite")

# Text decoder export (with int8 quantization):
text_decoder = model.language_model
edge_text = ai_edge_torch.convert(
    text_decoder.eval(),
    sample_args=(sample_text,),
    quant_config=ai_edge_torch.quantize.QuantizationConfig(
        ai_edge_torch.quantize.IntQuantizationConfig(num_bits=8)
    )
)
edge_text.export("gemma4-4b-text-int8.tflite")
```

**DLA acceleration for SigLIP on Thor:**

Jetson AGX Thor's DLA (Deep Learning Accelerator) can run the SigLIP ViT-L/14 encoder at lower power than the discrete GPU:

```text
SigLIP on Orin iGPU (FP16): ~80 ms, ~4W
SigLIP on Orin DLA (INT8):  ~120 ms, ~0.8W  ← 5× power reduction, 50% slower
SigLIP on Thor GPU (FP16):  ~40 ms, ~3W
SigLIP on Thor DLA (INT8):  ~60 ms, ~0.4W   ← 7.5× power reduction, 50% slower

RULE: Use DLA for SigLIP when power budget is critical (mobile robot, drone).
      Use GPU for SigLIP when latency budget is critical (manipulation robot).
```

---

## 6. Performance comparison: Gemma 4-VL vs other edge VLMs

| Model | Parameters | Runtime | Orin TTFT | Orin tok/s | KV at 4K ctx | Notes |
|-------|-----------|---------|-----------|-----------|-------------|-------|
| Gemma 4 4B-VL INT4 | 4B | llama.cpp | ~600 ms | 55 | 200 MB | Best throughput/quality |
| Gemma 4 1B-VL INT4 | 1B | llama.cpp | ~150 ms | 180 | 50 MB | Fastest, lower quality |
| Llama 3.2 11B-VL INT4 | 11B | ollama | ~900 ms | 25 | 400 MB | Stronger reasoning |
| Qwen2.5-VL 7B INT4 | 7B | llama.cpp | ~750 ms | 38 | 300 MB | Excellent OCR |
| Phi-4 14B-VL INT4 | 14B | TRT-LLM | ~1200 ms | 18 | 550 MB | Needs Thor or +GPU |
| Gemma 4 12B-VL INT4 | 12B | llama.cpp | ~1100 ms | 22 | 450 MB | Best on Thor |

**Gemma 4 4B-VL is the recommended target for Orin**. It hits 600 ms TTFT (acceptable for 1 Hz control), 55 tok/s decode for verbose descriptions, and 200 MB KV at 4K context — leaving ample space for the rest of the robot software stack.

---

## 7. Capstone: end-to-end robot perception system

### Capstone task

Build a perception module that:
1. Accepts a camera frame from a Jetson CSI camera
2. Runs Gemma 4 4B-VL to answer "Is there an obstacle in the robot's path?"
3. Returns a structured JSON answer: `{"obstacle": true/false, "description": "...", "confidence": 0-1, "latency_ms": ...}`
4. Operates at ≥ 1 Hz (TTFT ≤ 1000 ms)

### Reference implementation

```python
import json, time, re
import torch, PIL.Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

SYSTEM_PROMPT = """You are a robot perception module.
Answer ONLY with a JSON object: {"obstacle": true/false, "description": "...", "confidence": 0.0-1.0}
Keep description under 20 words. Be conservative: if unclear, obstacle=true."""

def build_perception_model(model_id="google/gemma-4-4b-it"):
    proc = AutoProcessor.from_pretrained(model_id)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    return proc, model

def perceive(image: PIL.Image.Image, proc, model) -> dict:
    t0 = time.perf_counter()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Is there an obstacle in the robot's path?"}
        ]}
    ]

    inputs = proc.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=80, do_sample=False)

    raw = proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Extract JSON from model output:
    match = re.search(r'\{.*?\}', raw, re.DOTALL)
    if match:
        result = json.loads(match.group())
    else:
        result = {"obstacle": True, "description": raw[:100], "confidence": 0.5}

    result["latency_ms"] = (time.perf_counter() - t0) * 1000
    return result

# Usage:
proc, model = build_perception_model()
image = PIL.Image.open("/tmp/robot_view.jpg").convert("RGB")
result = perceive(image, proc, model)
print(result)
# → {"obstacle": false, "description": "Clear path, table and boxes visible", "confidence": 0.9, "latency_ms": 582}
```

### Capstone verification checklist

- [ ] TTFT ≤ 1000 ms measured end-to-end (capture → JSON response) on Orin
- [ ] Obstacle=true rate for images with blockers ≥ 90% (test 20 samples)
- [ ] False positive rate (obstacle=true on clear path) ≤ 10% (test 20 samples)
- [ ] Continuous operation for 5 minutes without memory growth (check `tegrastats`)
- [ ] KV cache memory stays bounded (set `--ctx-size 4096` to cap it)
- [ ] Pipeline produces valid JSON for every response (test 50 diverse images)

---

## Key takeaways

- **SigLIP-400M** encodes images as 256 visual tokens (224×224) or 1024 tokens (448×448) that are prepended to the text context as standard input tokens — no special attention masking required.
- **TTFT for Gemma 4 4B-VL on Orin** is approximately 600 ms (80 ms SigLIP + 500 ms prefill + 20 ms decode), fitting within a 1 Hz robot control loop.
- **Always keep the vision encoder in FP16**; only quantize the text decoder to INT4. The SigLIP weights (880 MB FP16) add ~1 GB to the deployment footprint but are not large enough to justify the accuracy loss from INT4.
- **llama.cpp with --mmproj** is the fastest path to a working VLM deployment; Ollama wraps this for ease of use. Use HuggingFace Transformers for prototyping only (BF16 = 8.2 GB text decoder alone).
- **DLA for SigLIP** on Jetson Thor reduces encoder power by 7.5× at the cost of 50% higher latency — use on mobile or drone deployments where power dominates.
- **The GStreamer hardware pipeline** on Jetson resizes frames in the ISP/video converter, eliminating software resize cost and delivering 224×224 frames directly to CUDA.
- **The key latency knob** is not decode speed but **prefill** — 256 visual tokens dominate TTFT. Reduce text prompt length to save prefill time on the text side.

---

## Course completion checklist

You have completed the Gemma 4 Edge Deployment course when you can:

- [ ] Explain the KV-cache math for Gemma 4's interleaved attention and compute the exact footprint for any (model, batch, context) triple.
- [ ] Calibrate and produce a Q4_K_M GGUF from a Gemma 4 BF16 checkpoint with ≤ 0.3 MMLU degradation.
- [ ] Deploy Gemma 4 4B (text only) on Jetson Orin with llama.cpp, measured at ≥ 45 tok/s.
- [ ] Configure speculative decoding with Gemma 4 1B as draft, measure acceptance length τ, and verify the speedup matches the τ / (1 + K×c) formula.
- [ ] Deploy Gemma 4 4B-VL on Jetson Orin, measure end-to-end TTFT, and verify it meets a ≤ 1 Hz perception loop requirement.
- [ ] Complete the capstone perception module, pass the checklist, and report latency_ms in your notes.

---

## References

- "Sigmoid Loss for Language Image Pre-Training" (Zhai et al., 2023) — arXiv:2303.15343
- "Gemma 4 Technical Report" (Google DeepMind, April 2025) — check ai.google.dev/gemma
- "PaliGemma 2 VLM" (Beyer et al., 2024) — arXiv:2412.03555 (Gemma 4-VL builds on PaliGemma architecture)
- llama.cpp multimodal (--mmproj) — [github.com/ggml-org/llama.cpp/tree/master/examples/llava](https://github.com/ggml-org/llama.cpp)
- Google AI Edge / LiteRT — [ai.google.dev/edge/litert](https://ai.google.dev/edge/litert)
- Jetson Thor SDK (DLA v3) — NVIDIA JetPack 7.x SDK documentation
- [Qwen Inference Optimization → Lecture 05](../Qwen%20Inference%20Optimization/Lecture-05.md) — multimodal inference stack, same runtime layer

---

## Current as of 2026-06

Gemma 4 4B-VL available at `google/gemma-4-4b-it` (multimodal variant) on HuggingFace. SigLIP is integrated into the model; no separate download needed. llama.cpp mmproj support for Gemma 4-VL: check llama.cpp releases after v0.4.0. LiteRT AI Edge Torch: requires `ai-edge-torch >= 0.3.0` for Gemma 4 export. Ollama vision support: `ollama pull gemma4:4b` (check ollama.com/library for updated tags). Jetson AGX Thor: JetPack 7.0+ required for DLA v3 and unified memory >128 GB.

---

*Previous: [← Lecture 04 — Speculative Decoding with Gemma 4](Lecture-04.md) · Up: [Gemma 4 Edge Deployment](README.md)*
