# Edge AI Optimization — Concrete Guide

> **Focus:** Understand optimization from first principles using **tinygrad**, then apply the full porting pipeline to **Jetson Orin Nano 8GB** with TensorRT. Every concept has working code.

---

## Table of Contents

1. [Why Optimize? — The Edge Constraints](#1-why-optimize--the-edge-constraints)
2. [Quantization — From First Principles](#2-quantization--from-first-principles)
3. [Quantization in tinygrad](#3-quantization-in-tinygrad)
4. [Post-Training Quantization (PTQ) with TensorRT](#4-post-training-quantization-ptq-with-tensorrt)
5. [Quantization-Aware Training (QAT) with tinygrad](#5-quantization-aware-training-qat-with-tinygrad)
6. [Pruning — Structured and Unstructured](#6-pruning--structured-and-unstructured)
7. [Knowledge Distillation](#7-knowledge-distillation)
8. [Full Model Porting Pipeline](#8-full-model-porting-pipeline)
9. [TensorRT Engine Optimization — Jetson Deep Dive](#9-tensorrt-engine-optimization--jetson-deep-dive)
10. [DLA (Deep Learning Accelerator) on Orin Nano](#10-dla-deep-learning-accelerator-on-orin-nano)
11. [tinygrad on Jetson — CUDA Backend](#11-tinygrad-on-jetson--cuda-backend)
12. [Profiling and Benchmarking on Jetson](#12-profiling-and-benchmarking-on-jetson)
13. [DeepStream for Video Pipelines](#13-deepstream-for-video-pipelines)
14. [Projects](#14-projects)
15. [Resources](#15-resources)

### Deep Dive Subfolders

| Subfolder | Description |
|-----------|-------------|
| [tao-toolkit/](tao-toolkit/Guide.md) | NVIDIA TAO Toolkit — fine-tune NGC pre-trained models, prune, QAT, export to TensorRT, and deploy on Jetson without writing a training loop |

---

## 1. Why Optimize? — The Edge Constraints

### The Problem

A model that runs fine on a cloud A100 GPU will not run acceptably on Jetson Orin Nano without optimization.

```
A100 GPU (cloud):
  Memory:  80 GB HBM2e
  BW:      2 TB/s
  Power:   400W
  Cost:    ~$2/hour cloud

Orin Nano 8GB (edge):
  Memory:  8 GB LPDDR5 (shared CPU+GPU)
  BW:      68 GB/s
  Power:   5–15W
  Cost:    one-time hardware

YOLOv8x (unoptimized FP32):
  Model size: 136 MB
  GPU memory: ~1.2 GB
  FPS on A100: 500+
  FPS on Orin Nano FP32: ~4   ← unusable
  FPS on Orin Nano FP16: ~18
  FPS on Orin Nano INT8: ~32  ← acceptable
```

### Optimization Targets

| Target       | Technique                         | Typical Gain      |
|--------------|-----------------------------------|-------------------|
| Latency      | TensorRT + FP16/INT8              | 3–8×              |
| Memory       | Quantization, pruning             | 2–4× smaller      |
| Power        | Lower precision, DLA offload      | 2–3× less watts   |
| Throughput   | Batching, CUDA graphs             | 2–5× FPS          |

### The Accuracy-Efficiency Tradeoff

```
Accuracy
 99% |  FP32   ──────────────────────
 98% |         FP16 ─────────────────
 97% |              INT8 ────────────
 95% |                   INT4 ───────
 90% |                        PRUNE ─
      ──────────────────────────────→ Speed / Efficiency
```

The goal is to move right on this curve while staying above your minimum accuracy requirement.

---

## 2. Quantization — From First Principles

### What Quantization Does

Quantization maps floating-point values to integers:

```
FP32:   1.2847  stored as 4 bytes (32 bits)
INT8:   127     stored as 1 byte  (8 bits)

Compression: 4× smaller
Speed:       4–8× faster (integer ops + Tensor Core)
```

### The Math: Affine Quantization

Every tensor is quantized with two parameters: **scale (s)** and **zero-point (z)**:

```
Quantize:   q = round(x / s + z)    clip to [q_min, q_max]
Dequantize: x̂ = s × (q - z)

For INT8 (symmetric, zero-point = 0):
  s = max(|x|) / 127
  q = round(x / s)

For UINT8 (asymmetric):
  s = (x_max - x_min) / 255
  z = round(-x_min / s)
  q = round(x / s + z)
```

### Quantization Error

```python
import numpy as np

# Simulate quantizing a weight tensor
weights = np.random.randn(256, 256).astype(np.float32)

# Symmetric INT8 quantization
scale = np.max(np.abs(weights)) / 127.0
quantized = np.round(weights / scale).astype(np.int8)
dequantized = quantized.astype(np.float32) * scale

# Measure error
error = np.abs(weights - dequantized)
print(f"Max quantization error: {error.max():.6f}")
print(f"Mean quantization error: {error.mean():.6f}")
print(f"SNR: {20 * np.log10(np.std(weights) / np.std(error)):.1f} dB")

# Typical results:
# Max quantization error: 0.012
# Mean quantization error: 0.003
# SNR: ~40 dB  (good — well above audible noise floor analogy)
```

### Where Quantization Breaks

Some layers are sensitive to quantization:
- First and last layers (directly process raw inputs/outputs)
- Attention layers (large dynamic range)
- Batch normalization parameters

Mixed-precision quantization keeps sensitive layers in FP16 and quantizes the rest to INT8.

---

## 3. Quantization in tinygrad

Understanding quantization through tinygrad is the best way to internalize it before applying it via TensorRT.

### Implement INT8 Quantization from Scratch in tinygrad

```python
# quantization.py
from tinygrad.tensor import Tensor
import numpy as np

def quantize_tensor_int8(x: Tensor):
    """Symmetric per-tensor INT8 quantization"""
    x_np = x.numpy()
    scale = float(np.max(np.abs(x_np))) / 127.0
    scale = max(scale, 1e-8)    # avoid division by zero

    q = np.round(x_np / scale).astype(np.int8)
    q = np.clip(q, -128, 127)
    return q, scale

def dequantize_tensor(q: np.ndarray, scale: float) -> Tensor:
    return Tensor(q.astype(np.float32) * scale)

def quantized_linear(x: Tensor, W_q: np.ndarray, W_scale: float,
                     b: Tensor = None) -> Tensor:
    """
    Simulated quantized linear layer:
    1. Quantize input
    2. Integer matmul (simulated in float for this demo)
    3. Dequantize output
    """
    # Quantize input
    x_q, x_scale = quantize_tensor_int8(x)

    # Integer matmul (in practice, hardware does this natively)
    # output_q = x_q @ W_q  (INT8 @ INT8 = INT32 accumulation)
    out_q = x_q.astype(np.int32) @ W_q.astype(np.int32)

    # Dequantize: multiply by combined scale
    combined_scale = x_scale * W_scale
    out = Tensor(out_q.astype(np.float32) * combined_scale)

    if b is not None:
        out = out + b
    return out

# ── Demo ──────────────────────────────────────────────────────
# Original layer
W = Tensor.randn(128, 64)
x = Tensor.randn(32, 128)  # batch=32, input=128

# FP32 reference
out_fp32 = x.matmul(W)

# Quantized version
W_q, W_scale = quantize_tensor_int8(W)
out_int8 = quantized_linear(x, W_q, W_scale)

# Compare
diff = np.abs(out_fp32.numpy() - out_int8.numpy())
print(f"Mean error vs FP32: {diff.mean():.6f}")
print(f"Max error  vs FP32: {diff.max():.6f}")
```

### Per-Channel Quantization (Better Accuracy)

```python
def quantize_weight_per_channel(W: Tensor):
    """
    Per-output-channel quantization.
    Each output neuron gets its own scale → much better accuracy than per-tensor.
    """
    W_np = W.numpy()                         # shape [out, in]
    # Compute scale per output channel (row)
    scales = np.max(np.abs(W_np), axis=1, keepdims=True) / 127.0
    scales = np.maximum(scales, 1e-8)

    W_q = np.round(W_np / scales).astype(np.int8)
    W_q = np.clip(W_q, -128, 127)
    return W_q, scales.squeeze()

# Compare per-tensor vs per-channel accuracy
W = Tensor.randn(256, 256)
x = Tensor.randn(16, 256)
ref = x.matmul(W).numpy()

# Per-tensor
W_q_pt, W_s_pt = quantize_tensor_int8(W)
out_pt = x.numpy().astype(np.float32) @ W_q_pt.astype(np.float32) * W_s_pt

# Per-channel
W_q_pc, W_s_pc = quantize_weight_per_channel(W)
out_pc = (x.numpy().astype(np.float32) @ W_q_pc.astype(np.float32)) * W_s_pc

print(f"Per-tensor  mean error: {np.abs(ref - out_pt).mean():.6f}")
print(f"Per-channel mean error: {np.abs(ref - out_pc).mean():.6f}")
# Per-channel is typically 2-3× more accurate
```

### Quantizing an MLP in tinygrad

```python
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam
import numpy as np

class QuantizedMLP:
    """MLP where weights are stored as INT8, but inference simulated in FP32"""

    def __init__(self, layers):
        # Train in FP32
        self.fp32_weights = [Tensor.kaiming_uniform(layers[i], layers[i+1])
                             for i in range(len(layers)-1)]
        self.biases = [Tensor.zeros(layers[i+1])
                       for i in range(len(layers)-1)]
        self.quantized = False

    def quantize(self):
        """Quantize all weights to INT8 after training"""
        self.int8_weights = []
        self.scales = []
        for W in self.fp32_weights:
            W_q, scale = quantize_weight_per_channel(W)
            self.int8_weights.append(W_q)
            self.scales.append(scale)
        self.quantized = True
        print("Model quantized to INT8")

    def __call__(self, x):
        if not self.quantized:
            # FP32 training forward pass
            for i, (W, b) in enumerate(zip(self.fp32_weights, self.biases)):
                x = x.matmul(W) + b
                if i < len(self.fp32_weights) - 1:
                    x = x.relu()
        else:
            # Simulated INT8 inference forward pass
            for i, (W_q, scale, b) in enumerate(zip(self.int8_weights, self.scales, self.biases)):
                x = quantized_linear(x, W_q, scale, b)
                if i < len(self.int8_weights) - 1:
                    x = x.relu()
        return x.softmax()

    def parameters(self):
        return self.fp32_weights + self.biases

# Train in FP32
model = QuantizedMLP([784, 256, 128, 10])
optimizer = Adam(model.parameters(), lr=1e-3)

# ... (training loop here) ...

# After training: quantize and compare accuracy
model_q = QuantizedMLP([784, 256, 128, 10])
model_q.fp32_weights = model.fp32_weights
model_q.biases = model.biases
model_q.quantize()

# Memory comparison
fp32_bytes = sum(W.numpy().nbytes for W in model.fp32_weights)
int8_bytes  = sum(W.nbytes for W in model_q.int8_weights)
print(f"FP32 weights: {fp32_bytes/1024:.1f} KB")
print(f"INT8 weights: {int8_bytes/1024:.1f} KB")
print(f"Compression:  {fp32_bytes/int8_bytes:.1f}×")
```

---

## 4. Post-Training Quantization (PTQ) with TensorRT

### INT8 Calibration — Why It Matters

INT8 quantization needs to know the **range** of values that flow through each activation during inference. This is determined by running the model on a **calibration dataset** (100–1000 representative samples).

```
Without calibration:
  Scale = max possible value (very conservative)
  Most values underutilize the INT8 range → poor accuracy

With calibration:
  Scale = percentile of actual activation range
  INT8 range fully utilized → much better accuracy
```

### INT8 Calibration with TensorRT

```python
# int8_calibrator.py
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np

class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Entropy calibration: minimizes KL divergence between FP32 and INT8 distributions.
    Usually best for CNNs.
    """
    def __init__(self, calibration_data, cache_file='calib.cache'):
        super().__init__()
        self.data = calibration_data        # list of numpy arrays, shape [1, C, H, W]
        self.idx = 0
        self.cache_file = cache_file

        # Allocate device buffer for one batch
        self.device_input = cuda.mem_alloc(self.data[0].nbytes)

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.idx >= len(self.data):
            return None                     # signal end of calibration data

        batch = self.data[self.idx]
        cuda.memcpy_htod(self.device_input, batch)
        self.idx += 1
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)

def build_int8_engine(onnx_path, engine_path, calibration_data):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as net, \
         trt.OnnxParser(net, TRT_LOGGER) as parser, \
         builder.create_builder_config() as config:

        # 2 GB workspace
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

        # Enable INT8
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)          # FP16 fallback for INT8 unsupported layers

        # Attach calibrator
        calibrator = EntropyCalibrator(calibration_data)
        config.int8_calibrator = calibrator

        with open(onnx_path, 'rb') as f:
            parser.parse(f.read())

        engine_bytes = builder.build_serialized_network(net, config)
        with open(engine_path, 'wb') as f:
            f.write(engine_bytes)
        print(f"INT8 engine saved: {engine_path}")

# Prepare calibration data from your dataset
import cv2, os

def prepare_calibration_data(image_dir, n=500, size=(640, 640)):
    images = []
    for fname in os.listdir(image_dir)[:n]:
        img = cv2.imread(os.path.join(image_dir, fname))
        img = cv2.resize(img, size)
        img = img[:,:,::-1].transpose(2, 0, 1)          # BGR→RGB, HWC→CHW
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)                     # [1, 3, H, W]
        images.append(np.ascontiguousarray(img))
    return images

calib_data = prepare_calibration_data('/path/to/calib/images')
build_int8_engine('model.onnx', 'model_int8.engine', calib_data)
```

### Inspecting What TensorRT Quantized

```bash
# See which layers are INT8 vs FP16 vs FP32
trtexec --loadEngine=model_int8.engine \
        --verbose 2>&1 | grep -E "(INT8|FP16|FP32)" | head -40

# Build with layer timing info
trtexec --onnx=model.onnx \
        --int8 \
        --calib=calib.cache \
        --saveEngine=model_int8.engine \
        --verbose \
        --separateProfileRun \
        --avgRuns=100 2>&1 | grep "Timing"
```

### Accuracy vs Speed Comparison Script

```python
import time, numpy as np

def benchmark(engine_path, test_data, labels, n_runs=200):
    inferencer = TRTInferencer(engine_path)    # from Jetson Platform guide

    # Accuracy
    correct = 0
    for x, y in zip(test_data[:1000], labels[:1000]):
        pred = inferencer.infer(x).argmax()
        correct += (pred == y)
    accuracy = correct / 1000

    # Latency
    dummy = test_data[0]
    for _ in range(50):  # warmup
        inferencer.infer(dummy)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        inferencer.infer(dummy)
        times.append((time.perf_counter() - t0) * 1000)

    return accuracy, np.mean(times), 1000/np.mean(times)

for engine in ['model_fp32.engine', 'model_fp16.engine', 'model_int8.engine']:
    acc, lat, fps = benchmark(engine, X_test, Y_test)
    print(f"{engine:25s}  acc={acc:.3f}  lat={lat:.1f}ms  fps={fps:.1f}")

# Expected on Orin Nano 8GB (YOLOv8n example):
# model_fp32.engine     acc=0.921  lat=28.3ms  fps=35.3
# model_fp16.engine     acc=0.920  lat=12.1ms  fps=82.6
# model_int8.engine     acc=0.918  lat=7.4ms   fps=135.1
```

---

## 5. Quantization-Aware Training (QAT) with tinygrad

QAT simulates quantization noise **during training**, so the model learns to be robust to it. The result is INT8 accuracy almost matching FP32.

### The Straight-Through Estimator (STE)

The problem: `round()` has zero gradient everywhere — backprop would fail.

The solution: **straight-through estimator** — pass gradients through the rounding operation as if it were identity:

```
Forward:  q = round(x / s) * s   (quantize + dequantize)
Backward: dL/dx = dL/dq          (ignore the rounding, pass gradient through)
```

### QAT Implementation in tinygrad

```python
# qat.py
from tinygrad.tensor import Tensor
import numpy as np

class FakeQuantize:
    """
    Fake-quantize: simulates INT8 quantization in the forward pass,
    uses straight-through estimator in backward pass.
    """
    def __init__(self, num_bits=8, symmetric=True):
        self.num_bits = num_bits
        self.q_min = -(2 ** (num_bits - 1))     # -128 for INT8
        self.q_max =  (2 ** (num_bits - 1)) - 1  # 127 for INT8

        # Learnable scale (initialized to 1.0)
        self.scale = Tensor([1.0])

    def __call__(self, x: Tensor) -> Tensor:
        # Compute scale from current batch statistics
        x_max = float(x.abs().max().numpy())
        scale = max(x_max, 1e-8) / self.q_max
        self.scale = Tensor([scale])

        # Quantize: q = clip(round(x/s), q_min, q_max) * s
        x_scaled = x / scale
        # tinygrad doesn't have round() that supports backprop STE natively,
        # so we approximate: clamp then use the value
        x_clamped = x_scaled.clip(self.q_min, self.q_max)

        # Simulate quantization noise: add uniform noise ± 0.5 * scale
        # (approximates the effect of rounding for gradient purposes)
        noise = Tensor(np.random.uniform(-0.5, 0.5, x.shape).astype(np.float32))
        x_quant_sim = (x_clamped + noise) * scale

        return x_quant_sim

class QATLinear:
    """Linear layer with fake-quantized weights and activations"""
    def __init__(self, n_in, n_out):
        self.weight = Tensor.kaiming_uniform(n_in, n_out)
        self.bias   = Tensor.zeros(n_out)
        self.w_fq   = FakeQuantize(num_bits=8)
        self.a_fq   = FakeQuantize(num_bits=8)

    def __call__(self, x):
        w_q = self.w_fq(self.weight)    # fake-quantize weights
        out = x.matmul(w_q) + self.bias
        return self.a_fq(out)            # fake-quantize activations

    def parameters(self):
        return [self.weight, self.bias]

class QATMLP:
    def __init__(self, layers):
        self.layers = [QATLinear(layers[i], layers[i+1])
                       for i in range(len(layers)-1)]

    def __call__(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x).relu()
        return self.layers[-1](x).softmax()

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

# QAT Training Loop
from tinygrad.nn.optim import Adam

model = QATMLP([784, 256, 128, 10])
optimizer = Adam(model.parameters(), lr=5e-4)   # Lower LR for QAT

# Phase 1: Pretrain in FP32 (skip fake-quant initially)
# Phase 2: Enable QAT (fake-quant active) and fine-tune 3-5 epochs
# This two-phase approach gives best results

BATCH = 64
for epoch in range(5):    # QAT fine-tuning
    for i in range(0, len(X_train), BATCH):
        xb = Tensor(X_train[i:i+BATCH].astype(np.float32))
        yb_oh = np.zeros((BATCH, 10), dtype=np.float32)
        yb_oh[np.arange(BATCH), Y_train[i:i+BATCH]] = 1.0

        out = model(xb)
        loss = -(Tensor(yb_oh) * out.log()).sum(axis=1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"QAT Epoch {epoch+1}: loss={loss.numpy():.4f}")
```

---

## 6. Pruning — Structured and Unstructured

### Unstructured Pruning (Weight Magnitude)

Remove individual weights below a threshold. Easy to implement, hard to accelerate on GPU without sparse hardware.

```python
# magnitude_pruning.py
from tinygrad.tensor import Tensor
import numpy as np

def magnitude_prune(model, sparsity=0.5):
    """
    Prune the bottom `sparsity` fraction of weights by magnitude.
    Returns masks (1=keep, 0=prune).
    """
    masks = {}
    all_weights = []

    for name, W in model.named_weights():
        all_weights.append(np.abs(W.numpy()).flatten())

    # Find global threshold
    all_weights_flat = np.concatenate(all_weights)
    threshold = np.percentile(all_weights_flat, sparsity * 100)

    for name, W in model.named_weights():
        mask = (np.abs(W.numpy()) > threshold).astype(np.float32)
        masks[name] = mask
        actual_sparsity = 1.0 - mask.mean()
        print(f"  {name}: {actual_sparsity:.1%} sparse")

    return masks

def apply_masks(model, masks):
    """Zero out pruned weights"""
    for name, W in model.named_weights():
        if name in masks:
            W_pruned = W.numpy() * masks[name]
            # In-place update
            W.assign(Tensor(W_pruned))

# Usage with our MLP
masks = magnitude_prune(model, sparsity=0.7)   # prune 70% of weights
apply_masks(model, masks)

# After pruning, fine-tune for 1-2 epochs to recover accuracy
# During fine-tuning, re-apply masks after each update to keep weights zeroed
```

### Structured Pruning (Channel Pruning)

Remove entire neurons/channels. Hardware-friendly — the pruned model is actually smaller and faster.

```python
def structured_prune_layer(W: Tensor, keep_fraction=0.5):
    """
    Prune output neurons with smallest L1 norm of their weight vector.
    Returns pruned weight matrix.

    W shape: [n_in, n_out]
    Prunes output neurons (columns).
    """
    W_np = W.numpy()
    n_out = W_np.shape[1]
    n_keep = max(1, int(n_out * keep_fraction))

    # L1 norm of each output neuron's weights
    norms = np.sum(np.abs(W_np), axis=0)      # shape [n_out]

    # Keep top n_keep neurons
    keep_idx = np.argsort(norms)[-n_keep:]
    keep_idx = np.sort(keep_idx)

    return Tensor(W_np[:, keep_idx]), keep_idx

# Example: prune hidden layer
W1 = Tensor.randn(784, 256)   # first layer
W1_pruned, keep_idx = structured_prune_layer(W1, keep_fraction=0.5)
print(f"Layer 1: {W1.shape} → {W1_pruned.shape}")
# Layer 1: (784, 256) → (784, 128)

# Next layer input must match pruned output
W2 = Tensor.randn(256, 128)
W2_pruned = Tensor(W2.numpy()[keep_idx, :])   # keep matching rows
print(f"Layer 2: {W2.shape} → {W2_pruned.shape}")
# Layer 2: (256, 128) → (128, 128)

# Model is now physically smaller — real speedup
```

### L1 Regularization to Encourage Sparsity

Add L1 penalty during training to push weights toward zero, making them easier to prune later:

```python
# During training with L1 regularization
l1_lambda = 1e-4

for i in range(0, len(X_train), BATCH):
    xb = Tensor(X_train[i:i+BATCH].astype(np.float32))
    out = model(xb)
    task_loss = ...  # cross-entropy

    # L1 penalty: sum of absolute values of all weights
    l1_loss = sum(W.abs().sum() for W in model.parameters())
    total_loss = task_loss + l1_lambda * l1_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

---

## 7. Knowledge Distillation

Train a small **student** model to mimic a large **teacher** model. The student learns from soft probability outputs (which carry more information than hard labels).

```
Teacher (large, slow):      ResNet-50, 25M params, 80ms on Orin Nano
                 ↓ soft predictions (probabilities)
Student (small, fast):      MobileNet, 3M params, 8ms on Orin Nano
                 ↓ learns teacher's "knowledge"
Student accuracy ≈ Teacher accuracy   (within 1-2%)
Student speed    = 10×  faster
```

### Distillation Loss

```
L_distill = α × L_CE(student_pred, hard_labels)
           + (1-α) × L_KD(student_soft, teacher_soft, temperature T)

L_KD = KL divergence between teacher and student softmax outputs at temperature T

Temperature T:
  T=1: normal softmax (sharp)
  T>1: softer distribution (reveals more inter-class relationships)
  Typical: T=3 or T=4
```

### Knowledge Distillation in tinygrad

```python
# distillation.py
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam
import numpy as np

def softmax_with_temperature(logits: Tensor, T: float) -> Tensor:
    return (logits / T).softmax()

def kl_divergence(p: Tensor, q: Tensor, eps=1e-8) -> Tensor:
    """KL(p || q) = sum(p * log(p/q))"""
    return (p * (p + eps).log() - p * (q + eps).log()).sum(axis=1).mean()

def distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
    T: float = 4.0,
    alpha: float = 0.7
):
    """
    Combined distillation loss.
    alpha: weight of distillation loss (1-alpha = weight of cross-entropy)
    T: temperature for soft targets
    """
    # Soft targets (teacher knowledge)
    teacher_soft = softmax_with_temperature(teacher_logits, T)
    student_soft = softmax_with_temperature(student_logits, T)
    loss_kd = kl_divergence(teacher_soft, student_soft) * (T ** 2)

    # Hard targets (ground truth)
    student_prob = student_logits.softmax()
    loss_ce = -(labels * student_prob.log()).sum(axis=1).mean()

    return alpha * loss_kd + (1 - alpha) * loss_ce

# Teacher: large pretrained model (fixed, no gradient)
teacher = LargeMLP([784, 1024, 1024, 512, 10])
# ... load pretrained weights ...

# Student: small model (trained from scratch with distillation)
student = SmallMLP([784, 128, 64, 10])
optimizer = Adam(student.parameters(), lr=1e-3)

for epoch in range(20):
    for i in range(0, len(X_train), BATCH):
        xb = Tensor(X_train[i:i+BATCH].astype(np.float32))
        yb_oh = Tensor(one_hot(Y_train[i:i+BATCH], 10))

        # Teacher forward (no gradients needed)
        with Tensor.no_grad():
            teacher_logits = teacher(xb)

        # Student forward
        student_logits = student(xb)

        # Distillation loss
        loss = distillation_loss(
            student_logits, teacher_logits, yb_oh, T=4.0, alpha=0.7
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}: loss={loss.numpy():.4f}")
```

---

## 8. Full Model Porting Pipeline

### The Complete Journey

```
tinygrad model  ──→  PyTorch weights  ──→  ONNX  ──→  TensorRT  ──→  Jetson
   (training)         (export)         (universal)   (optimized)   (inference)
```

### Step 1: Train in tinygrad, Export Weights

```python
# export_weights.py
import numpy as np

def export_model_weights(model, path='model_weights.npz'):
    """Export all model weights as numpy arrays"""
    weights = {}
    for i, layer in enumerate(model.linears):
        weights[f'layer_{i}_W'] = layer.w.numpy()
        weights[f'layer_{i}_b'] = layer.b.numpy()

    np.savez(path, **weights)
    print(f"Saved {len(weights)} weight tensors to {path}")
    for k, v in weights.items():
        print(f"  {k}: {v.shape} {v.dtype}")
```

### Step 2: Load tinygrad Weights into PyTorch for ONNX Export

```python
# tinygrad_to_onnx.py
import torch
import torch.nn as nn
import numpy as np

class TorchMLP(nn.Module):
    """Identical architecture to tinygrad MLP"""
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(layers[i], layers[i+1])
            for i in range(len(layers)-1)
        ])

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.relu(layer(x))
        return self.layers[-1](x)

def load_tinygrad_weights(torch_model, npz_path):
    """Load tinygrad weights into PyTorch model"""
    weights = np.load(npz_path)

    for i, layer in enumerate(torch_model.layers):
        W = weights[f'layer_{i}_W']
        b = weights[f'layer_{i}_b']

        # tinygrad: weight shape [n_in, n_out]
        # PyTorch:  weight shape [n_out, n_in] (transposed!)
        layer.weight.data = torch.from_numpy(W.T)
        layer.bias.data   = torch.from_numpy(b)

    print("Weights loaded from tinygrad export")

# Export to ONNX
model_pt = TorchMLP([784, 256, 128, 10])
load_tinygrad_weights(model_pt, 'model_weights.npz')
model_pt.eval()

dummy_input = torch.randn(1, 784)
torch.onnx.export(
    model_pt,
    dummy_input,
    'model.onnx',
    opset_version=17,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
)

# Verify
import onnx, onnxruntime as ort
onnx.checker.check_model(onnx.load('model.onnx'))
sess = ort.InferenceSession('model.onnx')
out_onnx = sess.run(None, {'input': dummy_input.numpy()})[0]
out_torch = model_pt(dummy_input).detach().numpy()
print(f"ONNX vs PyTorch max diff: {np.abs(out_onnx - out_torch).max():.8f}")
```

### Step 3: Validate ONNX on Jetson Before TensorRT

```bash
# Install onnxruntime for Jetson (GPU version)
pip3 install onnxruntime-gpu    # NVIDIA provides builds for Jetson
# Or install from NVIDIA wheel:
# pip3 install onnxruntime_gpu-*.whl

# Quick validation
python3 -c "
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession('model.onnx', providers=['CUDAExecutionProvider'])
x = np.random.randn(1, 784).astype(np.float32)
out = sess.run(None, {'input': x})
print('ONNX Runtime OK, output shape:', out[0].shape)
"
```

### Step 4: Convert ONNX → TensorRT on Jetson

```bash
# Always build TensorRT engines on the target Jetson (hardware-specific)

# FP16 (best balance for most models)
trtexec --onnx=model.onnx \
        --saveEngine=model_fp16.engine \
        --fp16 \
        --minShapes=input:1x784 \
        --optShapes=input:64x784 \
        --maxShapes=input:256x784 \
        --verbose 2>&1 | tee build_fp16.log

# INT8 with calibration cache
trtexec --onnx=model.onnx \
        --saveEngine=model_int8.engine \
        --int8 \
        --calib=calib.cache \
        --fp16 \
        --verbose 2>&1 | tee build_int8.log

# Check engine info after build
trtexec --loadEngine=model_fp16.engine \
        --dumpLayerInfo \
        --dumpProfile 2>&1 | head -80
```

### Step 5: Cross-Check Accuracy at Each Stage

```python
# validate_pipeline.py
import numpy as np

# Load test data
X_test = np.load('x_test.npy').astype(np.float32)
Y_test = np.load('y_test.npy')

def accuracy(preds, labels):
    return (preds.argmax(axis=1) == labels).mean()

# Stage 1: tinygrad
tg_preds = model(Tensor(X_test)).numpy()
print(f"tinygrad FP32: {accuracy(tg_preds, Y_test):.3%}")

# Stage 2: ONNX Runtime
ort_preds = np.vstack([
    sess.run(None, {'input': X_test[i:i+64]})[0]
    for i in range(0, len(X_test), 64)
])
print(f"ONNX Runtime:  {accuracy(ort_preds, Y_test):.3%}")

# Stage 3: TensorRT FP16
fp16_preds = run_trt_batch('model_fp16.engine', X_test)
print(f"TRT FP16:      {accuracy(fp16_preds, Y_test):.3%}")

# Stage 4: TensorRT INT8
int8_preds = run_trt_batch('model_int8.engine', X_test)
print(f"TRT INT8:      {accuracy(int8_preds, Y_test):.3%}")

# Acceptable accuracy drop:
# ONNX vs tinygrad: < 0.01% (should be near-zero)
# FP16 vs FP32:     < 0.1%
# INT8 vs FP32:     < 1.0%
```

---

## 9. TensorRT Engine Optimization — Jetson Deep Dive

### Layer Fusion

TensorRT automatically fuses adjacent layers to reduce memory bandwidth and kernel launch overhead.

```
Before fusion (3 kernel launches):
  CONV → BN → ReLU

After fusion (1 kernel launch):
  CBR (Conv-BN-ReLU fused)

Result: removes 2 round-trips to GPU memory per fused block
On Orin Nano: ~15-25% latency reduction for CNN workloads
```

```bash
# See what TensorRT fused:
trtexec --onnx=model.onnx --fp16 \
        --dumpLayerInfo 2>&1 | grep -A1 "Fused"
```

### Timing Cache for Faster Engine Builds

TensorRT profiles many kernel variants during engine build. The timing cache saves these results so subsequent builds (e.g., same model, different batch size) are much faster:

```python
config.set_flag(trt.BuilderFlag.ENABLE_TACTIC_SOURCES)

# Save timing cache
timing_cache_file = 'timing.cache'
if os.path.exists(timing_cache_file):
    with open(timing_cache_file, 'rb') as f:
        timing_cache = config.create_timing_cache(f.read())
else:
    timing_cache = config.create_timing_cache(b'')

config.set_timing_cache(timing_cache, ignore_mismatch=False)

# Build engine...
engine_bytes = builder.build_serialized_network(net, config)

# Save updated timing cache
with open(timing_cache_file, 'wb') as f:
    f.write(config.get_timing_cache().serialize())
```

### Strongly Typed Mode (TensorRT 10+)

Gives you explicit control over tensor types instead of letting TensorRT choose:

```python
# Enable strongly typed mode
config.set_flag(trt.BuilderFlag.STRONGLY_TYPED)

# Now you must specify types for inputs and outputs explicitly
# Prevents unexpected precision downgrades in sensitive layers
```

### Dynamic Shapes for Variable Batch Inference

```python
# Build with dynamic shapes for production flexibility
profile = builder.create_optimization_profile()

profile.set_shape(
    'images',
    min=(1, 3, 640, 640),     # minimum batch size
    opt=(4, 3, 640, 640),     # most common (profile optimized for this)
    max=(16, 3, 640, 640)     # maximum batch size
)
config.add_optimization_profile(profile)

# At inference, set input shape dynamically
context.set_input_shape('images', (batch_size, 3, 640, 640))
```

### CUDA Graphs with TensorRT (Critical for Low Latency)

Without CUDA Graphs, each `execute_async_v2()` call has ~20–50µs CPU overhead for kernel launch. With CUDA Graphs, it's ~2µs:

```python
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np

class TRTWithCUDAGraph:
    def __init__(self, engine_path):
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self._allocate()

        # Capture CUDA Graph
        self._capture_graph()

    def _allocate(self):
        self.stream = cuda.Stream()
        self.inputs, self.outputs, self.bindings = [], [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.context.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            host = cuda.pagelocked_empty(trt.volume(shape), dtype)
            device = cuda.mem_alloc(host.nbytes)
            self.bindings.append(int(device))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({'host': host, 'device': device, 'name': name})
            else:
                self.outputs.append({'host': host, 'device': device, 'name': name})

    def _capture_graph(self):
        # Warmup (required before graph capture)
        for _ in range(3):
            self._execute()

        # Capture
        self.graph = cuda.Graph()
        stream_for_capture = cuda.Stream()
        cuda.start_graph_capture(stream_for_capture)
        self._execute(stream=stream_for_capture)
        self.graph_exec = cuda.end_graph_capture_and_instantiate(stream_for_capture)
        print("CUDA Graph captured")

    def _execute(self, stream=None):
        s = stream or self.stream
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], s)
        self.context.execute_async_v2(self.bindings, s.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], s)
        s.synchronize()

    def infer(self, data):
        np.copyto(self.inputs[0]['host'], data.ravel())
        # Replay graph — very low overhead
        self.graph_exec.launch(self.stream)
        self.stream.synchronize()
        return self.outputs[0]['host'].copy()
```

---

## 10. DLA (Deep Learning Accelerator) on Orin Nano

### What is DLA?

The DLA is a fixed-function neural network accelerator built into the Orin SoC. It runs independently from the GPU.

```
GPU:           1024 CUDA Ampere cores, 40 TOPS (shared with all workloads)
DLA:           Fixed-function INT8/FP16 engine, ~10 TOPS dedicated

Running on DLA:
  - Frees GPU for other tasks (sensor processing, computer vision)
  - Lower power than GPU for supported ops
  - Can run DLA + GPU simultaneously
```

### Which Layers Can Run on DLA

```
Supported:      Conv2d, Pooling, BatchNorm, ReLU, Sigmoid
                DepthwiseConv, FullyConnected (limited), Softmax

NOT Supported:  Custom plugins, dynamic shapes, many attention ops
                Operations with large memory footprint

Reality check: Most CNNs (ResNet, MobileNet, YOLO backbone) run well on DLA.
               Transformers do NOT run on DLA (attention is not supported).
```

### Build TensorRT Engine for DLA

```python
# build_dla_engine.py
import tensorrt as trt

def build_dla_engine(onnx_path, engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as net, \
         trt.OnnxParser(net, TRT_LOGGER) as parser, \
         builder.create_builder_config() as config:

        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        # DLA configuration
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = 0                          # Orin Nano has 1 DLA (core 0)
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK) # GPU handles unsupported layers
        config.set_flag(trt.BuilderFlag.FP16)         # DLA requires FP16 or INT8

        with open(onnx_path, 'rb') as f:
            parser.parse(f.read())

        engine_bytes = builder.build_serialized_network(net, config)
        with open(engine_path, 'wb') as f:
            f.write(engine_bytes)

# Alternatively with trtexec:
# trtexec --onnx=model.onnx \
#         --saveEngine=model_dla.engine \
#         --useDLACore=0 \
#         --fp16 \
#         --allowGPUFallback \
#         --verbose 2>&1 | grep "DLA"
```

### DLA + GPU Concurrent Execution

```python
import threading

# Run DLA inference and GPU inference on different models simultaneously
# DLA handles the backbone, GPU handles the detection head

dla_engine   = load_engine('backbone_dla.engine')
gpu_engine   = load_engine('detection_head_gpu.engine')

dla_context  = dla_engine.create_execution_context()
gpu_context  = gpu_engine.create_execution_context()

dla_stream   = cuda.Stream()
gpu_stream   = cuda.Stream()

def run_dla(input_data):
    # Backbone on DLA
    cuda.memcpy_htod_async(dla_input_buf, input_data, dla_stream)
    dla_context.execute_async_v2(dla_bindings, dla_stream.handle)
    cuda.memcpy_dtoh_async(features_cpu, dla_output_buf, dla_stream)
    dla_stream.synchronize()
    return features_cpu

def run_gpu(features):
    # Detection head on GPU
    cuda.memcpy_htod_async(gpu_input_buf, features, gpu_stream)
    gpu_context.execute_async_v2(gpu_bindings, gpu_stream.handle)
    cuda.memcpy_dtoh_async(detections_cpu, gpu_output_buf, gpu_stream)
    gpu_stream.synchronize()
    return detections_cpu

# In production: overlap DLA and GPU work
# While DLA processes frame N, GPU processes frame N-1's features
```

### DLA Benchmark Comparison

```bash
# Compare GPU vs DLA on Orin Nano
# (run while monitoring tegrastats for power)

# GPU only
trtexec --loadEngine=model_fp16_gpu.engine \
        --iterations=100 --avgRuns=100

# DLA only
trtexec --loadEngine=model_dla.engine \
        --iterations=100 --avgRuns=100

# Typical results for MobileNetV2 on Orin Nano:
#   GPU FP16:  8ms, ~300mW GPU power
#   DLA FP16: 12ms, ~100mW DLA power (saves GPU for other tasks)
#   Use DLA when: power matters, GPU needed elsewhere, supported ops only
#   Use GPU when: lowest latency needed, unsupported ops exist
```

---

## 11. tinygrad on Jetson — CUDA Backend

### Running tinygrad with CUDA on Jetson

tinygrad supports CUDA natively. On Jetson, this uses the Ampere GPU.

```bash
# Verify CUDA is available
python3 -c "from tinygrad.runtime.ops_cuda import CUDADevice; print('CUDA OK')"

# Set tinygrad to use CUDA backend
export CUDA=1    # or GPU=1
```

```python
# tinygrad_cuda_jetson.py
import os
os.environ['CUDA'] = '1'    # use CUDA backend

from tinygrad.tensor import Tensor
import numpy as np
import time

# All tensors default to CUDA device
x = Tensor.randn(64, 784)     # lives on Jetson GPU
W = Tensor.randn(784, 256)

# Operations execute on Ampere GPU
out = x.matmul(W).relu()

# Force computation and bring back to CPU
result = out.numpy()
print(f"Output shape: {result.shape}")

# Benchmark tinygrad CUDA vs CPU on Jetson
def benchmark_backend(backend_env, shape=(64, 784)):
    os.environ.clear()
    if backend_env:
        os.environ['CUDA'] = '1'

    from tinygrad.tensor import Tensor
    import importlib
    import tinygrad.tensor
    importlib.reload(tinygrad.tensor)

    x = Tensor.randn(*shape)
    W = Tensor.randn(shape[1], 256)

    # Warmup
    for _ in range(10):
        (x.matmul(W).relu()).numpy()

    t0 = time.perf_counter()
    for _ in range(100):
        (x.matmul(W).relu()).numpy()
    elapsed = (time.perf_counter() - t0) * 1000 / 100

    return elapsed

# Note: set backend before importing tinygrad, not mid-session
```

### Training a CNN in tinygrad on Jetson GPU

```python
# cnn_jetson.py
import os
os.environ['CUDA'] = '1'

from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d, BatchNorm2d
from tinygrad.nn.optim import Adam
import numpy as np

class ConvBlock:
    def __init__(self, in_ch, out_ch, stride=1):
        self.conv = Conv2d(in_ch, out_ch, 3, padding=1, stride=stride, bias=False)
        self.bn   = BatchNorm2d(out_ch)

    def __call__(self, x):
        return self.bn(self.conv(x)).relu()

    def parameters(self):
        return self.conv.weight, *self.bn.weight, *self.bn.bias

class TinyResNet:
    """Lightweight ResNet-style CNN for CIFAR-10"""
    def __init__(self, num_classes=10):
        self.c1 = ConvBlock(3, 32)
        self.c2 = ConvBlock(32, 64, stride=2)
        self.c3 = ConvBlock(64, 128, stride=2)
        self.c4 = ConvBlock(128, 256, stride=2)
        # After 3 stride-2 convolutions: 32×32 → 4×4
        self.fc = lambda x: x.reshape(x.shape[0], -1).linear(
            Tensor.kaiming_uniform(256 * 4 * 4, num_classes)
        )

    def __call__(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = x.mean(axis=(2, 3))    # global average pooling
        return x.softmax()

model = TinyResNet()
optimizer = Adam([p for block in [model.c1, model.c2, model.c3, model.c4]
                  for p in block.parameters()], lr=1e-3)

# Monitor Jetson GPU during training
# In another terminal: sudo tegrastats --interval 100
# You should see GR3D_FREQ increase to 60-100% during training
```

### Exporting tinygrad Model to ONNX for TensorRT

For production inference, train in tinygrad, export to ONNX, convert to TensorRT:

```python
# The general approach: save weights, load into PyTorch, export ONNX
# (same as Section 8 above)

# Alternatively: use tinygrad's built-in ONNX support
# tinygrad can run ONNX models natively:

from tinygrad.runtime.ops_cuda import CUDADevice
import tinygrad.frontend.onnx as onnx_runner
import onnx

model_onnx = onnx.load('model.onnx')
run_onnx = onnx_runner.get_run_onnx(model_onnx)

# Run inference with tinygrad CUDA backend
x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
output = run_onnx({'input': x})
print(output['output'].numpy())
```

---

## 12. Profiling and Benchmarking on Jetson

### tegrastats: The Essential Tool

```bash
# Always run tegrastats while developing — know your baseline
sudo tegrastats --interval 100

# Output fields:
# RAM 3045/7772MB          : used / total unified memory
# CPU [35%@1510, ...]      : % utilization @ MHz per core
# EMC_FREQ 38%             : External Memory Controller bandwidth utilization
# GR3D_FREQ 89%            : GPU utilization %
# CPU@42C GPU@44C tj@44C   : temperatures
# VDD_IN 6234mW            : total system power draw
# VDD_CPU_GPU_CV 2901mW    : CPU+GPU+CV power
# VDD_SOC 1158mW           : SoC power

# Log to file for analysis
sudo tegrastats --interval 100 | tee run_$(date +%s).log &
TEGRA_PID=$!

# Run your workload
python3 my_inference.py

# Stop logging
kill $TEGRA_PID

# Analyze: plot GPU%, temperature, power over time
python3 analyze_tegrastats.py run_*.log
```

```python
# analyze_tegrastats.py
import re
import matplotlib.pyplot as plt

def parse_tegrastats(log_file):
    gpu_util, gpu_temp, power = [], [], []

    with open(log_file) as f:
        for line in f:
            m_gpu  = re.search(r'GR3D_FREQ (\d+)%', line)
            m_temp = re.search(r'GPU@(\d+\.?\d*)C', line)
            m_pwr  = re.search(r'VDD_IN (\d+)mW', line)

            if m_gpu:  gpu_util.append(int(m_gpu.group(1)))
            if m_temp: gpu_temp.append(float(m_temp.group(1)))
            if m_pwr:  power.append(int(m_pwr.group(1)) / 1000)  # W

    return gpu_util, gpu_temp, power

gpu_util, gpu_temp, power = parse_tegrastats('run_log.log')

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
axes[0].plot(gpu_util);  axes[0].set_ylabel('GPU Util %')
axes[1].plot(gpu_temp);  axes[1].set_ylabel('GPU Temp °C')
axes[2].plot(power);     axes[2].set_ylabel('Total Power W')
plt.savefig('profile.png')
```

### trtexec: TensorRT Benchmarking

```bash
# Full benchmark report
trtexec --loadEngine=model_fp16.engine \
        --warmUp=500 \
        --iterations=1000 \
        --avgRuns=100 \
        --percentile=99 \
        --separateProfileRun 2>&1 | tail -30

# Output:
# [I] Latency: min = 11.2ms, max = 13.1ms, mean = 11.8ms
# [I] GPU Compute Time: min = 10.8ms, max = 12.5ms, mean = 11.3ms
# [I] H2D Latency: min = 0.15ms, max = 0.22ms
# [I] D2H Latency: min = 0.08ms, max = 0.12ms
# [I] Throughput: 84.7 qps

# Profile per-layer timing
trtexec --loadEngine=model_fp16.engine \
        --dumpProfile \
        --iterations=100 2>&1 | grep "Layer Time" | sort -t= -k2 -rn | head -20
```

### Nsight Systems: System-Level Profiling

```bash
# Install on Jetson
sudo apt-get install nsight-systems

# Profile your inference script
nsys profile \
    --trace=cuda,cudnn,tensorrt \
    --output=inference_profile \
    python3 inference.py

# View on Jetson display or copy to desktop
nsys-ui inference_profile.qdrep

# CLI report (no GUI needed)
nsys stats inference_profile.qdrep
```

### Memory Profiling (Critical on 8GB Unified Memory)

```bash
# Check GPU memory usage during inference
nvidia-smi -l 1    # if available on Orin
# or:
cat /sys/kernel/debug/nvmap/clients     # Jetson-specific

# In Python: track allocation
from tinygrad.tensor import Tensor

def get_gpu_mem_mb():
    """Read current GPU memory from Jetson sysfs"""
    try:
        with open('/sys/devices/gpu.0/mem_info_vram_used') as f:
            return int(f.read()) / (1024 * 1024)
    except:
        return None

before = get_gpu_mem_mb()
engine = load_trt_engine('model.engine')
after  = get_gpu_mem_mb()
print(f"Engine memory: {after - before:.1f} MB")
```

### Power Efficiency Metric: TOPS/W

```python
# Compute TOPS/W for your model on Jetson

# Step 1: Count operations (MACs) in your model
# For a linear layer [n_in, n_out]: n_in * n_out MACs
# For a conv layer: out_h * out_w * k_h * k_w * in_ch * out_ch MACs

def count_mlp_macs(layers):
    total = 0
    for i in range(len(layers)-1):
        total += layers[i] * layers[i+1]
    return total

# MLP [784→256→128→10]
macs = count_mlp_macs([784, 256, 128, 10])
tops_per_infer = macs * 2 / 1e12    # × 2: multiply + add

# Step 2: Measure FPS and power from tegrastats
fps = 135                  # from benchmark
power_w = 6.2              # from tegrastats VDD_IN

# TOPS/W
tops_total = tops_per_infer * fps
efficiency = tops_total / power_w
print(f"Efficiency: {efficiency:.4f} TOPS/W")
```

---

## 13. DeepStream for Video Pipelines

DeepStream is NVIDIA's GStreamer-based framework for building optimized multi-stream video AI pipelines. It is specifically designed for Jetson.

### When to Use DeepStream vs Raw TensorRT

```
Use DeepStream when:
  ✓ Processing live video streams (RTSP, CSI camera, USB camera)
  ✓ Multiple concurrent streams (4× cameras)
  ✓ Need tracking, re-identification, analytics
  ✓ Building production video AI systems

Use raw TensorRT when:
  ✓ Single-frame inference (non-video)
  ✓ Custom pipeline logic
  ✓ Robotic sensor fusion (LiDAR + camera)
  ✓ Need maximum control
```

### DeepStream Pipeline for Object Detection

```python
# deepstream_detect.py
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds   # DeepStream Python bindings

Gst.init(None)

pipeline = Gst.parse_launch("""
    nvarguscamerasrc sensor-id=0 !
    video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 !
    nvvideoconvert !
    video/x-raw(memory:NVMM),format=NV12 !
    m.sink_0 nvstreammux name=m batch-size=1 width=1280 height=720 !
    nvinfer config-file-path=yolo_config.txt !
    nvtracker tracker-width=640 tracker-height=360
              ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so !
    nvdsosd !
    nvvideoconvert !
    video/x-raw,format=RGBA !
    nveglglessink
""")

def on_detection(pad, info):
    """Process detections from nvinfer"""
    gst_buffer = info.get_buffer()
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    for frame_meta in pyds.NvDsFrameMetaList(batch_meta.frame_meta_list):
        for obj_meta in pyds.NvDsObjectMetaList(frame_meta.obj_meta_list):
            box = obj_meta.rect_params
            label = obj_meta.obj_label
            conf = obj_meta.confidence
            print(f"  {label}: conf={conf:.2f} box=[{box.left:.0f},{box.top:.0f},{box.width:.0f},{box.height:.0f}]")

pipeline.set_state(Gst.State.PLAYING)
GLib.MainLoop().run()
```

### DeepStream nvinfer Config (yolo_config.txt)

```ini
[property]
gpu-id=0
net-scale-factor=0.00392156     ; 1/255.0
model-engine-file=yolo_fp16.engine
batch-size=1
network-mode=2                  ; 0=FP32, 1=INT8, 2=FP16
num-detected-classes=80
gie-unique-id=1
output-blob-names=output

[class-attrs-all]
threshold=0.4
```

---

## 14. Projects

### Project 1: Quantize MNIST MLP in tinygrad
Implement symmetric INT8 quantization from scratch in pure tinygrad. Compare per-tensor vs per-channel accuracy. Plot the accuracy-compression curve for 4-bit through 16-bit.

**Goal:** understand exactly what TensorRT does internally when you pass `--int8`.

### Project 2: Full Porting Pipeline
Train a CNN on CIFAR-10 in tinygrad on Jetson GPU. Export → ONNX → TensorRT INT8. Verify accuracy at each stage. Must achieve <1% accuracy drop from tinygrad FP32 to TensorRT INT8.

### Project 3: Structured Pruning + Distillation
- Start with ResNet-18 (11M params, 82% CIFAR-10 accuracy)
- Prune 50% of channels → 3M params
- Distill from unpruned teacher → recover to 80%+ accuracy
- Measure FPS improvement on Orin Nano before/after

### Project 4: DLA vs GPU Benchmark
Take a MobileNetV2 model. Build three engines: GPU FP16, GPU INT8, DLA FP16. For each, measure:
- Inference latency (trtexec)
- Power draw (tegrastats VDD_IN)
- Compute TOPS/W efficiency
Determine the best operating point.

### Project 5: CUDA Graph Inference Node
Wrap TensorRT inference in a CUDA Graph. Measure launch overhead with and without graphs using Nsight Systems. Integrate into a ROS2 node and show latency improvement.

### Project 6: Mixed Precision Search
Some layers are sensitive to INT8. Write a script that:
1. Starts with full INT8
2. Progressively promotes layers to FP16 (one at a time)
3. Re-measures accuracy after each promotion
4. Stops when accuracy target is met

This is a simplified version of what tools like NVIDIA's AMO (Automatic Mixed Precision Optimizer) do.

---

## 15. Resources

### tinygrad
- **tinygrad source** — `tinygrad/tensor.py`: how ops become CUDA kernels
- **tinygrad examples**: `examples/mnist.py`, `examples/efficientnet.py`
- **tinygrad ONNX frontend**: `tinygrad/frontend/onnx.py`

### Quantization Theory
- **Nagel et al. "A White Paper on Neural Network Quantization"** (2021, Qualcomm): the definitive reference on post-training quantization methods
- **Krishnamoorthi "Quantizing Deep Convolutional Networks for Efficient Inference"** (Google): explains symmetric vs asymmetric, per-layer vs per-channel

### TensorRT
- **TensorRT Developer Guide** — docs.nvidia.com/deeplearning/tensorrt/developer-guide/
- **TensorRT OSS** — github.com/NVIDIA/TensorRT: plugin examples and INT8 calibration samples
- **trtexec** source in TensorRT OSS: shows exactly how benchmark tool works

### Jetson-Specific
- **Jetson Benchmarks** — developer.nvidia.com/embedded/jetson-benchmarks: official TOPS numbers per mode
- **Deep Learning Inference Benchmarking with TensorRT on Jetson** — NVIDIA Jetson AI Lab
- **NVIDIA NGC Containers** — ngc.nvidia.com: pre-optimized containers for Jetson (JetPack-aware, pre-built TRT engines)

### Papers
- **"EfficientNet: Rethinking Model Scaling"** — NAS + compound scaling for efficient CNNs
- **"MobileNetV2: Inverted Residuals and Linear Bottlenecks"** — depthwise separable convolution for edge
- **"Distilling the Knowledge in a Neural Network"** (Hinton et al.) — original distillation paper
- **"The Lottery Ticket Hypothesis"** — understanding pruning from a training perspective

---

## Deep Dive: Subfolders

### [NVIDIA TAO Toolkit](tao-toolkit/Guide.md)

The manual pipeline (tinygrad → ONNX → TensorRT) gives you full control. TAO Toolkit takes the other approach: **start from a pre-trained model from NGC, configure via YAML, and get a production INT8 TensorRT engine without writing training code**.

- **Transfer learning** from NGC pretrained models (ResNet, EfficientNet, YOLO, SSD) with 500–5,000 images instead of millions
- **One-command pruning**: `tao model yolo_v4 prune -prune_ratio 0.6` removes 60% of channels
- **QAT built-in**: `enable_qat: True` in the spec file — fake quantization inserted automatically
- **Direct TensorRT export**: `tao model yolo_v4 export` → calibrated `.engine` file
- **DeepStream drop-in**: generated model works as `nvinfer` primary GIE with a config file

```
TAO vs Manual tradeoff:
  TAO:    hours to working INT8 engine, limited to NGC architectures
  Manual: days to implement, unlimited architectural freedom
```

Use TAO for standard detection/classification on Jetson. Use the manual pipeline for custom architectures (BEVFusion, PointPillars, custom transformers).

---

*Previous: [2. AI Fundamentals](../2. AI Fundamentals - Neural Networks and Edge AI/Guide.md)*
*Next: [4. Sensor Fusion](../4. Sensor Fusion/Guide.md)*
