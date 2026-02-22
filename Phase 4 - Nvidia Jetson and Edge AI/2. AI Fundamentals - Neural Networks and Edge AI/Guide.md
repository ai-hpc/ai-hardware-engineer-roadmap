# AI Fundamentals: Neural Networks and Edge AI

> **Goal:** Build a 100% concrete, ground-up understanding of what AI is, what artificial neural networks are, how they learn, and how to implement everything hands-on using **tinygrad** — the minimal ML framework that exposes every fundamental operation.

---

## Table of Contents

1. [What is Edge AI?](#1-what-is-edge-ai)
2. [What is Artificial Intelligence?](#2-what-is-artificial-intelligence)
3. [What is a Neural Network?](#3-what-is-a-neural-network)
4. [The Neuron: Building Block](#4-the-neuron-building-block)
5. [Activation Functions](#5-activation-functions)
6. [The Multi-Layer Perceptron (MLP)](#6-the-multi-layer-perceptron-mlp)
7. [Forward Pass: Making a Prediction](#7-forward-pass-making-a-prediction)
8. [Loss Functions: Measuring Error](#8-loss-functions-measuring-error)
9. [Backpropagation: How Networks Learn](#9-backpropagation-how-networks-learn)
10. [Gradient Descent and Optimizers](#10-gradient-descent-and-optimizers)
11. [The Training Loop](#11-the-training-loop)
12. [Convolutional Neural Networks (CNNs)](#12-convolutional-neural-networks-cnns)
13. [Regularization and Generalization](#13-regularization-and-generalization)
14. [Hands-On with tinygrad](#14-hands-on-with-tinygrad)
15. [Projects](#15-projects)
16. [Resources](#16-resources)

### Deep Dive Subfolders

| Subfolder | Description |
|-----------|-------------|
| [pytorch-and-micrograd/](pytorch-and-micrograd/Guide.md) | Build autograd from scratch with micrograd, master PyTorch, then bridge to tinygrad — the essential prerequisite for deep tinygrad mastery |

---

## 1. What is Edge AI?

### Definition

**Edge AI** = running AI algorithms **locally on a device** (the "edge") instead of sending data to a remote cloud server.

```
Traditional Cloud AI:
  Device → [internet] → Cloud Server (GPU farm) → [internet] → Result
  Latency: 50–300ms   Privacy risk   Needs connectivity

Edge AI:
  Device → Local Chip (CPU/GPU/NPU/FPGA) → Result
  Latency: <1ms       Data stays local   Works offline
```

### Why Edge AI Exists

| Problem with Cloud AI        | Edge AI Solution                        |
|------------------------------|-----------------------------------------|
| Network latency (~100ms)     | Sub-millisecond local inference         |
| Bandwidth cost (video data)  | Only send results, not raw data         |
| Privacy (face/voice/medical) | Data never leaves the device            |
| Reliability (no internet)    | Works fully offline                     |
| Cloud cost at scale          | One-time hardware cost                  |

### Where Edge AI Runs

```
Tier 1 — Microcontrollers (MCU):
  STM32, Arduino, RP2040
  RAM: 256KB–512KB
  Power: <1W
  Use: keyword spotting, gesture detection

Tier 2 — Embedded Linux SBCs:
  Raspberry Pi, BeagleBone
  RAM: 1–8GB
  Power: 2–10W
  Use: image classification, object detection

Tier 3 — AI Accelerator SoCs:
  Nvidia Jetson, Google Coral (TPU), Apple Neural Engine
  RAM: 4–64GB
  Power: 5–30W
  Use: real-time video inference, NLP, robotics

Tier 4 — Edge Servers:
  FPGA + GPU combinations, industrial PCs
  Power: 50–300W
  Use: factory automation, autonomous vehicles
```

### The Edge AI Pipeline

```
1. Train model on a powerful workstation/cloud (large data, many epochs)
2. Optimize model for edge (quantization, pruning, distillation)
3. Convert model to edge runtime format (ONNX, TensorRT, TFLite)
4. Deploy to edge device
5. Run inference locally in real-time
```

Edge AI is the **destination** of everything in this roadmap. Understanding the AI fundamentals below is what makes edge deployment possible.

---

## 2. What is Artificial Intelligence?

### The Core Idea

Classical programming:
```
Rules + Data → Program → Output
```

Machine learning (the dominant AI approach today):
```
Data + Output (labels) → Learning Algorithm → Rules (model weights)
```

You don't write the rules. The algorithm **learns** the rules from examples.

### Types of Machine Learning

```
Supervised Learning:
  Input: labeled pairs (image, "cat") (image, "dog")
  Learns: mapping from inputs to labels
  Examples: image classification, speech recognition, fraud detection

Unsupervised Learning:
  Input: unlabeled data
  Learns: hidden structure (clusters, patterns)
  Examples: customer segmentation, anomaly detection

Reinforcement Learning:
  Agent learns by trial and error in an environment
  Reward signal guides behavior
  Examples: game playing (AlphaGo), robot locomotion, autonomous driving
```

### Where Neural Networks Fit

Neural networks are a **family of supervised (and self-supervised) learning algorithms** that are especially powerful for:
- Images and video (CNNs)
- Text and sequences (Transformers, RNNs)
- Audio (CNNs + Transformers)
- Tabular data (MLP)
- Graphs (GNNs)

They dominate modern AI because they can learn **arbitrary complex mappings** given enough data and compute.

---

## 3. What is a Neural Network?

### Biological Inspiration

The brain contains ~86 billion neurons. Each neuron:
- Receives electrical signals from other neurons via **dendrites**
- Sums up the signals
- If the sum exceeds a **threshold**, it **fires** (sends a signal to others) via the **axon**
- The strength of connections between neurons is called **synaptic weight**

Learning = changing synaptic weights.

### The Artificial Analogy

```
Biological Neuron          Artificial Neuron
─────────────────          ─────────────────
Dendrites                  Inputs  x₁, x₂, ..., xₙ
Synaptic weights           Weights w₁, w₂, ..., wₙ
Cell body summation        z = Σ(wᵢ · xᵢ) + b
Firing threshold           Activation function: a = f(z)
Axon output                Output a
```

### A Network of Neurons

Neurons are organized into **layers**:

```
Input Layer     Hidden Layers      Output Layer
    x₁  ─────→  [neuron]  ─────→
    x₂  ─────→  [neuron]  ─────→  [neuron] → ŷ
    x₃  ─────→  [neuron]  ─────→
               [neuron]
```

- **Input layer**: raw features (pixel values, sensor readings, etc.)
- **Hidden layers**: learn intermediate representations
- **Output layer**: final prediction (class probabilities, regression value)

The "deep" in **deep learning** = many hidden layers.

---

## 4. The Neuron: Building Block

### Mathematical Definition

Given inputs **x** = [x₁, x₂, ..., xₙ]:

```
Step 1 — Linear combination (weighted sum):
  z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
    = w·x + b          (dot product notation)

Step 2 — Activation:
  a = f(z)             (apply nonlinearity)
```

Where:
- **w** = weight vector (learned parameters)
- **b** = bias (learned scalar offset)
- **f** = activation function

### Why the Bias?

Without bias, the decision boundary must pass through the origin. With bias:
```
z = wx + b
```
The bias shifts the activation, allowing the network to learn when to activate regardless of input magnitude. Think of it as the neuron's "default activation level."

### Concrete Example

A single neuron classifying whether a tumor is malignant:
```
Inputs: x₁ = tumor size (cm), x₂ = patient age (years)
Weights: w₁ = 0.8, w₂ = 0.3
Bias: b = -2.0

z = 0.8 × 3.5 + 0.3 × 45 + (-2.0)
  = 2.8 + 13.5 - 2.0
  = 14.3

a = sigmoid(14.3) ≈ 0.9999  → malignant (high probability)
```

---

## 5. Activation Functions

Without activation functions, a stack of linear layers collapses to a single linear function. **Nonlinearity** is what gives neural networks their expressive power.

### Sigmoid

```
σ(z) = 1 / (1 + e^(-z))

Range: (0, 1)
Use: binary classification output
Problem: vanishing gradients for large |z|
```

```
σ(z)
 1.0 |          ___________
 0.5 |        /
 0.0 |_______/
      -5   0   5     z
```

### Tanh

```
tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))

Range: (-1, 1)
Use: hidden layers (historically), RNNs
Better than sigmoid: zero-centered
Problem: still has vanishing gradient
```

### ReLU (Rectified Linear Unit)

```
ReLU(z) = max(0, z)

Range: [0, ∞)
Use: default for hidden layers in modern networks
Advantage: no vanishing gradient for z > 0, computationally fast
Problem: "dying ReLU" — neurons stuck at 0 if z always negative
```

```
ReLU(z)
  |          /
  |         /
  |        /
  |_______/
  -2  -1  0  1  2   z
```

### Leaky ReLU

```
LeakyReLU(z) = z        if z > 0
              = 0.01·z   if z ≤ 0

Fixes dying ReLU by allowing small negative gradient
```

### GELU (Gaussian Error Linear Unit)

```
GELU(z) ≈ z · σ(1.702·z)

Used in: Transformers (BERT, GPT)
Smooth approximation of ReLU
```

### Softmax (Output Layer for Multi-class)

```
softmax(z)ᵢ = e^(zᵢ) / Σⱼ e^(zⱼ)

Converts raw scores to probabilities that sum to 1
Use: multi-class classification output
```

### Which to Use?

| Layer Type       | Recommended Activation |
|------------------|------------------------|
| Hidden (MLP/CNN) | ReLU or GELU           |
| Output (binary)  | Sigmoid                |
| Output (multi)   | Softmax                |
| Output (regression) | None (linear)       |
| RNN gates        | Tanh + Sigmoid         |

---

## 6. The Multi-Layer Perceptron (MLP)

### Architecture

An MLP with 2 hidden layers:

```
Input        Hidden 1      Hidden 2      Output
(3 neurons)  (4 neurons)   (4 neurons)   (2 neurons)

  x₁ ─┐
  x₂ ─┼──→ [h₁¹]          [h₁²]
  x₃ ─┘    [h₂¹] ──────→  [h₂²] ──────→ [o₁]
            [h₃¹]          [h₃²]          [o₂]
            [h₄¹]          [h₄²]
```

### Matrix Form

For a layer with n inputs and m neurons:

```
Z = X @ W + b

X: input matrix    shape [batch_size, n_inputs]
W: weight matrix   shape [n_inputs, n_neurons]
b: bias vector     shape [n_neurons]
Z: output          shape [batch_size, n_neurons]

Then: A = f(Z)   (apply activation element-wise)
```

This single matrix multiplication computes **all neurons in a layer at once** — highly parallelizable on GPU.

### Parameter Count

```
Layer (n_in → n_out):
  Weights: n_in × n_out
  Biases:  n_out
  Total:   n_in × n_out + n_out

Example MLP: 784 → 256 → 128 → 10
  Layer 1: 784×256 + 256 = 201,216
  Layer 2: 256×128 + 128 = 32,896
  Layer 3: 128×10  + 10  = 1,290
  Total:   235,402 parameters
```

---

## 7. Forward Pass: Making a Prediction

The forward pass computes the output for a given input, layer by layer.

### Step-by-Step (2-layer MLP)

```python
# Pseudocode
def forward(x):
    # Layer 1
    z1 = x @ W1 + b1       # linear
    a1 = relu(z1)           # activation

    # Layer 2
    z2 = a1 @ W2 + b2      # linear
    a2 = relu(z2)           # activation

    # Output layer
    z3 = a2 @ W3 + b3      # linear
    output = softmax(z3)    # probabilities

    return output
```

### What the Network Learns Layer by Layer

For image classification (MNIST digits):
```
Layer 1 (edges):    detects horizontal, vertical, diagonal edges
Layer 2 (shapes):   combines edges into curves, corners
Layer 3 (parts):    detects parts of digits (loops, lines)
Output (class):     combines parts into digit predictions
```

Each layer learns increasingly **abstract representations** of the input.

---

## 8. Loss Functions: Measuring Error

The loss (or cost) function measures **how wrong the model's predictions are**. Training = minimizing loss.

### Mean Squared Error (MSE) — Regression

```
MSE = (1/N) Σᵢ (yᵢ - ŷᵢ)²

yᵢ  = true value
ŷᵢ  = predicted value
N   = number of samples

Good for: predicting continuous values (house price, temperature)
```

### Binary Cross-Entropy — Binary Classification

```
BCE = -(1/N) Σᵢ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]

yᵢ ∈ {0, 1}       true label
ŷᵢ ∈ (0, 1)       predicted probability (after sigmoid)

Intuition: penalizes confident wrong predictions heavily
  Predicted 0.99 when truth is 0 → very high loss
  Predicted 0.5  when truth is 0 → moderate loss
```

### Categorical Cross-Entropy — Multi-class Classification

```
CCE = -(1/N) Σᵢ Σₖ yᵢₖ log(ŷᵢₖ)

yᵢₖ = 1 if sample i belongs to class k, else 0  (one-hot)
ŷᵢₖ = predicted probability for class k (after softmax)

Most common loss for image classification
```

### Why Cross-Entropy?

Cross-entropy comes from information theory. Minimizing it is equivalent to **maximum likelihood estimation** — finding parameters that make the observed data most probable. It penalizes the model proportionally to how surprised it was by the correct answer.

---

## 9. Backpropagation: How Networks Learn

### The Core Idea

Backpropagation = **chain rule of calculus** applied to compute gradients of the loss with respect to every weight in the network.

```
∂Loss/∂W = how much does the loss change when W changes slightly?

We want: decrease the loss
Strategy: move W in the direction that decreases Loss
Update:   W ← W - α · (∂Loss/∂W)
```

### Chain Rule Foundation

For a composed function f(g(x)):
```
df/dx = (df/dg) · (dg/dx)
```

In a neural network, the loss flows through many composed functions:
```
Loss = L(softmax(W₃ · relu(W₂ · relu(W₁ · x + b₁) + b₂) + b₃))
```

Backprop applies chain rule **backwards** from output to input.

### Computational Graph

Think of the network as a graph of operations:

```
x → [W₁ matmul] → z₁ → [relu] → a₁ → [W₂ matmul] → z₂ → [loss] → L
```

Forward pass: compute and **cache** every intermediate value.
Backward pass: compute gradient at each node using cached values + chain rule.

### Step-by-Step Backward Pass (1 layer example)

```
Forward:
  z = x·w + b
  a = relu(z)
  L = MSE(a, y)

Backward (chain rule):
  dL/da = 2(a - y)/N                    (MSE gradient)
  dL/dz = dL/da · d(relu)/dz           (relu gradient: 1 if z>0 else 0)
  dL/dw = dL/dz · x                    (matmul gradient)
  dL/db = dL/dz · 1                    (bias gradient)
  dL/dx = dL/dz · w                    (input gradient, for prev layer)
```

### Automatic Differentiation (Autograd)

Modern frameworks (including tinygrad) implement **autograd**: they build the computational graph during the forward pass, then automatically compute all gradients during `.backward()`.

You never implement backprop by hand — you just define the forward pass.

```python
# tinygrad autograd example
x = Tensor([2.0], requires_grad=True)
w = Tensor([3.0], requires_grad=True)
b = Tensor([1.0], requires_grad=True)

z = x * w + b          # forward: builds graph
loss = z.pow(2).mean() # loss computation

loss.backward()         # backward: compute all gradients

print(w.grad)  # dL/dw computed automatically
```

### Vanishing and Exploding Gradients

**Vanishing gradients**: in deep networks, gradients become extremely small as they propagate back through many layers.
- Cause: sigmoid/tanh saturate (derivative ≈ 0)
- Fix: ReLU, residual connections (ResNets), batch normalization

**Exploding gradients**: gradients grow exponentially.
- Cause: large weights × many layers
- Fix: gradient clipping, weight initialization schemes

---

## 10. Gradient Descent and Optimizers

### Gradient Descent (Base Algorithm)

```
For each training step:
  1. Compute loss on a batch of data
  2. Compute gradients via backprop
  3. Update every parameter:
     W ← W - α · ∂Loss/∂W

α = learning rate (hyperparameter, typically 1e-3 to 1e-4)
```

### Batch Variants

| Variant           | Batch Size    | Pros                      | Cons                        |
|-------------------|---------------|---------------------------|-----------------------------|
| Batch GD          | Full dataset  | Stable gradients          | Slow, memory-intensive      |
| Stochastic GD (SGD)| 1 sample    | Fast updates              | Very noisy                  |
| Mini-batch GD     | 32–256        | Balance of both           | Industry standard           |

### SGD with Momentum

```
vₜ = β·vₜ₋₁ + (1-β)·∇L     (momentum term, β ≈ 0.9)
W ← W - α·vₜ

Intuition: gradient as a ball rolling downhill, accumulates velocity
Benefit: faster convergence, escapes shallow local minima
```

### Adam (Adaptive Moment Estimation)

The most widely used optimizer:

```
mₜ = β₁·mₜ₋₁ + (1-β₁)·∇L          (1st moment: mean of gradients)
vₜ = β₂·vₜ₋₁ + (1-β₂)·(∇L)²       (2nd moment: variance of gradients)

m̂ₜ = mₜ/(1-β₁ᵗ)                   (bias correction)
v̂ₜ = vₜ/(1-β₂ᵗ)

W ← W - α · m̂ₜ / (√v̂ₜ + ε)

Defaults: β₁=0.9, β₂=0.999, ε=1e-8, α=1e-3
```

Adam adapts the learning rate **per parameter** based on historical gradients. Works well out of the box for most tasks.

### Learning Rate Scheduling

```
Constant LR:       α = 0.001 throughout
Step decay:        α = α₀ × 0.1 every 10 epochs
Cosine annealing:  α follows a cosine curve
Warmup + decay:    Start low, increase, then decrease (Transformers)
```

---

## 11. The Training Loop

### Complete Training Loop

```python
# Pseudocode (very close to tinygrad code)

model = MyNetwork()
optimizer = Adam(model.parameters(), lr=1e-3)
loss_fn = CrossEntropyLoss()

for epoch in range(num_epochs):
    # ── Training phase ──────────────────────────────
    model.train()
    for batch_x, batch_y in train_loader:

        # 1. Forward pass
        predictions = model(batch_x)

        # 2. Compute loss
        loss = loss_fn(predictions, batch_y)

        # 3. Zero gradients (clear previous step's gradients)
        optimizer.zero_grad()

        # 4. Backward pass (compute gradients)
        loss.backward()

        # 5. Update weights
        optimizer.step()

    # ── Validation phase ─────────────────────────────
    model.eval()
    val_loss, val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch}: loss={val_loss:.4f}, acc={val_acc:.2%}")
```

### Key Concepts in the Loop

**Epoch**: one complete pass through the entire training dataset.

**Batch**: a subset of the dataset processed together (e.g., 32 images). Enables:
- GPU parallelism
- Gradient averaging (more stable than single samples)
- Fitting large datasets in memory

**Overfitting**: model memorizes training data, fails on new data.
```
Training loss:   ↓ ↓ ↓ ↓ ↓ 0.01
Validation loss: ↓ ↓ ↑ ↑ ↑ 0.45   ← overfitting after this point
```

**Underfitting**: model too simple, fails on both training and validation.
```
Training loss:   0.40 (stays high)
Validation loss: 0.42 (also high)
```

### Hyperparameters vs Parameters

```
Parameters (learned by training):
  - Weights W
  - Biases b

Hyperparameters (set by you before training):
  - Learning rate α
  - Batch size
  - Number of layers
  - Number of neurons per layer
  - Number of epochs
  - Dropout rate
  - Weight decay
```

---

## 12. Convolutional Neural Networks (CNNs)

### Why Not Just Use MLP for Images?

A 224×224 RGB image = 224×224×3 = 150,528 inputs.
MLP first hidden layer with 1024 neurons = 150,528 × 1024 = **154 million parameters** for one layer.

Problems:
- Doesn't capture spatial structure (neighboring pixels matter together)
- No weight sharing (same edge detector at every location)
- Too many parameters → overfitting

### The Convolution Operation

A **filter (kernel)** slides over the input, computing a weighted sum at each position:

```
Input (5×5):          Filter (3×3):      Output (3×3):
 1  2  3  4  5         1  0  1            ?  ?  ?
 5  6  7  8  9         0  1  0            ?  ?  ?
 1  2  3  4  5         1  0  1            ?  ?  ?
 5  6  7  8  9
 1  2  3  4  5

At top-left position:
  (1×1)+(2×0)+(3×1) + (5×0)+(6×1)+(7×0) + (1×1)+(2×0)+(3×1)
  = 1+3+6+1+3 = 14
```

The filter is a **learned pattern detector**. The network learns filters that detect edges, textures, shapes automatically.

### CNN Architecture Components

```
Input Image
    ↓
Conv Layer (learns feature maps)
    ↓
Activation (ReLU)
    ↓
Pooling (downsample, reduce size)
    ↓
... (repeat)
    ↓
Flatten
    ↓
FC (Fully Connected) Layers
    ↓
Output (Softmax)
```

### Pooling

Max pooling reduces spatial dimensions, retaining the most prominent features:

```
Input (4×4):       Max Pool 2×2:
 1  3  2  4          3  4
 5  6  7  8    →     6  8
 3  1  4  2          3  4
 7  8  5  6          8  6
```

### CNN vs MLP for Images

```
MLP:
  - Every pixel connects to every neuron
  - No spatial structure preserved
  - Millions of redundant parameters

CNN:
  - Local connectivity (filter slides over image)
  - Weight sharing (same filter at all positions)
  - Translation invariant (cat in corner = cat in center)
  - Far fewer parameters
```

### Classic CNN Architectures

| Model      | Year | Layers | Params  | Top-5 Acc |
|------------|------|--------|---------|-----------|
| LeNet-5    | 1998 | 7      | 60K     | —         |
| AlexNet    | 2012 | 8      | 61M     | 84.6%     |
| VGG-16     | 2014 | 16     | 138M    | 92.7%     |
| ResNet-50  | 2015 | 50     | 25M     | 95.3%     |
| MobileNetV2| 2018 | 53     | 3.4M    | 93.4%     |
| EfficientNet| 2019| varies | varies  | 97.1%     |

MobileNet and EfficientNet are designed for **edge deployment** — accuracy vs. compute tradeoff.

---

## 13. Regularization and Generalization

### The Bias-Variance Tradeoff

```
Total Error = Bias² + Variance + Irreducible Noise

High Bias (underfitting):   model too simple, misses patterns
High Variance (overfitting): model too complex, memorizes noise
```

### Dropout

Randomly zero out neurons during training:

```python
# During training:
  For each neuron, with probability p, set output to 0
  Scale remaining neurons by 1/(1-p)

# During inference:
  No dropout (all neurons active)

Effect: Forces network to learn redundant representations
        Acts as ensemble of many sub-networks
Typical p: 0.2–0.5
```

### Weight Decay (L2 Regularization)

Add a penalty to the loss for large weights:

```
L_total = L_task + λ · Σ w²

λ = regularization strength (e.g., 1e-4)

Effect: Keeps weights small, reduces overfitting
Equivalent to Gaussian prior on weights (Bayesian view)
```

### Batch Normalization

Normalize the inputs to each layer:

```
For a mini-batch of activations z:
  μ = mean(z)
  σ² = var(z)
  z_norm = (z - μ) / √(σ² + ε)
  output = γ · z_norm + β        (γ, β are learned)

Benefits:
  - Faster training (higher learning rates)
  - Less sensitive to weight initialization
  - Mild regularization effect
  - Reduces internal covariate shift
```

### Early Stopping

```
Monitor validation loss during training
Stop training when validation loss stops improving

Epoch: 1  train_loss=2.3  val_loss=2.3
Epoch: 5  train_loss=1.2  val_loss=1.3
Epoch:10  train_loss=0.5  val_loss=0.8   ← save checkpoint here
Epoch:15  train_loss=0.2  val_loss=1.1   ← overfitting, stop
Epoch:20  train_loss=0.1  val_loss=1.4
```

### Data Augmentation

Artificially expand dataset by transforming existing samples:

```
Image augmentations:
  - Horizontal/vertical flip
  - Random crop and resize
  - Color jitter (brightness, contrast, saturation)
  - Random rotation
  - Gaussian noise

Effect: Model sees more variety → better generalization
Edge AI benefit: reduces need for large datasets
```

---

## 14. Hands-On with tinygrad

### Why tinygrad?

tinygrad is a minimal ML framework (~1000 lines of core code) that exposes every fundamental operation clearly. Unlike PyTorch or TensorFlow:

- No magic — you can read and understand the entire source
- Teaches you **exactly** what happens inside a neural network
- Compiles to CPU, GPU, CUDA, Metal, WebGPU
- Used in production at comma.ai (autonomous driving)

### tinygrad Core Concepts

```python
from tinygrad.tensor import Tensor
import numpy as np

# Tensor creation
x = Tensor([[1.0, 2.0, 3.0]])           # shape [1, 3]
w = Tensor.kaiming_uniform(3, 4)        # shape [3, 4]

# Operations (lazy by default)
z = x.matmul(w)                         # shape [1, 4]
a = z.relu()

# Realize (execute computation)
result = a.numpy()                       # triggers actual computation

# Gradient computation
loss = a.sum()
loss.backward()
print(w.grad.numpy())                   # ∂loss/∂w
```

### Implementing a Neuron from Scratch

```python
from tinygrad.tensor import Tensor

class Neuron:
    def __init__(self, n_inputs):
        # Initialize weights with small random values
        self.w = Tensor.randn(n_inputs, 1) * 0.01
        self.b = Tensor.zeros(1)

    def __call__(self, x):
        z = x.matmul(self.w) + self.b
        return z.relu()

    def parameters(self):
        return [self.w, self.b]

# Test
neuron = Neuron(3)
x = Tensor([[0.5, 1.2, -0.3]])
output = neuron(x)
print(output.numpy())  # shape [1, 1]
```

### Implementing a Layer

```python
class Linear:
    def __init__(self, n_in, n_out):
        # Kaiming initialization for ReLU
        self.w = Tensor.kaiming_uniform(n_in, n_out)
        self.b = Tensor.zeros(n_out)

    def __call__(self, x):
        return x.matmul(self.w) + self.b

    def parameters(self):
        return [self.w, self.b]
```

### Implementing an MLP

```python
class MLP:
    def __init__(self, layers):
        # layers = [784, 256, 128, 10]
        self.linears = [
            Linear(layers[i], layers[i+1])
            for i in range(len(layers)-1)
        ]

    def __call__(self, x):
        for i, layer in enumerate(self.linears[:-1]):
            x = layer(x).relu()          # hidden layers: ReLU
        x = self.linears[-1](x)          # output layer: no activation
        return x.softmax()               # softmax for probabilities

    def parameters(self):
        params = []
        for layer in self.linears:
            params.extend(layer.parameters())
        return params
```

### Training on MNIST

```python
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam
import numpy as np

# Load MNIST (use fetch from tinygrad)
from tinygrad.helpers import fetch
import gzip

def load_mnist():
    base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    X_train = np.frombuffer(gzip.open(fetch(base+"train-images-idx3-ubyte.gz")).read(), np.uint8, offset=16).reshape(-1, 784)
    Y_train = np.frombuffer(gzip.open(fetch(base+"train-labels-idx1-ubyte.gz")).read(), np.uint8, offset=8)
    X_test  = np.frombuffer(gzip.open(fetch(base+"t10k-images-idx3-ubyte.gz")).read(), np.uint8, offset=16).reshape(-1, 784)
    Y_test  = np.frombuffer(gzip.open(fetch(base+"t10k-labels-idx1-ubyte.gz")).read(), np.uint8, offset=8)
    return X_train/255.0, Y_train, X_test/255.0, Y_test

X_train, Y_train, X_test, Y_test = load_mnist()

# Define model
model = MLP([784, 256, 128, 10])
optimizer = Adam(model.parameters(), lr=1e-3)

# Training loop
BATCH = 64
EPOCHS = 10

for epoch in range(EPOCHS):
    # Shuffle
    idx = np.random.permutation(len(X_train))
    X_train, Y_train = X_train[idx], Y_train[idx]

    total_loss = 0
    for i in range(0, len(X_train), BATCH):
        xb = Tensor(X_train[i:i+BATCH].astype(np.float32))
        yb = Y_train[i:i+BATCH]

        # Forward pass
        out = model(xb)

        # Cross-entropy loss
        # One-hot encode labels
        yb_onehot = np.zeros((len(yb), 10), dtype=np.float32)
        yb_onehot[np.arange(len(yb)), yb] = 1.0
        yb_t = Tensor(yb_onehot)

        loss = -(yb_t * out.log()).sum(axis=1).mean()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.numpy()

    # Validation
    test_out = model(Tensor(X_test.astype(np.float32)))
    preds = test_out.numpy().argmax(axis=1)
    acc = (preds == Y_test).mean()
    print(f"Epoch {epoch+1}: loss={total_loss/(len(X_train)//BATCH):.4f}, test_acc={acc:.2%}")
```

### Understanding tinygrad's Autograd

```python
# tinygrad builds a computation graph lazily
# Every operation on a Tensor records itself

x = Tensor([2.0])
y = x * x       # records: mul(x, x)
z = y + x       # records: add(y, x)
loss = z.sum()

# .backward() traverses graph in reverse (topological sort)
# applying chain rule at each node
loss.backward()

print(x.grad)   # d(loss)/d(x) = d(x²+x)/dx = 2x+1 = 5.0
```

### Implementing a CNN in tinygrad

```python
from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d, BatchNorm2d
from tinygrad.nn.optim import Adam

class SimpleCNN:
    def __init__(self):
        # Conv layers
        self.c1 = Conv2d(1, 32, 3, padding=1)   # 1 channel in, 32 out, 3×3 kernel
        self.c2 = Conv2d(32, 64, 3, padding=1)

        # Fully connected
        self.fc1 = Linear(64 * 7 * 7, 128)
        self.fc2 = Linear(128, 10)

    def __call__(self, x):
        # x shape: [batch, 1, 28, 28]
        x = self.c1(x).relu().max_pool2d()       # → [batch, 32, 14, 14]
        x = self.c2(x).relu().max_pool2d()       # → [batch, 64, 7, 7]
        x = x.reshape(x.shape[0], -1)            # flatten → [batch, 3136]
        x = self.fc1(x).relu()
        x = self.fc2(x)
        return x.softmax()

    def parameters(self):
        params = []
        for layer in [self.c1, self.c2, self.fc1, self.fc2]:
            params.extend(layer.parameters() if hasattr(layer, 'parameters') else [layer.weight, layer.bias])
        return params
```

### Exploring tinygrad Internals

Study these files in the tinygrad source to deeply understand how ML works:

```
tinygrad/
  tensor.py          ← Tensor class, all ops, autograd engine
  lazybuffer.py      ← Lazy evaluation / computation graph
  ops.py             ← All primitive operations
  nn/
    optim.py         ← SGD, Adam, AdaGrad implementations
  runtime/
    ops_cpu.py       ← How ops execute on CPU
    ops_gpu.py       ← How ops execute on GPU
```

Reading `tensor.py` is one of the best ways to understand how autograd and deep learning frameworks work internally.

---

## 15. Projects

### Project 1: Neuron from Scratch (No Framework)
Implement a single neuron in pure Python/NumPy. Manually compute gradients and verify with finite differences.

```
Goal: understand forward pass, loss, gradient by hand
Dataset: XOR problem (4 samples)
Deliverable: working neuron with manual backprop
```

### Project 2: MLP for MNIST with tinygrad
Train a fully connected network on MNIST digit classification.

```
Goal: achieve >97% test accuracy
Architecture: 784 → 256 → 128 → 10
Deliverable: training script + loss/accuracy curves
```

### Project 3: CNN for MNIST/CIFAR-10 with tinygrad
Replace MLP with a convolutional network.

```
Goal: understand convolution, pooling, feature maps
Dataset: MNIST (>99%) or CIFAR-10 (>85%)
Deliverable: CNN training script, visualize learned filters
```

### Project 4: Implement Adam from Scratch
Implement the Adam optimizer manually in tinygrad (without using tinygrad's built-in Adam).

```
Goal: understand optimizer internals
Deliverable: custom Adam class matching tinygrad's results
```

### Project 5: Read tinygrad's Backprop
Trace through tinygrad source code for a simple operation (e.g., `x * x`) and document exactly what happens during `backward()`.

```
Goal: understand autograd mechanics
Deliverable: annotated source code walkthrough
Files: tensor.py, ops.py
```

### Project 6: MNIST on Jetson (Edge Deployment)
Train on desktop, export model weights, run inference on Jetson Nano.

```
Goal: complete edge AI pipeline
Steps:
  1. Train CNN on desktop
  2. Export weights as numpy arrays
  3. Load weights in tinygrad on Jetson
  4. Measure inference latency: CPU vs GPU
  5. Quantize to INT8, compare accuracy/speed
```

---

## 16. Resources

### Foundational Theory

- **3Blue1Brown — Neural Networks** (YouTube series): Best visual introduction to how neural networks work. Watch all 4 episodes before writing code.
  - "But what is a neural network?"
  - "Gradient descent, how neural networks learn"
  - "What is backpropagation really doing?"
  - "Backpropagation calculus"

- **Andrej Karpathy — micrograd** (GitHub + YouTube): Build autograd from scratch in ~150 lines. Essential for understanding backprop deeply. Then compare with tinygrad's implementation.
  - https://github.com/karpathy/micrograd

- **CS231n: Convolutional Neural Networks for Visual Recognition** (Stanford): The definitive CNN course. Lecture notes are excellent even without watching videos.
  - https://cs231n.github.io/

- **The Deep Learning Book** (Goodfellow, Bengio, Courville): Free online. Chapters 6-9 cover MLP, backprop, regularization, CNNs rigorously.
  - https://www.deeplearningbook.org/

### tinygrad-Specific

- **tinygrad source code**: The best tinygrad documentation is the source itself.
  - `tensor.py` for ops and autograd
  - `examples/` for training scripts
  - `test/` for understanding expected behavior

- **tinygrad MNIST example**: `tinygrad/examples/mnist.py` — canonical starting point
- **tinygrad CNN example**: `tinygrad/examples/efficientnet.py`

### Edge AI Context

- **TinyML book** (Pete Warden, Daniel Situnayake): Running ML on microcontrollers. Chapters 1-3 give strong context for why edge AI design choices matter.

- **AI at the Edge** (Daniel Situnayake, Jenny Plunkett): End-to-end edge AI system design.

### Math Prerequisites

- **Linear Algebra**: 3Blue1Brown "Essence of Linear Algebra" — especially matrices as transformations
- **Calculus**: Chain rule, partial derivatives — Khan Academy calculus or 3Blue1Brown "Essence of Calculus"
- **Statistics**: Probability, distributions — needed for loss functions and regularization

### Practice Datasets

| Dataset   | Task                | Samples | Input Size | Baseline |
|-----------|---------------------|---------|------------|----------|
| MNIST     | Digit classification | 70,000 | 28×28×1    | 99.7%    |
| Fashion-MNIST | Clothing classification | 70,000 | 28×28×1 | 94%   |
| CIFAR-10  | Object classification | 60,000 | 32×32×3   | 95%+     |
| Iris      | Flower classification | 150   | 4 features | 97%     |

---

## Quick Reference: Key Equations

```
Neuron forward:
  z = Wx + b
  a = f(z)

MSE Loss:
  L = (1/N) Σ(y - ŷ)²

Cross-entropy Loss:
  L = -(1/N) Σ y·log(ŷ)

Gradient descent:
  W ← W - α·∂L/∂W

Adam update:
  m = β₁m + (1-β₁)g
  v = β₂v + (1-β₂)g²
  W ← W - α·m̂/√(v̂+ε)

Convolution output size:
  out = floor((in + 2p - k) / s) + 1
  in=input, p=padding, k=kernel, s=stride
```

---

## Deep Dive: Subfolders

### [PyTorch + micrograd → tinygrad](pytorch-and-micrograd/Guide.md)

Before diving deep into tinygrad internals or porting models, you need a rock-solid foundation in how autograd actually works. This subfolder provides exactly that:

- **Part 1 — micrograd**: Reimplement Karpathy's scalar autograd engine from scratch. Every `+`, `*`, `tanh`, `exp` carries a `_backward` closure. You'll see chain rule as running code, not math notation.
- **Part 2 — PyTorch**: Aligned with the OpenCV PyTorch Bootcamp curriculum. Tensors, autograd, `nn.Module`, CNNs, transfer learning, object detection heads — with the same mental model as micrograd.
- **Part 3 — Bridge to tinygrad**: Side-by-side comparison of the same MLP in all three frameworks. What PyTorch hides (lazy evaluation, kernel fusion) that tinygrad exposes via `DEBUG=4`.

**Why this matters for tinygrad mastery:**
```
micrograd  →  you understand every gradient operation
PyTorch    →  you understand the production API and patterns
tinygrad   →  you see what frameworks hide, control the scheduler
```

---

*Next: [3. Edge AI Optimization](../3. Edge AI Optimization/Guide.md) — quantization, pruning, TensorRT, TFLite*
