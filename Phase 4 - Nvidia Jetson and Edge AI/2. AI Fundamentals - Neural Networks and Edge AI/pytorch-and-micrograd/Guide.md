# PyTorch and micrograd — Insight for Mastering tinygrad

> **Course reference:** [OpenCV PyTorch Bootcamp & Deep Learning](https://courses.opencv.org/courses/course-v1:PyTorch+Bootcamp+Deep-Learning/course/)
> **micrograd reference:** [karpathy/micrograd](https://github.com/karpathy/micrograd)
>
> **Why this guide exists:** tinygrad's design makes sense only when you understand *what problem it is solving*. That means understanding PyTorch's API (what tinygrad mimics) and micrograd's internals (what tinygrad extends). Read this before the tinygrad deep-dive.

---

## The Three-Framework Mental Model

```
micrograd                  PyTorch                   tinygrad
──────────────────         ───────────────────        ────────────────────
Scalar values only         Full tensor engine         Full tensor engine
~150 lines Python          ~3M lines C++/Python       ~5000 lines Python
No GPU                     GPU (opaque CUDA)          GPU (readable Python)
Pure education             Production                 Hackable production

Teaches:                   Teaches:                   Teaches:
  ∙ What autograd IS         ∙ The API you'll use       ∙ What the API DOES
  ∙ Chain rule concretely    ∙ Real training loops      ∙ Lazy eval, IR, kernels
  ∙ Computational graph      ∙ CNNs, transfer learning  ∙ Custom backends
  ∙ How backward() works     ∙ Industry patterns        ∙ Compiler internals
```

**Learning order:** micrograd → PyTorch → tinygrad

---

## Table of Contents

1. [micrograd — Autograd from 150 Lines](#1-micrograd--autograd-from-150-lines)
2. [Building a Neural Net on micrograd](#2-building-a-neural-net-on-micrograd)
3. [PyTorch Module 1 — Tensors](#3-pytorch-module-1--tensors)
4. [PyTorch Module 2 — Autograd](#4-pytorch-module-2--autograd)
5. [PyTorch Module 4 — Deep Learning Fundamentals](#5-pytorch-module-4--deep-learning-fundamentals)
6. [PyTorch Module 5 — CNNs](#6-pytorch-module-5--cnns)
7. [PyTorch Module 7 — Transfer Learning](#7-pytorch-module-7--transfer-learning)
8. [PyTorch Module 9 — Object Detection](#8-pytorch-module-9--object-detection)
9. [Side-by-Side: micrograd vs PyTorch vs tinygrad](#9-side-by-side-micrograd-vs-pytorch-vs-tinygrad)
10. [What PyTorch Hides — and tinygrad Exposes](#10-what-pytorch-hides--and-tinygrad-exposes)
11. [Exercises](#11-exercises)
12. [Resources](#12-resources)

---

## 1. micrograd — Autograd from 150 Lines

### What micrograd is

Karpathy's micrograd implements reverse-mode automatic differentiation (autograd) **at the scalar level**. Every number is a `Value` object that tracks:
- Its numeric data
- Its gradient (`grad`)
- What operation created it (`_op`)
- What inputs created it (`_prev`)

This is exactly what PyTorch's `Tensor` does — but PyTorch does it over multi-dimensional arrays with C++ kernels. micrograd does it over Python floats so you can read every line.

### Implement the `Value` class

```python
# micrograd.py  — the entire autograd engine in one class
import math

class Value:
    """
    A scalar value with autograd support.
    Wraps a Python float and tracks how it was computed.
    """

    def __init__(self, data, _children=(), _op='', label=''):
        self.data  = float(data)
        self.grad  = 0.0          # dL/d(self) — starts at zero
        self._op   = _op          # what operation produced this node
        self._prev = set(_children)
        self._label = label

        # _backward: function that computes gradients for this node's inputs
        # Default: leaf node, nothing to propagate back through
        self._backward = lambda: None

    # ── Forward operations ──────────────────────────────────────────────

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # d(out)/d(self)  = 1   →  self.grad += 1 * out.grad
            # d(out)/d(other) = 1   →  other.grad += 1 * out.grad
            self.grad  += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # d(a*b)/da = b  →  self.grad  += other.data * out.grad
            # d(a*b)/db = a  →  other.grad += self.data  * out.grad
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float))
        out = Value(self.data ** exponent, (self,), f'**{exponent}')

        def _backward():
            # d(x^n)/dx = n * x^(n-1)
            self.grad += (exponent * self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'ReLU')

        def _backward():
            # d(relu(x))/dx = 1 if x > 0 else 0
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # d(tanh(x))/dx = 1 - tanh(x)^2
            self.grad += (1.0 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')

        def _backward():
            # d(e^x)/dx = e^x
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data + 1e-10), (self,), 'log')

        def _backward():
            # d(ln(x))/dx = 1/x
            self.grad += (1.0 / (self.data + 1e-10)) * out.grad

        out._backward = _backward
        return out

    # ── Reverse operations (make Python operators work both ways) ───────
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __neg__(self):         return self * -1
    def __sub__(self, other):  return self + (-other)
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return Value(other) * self**-1

    # ── Backward pass ────────────────────────────────────────────────────

    def backward(self):
        """
        Compute gradients for all nodes in the computational graph.

        Algorithm:
          1. Build topological order of all nodes (leaf → root)
          2. Set self.grad = 1.0  (dL/dL = 1)
          3. Walk in REVERSE topological order
          4. Call each node's _backward() to propagate gradient to its inputs
        """
        topo = []
        visited = set()

        def build_topo(node):
            if id(node) not in visited:
                visited.add(id(node))
                for child in node._prev:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        self.grad = 1.0           # seed: dL/dL = 1

        for node in reversed(topo):
            node._backward()      # propagate gradient through this op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
```

### Trace a complete backward pass by hand

```python
# Verify against finite differences (numerical gradient check)

def numerical_gradient(f, x, h=1e-5):
    """Estimate df/dx numerically."""
    x_plus  = Value(x.data + h)
    x_minus = Value(x.data - h)
    return (f(x_plus).data - f(x_minus).data) / (2 * h)

# Test: f(x) = (x^2 + 3x + 2) * tanh(x)
x = Value(2.0)
y = (x**2 + 3*x + 2) * x.tanh()
y.backward()

print(f"Autograd gradient: {x.grad:.6f}")
print(f"Numerical gradient: {numerical_gradient(lambda v: (v**2 + 3*v + 2) * v.tanh(), x):.6f}")
# Both should match to 4+ decimal places

# Step-by-step trace — add this debugging helper
def trace_graph(root):
    """Print the computational graph."""
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    for n in nodes:
        print(f"  {n._label or id(n)}: data={n.data:.4f} grad={n.grad:.4f} op={n._op}")

x = Value(2.0, label='x')
w = Value(-3.0, label='w')
b = Value(1.0, label='b')
z = x * w + b;  z._label = 'z'
y = z.tanh();   y._label = 'y'
y.backward()
trace_graph(y)
```

### Why `+=` for gradients (not `=`)

```python
# A critical subtlety: nodes can be reused in the graph
a = Value(2.0)
b = a + a          # a appears TWICE as input to b
c = b * b
c.backward()

# When computing dc/da:
#   c = b^2         → dc/db = 2b
#   b = a + a       → db/da = 1 + 1 = 2
#   dc/da = dc/db * db/da = 2b * 2 = 4b = 4*(2+2) = 16

print(f"a.grad = {a.grad}")   # Should be 16.0
# If we used = instead of +=, second path would overwrite first → wrong answer

# This is why every _backward uses:  self.grad += ...  (not =)
```

---

## 2. Building a Neural Net on micrograd

### Neuron, Layer, MLP

```python
import random

class Neuron:
    def __init__(self, n_in, activation='tanh'):
        self.w   = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b   = Value(0.0)
        self.act = activation

    def __call__(self, x):
        # z = w·x + b
        z = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
        return z.tanh() if self.act == 'tanh' else z.relu() if self.act == 'relu' else z

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, n_in, n_out, **kwargs):
        self.neurons = [Neuron(n_in, **kwargs) for _ in range(n_out)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, n_in, layer_sizes):
        sizes = [n_in] + layer_sizes
        self.layers = [
            Layer(sizes[i], sizes[i+1],
                  activation='tanh' if i < len(layer_sizes)-1 else 'linear')
            for i in range(len(layer_sizes))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
```

### Training loop on XOR

```python
# XOR dataset — linearly inseparable, needs hidden layer
X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
Y = [0.0, 1.0, 1.0, 0.0]

model = MLP(2, [4, 1])     # 2 → 4 → 1

for epoch in range(200):
    # Forward pass
    preds = [model(x) for x in X]

    # MSE loss
    loss = sum((pred - y)**2 for pred, y in zip(preds, Y)) * (1/len(Y))

    # Backward
    model.zero_grad()
    loss.backward()

    # SGD update
    lr = 0.1
    for p in model.parameters():
        p.data -= lr * p.grad

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}: loss={loss.data:.4f}")

# Test
for x, y in zip(X, Y):
    pred = model(x)
    print(f"Input {x} → pred={pred.data:.3f}  truth={y}")
```

### What micrograd taught us about tinygrad

```
micrograd concept                tinygrad equivalent
──────────────────────           ─────────────────────────────────
Value._prev (set of inputs)      LazyBuffer.srcs (source buffers)
Value._backward (grad fn)        Tensor.grad_fn
build_topo() + reversed()        topological sort in realize()
self.grad += ...                  gradient accumulation in backward
Value.data = float               LazyBuffer → realized numpy/cuda array
```

---

## 3. PyTorch Module 1 — Tensors

### Tensor creation

```python
import torch
import numpy as np

# From data
a = torch.tensor([1.0, 2.0, 3.0])          # 1D, float32
b = torch.tensor([[1, 2], [3, 4]])           # 2D, int64

# Factory functions
zeros = torch.zeros(3, 4)                   # shape [3, 4], all zeros
ones  = torch.ones(2, 3)
eye   = torch.eye(4)                         # identity matrix
rand  = torch.rand(3, 3)                     # uniform [0, 1)
randn = torch.randn(3, 3)                    # standard normal

# From numpy (shares memory — zero copy)
arr = np.array([1.0, 2.0, 3.0])
t   = torch.from_numpy(arr)                 # no copy
t[0] = 99.0
print(arr[0])   # 99.0 — same memory!

# To numpy
np_arr = t.numpy()                          # CPU only

# Device
t_gpu = t.cuda()
t_cpu = t_gpu.cpu()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
t = torch.randn(3, 3, device=device)
```

### Tensor operations

```python
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

# Elementwise
print(a + b)
print(a * b)        # elementwise multiply (NOT matmul)

# Matrix multiply
print(a @ b)        # matmul — preferred syntax
print(torch.mm(a, b))
print(torch.matmul(a, b))

# Reduction
print(a.sum())                  # scalar sum
print(a.sum(dim=0))             # sum over rows → shape [2]
print(a.sum(dim=1))             # sum over cols → shape [2]
print(a.mean(), a.max(), a.min())

# Shape manipulation
t = torch.randn(2, 3, 4)
print(t.shape)                  # torch.Size([2, 3, 4])
print(t.reshape(6, 4).shape)    # [6, 4]
print(t.view(2, -1).shape)      # [2, 12]  (must be contiguous)
print(t.permute(1, 0, 2).shape) # [3, 2, 4]
print(t.transpose(0, 1).shape)  # [3, 2, 4]

# Adding dimensions
x = torch.randn(3)
print(x.unsqueeze(0).shape)     # [1, 3]
print(x.unsqueeze(1).shape)     # [3, 1]
print(x[None, :].shape)         # [1, 3]  — same as unsqueeze(0)

# Broadcasting
a = torch.randn(3, 1)
b = torch.randn(1, 4)
print((a + b).shape)            # [3, 4] — broadcasts
```

### dtype and device best practices

```python
# Always be explicit about dtype for edge/inference work
x = torch.tensor([1.0], dtype=torch.float32)   # not float64 (doubles GPU memory)
x = torch.tensor([1.0], dtype=torch.float16)   # FP16 for inference
x = torch.tensor([1],   dtype=torch.int8)      # INT8 for quantized inference

# Move to device
x = x.to(device)
x = x.to('cuda:0')             # specific GPU

# In-place ops (use with caution — breaks autograd graph)
x.add_(1.0)                    # in-place add
x.mul_(2.0)                    # in-place mul
```

---

## 4. PyTorch Module 2 — Autograd

### How autograd works in PyTorch

PyTorch autograd is the tensor-level equivalent of micrograd's `Value` class:

```
micrograd:               PyTorch:
Value.data       →       Tensor.data
Value.grad       →       Tensor.grad
Value._backward  →       Tensor.grad_fn (C++ function object)
Value._prev      →       grad_fn.next_functions
build_topo()     →       Engine.execute_graph()
value.backward() →       tensor.backward()
```

### requires_grad — opt-in differentiation

```python
x = torch.tensor([2.0], requires_grad=True)   # track this
w = torch.tensor([3.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=False)  # don't track bias here

# Forward pass — builds computational graph
z = x * w + b
y = z ** 2
loss = y.sum()

print(loss.grad_fn)                 # MulBackward0 (or similar)
print(loss.grad_fn.next_functions)  # shows the graph structure

# Backward pass
loss.backward()

print(f"x.grad = {x.grad}")    # dL/dx
print(f"w.grad = {w.grad}")    # dL/dw

# Verify manually:
# L = (x*w + b)^2 = (2*3+1)^2 = 49
# dL/dx = 2*(x*w+b)*w = 2*7*3 = 42
# dL/dw = 2*(x*w+b)*x = 2*7*2 = 28
```

### Gradient accumulation — understanding the += pattern

```python
x = torch.tensor([2.0], requires_grad=True)

# First backward
loss1 = x ** 2
loss1.backward()
print(f"After first backward:  x.grad = {x.grad}")   # 4.0

# Second backward WITHOUT zero_grad
loss2 = x ** 3
loss2.backward()
print(f"After second backward: x.grad = {x.grad}")   # 4.0 + 12.0 = 16.0 ← ACCUMULATES

# Why? In real training you batch gradients before updating weights
# Always zero gradients before each new backward:
x.grad.zero_()    # in-place zero
```

### No-grad context — inference and validation

```python
# During inference: don't build graph (saves memory, faster)
with torch.no_grad():
    output = model(input_data)    # no grad_fn attached

# Alternative decorator
@torch.no_grad()
def predict(model, x):
    return model(x)

# Detach a tensor from graph (useful for target values in RL)
target = output.detach()    # same data, no grad tracking
```

### Custom autograd function

Understanding how to write a custom `Function` reveals exactly how autograd works:

```python
class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # ctx.save_for_backward saves tensors needed in backward
        ctx.save_for_backward(x)
        return x.clamp(min=0)   # ReLU

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: gradient flowing IN from the next layer
        x, = ctx.saved_tensors
        # Gradient of ReLU: 1 where x > 0, else 0
        grad_input = grad_output.clone()
        grad_input[x < 0] = 0
        return grad_input    # gradient flowing OUT to previous layer

# Use it exactly like a built-in operation
x = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
y = MyReLU.apply(x)
y.sum().backward()
print(x.grad)   # tensor([0., 1., 1.])
```

---

## 5. PyTorch Module 4 — Deep Learning Fundamentals

### nn.Module — the building block

Every PyTorch model is a subclass of `nn.Module`. Understanding its internals is crucial for understanding tinygrad's equivalent.

```python
import torch.nn as nn

class Linear(nn.Module):
    """Re-implement nn.Linear from scratch to understand Module."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # nn.Parameter: a tensor that is registered as a parameter
        # (included in model.parameters(), moved with .to(), saved with state_dict())
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.empty(out_features)) if bias else None

        # Initialize (Kaiming uniform, same as PyTorch default)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # F.linear computes: x @ weight.T + bias
        return x @ self.weight.T + (self.bias if self.bias is not None else 0)

# nn.Module automatically handles:
#   .parameters()  — iterate all registered Parameter tensors
#   .to(device)    — move all parameters to device
#   .train()/.eval() — set training/eval mode (affects BatchNorm, Dropout)
#   .state_dict()  — serialize all parameters
#   .load_state_dict() — restore from checkpoint
```

### Building a complete MLP

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_features, hidden_sizes, out_features, dropout=0.2):
        super().__init__()
        sizes = [in_features] + hidden_sizes + [out_features]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:     # no activation/dropout after last layer
                layers.append(nn.BatchNorm1d(sizes[i+1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = MLP(784, [256, 128], 10)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(model)   # shows layer structure
```

### Loss functions

```python
# Classification
loss_fn = nn.CrossEntropyLoss()
# Input: logits [B, C] (raw scores, no softmax needed)
# Target: class indices [B] (long tensor)
logits = torch.randn(32, 10)   # batch=32, classes=10
labels = torch.randint(0, 10, (32,))
loss = loss_fn(logits, labels)

# CrossEntropyLoss internally does:
#   softmax(logits, dim=1) → log → NLLLoss
# Equivalent to:
#   F.cross_entropy(logits, labels)
#   -log_softmax(logits, dim=1).gather(1, labels.unsqueeze(1)).mean()

# Regression
loss_fn = nn.MSELoss()
preds   = torch.randn(32, 1)
targets = torch.randn(32, 1)
loss    = loss_fn(preds, targets)

# Binary classification
loss_fn = nn.BCEWithLogitsLoss()   # sigmoid inside (numerically stable)
logits  = torch.randn(32, 1)
targets = torch.randint(0, 2, (32, 1)).float()
loss    = loss_fn(logits, targets)
```

### Optimizers

```python
# SGD
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# Adam (most common for deep learning)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

# AdamW (Adam + decoupled weight decay — preferred for Transformers)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
# or
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=1e-2, steps_per_epoch=len(train_loader), epochs=30
)
```

### DataLoader

```python
from torch.utils.data import Dataset, DataLoader

class MNISTDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_ds = MNISTDataset(X_train, Y_train)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
# pin_memory=True: allocates batch in pinned (page-locked) CPU memory → faster GPU transfer
# num_workers=2: 2 background processes prefetch data while GPU trains
```

### Complete training loop

```python
def train_epoch(model, loader, loss_fn, optimizer, device):
    model.train()   # enables BatchNorm running stats update + Dropout
    total_loss, correct, total = 0, 0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        # 1. Forward
        logits = model(X)
        loss   = loss_fn(logits, y)

        # 2. Backward
        optimizer.zero_grad()   # CRITICAL: clear previous gradients
        loss.backward()

        # 3. Gradient clipping (optional but important for deep nets)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 4. Update
        optimizer.step()

        total_loss += loss.item()
        correct    += (logits.argmax(1) == y).sum().item()
        total      += len(y)

    return total_loss / len(loader), correct / total

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()   # disables Dropout, uses running stats in BatchNorm
    total_loss, correct, total = 0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        total_loss += loss_fn(logits, y).item()
        correct    += (logits.argmax(1) == y).sum().item()
        total      += len(y)
    return total_loss / len(loader), correct / total

# Run training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = MLP(784, [256, 128], 10).to(device)
opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)

for epoch in range(20):
    train_loss, train_acc = train_epoch(model, train_dl, loss_fn, opt, device)
    val_loss, val_acc     = evaluate(model, val_dl, loss_fn, device)
    scheduler.step()
    print(f"Epoch {epoch+1:2d}: train={train_acc:.3%}  val={val_acc:.3%}  lr={opt.param_groups[0]['lr']:.2e}")
```

---

## 6. PyTorch Module 5 — CNNs

### Convolution in PyTorch

```python
import torch.nn as nn

# nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

# Parameter count: out_ch × in_ch × kH × kW + out_ch (bias)
params = 64 * 3 * 3 * 3 + 64    # = 1,792

# Output size formula: floor((in + 2p - k) / s) + 1
# With padding=1, kernel=3, stride=1: output = input (same padding)

# Forward through a batch
x = torch.randn(8, 3, 224, 224)   # [B, C, H, W]
y = conv(x)
print(y.shape)   # [8, 64, 224, 224]
```

### Depthwise Separable Convolution (MobileNet building block)

```python
class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise: one filter per channel (groups=in_channels)
    Pointwise: 1×1 conv to mix channels

    Standard conv: in_ch × out_ch × k × k MACs per position
    DSConv:        in_ch × k × k + in_ch × out_ch MACs
    Speedup:       ~8× for k=3 vs standard conv
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise  = nn.Conv2d(in_ch, in_ch, 3, stride=stride,
                                     padding=1, groups=in_ch, bias=False)
        self.pointwise  = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.bn1(self.depthwise(x)).relu()
        x = self.bn2(self.pointwise(x)).relu()
        return x
```

### ResNet residual block

```python
class ResBlock(nn.Module):
    """
    Skip connection: output = F(x) + x
    Lets gradient flow directly through the + operation
    → solves vanishing gradient in deep networks
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = self.bn1(self.conv1(x)).relu()
        x = self.bn2(self.conv2(x))
        x = x + residual     # skip connection — gradient flows directly here
        return x.relu()
```

### Building a small CNN for MNIST

```python
class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                    # 28→14
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                    # 14→7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = SmallCNN()
x = torch.randn(32, 1, 28, 28)   # [B, C, H, W]
print(model(x).shape)             # [32, 10]

# Count parameters
total = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total:,}")   # ~421K
```

---

## 7. PyTorch Module 7 — Transfer Learning

Transfer learning is the most important practical technique for edge AI: you take a model pretrained on millions of images and adapt it to your specific task with a small dataset.

```python
import torchvision.models as models
import torch.nn as nn

# ── Strategy 1: Feature Extractor (freeze backbone) ──────────────────
# Use when: very small dataset (<1000 images)
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace the classification head (only this will train)
in_features = model.fc.in_features   # 512 for ResNet-18
model.fc = nn.Linear(in_features, num_classes)  # new head, requires_grad=True by default

# Only head parameters will update
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

# ── Strategy 2: Fine-tuning (unfreeze backbone) ──────────────────────
# Use when: medium dataset (>5000 images), or domain shift from ImageNet
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(512, num_classes)

# Lower LR for backbone (don't destroy pretrained features)
# Higher LR for head (train from scratch)
optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(),   'lr': 1e-4},
    {'params': model.layer3.parameters(),   'lr': 1e-5},
    {'params': model.fc.parameters(),       'lr': 1e-3},
], lr=1e-4)

# ── Strategy 3: Progressive unfreezing ────────────────────────────────
# Epoch 1-3:  train head only
# Epoch 4-6:  unfreeze layer4 + train
# Epoch 7-10: unfreeze layer3 + train
# This is the ULMFiT approach — prevents catastrophic forgetting

def unfreeze_layer(model, layer_name):
    for name, param in model.named_parameters():
        if layer_name in name:
            param.requires_grad = True
            print(f"Unfrozen: {name}")
```

### Model comparison for edge deployment

```python
# Compare accuracy vs speed vs size for transfer learning targets
import torchvision.models as models
import time

models_to_compare = {
    'ResNet-18':       (models.resnet18,       models.ResNet18_Weights.IMAGENET1K_V1),
    'MobileNetV3-S':   (models.mobilenet_v3_small,  models.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
    'EfficientNet-B0': (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
}

x = torch.randn(1, 3, 224, 224)
for name, (fn, weights) in models_to_compare.items():
    m = fn(weights=weights).eval()
    params = sum(p.numel() for p in m.parameters())

    # Latency
    with torch.no_grad():
        for _ in range(10): m(x)   # warmup
        t0 = time.perf_counter()
        for _ in range(100): m(x)
        lat = (time.perf_counter() - t0) * 10   # ms

    print(f"{name:20s}: {params/1e6:.1f}M params, {lat:.1f}ms/frame")

# Expected (CPU):
# ResNet-18:           11.7M params, 45ms/frame
# MobileNetV3-S:        2.5M params, 12ms/frame  ← best for edge
# EfficientNet-B0:      5.3M params, 30ms/frame
```

---

## 8. PyTorch Module 9 — Object Detection

### Understanding YOLO from PyTorch internals

```python
# The YOLO head: for each cell in the grid, predict B boxes
# Each box: [tx, ty, tw, th, objectness, class1...classN]

class YOLOHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        # Output per cell: anchors × (5 + num_classes)
        out_ch = num_anchors * (5 + num_classes)
        self.conv = nn.Conv2d(in_channels, out_ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        out = self.conv(x)   # [B, A*(5+C), H, W]
        # Reshape to [B, A, H, W, 5+C]
        out = out.view(B, self.num_anchors, 5 + self.num_classes, H, W)
        out = out.permute(0, 1, 3, 4, 2)   # [B, A, H, W, 5+C]

        # Apply sigmoid to tx, ty (center offset) and objectness
        out[..., :2]  = out[..., :2].sigmoid()    # tx, ty ∈ [0,1]
        out[..., 4:5] = out[..., 4:5].sigmoid()   # objectness
        out[..., 5:]  = out[..., 5:].sigmoid()    # class probs

        return out

# Load pretrained YOLOv8 with ultralytics
from ultralytics import YOLO
model = YOLO('yolov8n.pt')

# Access the underlying PyTorch model
pt_model = model.model
print(type(pt_model))   # ultralytics.nn.tasks.DetectionModel

# Export to ONNX (standard transfer for TensorRT)
model.export(format='onnx', opset=17, simplify=True)
```

### Custom dataset for fine-tuning YOLO

```yaml
# dataset.yaml — YOLO training config
path: /data/my_dataset
train: images/train
val:   images/val

nc: 3   # number of classes
names: ['car', 'pedestrian', 'cyclist']
```

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')   # start from pretrained nano model
results = model.train(
    data='dataset.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device='cuda',
    lr0=0.01,
    lrf=0.01,      # final LR = lr0 * lrf
    warmup_epochs=3,
    augment=True,  # built-in mosaic, copy-paste, mixup augmentations
    save=True,
    project='runs/detect',
    name='custom_yolo',
)
```

---

## 9. Side-by-Side: micrograd vs PyTorch vs tinygrad

### The same MLP in all three frameworks

```python
# ── micrograd ────────────────────────────────────────────────────────
from micrograd import Value, MLP as MgMLP

mg_model = MgMLP(2, [4, 1])
mg_x = [Value(0.5), Value(1.0)]
mg_out = mg_model(mg_x)
mg_loss = (mg_out - Value(1.0)) ** 2
mg_loss.backward()

# ── PyTorch ──────────────────────────────────────────────────────────
import torch, torch.nn as nn

pt_model = nn.Sequential(
    nn.Linear(2, 4), nn.Tanh(),
    nn.Linear(4, 1)
)
pt_x    = torch.tensor([[0.5, 1.0]])
pt_out  = pt_model(pt_x)
pt_loss = (pt_out - 1.0) ** 2
pt_loss.backward()

# ── tinygrad ─────────────────────────────────────────────────────────
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear as TgLinear

class TgMLP:
    def __init__(self):
        self.l1 = TgLinear(2, 4)
        self.l2 = TgLinear(4, 1)
    def __call__(self, x):
        return self.l2(self.l1(x).tanh())
    def parameters(self):
        return [*self.l1.weight, self.l1.bias, *self.l2.weight, self.l2.bias]

tg_model = TgMLP()
tg_x     = Tensor([[0.5, 1.0]])
tg_out   = tg_model(tg_x)
tg_loss  = (tg_out - 1.0).pow(2).mean()
tg_loss.backward()
```

### Autograd internals comparison

```python
# Inspect what's actually in the graph

# PyTorch: follow grad_fn chain
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x
print(y.grad_fn)                    # AddBackward0
print(y.grad_fn.next_functions)     # (PowBackward0, MulBackward0)
print(y.grad_fn.next_functions[0][0].next_functions)  # (AccumulateGrad,)

# tinygrad: inspect LazyBuffer
import os; os.environ['DEBUG'] = '1'   # show schedule
from tinygrad.tensor import Tensor

x = Tensor([2.0])
y = x ** 2 + x * 3
y.numpy()   # trigger computation — DEBUG=1 shows the ops

# With DEBUG=4: shows the generated GPU/CPU kernel code
os.environ['DEBUG'] = '4'
z = (x * x).sum().numpy()
```

### Gradient flow visualization

```python
import torch
from torchviz import make_dot   # pip install torchviz

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
x = torch.randn(1, 4)
y = model(x)
loss = y.sum()

# Render the computational graph
dot = make_dot(loss, params=dict(model.named_parameters()))
dot.render('graph', format='png')
# Open graph.png to see the full backward graph with all grad_fns
```

---

## 10. What PyTorch Hides — and tinygrad Exposes

This is the critical section for tinygrad mastery. For every PyTorch operation, ask: what does tinygrad do instead?

### Lazy evaluation

```python
# PyTorch: eager execution — operations run immediately
x = torch.randn(1000, 1000)
y = x @ x                     # matmul runs NOW, result in y immediately

# tinygrad: lazy execution — operations are scheduled, not run
from tinygrad.tensor import Tensor
x = Tensor.randn(1000, 1000)
y = x.matmul(x)               # NO computation yet — y is a LazyBuffer
                               # just a description of "matmul of x and x"
result = y.numpy()             # NOW the computation runs (realize())

# Why lazy?
#   → Operations can be fused before execution
#   → Redundant operations can be eliminated
#   → The scheduler can optimize the compute graph
#   → This is how XLA (TensorFlow/JAX), tinygrad, and torch.compile all work
```

### Kernel fusion

```python
# PyTorch (without torch.compile): 3 separate GPU kernel launches
x = torch.randn(1024, 1024, device='cuda')
y = x.relu()        # launch kernel 1: relu
z = y * 2           # launch kernel 2: mul
w = z + 1           # launch kernel 3: add
# Each launch has overhead: CPU→GPU sync, memory roundtrip

# tinygrad: these FUSE into a single kernel
import os; os.environ['CUDA'] = '1'; os.environ['DEBUG'] = '4'
from tinygrad.tensor import Tensor
x = Tensor.randn(1024, 1024)
w = (x.relu() * 2 + 1).numpy()   # ONE kernel: relu+mul+add fused
# DEBUG=4 shows the generated kernel source — single loop doing all three ops

# torch.compile() (PyTorch 2.0+) also does this, but via C++ Triton
# tinygrad does it in pure Python — readable
```

### The kernel code you can read

```python
# tinygrad lets you inspect the generated kernel
import os
os.environ['DEBUG'] = '4'
os.environ['CPU'] = '1'

from tinygrad.tensor import Tensor

a = Tensor.randn(4, 4)
b = Tensor.randn(4, 4)
c = (a @ b).relu()
c.numpy()

# DEBUG=4 output shows something like:
# void E_4_4(float* data0, const float* data1, const float* data2) {
#   for (int i = 0; i < 4; i++) {
#     for (int j = 0; j < 4; j++) {
#       float acc = 0.0f;
#       for (int k = 0; k < 4; k++) acc += data1[i*4+k] * data2[k*4+j];
#       data0[i*4+j] = fmax(0.0f, acc);  // relu fused!
#     }
#   }
# }
# You can READ this. Try doing that with PyTorch.
```

### Memory management: unified vs separate

```python
# PyTorch on GPU: CPU memory and GPU memory are separate
cpu_tensor = torch.randn(1000)
gpu_tensor = cpu_tensor.cuda()   # copy: CPU → GPU (PCIe transfer)
back       = gpu_tensor.cpu()    # copy: GPU → CPU

# tinygrad on Jetson: unified memory — no copy needed!
# (see Edge AI Optimization guide for full discussion)
# Jetson's LPDDR5 is shared between CPU and GPU
# tinygrad's cuda backend uses cuMemAllocManaged() for unified alloc

# PyTorch on Jetson: still works but you must use .cuda()/.cpu() explicitly
# tinygrad: if running on Jetson with CUDA backend, tensors are already accessible to both
```

---

## 11. Exercises

### Micrograd exercises

1. **Verify chain rule:** Implement `sin(x)` in the `Value` class. Check that the gradient matches `cos(x)` numerically.

2. **Computational graph visualization:** Use `graphviz` to draw the computational graph for `f(x, y) = (x^2 + y) * tanh(x - y)`. Annotate each node with its `data` and `grad` values after calling `f.backward()`.

3. **Manual batching:** Extend micrograd to support vectors of `Value`s (a `Tensor1D` class). Implement a batch forward pass and verify gradients are correct.

4. **Derive a layer from scratch:** Re-implement `BatchNorm1d` as a `Value`-level operation. Show that during inference (eval mode) it correctly uses running mean/variance instead of batch statistics.

### PyTorch exercises

5. **Reproduce micrograd:** Implement the XOR training from Section 2 using PyTorch (`nn.Linear`, `torch.tanh`, Adam). Verify it converges in <200 epochs with identical hyperparameters.

6. **Custom loss function:** Implement `FocalLoss` (used in RetinaNet for object detection) as a custom `nn.Module`. Verify it gives lower loss weight to easy examples.

7. **Gradient tape:** Using only `requires_grad`, `backward()`, and manual parameter updates (`param.data -= lr * param.grad`), train a 2-layer MLP on MNIST without using `nn.Module` or any optimizer. Verify >95% accuracy.

8. **Profile memory:** Use `torch.cuda.memory_summary()` to measure peak GPU memory for batch sizes 16, 32, 64, 128 during training a ResNet-18. Plot the result.

### Bridge to tinygrad

9. **Port micrograd to tinygrad tensors:** Rewrite the `Value` class using tinygrad `Tensor` ops instead of Python floats. The forward pass will use tinygrad; the backward pass is tinygrad's autograd.

10. **Read tinygrad's Linear:** Open `tinygrad/nn/__init__.py`. Find the `Linear` class. Map each line to its micrograd/PyTorch equivalent. Write annotations.

11. **Compare kernels:** Run the same matmul in PyTorch (profile with `torch.profiler`) and in tinygrad with `DEBUG=4`. Compare the generated/called kernel names.

---

## 12. Resources

### micrograd
- **karpathy/micrograd** — github.com/karpathy/micrograd: 150 lines — read every line
- **"The spelled-out intro to neural networks and backpropagation"** (Karpathy, YouTube, 2h25m): walks through building micrograd live. Watch before reading the source.
- **"Neural Networks: Zero to Hero"** (Karpathy series): micrograd → makemore → GPT. Best ML education series available.

### PyTorch (OpenCV Course aligned)
- **OpenCV PyTorch Bootcamp** — courses.opencv.org: the reference course for this guide. Covers Modules 1–10 with notebooks and assignments.
- **Official PyTorch Tutorials** — pytorch.org/tutorials: especially "Learning PyTorch with Examples" and "Writing Custom Datasets, DataLoaders and Transforms"
- **PyTorch Documentation** — pytorch.org/docs/stable: `torch.autograd`, `torch.nn`, `torch.optim`
- **"Deep Learning with PyTorch"** (Eli Stevens et al., Manning): free PDF at pytorch.org/deep-learning-with-pytorch

### tinygrad connection
- **tinygrad/tensor.py** — the 1000-line file that replaces micrograd's `Value` + PyTorch's `Tensor`
- **"tinygrad: a simple and powerful nn/ml framework"** — geohot's original blog post
- **DEBUG=1,2,3,4** environment variable: the best tinygrad tutorial is running it with increasing debug levels and reading the output

### Supplementary math
- **"Matrix Calculus for Deep Learning"** (Parr & Howard): the mathematical bridge between chain rule scalars (micrograd) and matrix/tensor calculus (PyTorch/tinygrad)
- **3Blue1Brown "Essence of Calculus"**: chain rule in 15 minutes, visually

---

*Parent: [AI Fundamentals Guide](../Guide.md)*
*Next: [tinygrad deep dive](../../Phase 5 - Advanced Topics and Specialization/4. Autonomous Driving/tinygrad/Guide.md)*
