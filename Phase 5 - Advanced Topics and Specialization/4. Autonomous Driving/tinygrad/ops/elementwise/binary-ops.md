# BinaryOps: Two Tensor Operations

## Overview

BinaryOps operate on **two tensors**, combining them element-by-element with automatic broadcasting.

```
Input A: [a, b, c, d]
Input B: [e, f, g, h]
BinaryOp: f(x, y)
Output:  [f(a,e), f(b,f), f(c,g), f(d,h)]
```

## Complete List of BinaryOps

### Arithmetic Operations

#### 1. ADD - Addition
```python
from tinygrad import Tensor

a = Tensor([1, 2, 3, 4])
b = Tensor([5, 6, 7, 8])
c = a + b  # or a.add(b)
# Result: [6, 8, 10, 12]
```

**Mathematical:** `c[i] = a[i] + b[i]`

**Use cases:**
- Residual connections (ResNet)
- Bias addition
- Gradient accumulation

#### 2. SUB - Subtraction
```python
a = Tensor([10, 20, 30, 40])
b = Tensor([1, 2, 3, 4])
c = a - b  # or a.sub(b)
# Result: [9, 18, 27, 36]
```

**Mathematical:** `c[i] = a[i] - b[i]`

**Use cases:**
- Loss calculation (prediction - target)
- Normalization (x - mean)
- Gradient computation

#### 3. MUL - Multiplication
```python
a = Tensor([1, 2, 3, 4])
b = Tensor([2, 3, 4, 5])
c = a * b  # or a.mul(b)
# Result: [2, 6, 12, 20]
```

**Mathematical:** `c[i] = a[i] * b[i]`

**Use cases:**
- Scaling
- Attention mechanisms
- Gating (LSTM, GRU)

#### 4. DIV - Division
```python
a = Tensor([10, 20, 30, 40])
b = Tensor([2, 4, 5, 8])
c = a / b  # or a.div(b)
# Result: [5, 5, 6, 5]
```

**Mathematical:** `c[i] = a[i] / b[i]`

**Use cases:**
- Normalization (x / std)
- Average pooling
- Learning rate scheduling

#### 5. MOD - Modulo
```python
a = Tensor([10, 11, 12, 13])
b = Tensor([3, 3, 3, 3])
c = a % b  # or a.mod(b)
# Result: [1, 2, 0, 1]
```

**Mathematical:** `c[i] = a[i] % b[i]`

**Use cases:**
- Positional encodings
- Cyclic patterns
- Index wrapping

### Comparison Operations

#### 6. CMPLT - Compare Less Than
```python
a = Tensor([1, 2, 3, 4])
b = Tensor([2, 2, 2, 2])
c = a < b  # or a.cmplt(b)
# Result: [True, False, False, False]
# As numbers: [1, 0, 0, 0]
```

**Mathematical:** `c[i] = (a[i] < b[i]) ? 1 : 0`

**Use cases:**
- Masking
- Conditional operations
- Thresholding

### Other Comparison Ops (Derived)

```python
# Greater than
a > b   # equivalent to: b < a

# Less than or equal
a <= b  # equivalent to: (a < b) | (a == b)

# Greater than or equal
a >= b  # equivalent to: b <= a

# Equal
a == b  # equivalent to: (a <= b) & (a >= b)

# Not equal
a != b  # equivalent to: ~(a == b)
```

### Special Operations

#### 7. MAX - Maximum
```python
a = Tensor([1, 5, 3, 2])
b = Tensor([4, 2, 6, 1])
c = a.maximum(b)
# Result: [4, 5, 6, 2]
```

**Mathematical:** `c[i] = max(a[i], b[i])`

**Use cases:**
- ReLU activation: `x.maximum(0)`
- Clipping: `x.maximum(min_val)`
- Max pooling (with ReduceOps)

## Broadcasting Rules

Broadcasting allows operations on different shaped tensors:

### Rule 1: Scalar Broadcasting
```python
a = Tensor([1, 2, 3, 4])
b = 10
c = a + b
# Result: [11, 12, 13, 14]

# b is broadcast to [10, 10, 10, 10]
```

### Rule 2: Dimension Broadcasting
```python
a = Tensor([[1, 2, 3],
            [4, 5, 6]])  # Shape: (2, 3)
b = Tensor([10, 20, 30])  # Shape: (3,)
c = a + b
# Result: [[11, 22, 33],
#          [14, 25, 36]]

# b is broadcast to [[10, 20, 30],
#                    [10, 20, 30]]
```

### Rule 3: Multiple Dimension Broadcasting
```python
a = Tensor([[1],
            [2],
            [3]])  # Shape: (3, 1)
b = Tensor([10, 20, 30, 40])  # Shape: (4,)
c = a + b
# Result: [[11, 21, 31, 41],
#          [12, 22, 32, 42],
#          [13, 23, 33, 43]]
# Shape: (3, 4)
```

### Broadcasting Visualization
```
Shape (3, 1):        Shape (1, 4):
[[1]                 [[10, 20, 30, 40]]
 [2]
 [3]]

After broadcasting both to (3, 4):
[[1, 1, 1, 1]        [[10, 20, 30, 40]
 [2, 2, 2, 2]    +    [10, 20, 30, 40]
 [3, 3, 3, 3]]        [10, 20, 30, 40]]

Result (3, 4):
[[11, 21, 31, 41]
 [12, 22, 32, 42]
 [13, 23, 33, 43]]
```

## Common Patterns

### Activation Functions

#### ReLU
```python
def relu(x):
    return x.maximum(0)

x = Tensor([-2, -1, 0, 1, 2])
y = relu(x)
# Result: [0, 0, 0, 1, 2]
```

#### Leaky ReLU
```python
def leaky_relu(x, alpha=0.01):
    return x.maximum(alpha * x)

x = Tensor([-2, -1, 0, 1, 2])
y = leaky_relu(x)
# Result: [-0.02, -0.01, 0, 1, 2]
```

#### ELU (Exponential Linear Unit)
```python
def elu(x, alpha=1.0):
    return x.maximum(0) + (x < 0) * alpha * (x.exp() - 1)

x = Tensor([-2, -1, 0, 1, 2])
y = elu(x)
# Result: [-0.865, -0.632, 0, 1, 2]
```

### Normalization

#### Batch Normalization (simplified)
```python
def batch_norm(x, mean, var, eps=1e-5):
    return (x - mean) / (var + eps).sqrt()

x = Tensor([[1, 2], [3, 4]])
mean = x.mean(axis=0)
var = x.var(axis=0)
normalized = batch_norm(x, mean, var)
```

#### Layer Normalization
```python
def layer_norm(x, eps=1e-5):
    mean = x.mean()
    var = ((x - mean) ** 2).mean()
    return (x - mean) / (var + eps).sqrt()
```

### Loss Functions

#### Mean Squared Error (MSE)
```python
def mse_loss(pred, target):
    return ((pred - target) ** 2).mean()

pred = Tensor([1.0, 2.0, 3.0])
target = Tensor([1.5, 2.5, 2.5])
loss = mse_loss(pred, target)
# Result: 0.25
```

#### Mean Absolute Error (MAE)
```python
def mae_loss(pred, target):
    return (pred - target).abs().mean()

pred = Tensor([1.0, 2.0, 3.0])
target = Tensor([1.5, 2.5, 2.5])
loss = mae_loss(pred, target)
# Result: 0.333
```

#### Huber Loss
```python
def huber_loss(pred, target, delta=1.0):
    diff = (pred - target).abs()
    quadratic = (diff < delta) * 0.5 * diff ** 2
    linear = (diff >= delta) * delta * (diff - 0.5 * delta)
    return (quadratic + linear).mean()
```

### Attention Mechanisms

#### Scaled Dot-Product Attention (simplified)
```python
def attention_scores(Q, K, d_k):
    # Q: (batch, seq_len, d_k)
    # K: (batch, seq_len, d_k)
    scores = (Q @ K.transpose(-2, -1)) / d_k.sqrt()
    return scores.softmax(axis=-1)
```

#### Gating Mechanism
```python
def gating(x, gate):
    """Element-wise gating (used in LSTM, GRU)"""
    return x * gate.sigmoid()

x = Tensor([1, 2, 3, 4])
gate = Tensor([0, 1, 2, 3])
output = gating(x, gate)
# Gate values: [0.5, 0.73, 0.88, 0.95]
# Output: [0.5, 1.46, 2.64, 3.8]
```

## Performance Considerations

### Fast Operations ⚡
```python
# Hardware optimized
a + b          # ADD
a * b          # MUL
a.maximum(b)   # MAX
```

### Medium Speed ⚠️
```python
# Requires more computation
a / b          # DIV (can use a * b.reciprocal())
a - b          # SUB (can use a + (-b))
a % b          # MOD
```

### Optimization Tips

#### Use MUL instead of DIV
```python
# Slower
y = x / scale

# Faster (if scale is constant)
inv_scale = 1.0 / scale
y = x * inv_scale
```

#### Fused Operations
```python
# Tinygrad automatically fuses these:
y = (x + 1) * 2 - 0.5
# Into a single kernel!
```

## Code Examples

### Example 1: Dropout
```python
def dropout(x, p=0.5, training=True):
    """Dropout regularization"""
    if not training:
        return x

    # Create random mask
    mask = Tensor.rand(*x.shape) > p
    # Scale by 1/(1-p) to maintain expected value
    return x * mask / (1 - p)

x = Tensor([[1, 2, 3, 4],
            [5, 6, 7, 8]])
dropped = dropout(x, p=0.5, training=True)
```

### Example 2: Gradient Clipping
```python
def clip_gradient(grad, clip_value=1.0):
    """Clip gradients by value"""
    return grad.maximum(-clip_value).minimum(clip_value)

grad = Tensor([-2, -0.5, 0, 0.5, 2])
clipped = clip_gradient(grad, clip_value=1.0)
# Result: [-1, -0.5, 0, 0.5, 1]
```

### Example 3: Smooth L1 Loss
```python
def smooth_l1_loss(pred, target, beta=1.0):
    """Smooth L1 loss (used in object detection)"""
    diff = (pred - target).abs()
    loss = (diff < beta) * (0.5 * diff ** 2 / beta) + \
           (diff >= beta) * (diff - 0.5 * beta)
    return loss.mean()
```

### Example 4: Cosine Similarity
```python
def cosine_similarity(a, b, eps=1e-8):
    """Cosine similarity between two vectors"""
    dot_product = (a * b).sum()
    norm_a = (a * a).sum().sqrt()
    norm_b = (b * b).sum().sqrt()
    return dot_product / (norm_a * norm_b + eps)

a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])
sim = cosine_similarity(a, b)
# Result: 0.974 (highly similar)
```

## Visual Examples

### ADD Operation
```
A: [1, 2, 3, 4]
B: [5, 6, 7, 8]
   ↓  ↓  ↓  ↓
   +  +  +  +
   ↓  ↓  ↓  ↓
C: [6, 8, 10, 12]
```

### MUL with Broadcasting
```
A (2x3):        B (3,):
[[1, 2, 3]      [10, 20, 30]
 [4, 5, 6]]

B broadcast to (2x3):
[[10, 20, 30]
 [10, 20, 30]]

Result (2x3):
[[10, 40, 90]
 [40, 100, 180]]
```

### MAX Operation
```
A: [1, 5, 3, 2]
B: [4, 2, 6, 1]
   ↓  ↓  ↓  ↓
  max max max max
   ↓  ↓  ↓  ↓
C: [4, 5, 6, 2]
```

## Debugging Tips

### Check Broadcasting
```python
a = Tensor.randn(3, 1)
b = Tensor.randn(4)

# Check if shapes are compatible
print(f"A shape: {a.shape}")  # (3, 1)
print(f"B shape: {b.shape}")  # (4,)

c = a + b
print(f"Result shape: {c.shape}")  # (3, 4)
```

### Visualize Operations
```python
import os
os.environ['DEBUG'] = '3'

a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])
c = (a + b) * 2
c.realize()
# Shows kernel fusion!
```

## Key Takeaways

1. **BinaryOps combine two tensors** element-by-element
2. **7 primitive BinaryOps**: ADD, SUB, MUL, DIV, MOD, CMPLT, MAX
3. **Broadcasting** allows operations on different shapes
4. **Highly composable** - build complex operations from simple ones
5. **Automatic fusion** - multiple ops combined into single kernel
6. **Foundation for neural networks** - activations, losses, attention

## Next Steps

- Learn about [TernaryOps](ternary-ops.md) for three-tensor operations
- Explore how BinaryOps combine with ReduceOps
- Study kernel fusion for performance optimization
