# UnaryOps: Single Tensor Operations

## Overview

UnaryOps operate on a **single tensor**, applying the same operation to each element independently.

```
Input:  [a, b, c, d]
UnaryOp: f(x)
Output: [f(a), f(b), f(c), f(d)]
```

## Complete List of UnaryOps

### Mathematical Operations

#### 1. EXP2 - Base-2 Exponential
```python
# Mathematical: 2^x
from tinygrad import Tensor

x = Tensor([0, 1, 2, 3])
y = x.exp2()
# Result: [1, 2, 4, 8]
```

**Use cases:**
- Fast exponential (hardware optimized)
- Power-of-2 calculations
- Binary scaling

#### 2. LOG2 - Base-2 Logarithm
```python
# Mathematical: log₂(x)
x = Tensor([1, 2, 4, 8])
y = x.log2()
# Result: [0, 1, 2, 3]
```

**Use cases:**
- Entropy calculations
- Information theory
- Bit depth calculations

#### 3. SQRT - Square Root
```python
# Mathematical: √x
x = Tensor([1, 4, 9, 16])
y = x.sqrt()
# Result: [1, 2, 3, 4]
```

**Use cases:**
- Standard deviation
- Euclidean distance
- Normalization

#### 4. RECIP - Reciprocal
```python
# Mathematical: 1/x
x = Tensor([1, 2, 4, 8])
y = x.reciprocal()
# Result: [1, 0.5, 0.25, 0.125]
```

**Use cases:**
- Division (x/y = x * recip(y))
- Inverse operations
- Normalization

#### 5. NEG - Negation
```python
# Mathematical: -x
x = Tensor([1, -2, 3, -4])
y = -x  # or x.neg()
# Result: [-1, 2, -3, 4]
```

**Use cases:**
- Sign flipping
- Gradient reversal
- Subtraction (a - b = a + (-b))

#### 6. SIN - Sine
```python
# Mathematical: sin(x)
import math
x = Tensor([0, math.pi/2, math.pi])
y = x.sin()
# Result: [0, 1, 0]
```

**Use cases:**
- Positional encodings (Transformers)
- Signal processing
- Periodic activations

### Type Operations

#### 7. CAST - Type Conversion
```python
# Convert between data types
from tinygrad import dtypes

x = Tensor([1.5, 2.7, 3.9])
y = x.cast(dtypes.int32)
# Result: [1, 2, 3]

z = x.cast(dtypes.float16)  # Half precision
```

**Use cases:**
- Precision control (FP32 → FP16)
- Integer operations
- Memory optimization

## Derived UnaryOps

These are built from the basic UnaryOps:

### EXP - Natural Exponential
```python
# e^x = 2^(x * log₂(e))
def exp(x):
    return (x * 1.442695040888963).exp2()  # log₂(e) ≈ 1.4427

x = Tensor([0, 1, 2])
y = x.exp()
# Result: [1, 2.718, 7.389]
```

### LOG - Natural Logarithm
```python
# ln(x) = log₂(x) / log₂(e)
def log(x):
    return x.log2() * 0.6931471805599453  # ln(2)

x = Tensor([1, 2.718, 7.389])
y = x.log()
# Result: [0, 1, 2]
```

### ABS - Absolute Value
```python
# |x| = max(x, -x)
def abs(x):
    return x.maximum(-x)  # Uses BinaryOp

x = Tensor([-2, -1, 0, 1, 2])
y = x.abs()
# Result: [2, 1, 0, 1, 2]
```

### RELU - Rectified Linear Unit
```python
# ReLU(x) = max(x, 0)
def relu(x):
    return x.maximum(0)  # Uses BinaryOp

x = Tensor([-2, -1, 0, 1, 2])
y = x.relu()
# Result: [0, 0, 0, 1, 2]
```

### SIGMOID - Logistic Function
```python
# σ(x) = 1 / (1 + e^(-x))
def sigmoid(x):
    return (1 + (-x).exp()).reciprocal()

x = Tensor([-2, -1, 0, 1, 2])
y = x.sigmoid()
# Result: [0.119, 0.268, 0.5, 0.731, 0.881]
```

### TANH - Hyperbolic Tangent
```python
# tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
def tanh(x):
    exp_x = x.exp()
    exp_neg_x = (-x).exp()
    return (exp_x - exp_neg_x) / (exp_x + exp_neg_x)

# Or more efficiently:
def tanh_fast(x):
    return 2 * (2*x).sigmoid() - 1

x = Tensor([-2, -1, 0, 1, 2])
y = x.tanh()
# Result: [-0.964, -0.762, 0, 0.762, 0.964]
```

## Visual Examples

### SQRT Operation
```
Input:  [1.0, 4.0, 9.0, 16.0]
         ↓    ↓    ↓     ↓
        √    √    √     √
         ↓    ↓    ↓     ↓
Output: [1.0, 2.0, 3.0, 4.0]
```

### EXP2 Operation
```
Input:  [0, 1, 2, 3]
         ↓  ↓  ↓  ↓
        2⁰ 2¹ 2² 2³
         ↓  ↓  ↓  ↓
Output: [1, 2, 4, 8]
```

### SIGMOID Operation
```
Input:  [-∞, -2, -1, 0, 1, 2, +∞]
         ↓    ↓   ↓  ↓  ↓  ↓   ↓
       σ(x) = 1/(1+e^(-x))
         ↓    ↓   ↓  ↓  ↓  ↓   ↓
Output: [0, 0.12, 0.27, 0.5, 0.73, 0.88, 1]

Graph:
1.0 |           ___---
    |       __--
0.5 |    __-
    | __-
0.0 |--
    -2  -1  0  1  2
```

## Performance Considerations

### Fast Operations ⚡
```python
# Hardware optimized
x.neg()        # Simple sign flip
x.reciprocal() # Fast division alternative
x.cast()       # Type conversion
```

### Medium Speed ⚠️
```python
# Requires computation
x.sqrt()       # Square root
x.exp2()       # Exponential
x.log2()       # Logarithm
```

### Slower Operations 🐌
```python
# Complex transcendental functions
x.sin()        # Trigonometric
x.exp()        # Natural exponential (derived)
x.tanh()       # Hyperbolic (multiple ops)
```

## Code Examples

### Example 1: Custom Activation Function
```python
from tinygrad import Tensor

def swish(x):
    """Swish activation: x * sigmoid(x)"""
    return x * x.sigmoid()

def mish(x):
    """Mish activation: x * tanh(softplus(x))"""
    return x * (x.softplus()).tanh()

x = Tensor([-2, -1, 0, 1, 2])
print("Swish:", swish(x).numpy())
print("Mish:", mish(x).numpy())
```

### Example 2: Normalization
```python
def normalize(x):
    """Normalize to [0, 1] range"""
    min_val = x.min()
    max_val = x.max()
    return (x - min_val) / (max_val - min_val)

x = Tensor([10, 20, 30, 40, 50])
normalized = normalize(x)
print(normalized.numpy())  # [0, 0.25, 0.5, 0.75, 1.0]
```

### Example 3: Gradient Clipping
```python
def clip_by_norm(x, max_norm=1.0):
    """Clip tensor by L2 norm"""
    norm = x.square().sum().sqrt()
    scale = (max_norm / norm).minimum(1.0)
    return x * scale

x = Tensor([3.0, 4.0])  # Norm = 5.0
clipped = clip_by_norm(x, max_norm=1.0)
print(clipped.numpy())  # [0.6, 0.8], norm = 1.0
```

## Common Patterns

### Activation Functions
```python
# All built from UnaryOps
relu(x)      = x.maximum(0)
sigmoid(x)   = (1 + (-x).exp()).reciprocal()
tanh(x)      = 2 * (2*x).sigmoid() - 1
softplus(x)  = x.exp().log()  # log(1 + e^x)
```

### Normalization
```python
# Standard score
z_score = (x - mean) / std

# Min-max scaling
scaled = (x - x.min()) / (x.max() - x.min())
```

### Numerical Stability
```python
# Stable log-sum-exp
def logsumexp(x):
    max_x = x.max()
    return max_x + (x - max_x).exp().sum().log()

# Stable softmax
def softmax(x):
    exp_x = (x - x.max()).exp()
    return exp_x / exp_x.sum()
```

## Debugging Tips

### Visualize Operations
```python
import os
os.environ['DEBUG'] = '3'

x = Tensor([1, 2, 3, 4])
y = x.sqrt().exp().log()
y.realize()
# Prints the kernel code and operations
```

### Check for NaN/Inf
```python
def safe_log(x, eps=1e-8):
    """Prevent log(0) = -inf"""
    return (x + eps).log()

def safe_sqrt(x, eps=1e-8):
    """Prevent sqrt(negative)"""
    return (x.maximum(0) + eps).sqrt()
```

## Key Takeaways

1. **UnaryOps operate element-by-element** on a single tensor
2. **7 primitive UnaryOps** in tinygrad (EXP2, LOG2, SQRT, RECIP, NEG, SIN, CAST)
3. **All other unary operations** are derived from these primitives
4. **Highly parallelizable** - perfect for GPU execution
5. **Compose into complex functions** - activations, normalizations, etc.

## Next Steps

- Learn about [BinaryOps](binary-ops.md) for two-tensor operations
- Understand how UnaryOps combine with other operation types
- Explore kernel fusion for performance optimization
