# TernaryOps: Three Tensor Operations

## Overview

TernaryOps operate on **three tensors**, typically for conditional or fused multiply-accumulate operations.

```
Input A: [a, b, c, d]
Input B: [e, f, g, h]
Input C: [i, j, k, l]
TernaryOp: f(x, y, z)
Output:  [f(a,e,i), f(b,f,j), f(c,g,k), f(d,h,l)]
```

## Complete List of TernaryOps

### 1. WHERE - Conditional Selection

The most important TernaryOp!

```python
from tinygrad import Tensor

# WHERE(condition, if_true, if_false)
condition = Tensor([True, False, True, False])
if_true = Tensor([1, 2, 3, 4])
if_false = Tensor([5, 6, 7, 8])

result = condition.where(if_true, if_false)
# Result: [1, 6, 3, 8]
```

**Mathematical:**
```
result[i] = condition[i] ? if_true[i] : if_false[i]
```

**Equivalent to:**
```python
# In Python/NumPy
result = np.where(condition, if_true, if_false)

# In C/C++
result[i] = condition[i] ? if_true[i] : if_false[i];
```

### 2. MULACC - Multiply-Accumulate

Fused multiply-add operation (FMA).

```python
# MULACC(a, b, c) = a * b + c
a = Tensor([1, 2, 3, 4])
b = Tensor([2, 3, 4, 5])
c = Tensor([10, 10, 10, 10])

result = a.mulacc(b, c)
# Result: [12, 16, 22, 30]
# Computed as: [1*2+10, 2*3+10, 3*4+10, 4*5+10]
```

**Mathematical:**
```
result[i] = a[i] * b[i] + c[i]
```

**Why it matters:**
- Single hardware instruction on modern CPUs/GPUs
- More accurate (no intermediate rounding)
- Faster than separate MUL + ADD

## WHERE: Deep Dive

### Basic Usage

```python
# Select based on condition
x = Tensor([-2, -1, 0, 1, 2])
condition = x > 0
result = condition.where(x, 0)
# Result: [0, 0, 0, 1, 2]
# This is ReLU!
```

### Broadcasting with WHERE

```python
# Condition, true, and false can have different shapes
condition = Tensor([[True, False],
                    [False, True]])  # Shape: (2, 2)
if_true = Tensor([1, 2])             # Shape: (2,)
if_false = 0                         # Scalar

result = condition.where(if_true, if_false)
# Result: [[1, 0],
#          [0, 2]]
```

### Multiple Conditions

```python
# Chained WHERE for multiple conditions
x = Tensor([-2, -1, 0, 1, 2])

# Classify: negative, zero, positive
negative = x < 0
zero = x == 0

result = negative.where(-1, zero.where(0, 1))
# Result: [-1, -1, 0, 1, 1]
```

## Common Patterns

### Activation Functions

#### ReLU
```python
def relu(x):
    return (x > 0).where(x, 0)
    # Or more efficiently: x.maximum(0)

x = Tensor([-2, -1, 0, 1, 2])
y = relu(x)
# Result: [0, 0, 0, 1, 2]
```

#### Leaky ReLU
```python
def leaky_relu(x, alpha=0.01):
    return (x > 0).where(x, alpha * x)

x = Tensor([-2, -1, 0, 1, 2])
y = leaky_relu(x)
# Result: [-0.02, -0.01, 0, 1, 2]
```

#### Hard Sigmoid
```python
def hard_sigmoid(x):
    """Piecewise linear approximation of sigmoid"""
    return (x < -2.5).where(0,
           (x > 2.5).where(1,
           0.2 * x + 0.5))

x = Tensor([-3, -1, 0, 1, 3])
y = hard_sigmoid(x)
# Result: [0, 0.3, 0.5, 0.7, 1]
```

#### Hard Swish
```python
def hard_swish(x):
    """Efficient approximation of Swish"""
    return x * (x + 3).relu().minimum(6) / 6

# Equivalent using WHERE:
def hard_swish_where(x):
    return (x <= -3).where(0,
           (x >= 3).where(x,
           x * (x + 3) / 6))
```

### Clipping Operations

#### Clip by Value
```python
def clip(x, min_val, max_val):
    """Clip values to [min_val, max_val]"""
    return (x < min_val).where(min_val,
           (x > max_val).where(max_val, x))

x = Tensor([-5, -1, 0, 1, 5])
clipped = clip(x, -2, 2)
# Result: [-2, -1, 0, 1, 2]
```

#### Gradient Clipping
```python
def clip_gradient(grad, threshold):
    """Clip gradients by norm"""
    norm = grad.square().sum().sqrt()
    scale = (norm > threshold).where(threshold / norm, 1.0)
    return grad * scale
```

### Masking Operations

#### Attention Mask
```python
def apply_attention_mask(scores, mask, neg_inf=-1e9):
    """Apply mask to attention scores"""
    # mask: 1 for valid, 0 for masked
    return mask.where(scores, neg_inf)

scores = Tensor([[1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0]])
mask = Tensor([[1, 1, 0],
               [1, 0, 0]])

masked_scores = apply_attention_mask(scores, mask)
# Result: [[1.0, 2.0, -1e9],
#          [4.0, -1e9, -1e9]]
```

#### Dropout
```python
def dropout(x, p=0.5, training=True):
    """Dropout with WHERE"""
    if not training:
        return x

    mask = Tensor.rand(*x.shape) > p
    return mask.where(x / (1 - p), 0)
```

### Numerical Stability

#### Safe Division
```python
def safe_divide(a, b, eps=1e-8):
    """Prevent division by zero"""
    return a / (b.abs() < eps).where(eps, b)

a = Tensor([1, 2, 3])
b = Tensor([0, 2, 0])
result = safe_divide(a, b)
# Result: [1e8, 1, 3e8]
```

#### Safe Log
```python
def safe_log(x, eps=1e-8):
    """Prevent log(0)"""
    return (x < eps).where(eps, x).log()
```

#### Safe Sqrt
```python
def safe_sqrt(x, eps=1e-8):
    """Prevent sqrt(negative)"""
    return (x < 0).where(0, x).sqrt()
```

## MULACC: Deep Dive

### Basic Usage

```python
# Fused multiply-add
a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])
c = Tensor([10, 20, 30])

result = a.mulacc(b, c)
# Result: [14, 30, 48]
# = [1*4+10, 2*5+20, 3*6+30]
```

### Why MULACC Matters

#### Performance
```python
# Slower: two operations
result = a * b + c

# Faster: single fused operation
result = a.mulacc(b, c)
```

#### Accuracy
```python
# MULACC has better numerical precision
# No intermediate rounding between multiply and add
```

### Use Cases

#### Polynomial Evaluation
```python
def polynomial(x, coeffs):
    """Evaluate polynomial using Horner's method"""
    # p(x) = a₀ + a₁x + a₂x² + a₃x³
    result = coeffs[-1]
    for coeff in reversed(coeffs[:-1]):
        result = x.mulacc(result, coeff)
    return result

x = Tensor([1, 2, 3])
coeffs = [1, 2, 3, 4]  # 1 + 2x + 3x² + 4x³
y = polynomial(x, coeffs)
```

#### Weighted Sum
```python
def weighted_sum(values, weights, bias):
    """Compute weighted sum with bias"""
    return values.mulacc(weights, bias)

# Linear layer (simplified)
def linear(x, weight, bias):
    return x.mulacc(weight, bias)
```

## Visual Examples

### WHERE Operation
```
Condition: [T, F, T, F]
If True:   [1, 2, 3, 4]
If False:  [5, 6, 7, 8]
           ↓  ↓  ↓  ↓
         [1, 6, 3, 8]

Visualization:
T → pick from if_true
F → pick from if_false
```

### MULACC Operation
```
A: [1, 2, 3]
B: [4, 5, 6]
C: [10, 20, 30]

Step 1 (multiply):
A * B = [4, 10, 18]

Step 2 (add):
[4, 10, 18] + [10, 20, 30] = [14, 30, 48]

But MULACC does both in ONE operation!
```

### Chained WHERE
```
x = [-2, -1, 0, 1, 2]

Step 1: x < 0
Condition: [T, T, F, F, F]
Result:    [-1, -1, ?, ?, ?]

Step 2: x == 0 (for remaining)
Condition: [F, F, T, F, F]
Result:    [-1, -1, 0, ?, ?]

Step 3: else
Result:    [-1, -1, 0, 1, 1]
```

## Code Examples

### Example 1: Piecewise Function
```python
def piecewise_linear(x):
    """
    f(x) = -x     if x < -1
           0      if -1 <= x <= 1
           x      if x > 1
    """
    return (x < -1).where(-x,
           (x > 1).where(x, 0))

x = Tensor([-2, -1, 0, 1, 2])
y = piecewise_linear(x)
# Result: [2, 0, 0, 0, 2]
```

### Example 2: Smooth Clipping
```python
def smooth_clip(x, min_val, max_val, smoothness=0.1):
    """Smooth clipping with soft boundaries"""
    # Use sigmoid for smooth transition
    lower = (x - min_val) / smoothness
    upper = (max_val - x) / smoothness

    return (x < min_val).where(
        min_val + smoothness * lower.sigmoid(),
        (x > max_val).where(
            max_val - smoothness * upper.sigmoid(),
            x
        )
    )
```

### Example 3: Gumbel-Softmax
```python
def gumbel_softmax(logits, temperature=1.0):
    """Differentiable sampling"""
    # Add Gumbel noise
    gumbel = -(-Tensor.rand(*logits.shape).log()).log()
    y = (logits + gumbel) / temperature
    return y.softmax()
```

### Example 4: Focal Loss
```python
def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """Focal loss for imbalanced classification"""
    ce = -(target * pred.log() + (1 - target) * (1 - pred).log())
    p_t = target.where(pred, 1 - pred)
    alpha_t = target.where(alpha, 1 - alpha)
    focal_weight = (1 - p_t) ** gamma
    return (alpha_t * focal_weight * ce).mean()
```

## Performance Considerations

### WHERE Performance
```python
# Fast: simple condition
result = (x > 0).where(x, 0)

# Slower: complex nested conditions
result = (x < -1).where(-1,
         (x > 1).where(1,
         (x == 0).where(0, x)))

# Better: use clip operations when possible
result = x.clip(-1, 1)
```

### MULACC Performance
```python
# Slower: separate operations
y = a * b + c

# Faster: fused operation
y = a.mulacc(b, c)

# Even better: let tinygrad fuse automatically
y = a * b + c  # Tinygrad will optimize this!
```

## Debugging Tips

### Visualize WHERE Logic
```python
x = Tensor([-2, -1, 0, 1, 2])
condition = x > 0

print("Input:", x.numpy())
print("Condition:", condition.numpy())
print("Result:", condition.where(x, 0).numpy())
```

### Check MULACC Equivalence
```python
a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])
c = Tensor([10, 20, 30])

# These should be identical
result1 = a * b + c
result2 = a.mulacc(b, c)

print("Difference:", (result1 - result2).abs().max().numpy())
# Should be 0 (or very close)
```

## Key Takeaways

1. **TernaryOps operate on three tensors**
2. **WHERE** is the most important - conditional selection
3. **MULACC** is fused multiply-add for performance
4. **WHERE enables**: masking, clipping, piecewise functions
5. **MULACC enables**: efficient linear operations, polynomials
6. **Broadcasting works** with all three inputs
7. **Composable** with other operation types

## Next Steps

- Combine with [UnaryOps](unary-ops.md) and [BinaryOps](binary-ops.md)
- Learn about [ReduceOps](../02-reduce-ops.md) for aggregation
- Explore [MovementOps](../03-movement-ops.md) for reshaping
- Study how these compose into neural network layers

## Summary Table

| Operation | Inputs | Purpose | Example |
|-----------|--------|---------|---------|
| WHERE | 3 (cond, true, false) | Conditional selection | ReLU, masking, clipping |
| MULACC | 3 (a, b, c) | Fused multiply-add | Linear layers, polynomials |

Both operations support full broadcasting across all inputs!
