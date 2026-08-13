# ElementwiseOps: Complete Guide

ElementwiseOps work **element-by-element** on tensors. Each element is processed independently — perfect for GPU parallelization. There are three subtypes: Unary (1 input), Binary (2 inputs), Ternary (3 inputs).

---

## Overview

```
UnaryOp:    [a, b, c, d]           → [f(a), f(b), f(c), f(d)]
BinaryOp:   [a,b,c,d] + [e,f,g,h] → [a+e, b+f, c+g, d+h]
TernaryOp:  [T,F,T,F].where([a,b,c,d],[e,f,g,h]) → [a, f, c, h]
```

**Key property:** Multiple elementwise ops are automatically fused into a single GPU kernel:
```python
y = ((x + 1) * 2 - 0.5).relu()  # → 1 kernel, not 4
```

---

## Part 1: UnaryOps (7 primitives)

UnaryOps apply a function to each element of a single tensor.

### Primitive UnaryOps

| Op | Code | Math | Example |
|----|------|------|---------|
| EXP2 | `x.exp2()` | 2^x | `[0,1,2]` → `[1,2,4]` |
| LOG2 | `x.log2()` | log₂(x) | `[1,2,4]` → `[0,1,2]` |
| SQRT | `x.sqrt()` | √x | `[1,4,9]` → `[1,2,3]` |
| RECIP | `x.reciprocal()` | 1/x | `[1,2,4]` → `[1,0.5,0.25]` |
| NEG | `-x` | -x | `[1,-2]` → `[-1,2]` |
| SIN | `x.sin()` | sin(x) | `[0,π/2]` → `[0,1]` |
| CAST | `x.cast(dtype)` | type(x) | `[1.5]` → `[1]` (int) |

### Code examples

```python
from tinygrad import Tensor, dtypes

x = Tensor([0, 1, 2, 3])
print(x.exp2().numpy())        # [1, 2, 4, 8]
print(x.log2().numpy())        # requires x > 0

x = Tensor([1, 4, 9, 16], dtype=dtypes.float32)
print(x.sqrt().numpy())        # [1, 2, 3, 4]
print(x.reciprocal().numpy())  # [1, 0.25, 0.111, 0.0625]

x = Tensor([1.5, 2.7, 3.9])
print(x.cast(dtypes.int32).numpy())   # [1, 2, 3]
print(x.cast(dtypes.float16).numpy()) # half precision
```

### Derived UnaryOps (built from primitives)

| Op | Code | Built From |
|----|------|------------|
| EXP | `x.exp()` | `(x * log₂(e)).exp2()` |
| LOG | `x.log()` | `x.log2() * ln(2)` |
| ABS | `x.abs()` | `x.maximum(-x)` |
| RELU | `x.relu()` | `x.maximum(0)` |
| SIGMOID | `x.sigmoid()` | `(1 + (-x).exp()).reciprocal()` |
| TANH | `x.tanh()` | `2 * (2*x).sigmoid() - 1` |

```python
x = Tensor([-2, -1, 0, 1, 2])
print(x.relu().numpy())     # [0, 0, 0, 1, 2]
print(x.sigmoid().numpy())  # [0.119, 0.268, 0.5, 0.731, 0.881]
print(x.tanh().numpy())     # [-0.964, -0.762, 0, 0.762, 0.964]
```

### Performance

- **Fast**: NEG, CAST, RECIP (simple hardware ops)
- **Medium**: SQRT, EXP2, LOG2 (one instruction)
- **Slower**: SIN, EXP, TANH (derived, multiple ops)

---

## Part 2: BinaryOps (7 primitives)

BinaryOps combine two tensors element-by-element with automatic broadcasting.

### Primitive BinaryOps

| Op | Code | Math | Example |
|----|------|------|---------|
| ADD | `a + b` | a + b | `[1,2] + [3,4]` → `[4,6]` |
| SUB | `a - b` | a - b | `[5,6] - [1,2]` → `[4,4]` |
| MUL | `a * b` | a × b | `[2,3] * [4,5]` → `[8,15]` |
| DIV | `a / b` | a ÷ b | `[10,20] / [2,4]` → `[5,5]` |
| MOD | `a % b` | a mod b | `[10,11] % [3,3]` → `[1,2]` |
| MAX | `a.maximum(b)` | max(a,b) | `[1,5] max [4,2]` → `[4,5]` |
| CMPLT | `a < b` | a < b | `[1,3] < [2,2]` → `[T,F]` |

### Code examples

```python
from tinygrad import Tensor

a = Tensor([1, 2, 3, 4])
b = Tensor([5, 6, 7, 8])
print((a + b).numpy())          # [6, 8, 10, 12]
print((a * b).numpy())          # [5, 12, 21, 32]
print(a.maximum(b).numpy())     # [5, 6, 7, 8]
print((a < b).numpy())          # [1, 1, 1, 1]
```

### Derived BinaryOps

| Op | Code | Built From |
|----|------|------------|
| GT | `a > b` | `b < a` |
| LE | `a <= b` | `~(a > b)` |
| GE | `a >= b` | `~(a < b)` |
| EQ | `a == b` | `(a <= b) & (a >= b)` |
| NE | `a != b` | `~(a == b)` |
| MIN | `a.minimum(b)` | `-((-a).maximum(-b))` |
| POW | `a ** b` | `(a.log() * b).exp()` |

### Broadcasting rules

```python
# Scalar: broadcasts to all elements
[1,2,3] + 10 = [11,12,13]

# Vector: broadcasts along missing dims
a = Tensor([[1,2,3],[4,5,6]])  # (2,3)
b = Tensor([10,20,30])          # (3,) → broadcasts to (2,3)
print((a + b).numpy())
# [[11,22,33],[14,25,36]]

# Two 1-element dims: outer product style
a = Tensor([[1],[2],[3]])  # (3,1)
b = Tensor([10,20,30,40]) # (4,)
print((a + b).numpy())    # (3,4)
```

### Common uses

```python
# ReLU
def relu(x): return x.maximum(0)

# Leaky ReLU
def leaky_relu(x, alpha=0.01): return x.maximum(alpha * x)

# Residual connection
y = x + residual

# Dropout mask
mask = Tensor.rand(*x.shape) > p
out = mask * x / (1 - p)

# MSE loss
loss = ((pred - target) ** 2).mean()
```

---

## Part 3: TernaryOps (2 primitives)

TernaryOps take three tensor inputs.

### WHERE — Conditional Selection

```python
result[i] = condition[i] ? if_true[i] : if_false[i]
```

```python
from tinygrad import Tensor

condition = Tensor([True, False, True, False])
if_true   = Tensor([1, 2, 3, 4])
if_false  = Tensor([5, 6, 7, 8])

result = condition.where(if_true, if_false)
# Result: [1, 6, 3, 8]
```

**Common patterns:**

```python
# ReLU via WHERE
x.relu()  # equivalent to (x > 0).where(x, 0)

# Causal attention mask
scores = (x < 0).where(Tensor.full(scores.shape, float('-inf')), scores)

# Clipping
x_clipped = (x < min_val).where(min_val, (x > max_val).where(max_val, x))

# Dropout
mask = (Tensor.rand(*x.shape) > p)
out = mask.where(x / (1-p), 0)

# Chained (piecewise function)
result = (x < -1).where(-1, (x > 1).where(1, x))  # clamp to [-1, 1]
```

### MULACC — Fused Multiply-Add

```python
result[i] = a[i] * b[i] + c[i]  # single hardware instruction
```

```python
a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])
c = Tensor([10, 20, 30])
result = a.mulacc(b, c)
# [14, 30, 48]  — faster and more numerically accurate than a*b + c
```

**Uses:** polynomial evaluation (Horner's method), linear layers, weighted sums.

---

## Quick Reference

### Activation Functions

```python
relu(x)     = x.maximum(0)
sigmoid(x)  = (1 + (-x).exp()).reciprocal()
tanh(x)     = 2 * (2*x).sigmoid() - 1
swish(x)    = x * x.sigmoid()
gelu(x)     = 0.5*x*(1 + (x * 0.7979 * (1 + 0.044715 * x * x)).tanh())
leaky(x)    = x.maximum(0.01 * x)
hard_sig(x) = (x < -2.5).where(0, (x > 2.5).where(1, 0.2*x + 0.5))
```

### Normalization

```python
z_score  = (x - x.mean()) / x.std()
min_max  = (x - x.min()) / (x.max() - x.min())
layer_norm = (x - x.mean(-1, True)) / (x.var(-1, True) + 1e-5).sqrt()
```

### Loss Functions

```python
mse = ((pred - target) ** 2).mean()
mae = (pred - target).abs().mean()
bce = -(target * pred.log() + (1-target) * (1-pred).log()).mean()
```

### Clipping and Masking

```python
x.maximum(min_val).minimum(max_val)   # clip
mask.where(x, 0)                       # apply binary mask
mask.where(scores, float('-inf'))      # attention masking
```

### Performance Tips

| Prefer | Instead of |
|--------|-----------|
| `x * (1/scale)` | `x / scale` (MUL faster than DIV) |
| Chain ops, single `.realize()` | Intermediate `.realize()` calls |
| `a.mulacc(b, c)` | `a * b + c` (avoids intermediate allocation) |
| `x.maximum(0)` | `(x > 0).where(x, 0)` (same result, simpler) |

### All 16 Primitives at a Glance

```
UnaryOps  (7): EXP2  LOG2  SQRT  RECIP  NEG  SIN  CAST
BinaryOps (7): ADD   SUB   MUL   DIV    MOD  MAX  CMPLT
TernaryOps(2): WHERE  MULACC
```

These 16 primitives, combined with ReduceOps and MovementOps, build all of deep learning.
