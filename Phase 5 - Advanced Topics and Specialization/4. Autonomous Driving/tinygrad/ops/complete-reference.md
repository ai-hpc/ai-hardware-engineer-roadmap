# Complete Tinygrad Operations Reference

## All Three Operation Types

Tinygrad has **only 3 types of operations** that compose into everything else!

## 1. ElementwiseOps (Element-by-Element)

### UnaryOps (1 input)
| Op | Code | Math | Example |
|----|------|------|---------|
| EXP2 | `x.exp2()` | 2^x | `[0,1,2]` → `[1,2,4]` |
| LOG2 | `x.log2()` | log₂(x) | `[1,2,4]` → `[0,1,2]` |
| SQRT | `x.sqrt()` | √x | `[1,4,9]` → `[1,2,3]` |
| RECIP | `x.reciprocal()` | 1/x | `[1,2,4]` → `[1,0.5,0.25]` |
| NEG | `-x` | -x | `[1,-2]` → `[-1,2]` |
| SIN | `x.sin()` | sin(x) | `[0,π/2]` → `[0,1]` |
| CAST | `x.cast(dtype)` | type(x) | `[1.5]` → `[1]` (int) |

### BinaryOps (2 inputs)
| Op | Code | Math | Example |
|----|------|------|---------|
| ADD | `a + b` | a + b | `[1,2] + [3,4]` → `[4,6]` |
| SUB | `a - b` | a - b | `[5,6] - [1,2]` → `[4,4]` |
| MUL | `a * b` | a × b | `[2,3] * [4,5]` → `[8,15]` |
| DIV | `a / b` | a ÷ b | `[10,20] / [2,4]` → `[5,5]` |
| MOD | `a % b` | a mod b | `[10,11] % [3,3]` → `[1,2]` |
| MAX | `a.maximum(b)` | max(a,b) | `[1,5] max [4,2]` → `[4,5]` |
| CMPLT | `a < b` | a < b | `[1,3] < [2,2]` → `[T,F]` |

### TernaryOps (3 inputs)
| Op | Code | Math | Example |
|----|------|------|---------|
| WHERE | `c.where(a,b)` | c ? a : b | `[T,F].where([1,2],[3,4])` → `[1,4]` |
| MULACC | `a.mulacc(b,c)` | a×b + c | `[1,2].mulacc([3,4],[5,6])` → `[8,14]` |

## 2. ReduceOps (Dimension Reduction)

| Op | Code | Math | Example |
|----|------|------|---------|
| SUM | `x.sum(axis)` | Σx | `[1,2,3,4]` → `10` |
| MAX | `x.max(axis)` | max(x) | `[1,5,3,2]` → `5` |

### Derived ReduceOps
| Op | Built From | Example |
|----|------------|---------|
| MEAN | `x.sum() / x.numel()` | `[1,2,3,4]` → `2.5` |
| MIN | `-(-x).max()` | `[1,5,3,2]` → `1` |
| VAR | `((x - mean)**2).mean()` | Variance |
| STD | `var.sqrt()` | Standard deviation |
| PROD | `x.log().sum().exp()` | Product |

## 3. MovementOps (Zero-Copy Reshaping)

| Op | Code | Purpose | Zero-Copy |
|----|------|---------|-----------|
| RESHAPE | `x.reshape(shape)` | Change shape | ✅ |
| PERMUTE | `x.permute(dims)` | Reorder dimensions | ✅ |
| EXPAND | `x.expand(shape)` | Broadcast | ✅ |
| SHRINK | `x[slice]` | Extract region | ✅ |
| PAD | `x.pad(padding)` | Add padding | ❌ |
| FLIP | `x.flip(axis)` | Reverse | ✅ |
| STRIDE | `x[::n]` | Skip elements | ✅ |

## How Complex Operations Are Built

### Activation Functions
```python
# All from ElementwiseOps!
relu(x)      = x.maximum(0)                    # BinaryOp
sigmoid(x)   = (1 + (-x).exp()).reciprocal()   # UnaryOps
tanh(x)      = 2 * (2*x).sigmoid() - 1         # UnaryOps + BinaryOps
gelu(x)      = 0.5*x*(1+(x*0.7979*(1+0.044715*x*x)).tanh())
```

### Normalization
```python
# ElementwiseOps + ReduceOps
layer_norm(x) = (x - x.mean()) / x.std()       # BinaryOps + ReduceOps
batch_norm(x) = (x - mean) / sqrt(var + eps)   # BinaryOps + ReduceOps
```

### Pooling
```python
# MovementOps + ReduceOps
avg_pool(x) = x.reshape(...).mean(axis=...)    # RESHAPE + SUM
max_pool(x) = x.reshape(...).max(axis=...)     # RESHAPE + MAX
```

### Matrix Multiplication
```python
# MovementOps + ElementwiseOps + ReduceOps
matmul(a, b) = (a.reshape(...) * b.reshape(...)).sum(axis=...)
# RESHAPE + EXPAND + MUL + SUM
```

### Convolution
```python
# MovementOps + ElementwiseOps + ReduceOps
conv2d(x, w) = im2col(x) @ w.reshape(...)
# RESHAPE + PERMUTE + EXPAND + MUL + SUM
```

### Attention
```python
# All three types!
attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
# PERMUTE + matmul + DIV + softmax + matmul
# Uses: MovementOps, ElementwiseOps, ReduceOps
```

## Operation Composition Examples

### Example 1: Softmax
```python
def softmax(x, axis=-1):
    # 1. ReduceOp: find max
    max_x = x.max(axis=axis, keepdim=True)

    # 2. BinaryOp: subtract
    # 3. UnaryOp: exp
    exp_x = (x - max_x).exp()

    # 4. ReduceOp: sum
    sum_exp = exp_x.sum(axis=axis, keepdim=True)

    # 5. BinaryOp: divide
    return exp_x / sum_exp

# Uses: MAX, SUB, EXP, SUM, DIV
```

### Example 2: Batch Normalization
```python
def batch_norm(x, eps=1e-5):
    # x: (batch, channels, height, width)

    # 1. ReduceOps: compute statistics
    mean = x.mean(axis=(0, 2, 3), keepdim=True)
    var = x.var(axis=(0, 2, 3), keepdim=True)

    # 2. BinaryOps: normalize
    # 3. UnaryOp: sqrt
    return (x - mean) / (var + eps).sqrt()

# Uses: SUM (for mean/var), SUB, DIV, SQRT, ADD
```

### Example 3: Convolution (Simplified)
```python
def conv2d(x, weight):
    # x: (batch, in_channels, h, w)
    # weight: (out_channels, in_channels, kh, kw)

    # 1. MovementOps: reshape for matmul
    x_col = im2col(x)  # RESHAPE + PERMUTE
    w_row = weight.reshape(weight.shape[0], -1)  # RESHAPE

    # 2. ElementwiseOps + ReduceOps: matmul
    out = x_col @ w_row.T  # MUL + SUM

    # 3. MovementOps: reshape output
    return out.reshape(...)  # RESHAPE

# Uses: RESHAPE, PERMUTE, MUL, SUM
```

### Example 4: Multi-Head Attention
```python
def multi_head_attention(Q, K, V, num_heads):
    batch, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    # 1. MovementOps: split into heads
    Q = Q.reshape(batch, seq_len, num_heads, d_k)
    Q = Q.permute(0, 2, 1, 3)  # (batch, heads, seq, d_k)

    # Same for K, V
    K = K.reshape(batch, seq_len, num_heads, d_k).permute(0, 2, 1, 3)
    V = V.reshape(batch, seq_len, num_heads, d_k).permute(0, 2, 1, 3)

    # 2. ElementwiseOps + ReduceOps: attention scores
    scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)
    # PERMUTE + MUL + SUM + DIV

    # 3. Softmax (ReduceOps + ElementwiseOps)
    attn = scores.softmax(axis=-1)

    # 4. Apply attention
    out = attn @ V  # MUL + SUM

    # 5. MovementOps: concatenate heads
    out = out.permute(0, 2, 1, 3)
    out = out.reshape(batch, seq_len, d_model)

    return out

# Uses: RESHAPE, PERMUTE, MUL, SUM, DIV, MAX, EXP
```

## Performance Characteristics

### Fast ⚡
- **UnaryOps**: NEG, CAST
- **BinaryOps**: ADD, MUL, MAX
- **MovementOps**: All (zero-copy!)
- **Fused operations**: Automatic kernel fusion

### Medium ⚠️
- **UnaryOps**: SQRT, EXP2, LOG2
- **BinaryOps**: DIV, MOD
- **ReduceOps**: SUM, MAX (depends on size)

### Slower 🐌
- **UnaryOps**: SIN (transcendental)
- **Complex compositions**: Multiple unfused ops
- **Large reductions**: Across many dimensions

## Memory Efficiency

### Zero-Copy Operations
```python
# All MovementOps (except PAD) are zero-copy
x = Tensor.randn(1000, 1000)
y = x.reshape(100, 10000)      # FREE
z = y.transpose()               # FREE
w = z.expand(2, 10000, 100)    # FREE
```

### Kernel Fusion
```python
# Tinygrad automatically fuses ElementwiseOps
y = ((x + 1) * 2 - 0.5).relu()
# Fused into ONE kernel!
# No intermediate memory allocations
```

### Memory Layout
```python
# Contiguous memory is faster
x = Tensor.randn(100, 100)
y = x.transpose()  # Not contiguous
z = y.contiguous()  # Make contiguous for speed
```

## Debugging Checklist

### Check Operation Types
```python
import os
os.environ['DEBUG'] = '3'

x = Tensor([1, 2, 3])
y = (x + 1) * 2  # ElementwiseOps
z = y.sum()      # ReduceOp
w = z.reshape(1) # MovementOp

w.realize()  # See all operations and fusion
```

### Verify Shapes
```python
x = Tensor.randn(2, 3, 4)
print(f"Original: {x.shape}")

y = x.sum(axis=1)
print(f"After reduce: {y.shape}")  # (2, 4)

z = y.reshape(8)
print(f"After reshape: {z.shape}")  # (8,)
```

### Check Zero-Copy
```python
x = Tensor([1, 2, 3, 4])
y = x.reshape(2, 2)

# Movement ops are zero-copy until realize()
print(f"Shape: {y.shape}")
print(f"Dtype: {y.dtype}")
# Data not copied until y.realize()
```

## Quick Reference by Use Case

| Need | Use | Operations |
|------|-----|------------|
| Activation | ReLU, Sigmoid, Tanh | UnaryOps + BinaryOps |
| Normalization | BatchNorm, LayerNorm | ReduceOps + BinaryOps |
| Pooling | MaxPool, AvgPool | MovementOps + ReduceOps |
| Convolution | Conv2D | All three types |
| Attention | Multi-head attention | All three types |
| Loss | MSE, CrossEntropy | BinaryOps + ReduceOps |
| Reshape | Flatten, Transpose | MovementOps |

## The Big Picture

```
┌─────────────────────────────────────────────────┐
│         TINYGRAD: 3 OPERATION TYPES             │
├─────────────────────────────────────────────────┤
│                                                 │
│  1. ElementwiseOps (element-by-element)         │
│     • UnaryOps (7 primitives)                   │
│     • BinaryOps (7 primitives)                  │
│     • TernaryOps (2 primitives)                 │
│                                                 │
│  2. ReduceOps (dimension reduction)             │
│     • SUM, MAX (2 primitives)                   │
│                                                 │
│  3. MovementOps (zero-copy reshaping)           │
│     • RESHAPE, PERMUTE, EXPAND, etc.            │
│                                                 │
├─────────────────────────────────────────────────┤
│              COMPOSE INTO                       │
├─────────────────────────────────────────────────┤
│                                                 │
│  • Activations (ReLU, Sigmoid, GELU)            │
│  • Normalization (BatchNorm, LayerNorm)         │
│  • Pooling (MaxPool, AvgPool)                   │
│  • Convolution (Conv2D, Conv3D)                 │
│  • Matrix Multiplication (MatMul)               │
│  • Attention (Multi-head, Self-attention)       │
│  • Loss Functions (MSE, CrossEntropy)           │
│  • Everything else!                             │
│                                                 │
└─────────────────────────────────────────────────┘
```

## Summary

- **16 primitive operations** total (7 Unary + 7 Binary + 2 Ternary)
- **2 reduce operations** (SUM, MAX)
- **7 movement operations** (mostly zero-copy)
- **Everything else is composed** from these primitives
- **Automatic kernel fusion** for performance
- **Lazy evaluation** for optimization
- **ShapeTracker** for zero-copy transformations

This is the genius of tinygrad: **extreme simplicity that composes into complexity**!

## Next Steps

1. Master each operation type individually
2. Understand how they compose
3. Build neural network layers
4. Optimize performance
5. Contribute to tinygrad!

📖 See detailed guides:
- [ElementwiseOps](01-elementwise-ops.md)
- [ReduceOps](02-reduce-ops.md)
- [MovementOps](03-movement-ops.md)
