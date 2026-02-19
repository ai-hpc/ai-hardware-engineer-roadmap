# ElementwiseOps Quick Reference

## UnaryOps (1 input)

| Operation | Code | Math | Example |
|-----------|------|------|---------|
| EXP2 | `x.exp2()` | 2^x | `[0,1,2]` → `[1,2,4]` |
| LOG2 | `x.log2()` | log₂(x) | `[1,2,4]` → `[0,1,2]` |
| SQRT | `x.sqrt()` | √x | `[1,4,9]` → `[1,2,3]` |
| RECIP | `x.reciprocal()` | 1/x | `[1,2,4]` → `[1,0.5,0.25]` |
| NEG | `-x` | -x | `[1,-2,3]` → `[-1,2,-3]` |
| SIN | `x.sin()` | sin(x) | `[0,π/2,π]` → `[0,1,0]` |
| CAST | `x.cast(dtype)` | type(x) | `[1.5,2.7]` → `[1,2]` (int) |

### Derived UnaryOps

| Operation | Code | Built From |
|-----------|------|------------|
| EXP | `x.exp()` | `(x * 1.4427).exp2()` |
| LOG | `x.log()` | `x.log2() * 0.6931` |
| ABS | `x.abs()` | `x.maximum(-x)` |
| RELU | `x.relu()` | `x.maximum(0)` |
| SIGMOID | `x.sigmoid()` | `(1 + (-x).exp()).reciprocal()` |
| TANH | `x.tanh()` | `2 * (2*x).sigmoid() - 1` |

## BinaryOps (2 inputs)

| Operation | Code | Math | Example |
|-----------|------|------|---------|
| ADD | `a + b` | a + b | `[1,2] + [3,4]` → `[4,6]` |
| SUB | `a - b` | a - b | `[5,6] - [1,2]` → `[4,4]` |
| MUL | `a * b` | a × b | `[2,3] * [4,5]` → `[8,15]` |
| DIV | `a / b` | a ÷ b | `[10,20] / [2,4]` → `[5,5]` |
| MOD | `a % b` | a mod b | `[10,11] % [3,3]` → `[1,2]` |
| MAX | `a.maximum(b)` | max(a,b) | `[1,5] max [4,2]` → `[4,5]` |
| CMPLT | `a < b` | a < b | `[1,3] < [2,2]` → `[T,F]` |

### Derived BinaryOps

| Operation | Code | Built From |
|-----------|------|------------|
| GT | `a > b` | `b < a` |
| LE | `a <= b` | `~(a > b)` |
| GE | `a >= b` | `~(a < b)` |
| EQ | `a == b` | `(a <= b) & (a >= b)` |
| NE | `a != b` | `~(a == b)` |
| MIN | `a.minimum(b)` | `-((-a).maximum(-b))` |
| POW | `a ** b` | `(a.log() * b).exp()` |

## TernaryOps (3 inputs)

| Operation | Code | Math | Example |
|-----------|------|------|---------|
| WHERE | `cond.where(a, b)` | cond ? a : b | `[T,F].where([1,2],[3,4])` → `[1,4]` |
| MULACC | `a.mulacc(b, c)` | a×b + c | `[1,2].mulacc([3,4],[5,6])` → `[8,14]` |

## Broadcasting Examples

```python
# Scalar
[1,2,3] + 10 = [11,12,13]

# Vector
[[1,2,3],     [10]      [[11,12,13],
 [4,5,6]]  +  [20]  =    [24,25,26]]

# Matrix
(3,1) + (1,4) = (3,4)
```

## Common Activation Functions

```python
# ReLU
x.maximum(0)

# Leaky ReLU
x.maximum(0.01 * x)

# Sigmoid
(1 + (-x).exp()).reciprocal()

# Tanh
2 * (2*x).sigmoid() - 1

# Swish
x * x.sigmoid()

# GELU (approx)
0.5 * x * (1 + (x * 0.7979 * (1 + 0.044715 * x * x)).tanh())

# Hard Sigmoid
(x < -2.5).where(0, (x > 2.5).where(1, 0.2 * x + 0.5))
```

## Common Patterns

### Normalization
```python
# Z-score
(x - x.mean()) / x.std()

# Min-max
(x - x.min()) / (x.max() - x.min())

# Layer norm
(x - x.mean()) / (x.var() + eps).sqrt()
```

### Loss Functions
```python
# MSE
((pred - target) ** 2).mean()

# MAE
(pred - target).abs().mean()

# Binary Cross Entropy
-(target * pred.log() + (1-target) * (1-pred).log()).mean()
```

### Clipping
```python
# Clip by value
x.maximum(min_val).minimum(max_val)

# Clip by value (WHERE)
(x < min_val).where(min_val, (x > max_val).where(max_val, x))
```

### Masking
```python
# Apply mask
mask.where(x, 0)

# Attention mask
mask.where(scores, -1e9)

# Dropout
(Tensor.rand(*x.shape) > p).where(x / (1-p), 0)
```

## Performance Tips

### ✅ Fast
- Simple ops: ADD, MUL, MAX
- Fused operations (automatic)
- Aligned memory access

### ⚠️ Medium
- DIV (use MUL with reciprocal)
- Transcendental: EXP, LOG, SIN
- Complex WHERE chains

### 🚀 Optimization
```python
# Use reciprocal for division
y = x * (1/scale)  # faster than x / scale

# Let tinygrad fuse
y = (x + 1) * 2 - 0.5  # fused into one kernel

# Avoid nested WHERE
# Use clip operations when possible
```

## Debugging

```python
# Enable debug output
import os
os.environ['DEBUG'] = '3'

# Check shapes
print(f"Shape: {x.shape}")

# Check values
print(f"Values: {x.numpy()}")

# Check operation fusion
x = Tensor([1,2,3])
y = (x + 1) * 2
y.realize()  # Shows fused kernel
```

## Memory Layout

```
UnaryOp:   [a,b,c,d] → [f(a),f(b),f(c),f(d)]
           Same shape, element-wise

BinaryOp:  [a,b,c,d] + [e,f,g,h] → [a+e,b+f,c+g,d+h]
           Broadcast to same shape, element-wise

TernaryOp: [T,F,T,F].where([a,b,c,d],[e,f,g,h]) → [a,f,c,h]
           Broadcast all three, element-wise
```

## Cheat Sheet

| Need | Use |
|------|-----|
| Activation | UnaryOps + BinaryOps |
| Normalization | BinaryOps (SUB, DIV) |
| Loss | BinaryOps + ReduceOps |
| Masking | WHERE (TernaryOp) |
| Clipping | MAX/MIN or WHERE |
| Gating | MUL (BinaryOp) |
| Conditional | WHERE (TernaryOp) |
| Fused ops | MULACC (TernaryOp) |

## Next Steps

- 📖 [Detailed UnaryOps Guide](unary-ops.md)
- 📖 [Detailed BinaryOps Guide](binary-ops.md)
- 📖 [Detailed TernaryOps Guide](ternary-ops.md)
- 📖 [ReduceOps Overview](../02-reduce-ops.md)
- 📖 [MovementOps Overview](../03-movement-ops.md)
