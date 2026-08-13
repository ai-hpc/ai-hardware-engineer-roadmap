# Tinygrad Operations Reference

Tinygrad builds **all of deep learning** from exactly 3 operation types and 25 primitive ops. No CONV or MATMUL primitives — they're composed from the basics.

---

## The Three Types

### 1. ElementwiseOps — element-by-element

| Subtype | Primitives | Examples |
|---------|------------|---------|
| UnaryOps (1 input) | 7: EXP2, LOG2, SQRT, RECIP, NEG, SIN, CAST | `x.relu()`, `x.sigmoid()`, `x.tanh()` |
| BinaryOps (2 inputs) | 7: ADD, SUB, MUL, DIV, MOD, MAX, CMPLT | `a + b`, `a * b`, `a.maximum(b)` |
| TernaryOps (3 inputs) | 2: WHERE, MULACC | `cond.where(a, b)`, `a.mulacc(b, c)` |

Key property: multiple elementwise ops fuse automatically into **one GPU kernel**.

### 2. ReduceOps — collapse a dimension

| Primitive | Code | Example |
|-----------|------|---------|
| SUM | `x.sum(axis)` | `[1,2,3,4]` → `10` |
| MAX | `x.max(axis)` | `[1,5,3,2]` → `5` |

Derived: MEAN, MIN, VAR, STD, PROD — all built from SUM and MAX.

Reductions **break fusion boundaries**: ops before a reduce fuse together, then a new kernel starts after.

### 3. MovementOps — zero-copy reshaping

| Op | Code | Zero-Copy |
|----|------|-----------|
| RESHAPE | `x.reshape(shape)` | ✅ |
| PERMUTE | `x.permute(dims)` | ✅ |
| EXPAND | `x.expand(shape)` | ✅ |
| SHRINK | `x[slice]` | ✅ |
| FLIP | `x.flip(axis)` | ✅ |
| STRIDE | `x[::n]` | ✅ |
| PAD | `x.pad(padding)` | ❌ |

ShapeTracker tracks strides/offsets so no data moves until `.realize()`.

---

## How Complex Ops Decompose

```
MATMUL(A, B) = RESHAPE + EXPAND + MUL + SUM
CONV2D       = RESHAPE + PERMUTE + MUL + SUM
SOFTMAX      = MAX + SUB + EXP + SUM + DIV
LAYERNORM    = SUM (mean) + SUB + POW + SUM (var) + SQRT + DIV
ATTENTION    = PERMUTE + MATMUL + DIV + SOFTMAX + MATMUL
```

### Activation Functions

```python
relu(x)    = x.maximum(0)                          # 1 BinaryOp
sigmoid(x) = (1 + (-x).exp()).reciprocal()         # 3 UnaryOps
tanh(x)    = 2 * (2*x).sigmoid() - 1              # composed
swish(x)   = x * x.sigmoid()                       # MUL + sigmoid
gelu(x)    = 0.5*x*(1 + (x*0.7979*(1+0.044715*x*x)).tanh())
```

### Pooling

```python
def max_pool2d(x, k=2):
    b, c, h, w = x.shape
    return x.reshape(b, c, h//k, k, w//k, k).max(axis=(3, 5))
    # MovementOp + ReduceOp

def avg_pool2d(x, k=2):
    b, c, h, w = x.shape
    return x.reshape(b, c, h//k, k, w//k, k).mean(axis=(3, 5))
```

### Softmax (numerically stable)

```python
def softmax(x, axis=-1):
    max_x = x.max(axis=axis, keepdim=True)  # ReduceOp
    exp_x = (x - max_x).exp()               # BinaryOp + UnaryOp
    return exp_x / exp_x.sum(axis=axis, keepdim=True)  # ReduceOp + BinaryOp
```

---

## Kernel Fusion

```python
# These three operations...
y = x + 1
z = y * 2
w = z.relu()
# ...are compiled into ONE kernel automatically:
# w = max((x + 1) * 2, 0)
```

Reduction ops break fusion. This produces 2 kernels:
```python
# Kernel 1: x + 1 (elementwise)
# Kernel 2: sum (reduction) + divide (elementwise)
result = (x + 1).sum() / n
```

---

## Lazy Evaluation

```python
x = Tensor([1, 2, 3])
y = x + 1      # graph built, nothing computed
z = y * 2      # graph extended, nothing computed
result = z.realize()  # compiled and executed here
```

Check the schedule (kernel plan) before running:
```python
sched = z.schedule()
print(f"{len(sched)} kernel(s)")
```

---

## The Big Picture

```
16 ElementwiseOps  +  2 ReduceOps  +  7 MovementOps
                         ↓
        Activations, Normalizations, Pooling
        Convolutions, MatMul, Attention
        Loss Functions — everything in deep learning
```

---

## Detailed Guides

| Topic | File |
|-------|------|
| ElementwiseOps (Unary, Binary, Ternary) | [elementwise.md](elementwise.md) |
| ReduceOps (SUM, MAX, derived) | [reduce.md](reduce.md) |
| MovementOps (ShapeTracker, zero-copy) | [movement.md](movement.md) |
