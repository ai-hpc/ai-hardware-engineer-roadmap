# Tinygrad Operations Reference

This reference groups tensor mathematics into elementwise, reduction, and movement operations. The lists below are a teaching decomposition and historical names, not a complete or version-independent list of current UOps.

---

## The Three Types

### 1. ElementwiseOps — element-by-element

| Subtype | Primitives | Examples |
|---------|------------|---------|
| UnaryOps (1 input) | 7: EXP2, LOG2, SQRT, RECIP, NEG, SIN, CAST | `x.relu()`, `x.sigmoid()`, `x.tanh()` |
| BinaryOps (2 inputs) | 7: ADD, SUB, MUL, DIV, MOD, MAX, CMPLT | `a + b`, `a * b`, `a.maximum(b)` |
| TernaryOps (3 inputs) | 2: WHERE, MULACC | `cond.where(a, b)`, `a.mulacc(b, c)` |

Compatible elementwise operations may fuse; inspect the schedule for the selected shapes, backend, and revision.

### 2. ReduceOps — collapse a dimension

| Primitive | Code | Example |
|-----------|------|---------|
| SUM | `x.sum(axis)` | `[1,2,3,4]` → `10` |
| MAX | `x.max(axis)` | `[1,5,3,2]` → `5` |

Derived: MEAN, MIN, VAR, STD, PROD — all built from SUM and MAX.

Reductions impose dependencies, but do not inherently require a separate kernel. Producer/consumer fusion depends on the scheduler, shape, and resource constraints.

### 3. MovementOps — logical index transformations

| Op | Code | Zero-Copy |
|----|------|-----------|
| RESHAPE | `x.reshape(shape)` | ✅ |
| PERMUTE | `x.permute(dims)` | ✅ |
| EXPAND | `x.expand(shape)` | ✅ |
| SHRINK | `x[slice]` | ✅ |
| FLIP | `x.flip(axis)` | ✅ |
| STRIDE | `x[::n]` | ✅ |
| PAD | `x.pad(padding)` | ❌ |

These operations can be represented as index transformations. Padding can also be expressed through masked reads, while some consumers force contiguous materialization. Treat the table as conceptual view capability, not a guarantee of zero physical copies.

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

Predict a schedule, then inspect it: this expression may fuse into one reduction kernel or be split by the implementation.
```python
# Elementwise producer, reduction, and scalar consumer: kernel count is measured
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

## Practice and assessment

**Entry diagnostic (prerequisite check):** Trace a tensor expression into operations, scheduling, and generated execution. If this is unfamiliar, use the prerequisite material above or the [roadmap diagnostics](../../../../Learning-Guide.md#diagnose-before-reading) before starting the lab.

1. **Baseline lab:** Follow one matmul or reduction through the pinned tinygrad source and compare with a reference.
2. **Guided variation:** Change shape/layout and inspect the resulting schedule and kernel choices.
3. **Independent transfer:** repeat the method on a new input or configuration and explain where your original assumptions stop applying.

**Evidence to submit:** Source commit, correctness, kernel count, compile/warmup time, execution latency, and memory. Keep raw results and reproduction steps; distinguish measured results from estimates or work not run.

**Self-check:** Which observation would disprove your explanation of the result? Identify a competing explanation and the smallest experiment that separates them. A passing answer names a concrete predicted difference, not only another tool to try.

**Exit criteria:** reproduce the baseline, demonstrate the negative/variation check, and meet the [shared completion rubric](../../../../Learning-Guide.md#completion-rubric). Use the [lesson cycle](../../../../Learning-Guide.md#a-repeatable-lesson-cycle) for retrieval practice and spaced review.
