# ReduceOps: Dimension Reduction

ReduceOps aggregate values along one or more dimensions, returning a smaller tensor. Tinygrad has **2 primitive ReduceOps** — SUM and MAX — from which all others derive.

---

## The Two Primitives

### SUM

```python
from tinygrad import Tensor

x = Tensor([[1, 2, 3],
            [4, 5, 6]])

x.sum()         # 21  — reduce all
x.sum(axis=0)   # [5, 7, 9]  — reduce rows
x.sum(axis=1)   # [6, 15]    — reduce columns
x.sum(axis=1, keepdim=True)  # [[6], [15]]  — keep dims for broadcasting
```

```
axis=0 (sum columns):       axis=1 (sum rows):
[[1,2,3],                   [[1+2+3] = [6]
 [4,5,6]]                    [4+5+6] = [15]]
  ↓↓↓
[5,7,9]
```

### MAX

```python
x = Tensor([[1, 5, 3],
            [4, 2, 6]])

x.max()         # 6
x.max(axis=0)   # [4, 5, 6]
x.max(axis=1)   # [5, 6]
```

---

## Derived Reductions

| Op | Built From | Example |
|----|------------|---------|
| `x.mean()` | `x.sum() / x.numel()` | `[1,2,3,4]` → `2.5` |
| `x.min()` | `-(-x).max()` | `[1,5,3,2]` → `1` |
| `x.var()` | `((x - mean)**2).mean()` | variance |
| `x.std()` | `x.var().sqrt()` | standard deviation |
| `x.prod()` | `x.log().sum().exp()` | product |

---

## Axes and keepdim

```python
x = Tensor.randn(2, 3, 4)

x.sum(axis=0)              # (3, 4)
x.sum(axis=1)              # (2, 4)
x.sum(axis=2)              # (2, 3)
x.sum()                    # scalar

x.sum(axis=1, keepdim=True)  # (2, 1, 4)  ← useful for broadcasting
```

`keepdim=True` preserves the reduced dimension as size 1, making subsequent broadcasting straightforward.

---

## Common Patterns

### Pooling

```python
# Average pooling (2D)
def avg_pool2d(x, k=2):
    b, c, h, w = x.shape
    return x.reshape(b, c, h//k, k, w//k, k).mean(axis=(3, 5))

# Max pooling (2D)
def max_pool2d(x, k=2):
    b, c, h, w = x.shape
    return x.reshape(b, c, h//k, k, w//k, k).max(axis=(3, 5))

# Global average pooling
def gap(x):  # (B,C,H,W) → (B,C)
    return x.mean(axis=(2, 3))
```

### Normalization

```python
# Layer norm
def layer_norm(x, eps=1e-5):
    mean = x.mean(axis=-1, keepdim=True)
    var = ((x - mean)**2).mean(axis=-1, keepdim=True)
    return (x - mean) / (var + eps).sqrt()

# Batch norm (simplified)
def batch_norm(x, eps=1e-5):
    mean = x.mean(axis=(0, 2, 3), keepdim=True)
    var = x.var(axis=(0, 2, 3), keepdim=True)
    return (x - mean) / (var + eps).sqrt()

# RMS norm
def rms_norm(x, eps=1e-6):
    rms = (x * x).mean(axis=-1, keepdim=True).add(eps).sqrt()
    return x / rms
```

### Attention

```python
# Numerically stable softmax
def softmax(x, axis=-1):
    max_x = x.max(axis=axis, keepdim=True)
    exp_x = (x - max_x).exp()
    return exp_x / exp_x.sum(axis=axis, keepdim=True)

# Log-sum-exp
def logsumexp(x, axis=-1):
    max_x = x.max(axis=axis, keepdim=True)
    return max_x.squeeze(axis) + (x - max_x).exp().sum(axis=axis).log()
```

### Loss Functions

```python
mse = ((pred - target)**2).mean()
mae = (pred - target).abs().mean()

def cross_entropy(logits, targets):
    log_probs = logits.log_softmax(axis=-1)
    return -(targets * log_probs).sum(axis=-1).mean()
```

---

## Numerical Stability

```python
# Unstable softmax (overflow for large x)
def softmax_bad(x):
    return x.exp() / x.exp().sum(axis=-1, keepdim=True)

# Stable (subtract max first)
def softmax_stable(x, axis=-1):
    x = x - x.max(axis=axis, keepdim=True)
    exp_x = x.exp()
    return exp_x / exp_x.sum(axis=axis, keepdim=True)

# Always add eps before log/sqrt
safe_log  = (x + 1e-8).log()
safe_sqrt = (x.maximum(0) + 1e-8).sqrt()
```

---

## Performance Notes

- ReduceOps break **fusion boundaries** — elementwise ops before a reduce fuse together, and elementwise ops after a reduce start a new kernel
- Large reductions across many elements are slower; prefer reducing the last (innermost) axis for better cache locality
- `keepdim=True` avoids a reshape and is slightly more efficient when you need to broadcast back

---

## Key Takeaways

1. **2 primitives**: SUM and MAX
2. All other reductions (MEAN, MIN, VAR, STD, PROD) are derived
3. `axis` selects which dimension to collapse; `keepdim` preserves shape for broadcasting
4. Reductions break kernel fusion — each reduce produces a separate kernel
5. Numerical stability requires subtracting max before exponentiation in softmax/logsumexp
