# MovementOps: Zero-Copy Data Reorganization

MovementOps reorganize tensor data **without copying memory**. They update ShapeTracker metadata — the actual bytes stay in place until a compute op forces materialization.

---

## The Magic: ShapeTracker

```python
x = Tensor([1, 2, 3, 4, 5, 6])
# Memory: [1, 2, 3, 4, 5, 6]

y = x.reshape(2, 3)
# Memory: [1, 2, 3, 4, 5, 6]  ← UNCHANGED
# View: 2×3 matrix

z = y.transpose()
# Memory: [1, 2, 3, 4, 5, 6]  ← STILL UNCHANGED
# View: 3×2 matrix (reads in column-major order via stride)
```

---

## All MovementOps

| Op | Code | Zero-Copy | Description |
|----|------|-----------|-------------|
| RESHAPE | `x.reshape(shape)` | ✅ | Change shape, same total elements |
| PERMUTE | `x.permute(dims)` | ✅ | Reorder axes |
| EXPAND | `x.expand(shape)` | ✅ | Broadcast a dimension |
| SHRINK | `x[slice]` | ✅ | Extract a subregion |
| FLIP | `x.flip(axis)` | ✅ | Reverse along an axis |
| STRIDE | `x[::n]` | ✅ | Take every nth element |
| PAD | `x.pad(padding)` | ❌ | Adds zeros (needs new memory) |

---

## RESHAPE

```python
x = Tensor([1, 2, 3, 4, 5, 6])     # (6,)
y = x.reshape(2, 3)                  # (2, 3)
z = x.reshape(3, 2)                  # (3, 2)
w = x.reshape(6, 1)                  # (6, 1)

# Infer one dim with -1
x = Tensor.randn(24)
x.reshape(4, -1)   # (4, 6)
x.reshape(-1, 8)   # (3, 8)

# Flatten
x = Tensor.randn(2, 3, 4)
x.reshape(x.shape[0], -1)  # (2, 12)
```

### PERMUTE

```python
# Transpose (2D)
x = Tensor.randn(3, 4)
y = x.permute(1, 0)    # (4, 3) — equivalent to x.T or x.transpose()

# 4D: NCHW → NHWC
x = Tensor.randn(1, 3, 224, 224)
y = x.permute(0, 2, 3, 1)   # (1, 224, 224, 3)

# Batch matrix transpose (last two dims)
x = Tensor.randn(32, 8, 64, 64)
y = x.transpose(-2, -1)  # (32, 8, 64, 64) — swaps last 2 dims
```

### EXPAND

```python
# Broadcast a dim of size 1
x = Tensor.randn(3, 1)
y = x.expand(3, 5)   # (3, 5) — no data copied

# Add a new dim with reshape first
x = Tensor([1, 2, 3])     # (3,)
x_col = x.reshape(3, 1).expand(3, 4)  # (3, 4) outer broadcast
```

### SHRINK (slice)

```python
x = Tensor.randn(10, 10)
y = x[2:8, 3:7]     # (6, 4) — zero-copy view

# Using .shrink() directly (tinygrad internal)
y = x.shrink(((2, 8), (3, 7)))
```

### FLIP and STRIDE

```python
x = Tensor([1, 2, 3, 4, 5, 6])
x.flip(0)          # [6, 5, 4, 3, 2, 1]
x[::2]             # [1, 3, 5]  — every 2nd element

x = Tensor.randn(8, 8)
x[::2, ::2]        # (4, 4) — downsample by 2
```

### PAD (only non-zero-copy MovementOp)

```python
x = Tensor([[1, 2], [3, 4]])   # (2, 2)
y = x.pad(((1, 1), (1, 1)))   # pad 1 on each side → (4, 4)
# [[0, 0, 0, 0],
#  [0, 1, 2, 0],
#  [0, 3, 4, 0],
#  [0, 0, 0, 0]]
```

---

## Common Patterns

### Image Processing

```python
# NCHW ↔ NHWC
def nchw_to_nhwc(x): return x.permute(0, 2, 3, 1)
def nhwc_to_nchw(x): return x.permute(0, 3, 1, 2)

# Padding for convolution
def conv_pad(x, p=1):
    return x.pad(((0,0),(0,0),(p,p),(p,p)))
```

### Attention

```python
# Split into heads: (B, T, D) → (B, H, T, d)
def split_heads(x, num_heads):
    B, T, D = x.shape
    d = D // num_heads
    return x.reshape(B, T, num_heads, d).permute(0, 2, 1, 3)

# Expand attention mask: (B, T) → (B, 1, 1, T)
def expand_mask(mask):
    return mask.reshape(mask.shape[0], 1, 1, mask.shape[1])
```

### Manual Matmul from Primitives

```python
# A (M,K) @ B (K,N) via expand + mul + sum
A_exp = A.reshape(M, K, 1).expand(M, K, N)
B_exp = B.reshape(1, K, N).expand(M, K, N)
C = (A_exp * B_exp).sum(axis=1)
```

---

## Performance Notes

### Chain operations, realize once

```python
# Bad: multiple realizes, data copied each time
y = x.reshape(100, 10000).realize()
z = y.transpose().realize()

# Good: chain, single realize
result = x.reshape(100, 10000).transpose().realize()
```

### Contiguity

After `permute`, memory is non-contiguous. If subsequent operations need contiguous access, call `.contiguous()`:

```python
x = Tensor.randn(100, 100)
y = x.permute(1, 0).contiguous()  # forces a copy, but now cache-friendly
```

### Zero-copy chain example

```python
x = Tensor.randn(1000, 1000)
y = x.reshape(100, 10000)   # FREE
z = y.transpose()            # FREE
w = z.expand(2, 10000, 100) # FREE
v = w[0, :500, :50]         # FREE (shrink)
result = v.realize()         # data moves HERE, only once
```

---

## Key Takeaways

1. All MovementOps except PAD are **zero-copy** — only metadata changes
2. **ShapeTracker** stores strides/offsets to describe the view
3. Data only materializes at `.realize()` or when a compute op needs it
4. Chain movment ops freely — they compose into a single ShapeTracker view
5. `permute` makes memory non-contiguous; use `.contiguous()` when needed for perf
