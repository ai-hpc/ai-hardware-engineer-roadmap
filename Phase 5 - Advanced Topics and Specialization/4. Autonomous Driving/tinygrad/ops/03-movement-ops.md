# MovementOps: Zero-Copy Data Reorganization

## Overview

MovementOps are **virtual operations** that reorganize tensor data without copying memory. They use **ShapeTracker** to track how data should be viewed, making them extremely efficient.

```
Input:  [1, 2, 3, 4, 5, 6]  (Shape: 6)
RESHAPE: [[1, 2, 3],
          [4, 5, 6]]        (Shape: 2x3)
NO DATA COPIED! Just metadata changed.
```

## The Magic: ShapeTracker

ShapeTracker maintains a "view" of the data without moving it:

```python
from tinygrad import Tensor

x = Tensor([1, 2, 3, 4, 5, 6])
# Data in memory: [1, 2, 3, 4, 5, 6]

y = x.reshape(2, 3)
# Data in memory: [1, 2, 3, 4, 5, 6]  (SAME!)
# View: 2x3 matrix

z = y.transpose()
# Data in memory: [1, 2, 3, 4, 5, 6]  (STILL SAME!)
# View: 3x2 matrix
```

## Complete List of MovementOps

### 1. RESHAPE - Change Shape
```python
from tinygrad import Tensor

# 1D to 2D
x = Tensor([1, 2, 3, 4, 5, 6])
y = x.reshape(2, 3)
# [[1, 2, 3],
#  [4, 5, 6]]

# 2D to 3D
x = Tensor.randn(12)
y = x.reshape(2, 3, 2)
# Shape: (2, 3, 2)

# Automatic dimension inference
x = Tensor.randn(24)
y = x.reshape(4, -1)  # -1 infers 6
# Shape: (4, 6)
```

**Rule:** Total elements must remain the same!

### 2. PERMUTE - Reorder Dimensions
```python
# Transpose (2D)
x = Tensor([[1, 2, 3],
            [4, 5, 6]])  # Shape: (2, 3)
y = x.transpose()  # or x.permute(1, 0)
# [[1, 4],
#  [2, 5],
#  [3, 6]]  # Shape: (3, 2)

# 3D permutation
x = Tensor.randn(2, 3, 4)  # (batch, height, width)
y = x.permute(0, 2, 1)     # (batch, width, height)
# Shape: (2, 4, 3)

# Channel-first to channel-last
x = Tensor.randn(1, 3, 224, 224)  # (N, C, H, W)
y = x.permute(0, 2, 3, 1)         # (N, H, W, C)
# Shape: (1, 224, 224, 3)
```

### 3. EXPAND - Broadcast Dimensions
```python
# Add dimensions by broadcasting
x = Tensor([1, 2, 3])  # Shape: (3,)
y = x.expand(4, 3)     # Shape: (4, 3)
# [[1, 2, 3],
#  [1, 2, 3],
#  [1, 2, 3],
#  [1, 2, 3]]

# Expand specific dimensions
x = Tensor([[1], [2], [3]])  # Shape: (3, 1)
y = x.expand(3, 5)            # Shape: (3, 5)
# [[1, 1, 1, 1, 1],
#  [2, 2, 2, 2, 2],
#  [3, 3, 3, 3, 3]]
```

**Note:** No data duplication! Just metadata.

### 4. SHRINK (SLICE) - Extract Subregion
```python
# Slice tensor
x = Tensor([[1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12]])  # Shape: (3, 4)

# Python slicing
y = x[1:3, 1:3]
# [[6, 7],
#  [10, 11]]  # Shape: (2, 2)

# Shrink operation (internal)
y = x.shrink(((1, 3), (1, 3)))
# Same result, zero-copy!
```

### 5. PAD - Add Padding
```python
# Add zeros around tensor
x = Tensor([[1, 2],
            [3, 4]])  # Shape: (2, 2)

y = x.pad(((1, 1), (1, 1)))  # Pad 1 on all sides
# [[0, 0, 0, 0],
#  [0, 1, 2, 0],
#  [0, 3, 4, 0],
#  [0, 0, 0, 0]]  # Shape: (4, 4)

# Asymmetric padding
y = x.pad(((0, 1), (1, 2)))  # (top, bottom), (left, right)
```

### 6. FLIP - Reverse Dimensions
```python
# Flip along axis
x = Tensor([1, 2, 3, 4])
y = x.flip(axis=0)
# [4, 3, 2, 1]

# Flip 2D
x = Tensor([[1, 2, 3],
            [4, 5, 6]])
y = x.flip(axis=0)  # Flip rows
# [[4, 5, 6],
#  [1, 2, 3]]

z = x.flip(axis=1)  # Flip columns
# [[3, 2, 1],
#  [6, 5, 4]]
```

### 7. STRIDE - Skip Elements
```python
# Take every nth element
x = Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = x[::2]  # Every 2nd element
# [0, 2, 4, 6, 8]

# 2D striding
x = Tensor.randn(8, 8)
y = x[::2, ::2]  # Downsample by 2
# Shape: (4, 4)
```

## Visual Examples

### RESHAPE
```
Input (6,):
[1, 2, 3, 4, 5, 6]

Reshape to (2, 3):
[[1, 2, 3],
 [4, 5, 6]]

Reshape to (3, 2):
[[1, 2],
 [3, 4],
 [5, 6]]

Memory layout: [1, 2, 3, 4, 5, 6] (UNCHANGED!)
```

### PERMUTE (Transpose)
```
Input (2, 3):
[[1, 2, 3],
 [4, 5, 6]]

Transpose (3, 2):
[[1, 4],
 [2, 5],
 [3, 6]]

Memory: [1, 2, 3, 4, 5, 6] (SAME!)
View changed: read in different order
```

### EXPAND
```
Input (3,):
[1, 2, 3]

Expand to (4, 3):
[[1, 2, 3],  ← Same data
 [1, 2, 3],  ← Same data
 [1, 2, 3],  ← Same data
 [1, 2, 3]]  ← Same data

Memory: [1, 2, 3] (ONLY ONCE!)
```

### SHRINK (Slice)
```
Input (4, 4):
[[1,  2,  3,  4],
 [5,  6,  7,  8],
 [9,  10, 11, 12],
 [13, 14, 15, 16]]

Shrink [1:3, 1:3]:
[[6,  7],
 [10, 11]]

Memory: All 16 elements still there
View: Only see 4 elements
```

## Common Patterns

### Image Processing

#### Channel Conversion
```python
# RGB to BGR
def rgb_to_bgr(x):
    """Convert RGB to BGR by flipping channel dimension"""
    # x shape: (batch, 3, height, width)
    return x[:, [2, 1, 0], :, :]

# Or using flip
def rgb_to_bgr_flip(x):
    return x.flip(axis=1)
```

#### Image Transpose
```python
# NCHW to NHWC (PyTorch to TensorFlow format)
def nchw_to_nhwc(x):
    """(N, C, H, W) -> (N, H, W, C)"""
    return x.permute(0, 2, 3, 1)

# NHWC to NCHW
def nhwc_to_nchw(x):
    """(N, H, W, C) -> (N, C, H, W)"""
    return x.permute(0, 3, 1, 2)

x = Tensor.randn(1, 3, 224, 224)  # PyTorch format
y = nchw_to_nhwc(x)               # TensorFlow format
# Shape: (1, 224, 224, 3)
```

#### Padding for Convolution
```python
def pad_for_conv(x, padding=1):
    """Pad image for convolution"""
    # x shape: (batch, channels, height, width)
    return x.pad(((0, 0), (0, 0),
                  (padding, padding),
                  (padding, padding)))

x = Tensor.randn(1, 3, 224, 224)
padded = pad_for_conv(x, padding=1)
# Shape: (1, 3, 226, 226)
```

### Matrix Operations

#### Batch Matrix Multiply Setup
```python
def prepare_batch_matmul(a, b):
    """Prepare tensors for batch matrix multiplication"""
    # a: (batch, m, k)
    # b: (batch, k, n)
    # Ensure last two dims are correct
    return a, b.transpose(-2, -1) if b.shape[-2] != a.shape[-1] else b

a = Tensor.randn(32, 10, 20)
b = Tensor.randn(32, 30, 20)
a, b = prepare_batch_matmul(a, b)
result = a @ b  # (32, 10, 30)
```

#### Flatten for Linear Layer
```python
def flatten(x):
    """Flatten all dimensions except batch"""
    batch_size = x.shape[0]
    return x.reshape(batch_size, -1)

x = Tensor.randn(32, 3, 224, 224)
flat = flatten(x)
# Shape: (32, 150528)  # 3*224*224
```

### Sequence Processing

#### Attention Mask Expansion
```python
def expand_attention_mask(mask, num_heads):
    """Expand mask for multi-head attention"""
    # mask: (batch, seq_len)
    # output: (batch, num_heads, seq_len, seq_len)
    batch, seq_len = mask.shape

    # Add dimensions
    mask = mask.reshape(batch, 1, 1, seq_len)

    # Expand
    mask = mask.expand(batch, num_heads, seq_len, seq_len)
    return mask

mask = Tensor([[1, 1, 1, 0, 0]])  # (1, 5)
expanded = expand_attention_mask(mask, num_heads=8)
# Shape: (1, 8, 5, 5)
```

#### Sequence Padding
```python
def pad_sequence(x, max_len):
    """Pad sequence to max length"""
    current_len = x.shape[0]
    if current_len >= max_len:
        return x[:max_len]

    pad_len = max_len - current_len
    return x.pad(((0, pad_len),) + ((0, 0),) * (len(x.shape) - 1))

seq = Tensor.randn(10, 512)  # 10 tokens, 512 dims
padded = pad_sequence(seq, max_len=20)
# Shape: (20, 512)
```

### Advanced Reshaping

#### Im2Col for Convolution
```python
def im2col(x, kernel_size, stride=1):
    """Convert image to column matrix for convolution"""
    batch, channels, h, w = x.shape
    kh, kw = kernel_size

    # Calculate output dimensions
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1

    # Reshape to expose patches
    x = x.reshape(batch, channels, out_h, stride, out_w, stride, kh, kw)
    x = x.permute(0, 2, 4, 1, 6, 7, 3, 5)
    x = x.reshape(batch * out_h * out_w, channels * kh * kw)

    return x
```

#### Unfold (Sliding Window)
```python
def unfold(x, kernel_size, stride=1):
    """Extract sliding windows"""
    # x: (batch, channels, height, width)
    # Returns: (batch, channels*kh*kw, num_windows)
    batch, channels, h, w = x.shape
    kh, kw = kernel_size

    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1

    # Use reshape and permute to extract windows
    windows = x.reshape(batch, channels, out_h, kh, out_w, kw)
    windows = windows.permute(0, 1, 3, 5, 2, 4)
    windows = windows.reshape(batch, channels * kh * kw, out_h * out_w)

    return windows
```

## Performance Considerations

### Zero-Copy Operations ⚡
```python
# All these are FREE (no data movement)
x = Tensor.randn(1000, 1000)

y = x.reshape(100, 10000)      # Zero-copy
z = y.transpose()               # Zero-copy
w = z.expand(2, 10000, 100)    # Zero-copy
v = w[0, :500, :50]            # Zero-copy (shrink)

# Data only moves when you realize()
result = v.realize()
```

### When Data Moves 🐌
```python
# These operations require actual data movement
x = Tensor.randn(100, 100)

# Pad requires new memory
y = x.pad(((10, 10), (10, 10)))  # Data copied

# Non-contiguous access after complex ops
z = x.transpose().reshape(50, 200)  # May need copy

# Explicit realize forces computation
w = x.reshape(50, 200).realize()  # Data moved
```

### Optimization Tips

#### Minimize Realizes
```python
# Bad: Multiple realizes
x = Tensor.randn(1000, 1000)
y = x.reshape(100, 10000).realize()
z = y.transpose().realize()
w = z.reshape(10000, 100).realize()

# Good: Chain operations, single realize
x = Tensor.randn(1000, 1000)
w = x.reshape(100, 10000).transpose().reshape(10000, 100).realize()
```

#### Contiguous Memory
```python
# After complex movement ops, make contiguous
x = Tensor.randn(100, 100)
y = x.transpose().reshape(50, 200)

# Make contiguous for better performance
y = y.contiguous()
```

## Code Examples

### Example 1: Batch Processing
```python
def batch_images(images, batch_size):
    """Batch list of images into tensor"""
    # images: list of (C, H, W) tensors
    stacked = Tensor.stack(images)  # (N, C, H, W)

    num_batches = len(images) // batch_size
    remainder = len(images) % batch_size

    # Reshape into batches
    batched = stacked[:num_batches * batch_size]
    batched = batched.reshape(num_batches, batch_size, *stacked.shape[1:])

    return batched

images = [Tensor.randn(3, 224, 224) for _ in range(100)]
batches = batch_images(images, batch_size=10)
# Shape: (10, 10, 3, 224, 224)
```

### Example 2: Sliding Window
```python
def sliding_window(x, window_size, stride=1):
    """Create sliding windows over sequence"""
    seq_len = x.shape[0]
    num_windows = (seq_len - window_size) // stride + 1

    # Create indices for windows
    windows = []
    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        windows.append(x[start:end])

    return Tensor.stack(windows)

seq = Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
windows = sliding_window(seq, window_size=3, stride=2)
# [[1, 2, 3],
#  [3, 4, 5],
#  [5, 6, 7],
#  [7, 8, 9]]
```

### Example 3: Tensor Tiling
```python
def tile(x, reps):
    """Tile tensor along dimensions"""
    # x: (a, b, c)
    # reps: (r1, r2, r3)
    # output: (a*r1, b*r2, c*r3)

    for axis, rep in enumerate(reps):
        x = x.expand(*x.shape[:axis], rep, *x.shape[axis:])
        x = x.reshape(*x.shape[:axis],
                      x.shape[axis] * x.shape[axis+1],
                      *x.shape[axis+2:])
    return x

x = Tensor([[1, 2], [3, 4]])
tiled = tile(x, (2, 3))
# [[1, 2, 1, 2, 1, 2],
#  [3, 4, 3, 4, 3, 4],
#  [1, 2, 1, 2, 1, 2],
#  [3, 4, 3, 4, 3, 4]]
```

### Example 4: Einsum-style Reshape
```python
def rearrange(x, pattern):
    """Rearrange tensor dimensions (simplified einops)"""
    # Example: "b c h w -> b (c h w)"
    if pattern == "b c h w -> b (c h w)":
        b, c, h, w = x.shape
        return x.reshape(b, c * h * w)

    # Example: "b (h w) c -> b c h w"
    elif "b (h w) c -> b c h w" in pattern:
        b, hw, c = x.shape
        h = int(hw ** 0.5)
        w = hw // h
        return x.reshape(b, h, w, c).permute(0, 3, 1, 2)

    return x

x = Tensor.randn(32, 3, 224, 224)
flat = rearrange(x, "b c h w -> b (c h w)")
# Shape: (32, 150528)
```

## Debugging Tips

### Visualize Shape Changes
```python
x = Tensor.randn(2, 3, 4)
print(f"Original: {x.shape}")

y = x.reshape(6, 4)
print(f"After reshape: {y.shape}")

z = y.transpose()
print(f"After transpose: {z.shape}")

w = z.expand(2, 4, 6)
print(f"After expand: {w.shape}")
```

### Check Zero-Copy
```python
x = Tensor.randn(10, 10)
y = x.reshape(5, 20).transpose().expand(3, 20, 5)

# Movement ops are zero-copy until realize()
print(f"Shape: {y.shape}")
print(f"Dtype: {y.dtype}")
# Data not copied until y.realize()
```

### Verify Zero-Copy
```python
import numpy as np

x = Tensor([1, 2, 3, 4, 5, 6])
y = x.reshape(2, 3)

# Check if they share memory (in numpy)
x_np = x.numpy()
y_np = y.numpy()
print(f"Share memory: {np.shares_memory(x_np, y_np)}")
```

## Key Takeaways

1. **MovementOps are zero-copy** - only metadata changes
2. **ShapeTracker** tracks how to view the data
3. **No data movement** until realize() or computation
4. **Seven main operations**: RESHAPE, PERMUTE, EXPAND, SHRINK, PAD, FLIP, STRIDE
5. **Highly composable** - chain multiple operations
6. **Critical for efficiency** - avoid unnecessary copies
7. **Foundation for**: convolution, attention, pooling
8. **Understand contiguity** - affects performance

## Next Steps

- Combine with [ElementwiseOps](01-elementwise-ops.md) and [ReduceOps](02-reduce-ops.md)
- Build complete neural network layers
- Understand how CONV and MATMUL are implemented
- Optimize memory access patterns
- Study ShapeTracker internals

## Summary Table

| Operation | Purpose | Zero-Copy | Example |
|-----------|---------|-----------|---------|
| RESHAPE | Change shape | ✅ | (6,) → (2,3) |
| PERMUTE | Reorder dims | ✅ | (2,3) → (3,2) |
| EXPAND | Broadcast | ✅ | (3,) → (4,3) |
| SHRINK | Extract region | ✅ | [1:3, 1:3] |
| PAD | Add padding | ❌ | Add zeros |
| FLIP | Reverse | ✅ | Reverse axis |
| STRIDE | Skip elements | ✅ | Every 2nd |

All operations except PAD are zero-copy!
