# ReduceOps: Dimension Reduction Operations

## Overview

ReduceOps operate on a **single tensor** and return a **smaller tensor** by aggregating values along specified dimensions.

```
Input (4 elements):  [1, 2, 3, 4]
SUM (reduce all):    10
Output (1 element):  [10]

Input (2x3 matrix):  [[1, 2, 3],
                      [4, 5, 6]]
SUM (axis=0):        [5, 7, 9]
SUM (axis=1):        [6, 15]
SUM (all):           21
```

## Complete List of ReduceOps

### 1. SUM - Summation
```python
from tinygrad import Tensor

# Sum all elements
x = Tensor([1, 2, 3, 4])
total = x.sum()
# Result: 10

# Sum along axis
x = Tensor([[1, 2, 3],
            [4, 5, 6]])
sum_axis0 = x.sum(axis=0)  # [5, 7, 9]
sum_axis1 = x.sum(axis=1)  # [6, 15]
```

**Mathematical:** `output = Σ input[i]`

**Use cases:**
- Total loss calculation
- Pooling operations
- Attention weights normalization

### 2. MAX - Maximum
```python
# Find maximum value
x = Tensor([1, 5, 3, 2])
max_val = x.max()
# Result: 5

# Max along axis
x = Tensor([[1, 5, 3],
            [4, 2, 6]])
max_axis0 = x.max(axis=0)  # [4, 5, 6]
max_axis1 = x.max(axis=1)  # [5, 6]
```

**Mathematical:** `output = max(input[i])`

**Use cases:**
- Max pooling
- Finding peaks
- Numerical stability (log-sum-exp trick)

### Derived ReduceOps

#### MEAN - Average
```python
def mean(x, axis=None):
    return x.sum(axis=axis) / x.shape[axis if axis else 0]

x = Tensor([1, 2, 3, 4])
avg = x.mean()
# Result: 2.5
```

#### MIN - Minimum
```python
def min(x, axis=None):
    return -(-x).max(axis=axis)

x = Tensor([1, 5, 3, 2])
min_val = x.min()
# Result: 1
```

#### PROD - Product
```python
def prod(x, axis=None):
    return x.log().sum(axis=axis).exp()

x = Tensor([2, 3, 4])
product = x.prod()
# Result: 24
```

#### VAR - Variance
```python
def var(x, axis=None):
    mean = x.mean(axis=axis, keepdim=True)
    return ((x - mean) ** 2).mean(axis=axis)

x = Tensor([1, 2, 3, 4, 5])
variance = x.var()
# Result: 2.0
```

#### STD - Standard Deviation
```python
def std(x, axis=None):
    return x.var(axis=axis).sqrt()

x = Tensor([1, 2, 3, 4, 5])
std_dev = x.std()
# Result: 1.414
```

## Understanding Axes

### 1D Tensor (Vector)
```python
x = Tensor([1, 2, 3, 4])
# Shape: (4,)

x.sum()  # Reduce all: 10
# Shape: ()
```

### 2D Tensor (Matrix)
```python
x = Tensor([[1, 2, 3],
            [4, 5, 6]])
# Shape: (2, 3)

x.sum(axis=0)  # Sum columns: [5, 7, 9]
# Shape: (3,)

x.sum(axis=1)  # Sum rows: [6, 15]
# Shape: (2,)

x.sum()  # Sum all: 21
# Shape: ()
```

### 3D Tensor
```python
x = Tensor.randn(2, 3, 4)
# Shape: (2, 3, 4)

x.sum(axis=0)  # Shape: (3, 4)
x.sum(axis=1)  # Shape: (2, 4)
x.sum(axis=2)  # Shape: (2, 3)
x.sum()        # Shape: ()
```

## Visual Examples

### SUM Operation
```
Input (2x3):
[[1, 2, 3],
 [4, 5, 6]]

axis=0 (sum columns):
 ↓  ↓  ↓
[5, 7, 9]

axis=1 (sum rows):
[1+2+3] → [6]
[4+5+6] → [15]

all (sum everything):
1+2+3+4+5+6 → 21
```

### MAX Operation
```
Input (2x3):
[[1, 5, 3],
 [4, 2, 6]]

axis=0 (max of columns):
 ↓  ↓  ↓
[4, 5, 6]

axis=1 (max of rows):
max(1,5,3) → [5]
max(4,2,6) → [6]
```

### Multi-Axis Reduction
```
Input (2x3x4):
[[[...], [...], [...], [...]],
 [[...], [...], [...], [...]]]

axis=(0,1) - reduce first two dimensions:
Result shape: (4,)

axis=(1,2) - reduce last two dimensions:
Result shape: (2,)
```

## Common Patterns

### Pooling Operations

#### Average Pooling
```python
def avg_pool2d(x, kernel_size=2):
    """Simple 2D average pooling"""
    # Reshape to expose pooling windows
    b, c, h, w = x.shape
    x = x.reshape(b, c, h//kernel_size, kernel_size,
                  w//kernel_size, kernel_size)
    # Average over kernel dimensions
    return x.mean(axis=(3, 5))

x = Tensor.randn(1, 3, 8, 8)  # Batch, channels, height, width
pooled = avg_pool2d(x, kernel_size=2)
# Shape: (1, 3, 4, 4)
```

#### Max Pooling
```python
def max_pool2d(x, kernel_size=2):
    """Simple 2D max pooling"""
    b, c, h, w = x.shape
    x = x.reshape(b, c, h//kernel_size, kernel_size,
                  w//kernel_size, kernel_size)
    return x.max(axis=(3, 5))

x = Tensor.randn(1, 3, 8, 8)
pooled = max_pool2d(x, kernel_size=2)
# Shape: (1, 3, 4, 4)
```

#### Global Average Pooling
```python
def global_avg_pool(x):
    """Average over spatial dimensions"""
    # x shape: (batch, channels, height, width)
    return x.mean(axis=(2, 3))

x = Tensor.randn(32, 512, 7, 7)
features = global_avg_pool(x)
# Shape: (32, 512)
```

### Normalization

#### Batch Normalization
```python
def batch_norm(x, eps=1e-5):
    """Normalize over batch dimension"""
    # x shape: (batch, channels, height, width)
    mean = x.mean(axis=(0, 2, 3), keepdim=True)
    var = x.var(axis=(0, 2, 3), keepdim=True)
    return (x - mean) / (var + eps).sqrt()
```

#### Layer Normalization
```python
def layer_norm(x, eps=1e-5):
    """Normalize over feature dimensions"""
    # x shape: (batch, features)
    mean = x.mean(axis=-1, keepdim=True)
    var = x.var(axis=-1, keepdim=True)
    return (x - mean) / (var + eps).sqrt()
```

#### Instance Normalization
```python
def instance_norm(x, eps=1e-5):
    """Normalize each instance independently"""
    # x shape: (batch, channels, height, width)
    mean = x.mean(axis=(2, 3), keepdim=True)
    var = x.var(axis=(2, 3), keepdim=True)
    return (x - mean) / (var + eps).sqrt()
```

### Loss Functions

#### Mean Squared Error
```python
def mse_loss(pred, target):
    """MSE = mean((pred - target)^2)"""
    return ((pred - target) ** 2).mean()

pred = Tensor([[1.0, 2.0], [3.0, 4.0]])
target = Tensor([[1.5, 2.5], [2.5, 3.5]])
loss = mse_loss(pred, target)
```

#### Cross Entropy Loss
```python
def cross_entropy(logits, targets):
    """Cross entropy with log-softmax"""
    log_probs = logits.log_softmax(axis=-1)
    return -(targets * log_probs).sum(axis=-1).mean()
```

#### L1 Loss (MAE)
```python
def l1_loss(pred, target):
    """L1 = mean(|pred - target|)"""
    return (pred - target).abs().mean()
```

### Attention Mechanisms

#### Softmax
```python
def softmax(x, axis=-1):
    """Numerically stable softmax"""
    # Subtract max for numerical stability
    exp_x = (x - x.max(axis=axis, keepdim=True)).exp()
    return exp_x / exp_x.sum(axis=axis, keepdim=True)

logits = Tensor([[1, 2, 3], [4, 5, 6]])
probs = softmax(logits, axis=-1)
# Each row sums to 1.0
```

#### Log-Sum-Exp
```python
def logsumexp(x, axis=-1):
    """Numerically stable log(sum(exp(x)))"""
    max_x = x.max(axis=axis, keepdim=True)
    return max_x + (x - max_x).exp().sum(axis=axis, keepdim=True).log()
```

#### Attention Weights
```python
def attention(Q, K, V, mask=None):
    """Scaled dot-product attention"""
    # Q, K, V: (batch, heads, seq_len, d_k)
    d_k = Q.shape[-1]

    # Compute attention scores
    scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)

    # Apply mask if provided
    if mask is not None:
        scores = mask.where(scores, -1e9)

    # Softmax over last dimension
    attn_weights = scores.softmax(axis=-1)

    # Apply attention to values
    return attn_weights @ V
```

## Advanced Patterns

### Variance and Covariance
```python
def covariance(x, y):
    """Covariance between two tensors"""
    mean_x = x.mean()
    mean_y = y.mean()
    return ((x - mean_x) * (y - mean_y)).mean()

def correlation(x, y):
    """Pearson correlation coefficient"""
    cov = covariance(x, y)
    std_x = x.std()
    std_y = y.std()
    return cov / (std_x * std_y)
```

### Percentiles (Approximate)
```python
def percentile(x, p):
    """Approximate percentile using sorting"""
    sorted_x = x.sort()
    idx = int(len(sorted_x) * p / 100)
    return sorted_x[idx]
```

### Entropy
```python
def entropy(probs, axis=-1):
    """Shannon entropy: -sum(p * log(p))"""
    return -(probs * probs.log()).sum(axis=axis)

probs = Tensor([[0.7, 0.2, 0.1],
                [0.33, 0.33, 0.34]])
h = entropy(probs, axis=-1)
```

### KL Divergence
```python
def kl_divergence(p, q, axis=-1):
    """KL(P||Q) = sum(p * log(p/q))"""
    return (p * (p.log() - q.log())).sum(axis=axis)
```

## Performance Considerations

### Memory Efficiency

#### keepdim Parameter
```python
# Without keepdim - shape changes
x = Tensor.randn(2, 3, 4)
mean = x.mean(axis=1)  # Shape: (2, 4)

# With keepdim - maintains dimensions
mean = x.mean(axis=1, keepdim=True)  # Shape: (2, 1, 4)
# Easier for broadcasting!
```

#### Fused Operations
```python
# Tinygrad fuses these automatically
mean = x.sum() / x.numel()  # Fused into one kernel

# Variance computation is also fused
var = ((x - x.mean()) ** 2).mean()
```

### Numerical Stability

#### Stable Softmax
```python
# Unstable - can overflow
def softmax_unstable(x):
    exp_x = x.exp()
    return exp_x / exp_x.sum(axis=-1, keepdim=True)

# Stable - subtract max first
def softmax_stable(x):
    max_x = x.max(axis=-1, keepdim=True)
    exp_x = (x - max_x).exp()
    return exp_x / exp_x.sum(axis=-1, keepdim=True)
```

#### Stable Log-Sum-Exp
```python
# For numerical stability in log-space
def stable_logsumexp(x, axis=-1):
    max_x = x.max(axis=axis, keepdim=True)
    return max_x.squeeze() + (x - max_x).exp().sum(axis=axis).log()
```

## Code Examples

### Example 1: Custom Pooling
```python
def adaptive_avg_pool(x, output_size):
    """Adaptive average pooling to fixed output size"""
    b, c, h, w = x.shape
    oh, ow = output_size

    # Calculate stride and kernel size
    stride_h = h // oh
    stride_w = w // ow
    kernel_h = h - (oh - 1) * stride_h
    kernel_w = w - (ow - 1) * stride_w

    # Reshape and pool
    x = x.reshape(b, c, oh, kernel_h, ow, kernel_w)
    return x.mean(axis=(3, 5))

x = Tensor.randn(1, 512, 14, 14)
pooled = adaptive_avg_pool(x, (7, 7))
# Shape: (1, 512, 7, 7)
```

### Example 2: Batch Statistics
```python
def compute_batch_stats(x):
    """Compute statistics over batch"""
    return {
        'mean': x.mean(axis=0),
        'var': x.var(axis=0),
        'min': x.min(axis=0),
        'max': x.max(axis=0),
        'std': x.std(axis=0)
    }

batch = Tensor.randn(32, 10)  # 32 samples, 10 features
stats = compute_batch_stats(batch)
```

### Example 3: Weighted Average
```python
def weighted_average(values, weights, axis=-1):
    """Compute weighted average"""
    return (values * weights).sum(axis=axis) / weights.sum(axis=axis)

values = Tensor([1, 2, 3, 4])
weights = Tensor([0.1, 0.2, 0.3, 0.4])
avg = weighted_average(values, weights)
# Result: 3.0
```

### Example 4: Moving Average
```python
def exponential_moving_average(x, alpha=0.9):
    """Compute EMA over sequence"""
    ema = x[0]
    result = [ema]

    for val in x[1:]:
        ema = alpha * ema + (1 - alpha) * val
        result.append(ema)

    return Tensor(result)
```

## Debugging Tips

### Check Reduction Dimensions
```python
x = Tensor.randn(2, 3, 4)
print(f"Original shape: {x.shape}")

reduced = x.sum(axis=1)
print(f"After sum(axis=1): {reduced.shape}")  # (2, 4)

reduced = x.sum(axis=1, keepdim=True)
print(f"With keepdim: {reduced.shape}")  # (2, 1, 4)
```

### Verify Numerical Stability
```python
# Check for NaN or Inf
x = Tensor([1e10, 1e10, 1e10])
softmax_result = x.softmax()
print(f"Has NaN: {softmax_result.numpy().isnan().any()}")

# Use stable version
stable_result = (x - x.max()).softmax()
print(f"Stable result: {stable_result.numpy()}")
```

## Key Takeaways

1. **ReduceOps aggregate values** along specified dimensions
2. **Two primitive ReduceOps**: SUM and MAX
3. **All other reductions** are derived (MEAN, MIN, VAR, STD, etc.)
4. **Axis parameter** controls which dimension to reduce
5. **keepdim** maintains dimensions for broadcasting
6. **Numerical stability** is crucial (subtract max, use eps)
7. **Automatic fusion** with elementwise operations
8. **Foundation for**: pooling, normalization, losses, attention

## Next Steps

- Learn about [MovementOps](03-movement-ops.md) for reshaping
- Combine with [ElementwiseOps](01-elementwise-ops.md)
- Build complete neural network layers
- Optimize performance with kernel fusion
