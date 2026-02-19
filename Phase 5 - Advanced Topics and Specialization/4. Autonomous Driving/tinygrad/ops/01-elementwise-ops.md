# ElementwiseOps: The Foundation of Tinygrad

## What are ElementwiseOps?

ElementwiseOps are operations that work **element-by-element** on tensors. They process each element independently, making them highly parallelizable on GPUs.

## The Three Categories

### 1. UnaryOps (1 input tensor)
Operations on a single tensor, applied to each element.

```
Input:  [1, 2, 3, 4]
SQRT:   [1, 1.414, 1.732, 2]
```

### 2. BinaryOps (2 input tensors)
Operations between two tensors, element-by-element with broadcasting.

```
Input A: [1, 2, 3, 4]
Input B: [5, 6, 7, 8]
ADD:     [6, 8, 10, 12]
```

### 3. TernaryOps (3 input tensors)
Operations on three tensors, typically conditional operations.

```
Condition: [True, False, True, False]
If True:   [1, 2, 3, 4]
If False:  [5, 6, 7, 8]
WHERE:     [1, 6, 3, 8]
```

## Why ElementwiseOps Matter

### 1. GPU Parallelization
Each element can be computed independently:
```
CPU: for i in range(n): output[i] = sqrt(input[i])
GPU: All elements computed simultaneously!
```

### 2. Memory Efficiency
- Input and output have the same shape (for unary)
- Predictable memory access patterns
- Cache-friendly operations

### 3. Composability
Complex operations are built from elementwise ops:
```python
# ReLU activation
def relu(x):
    return x.maximum(0)  # BinaryOp: MAX(x, 0)

# Sigmoid activation
def sigmoid(x):
    return 1 / (1 + (-x).exp())  # UnaryOps: NEG, EXP, RECIP
```

## Broadcasting Rules

BinaryOps and TernaryOps support broadcasting:

```python
# Scalar broadcast
[1, 2, 3] + 10 = [11, 12, 13]

# Vector broadcast
[[1, 2, 3],     [10]      [[11, 12, 13],
 [4, 5, 6]]  +  [20]  =    [24, 25, 26]]

# Dimension expansion
Shape (3, 1) + Shape (1, 4) = Shape (3, 4)
```

## Performance Characteristics

### Fast ✅
- Simple operations (ADD, MUL)
- Fused operations (ADD + MUL in one kernel)
- Aligned memory access

### Slower ⚠️
- Transcendental functions (EXP, LOG, SIN)
- Complex operations (POW with non-integer exponents)
- Misaligned memory access

## Kernel Fusion

Tinygrad automatically fuses elementwise operations:

```python
# Three separate operations
y = x + 1
z = y * 2
w = z.relu()

# Tinygrad fuses into ONE kernel:
# w = max((x + 1) * 2, 0)
```

This eliminates intermediate memory reads/writes!

## Common Patterns

### Activation Functions
```python
# All built from elementwise ops
relu(x)      = x.maximum(0)           # BinaryOp
sigmoid(x)   = 1 / (1 + (-x).exp())   # UnaryOps
tanh(x)      = x.tanh()               # UnaryOp
gelu(x)      = x * 0.5 * (1 + erf(x)) # BinaryOps + UnaryOp
```

### Normalization
```python
# Layer normalization (simplified)
mean = x.mean()
std = x.std()
normalized = (x - mean) / std  # BinaryOps: SUB, DIV
```

### Loss Functions
```python
# MSE Loss
mse = ((pred - target) ** 2).mean()  # BinaryOps: SUB, POW + ReduceOp

# Binary Cross Entropy
bce = -(target * pred.log() + (1-target) * (1-pred).log())
```

## Implementation in Tinygrad

ElementwiseOps are defined in `tinygrad/ops.py`:

```python
class UnaryOps(Enum):
    EXP2 = auto(); LOG2 = auto(); CAST = auto()
    SIN = auto(); SQRT = auto(); RECIP = auto(); NEG = auto()

class BinaryOps(Enum):
    ADD = auto(); SUB = auto(); MUL = auto(); DIV = auto()
    MAX = auto(); MOD = auto(); CMPLT = auto()

class TernaryOps(Enum):
    WHERE = auto(); MULACC = auto()
```

## Detailed Guides

For in-depth explanations with code examples:

- 📖 [UnaryOps Guide](elementwise/unary-ops.md)
- 📖 [BinaryOps Guide](elementwise/binary-ops.md)
- 📖 [TernaryOps Guide](elementwise/ternary-ops.md)

## Key Takeaways

1. **ElementwiseOps work element-by-element** - highly parallelizable
2. **Three types**: Unary (1 input), Binary (2 inputs), Ternary (3 inputs)
3. **Broadcasting** allows operations on different shaped tensors
4. **Kernel fusion** combines multiple ops into one for efficiency
5. **All activations and many operations** are built from elementwise ops

## Next Steps

Start with [UnaryOps](elementwise/unary-ops.md) to understand the simplest operations, then progress to BinaryOps and TernaryOps.
