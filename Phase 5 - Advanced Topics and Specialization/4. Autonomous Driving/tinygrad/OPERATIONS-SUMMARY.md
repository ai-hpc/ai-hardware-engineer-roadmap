# Tinygrad Operations: Complete Summary

## 🎯 The Core Insight

Tinygrad has **ONLY 3 types of operations** that build everything:

1. **ElementwiseOps** - Element-by-element operations
2. **ReduceOps** - Dimension reduction
3. **MovementOps** - Zero-copy reshaping

## 📊 Complete Operation Count

### ElementwiseOps: 16 primitives
- **UnaryOps**: 7 (EXP2, LOG2, SQRT, RECIP, NEG, SIN, CAST)
- **BinaryOps**: 7 (ADD, SUB, MUL, DIV, MOD, MAX, CMPLT)
- **TernaryOps**: 2 (WHERE, MULACC)

### ReduceOps: 2 primitives
- SUM, MAX

### MovementOps: 7 operations
- RESHAPE, PERMUTE, EXPAND, SHRINK, PAD, FLIP, STRIDE

**Total: 25 operations** to build all of deep learning!

## 🔨 How Everything Is Built

### No CONV or MATMUL Primitives!

```
MATMUL = RESHAPE + EXPAND + MUL + SUM
CONV2D = RESHAPE + PERMUTE + MUL + SUM
```

### Activation Functions
```python
ReLU      = x.maximum(0)                    # 1 BinaryOp
Sigmoid   = (1 + (-x).exp()).reciprocal()   # 3 UnaryOps
Tanh      = 2 * (2*x).sigmoid() - 1         # Composed
GELU      = 0.5*x*(1+(x*0.7979*(1+0.044715*x*x)).tanh())
```

### Normalization
```python
LayerNorm  = (x - x.mean()) / x.std()       # ReduceOps + BinaryOps
BatchNorm  = (x - mean) / sqrt(var + eps)   # ReduceOps + BinaryOps + UnaryOps
```

### Pooling
```python
MaxPool = x.reshape(...).max(axis=...)      # MovementOp + ReduceOp
AvgPool = x.reshape(...).mean(axis=...)     # MovementOp + ReduceOp
```

### Attention
```python
Attention = softmax(Q @ K.T / sqrt(d_k)) @ V
# Uses: PERMUTE, MUL, SUM, DIV, MAX, EXP
# All three operation types!
```

## 📁 Documentation Structure

```
ops/
├── README.md                    # Overview
├── complete-reference.md        # This summary
├── 01-elementwise-ops.md        # ElementwiseOps guide
├── 02-reduce-ops.md             # ReduceOps guide
├── 03-movement-ops.md           # MovementOps guide
└── elementwise/
    ├── unary-ops.md             # 7 UnaryOps detailed
    ├── binary-ops.md            # 7 BinaryOps detailed
    ├── ternary-ops.md           # 2 TernaryOps detailed
    └── quick-reference.md       # Cheat sheet
```

## 🚀 Quick Start Guide

### 1. Learn ElementwiseOps First
Start with the most common operations:
- Read `elementwise/quick-reference.md` for overview
- Study `elementwise/unary-ops.md` for single-tensor ops
- Study `elementwise/binary-ops.md` for two-tensor ops
- Study `elementwise/ternary-ops.md` for conditional ops

### 2. Then ReduceOps
Learn aggregation:
- Read `02-reduce-ops.md`
- Understand SUM and MAX
- Learn derived ops (MEAN, VAR, STD)

### 3. Finally MovementOps
Master zero-copy transformations:
- Read `03-movement-ops.md`
- Understand ShapeTracker
- Learn RESHAPE, PERMUTE, EXPAND

### 4. See How They Compose
- Read `complete-reference.md`
- See how complex ops are built
- Build your own neural network layers

## 💡 Key Concepts

### Lazy Evaluation
```python
x = Tensor([1, 2, 3])
y = x + 1          # Not executed yet!
z = y * 2          # Still not executed!
result = z.realize()  # NOW it executes (fused!)
```

### Kernel Fusion
```python
# These three operations...
y = x + 1
z = y * 2
w = z.relu()

# ...are fused into ONE kernel automatically!
w = max((x + 1) * 2, 0)
```

### Zero-Copy MovementOps
```python
x = Tensor.randn(1000, 1000)
y = x.reshape(100, 10000)      # FREE (no copy)
z = y.transpose()               # FREE (no copy)
w = z.expand(2, 10000, 100)    # FREE (no copy)
# Data only moves when you realize()
```

### Broadcasting
```python
# Scalar
[1,2,3] + 10 = [11,12,13]

# Vector
[[1,2,3],     [10]      [[11,12,13],
 [4,5,6]]  +  [20]  =    [24,25,26]]

# Works with all BinaryOps and TernaryOps!
```

## 📈 Performance Tips

### ✅ Fast Operations
- Simple ElementwiseOps: ADD, MUL, NEG
- All MovementOps (zero-copy!)
- Fused operations (automatic)

### ⚠️ Medium Speed
- Transcendental UnaryOps: EXP, LOG, SIN
- Division (use MUL with reciprocal)
- Large ReduceOps

### 🚀 Optimization Strategies
1. Let tinygrad fuse operations automatically
2. Use MovementOps instead of copying data
3. Minimize realizes (chain operations)
4. Use contiguous memory when needed
5. Prefer MUL over DIV when possible

## 🎓 Learning Path

### Beginner (1-2 hours)
1. ✅ Read `elementwise/quick-reference.md`
2. ✅ Try basic operations in Python
3. ✅ Understand broadcasting

### Intermediate (3-5 hours)
4. ✅ Study each ElementwiseOp type
5. ✅ Learn ReduceOps for aggregation
6. ✅ Master MovementOps for reshaping
7. ✅ Build simple activations and losses

### Advanced (5-10 hours)
8. ✅ Understand kernel fusion
9. ✅ Build complete layers (Conv, Attention)
10. ✅ Optimize performance
11. ✅ Read tinygrad source code

## 🔍 Common Patterns

### Pattern 1: Activation Function
```python
def swish(x):
    return x * x.sigmoid()
# Uses: MUL (BinaryOp), sigmoid (UnaryOps)
```

### Pattern 2: Normalization
```python
def layer_norm(x, eps=1e-5):
    mean = x.mean(axis=-1, keepdim=True)
    var = x.var(axis=-1, keepdim=True)
    return (x - mean) / (var + eps).sqrt()
# Uses: SUM (ReduceOp), SUB, DIV (BinaryOps), SQRT (UnaryOp)
```

### Pattern 3: Pooling
```python
def max_pool2d(x, kernel_size=2):
    b, c, h, w = x.shape
    x = x.reshape(b, c, h//kernel_size, kernel_size,
                  w//kernel_size, kernel_size)
    return x.max(axis=(3, 5))
# Uses: RESHAPE (MovementOp), MAX (ReduceOp)
```

### Pattern 4: Attention
```python
def attention(Q, K, V):
    scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)
    attn = scores.softmax(axis=-1)
    return attn @ V
# Uses: PERMUTE, MUL, SUM, DIV, MAX, EXP
```

## 🎯 The Big Picture

```
┌──────────────────────────────────────┐
│   16 ElementwiseOps (primitives)     │
│   + 2 ReduceOps (primitives)         │
│   + 7 MovementOps                    │
│   = 25 operations                    │
├──────────────────────────────────────┤
│              ↓ COMPOSE               │
├──────────────────────────────────────┤
│   • Activations                      │
│   • Normalizations                   │
│   • Pooling                          │
│   • Convolutions                     │
│   • Matrix Multiplication            │
│   • Attention                        │
│   • Loss Functions                   │
│   • EVERYTHING in Deep Learning!     │
└──────────────────────────────────────┘
```

## 📚 Detailed Guides

| Topic | File | What You'll Learn |
|-------|------|-------------------|
| Overview | `01-elementwise-ops.md` | ElementwiseOps concepts |
| UnaryOps | `elementwise/unary-ops.md` | 7 primitive UnaryOps |
| BinaryOps | `elementwise/binary-ops.md` | 7 primitive BinaryOps |
| TernaryOps | `elementwise/ternary-ops.md` | WHERE and MULACC |
| ReduceOps | `02-reduce-ops.md` | SUM, MAX, and derived ops |
| MovementOps | `03-movement-ops.md` | Zero-copy transformations |
| Quick Ref | `elementwise/quick-reference.md` | Cheat sheet |
| Complete | `complete-reference.md` | How everything composes |

## 🎉 Why This Matters

### For Learning
- **Transparent**: See exactly how deep learning works
- **Simple**: Only 25 operations to understand
- **Composable**: Build complex from simple

### For Research
- **Hackable**: Easy to modify and experiment
- **Fast iteration**: Simple codebase
- **Custom ops**: Add new operations easily

### For Production
- **Performance**: Automatic kernel fusion
- **Portable**: Multiple backend support
- **Efficient**: Lazy evaluation, zero-copy

## 🚀 Next Steps

1. **Start coding**: Try examples in each guide
2. **Build layers**: Implement Conv2D, Attention
3. **Optimize**: Learn kernel fusion
4. **Contribute**: Add to tinygrad!

## 📖 Resources

- [Tinygrad GitHub](https://github.com/tinygrad/tinygrad)
- [Official Docs](https://docs.tinygrad.org)
- [Discord Community](https://discord.gg/tinygrad)
- [Tinygrad Homepage](https://tinygrad.org)

---

**Remember**: Everything in deep learning can be built from just 3 operation types!

That's the genius of tinygrad. 🎯
