# Tinygrad Operations Deep Dive

This directory contains detailed explanations of tinygrad's three core operation types.

## Directory Structure

```
ops/
├── README.md                    # This file
├── 01-elementwise-ops.md        # ElementwiseOps overview
├── 02-reduce-ops.md             # ReduceOps overview
├── 03-movement-ops.md           # MovementOps overview
└── elementwise/
    ├── unary-ops.md             # UnaryOps detailed guide
    ├── binary-ops.md            # BinaryOps detailed guide
    └── ternary-ops.md           # TernaryOps detailed guide
```

## The Three Operation Types

Tinygrad breaks down ALL neural network operations into just three categories:

### 1. ElementwiseOps
Operations that work element-by-element on tensors.
- **UnaryOps**: Single input tensor (SQRT, LOG, EXP, etc.)
- **BinaryOps**: Two input tensors (ADD, MUL, DIV, etc.)
- **TernaryOps**: Three input tensors (WHERE, MULACC, etc.)

📖 See: `01-elementwise-ops.md` and `elementwise/` directory

### 2. ReduceOps
Operations that reduce tensor dimensions.
- Examples: SUM, MAX, MIN
- Reduce along specified axes

📖 See: `02-reduce-ops.md`

### 3. MovementOps
Virtual operations that reorganize data without copying.
- Examples: RESHAPE, PERMUTE, EXPAND, SLICE
- Zero-copy with ShapeTracker

📖 See: `03-movement-ops.md`

## The Big Question

**Where are CONV and MATMUL?**

They don't exist as primitives! They're built from these three operation types:
- **MATMUL** = RESHAPE + EXPAND + MUL + SUM (ReduceOp)
- **CONV** = Similar decomposition using movement + elementwise + reduce

This is the genius of tinygrad - extreme simplicity that composes into complex operations.

## Quick Reference

| Operation Type | Input Tensors | Output Size | Examples |
|---------------|---------------|-------------|----------|
| UnaryOps | 1 | Same as input | SQRT, EXP, LOG, RELU |
| BinaryOps | 2 | Broadcast result | ADD, MUL, SUB, DIV |
| TernaryOps | 3 | Broadcast result | WHERE, MULACC |
| ReduceOps | 1 | Smaller (reduced) | SUM, MAX, MEAN |
| MovementOps | 1 | Different shape | RESHAPE, PERMUTE |

## Learning Path

1. Start with **UnaryOps** - simplest operations
2. Move to **BinaryOps** - understand broadcasting
3. Learn **TernaryOps** - conditional operations
4. Understand **ReduceOps** - dimension reduction
5. Master **MovementOps** - zero-copy transformations
6. See how they compose into complex operations

## Code Examples

Each markdown file contains:
- Detailed explanations
- Mathematical definitions
- Python code examples
- Visual diagrams (ASCII art)
- Real-world use cases
- Performance considerations

Start with `elementwise/unary-ops.md` for the most basic operations!
