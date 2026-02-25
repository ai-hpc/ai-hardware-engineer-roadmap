"""
Project 2: The Three Op Types
==============================
Goal: Understand ElementwiseOps, ReduceOps, and MovementOps by inspecting
what tinygrad generates and how they compose.

Run with:
  DEBUG=1 python3 02_op_types.py     # kernel counts
  DEBUG=3 python3 02_op_types.py     # generated C code
"""

import os
import numpy as np


# ---------------------------------------------------------------------------
# Task 1: ElementwiseOps — each element computed independently
# ---------------------------------------------------------------------------
def task1_elementwise():
    print("\n=== Task 1: ElementwiseOps ===")
    from tinygrad import Tensor

    x = Tensor.randn(8, 8)

    print("UnaryOps (1 input):")
    ops = {
        "relu":      x.relu(),
        "exp":       x.exp(),
        "log":       x.abs().log(),
        "sqrt":      x.abs().sqrt(),
        "neg":       (-x),
    }
    for name, t in ops.items():
        result = t.numpy()
        print(f"  {name:8s}: shape={result.shape}, mean={result.mean():.4f}")

    print("\nBinaryOps (2 inputs):")
    y = Tensor.randn(8, 8)
    bin_ops = {
        "add":   (x + y),
        "mul":   (x * y),
        "sub":   (x - y),
        "div":   (x / (y.abs() + 0.1)),
        "max":   x.maximum(y),
    }
    for name, t in bin_ops.items():
        print(f"  {name:8s}: shape={t.numpy().shape}")

    print("\nTernaryOp — WHERE (conditional select):")
    mask = (x > 0)
    selected = mask.where(x, -x)  # abs(x) via WHERE
    ref = x.abs().numpy()
    result = selected.numpy()
    print(f"  |x| via WHERE: max_error={np.abs(result - ref).max():.2e}")

    print("\nKEY: All elementwise ops fuse into 1 kernel when unrealized.")
    print("Run with DEBUG=1 to see.")


# ---------------------------------------------------------------------------
# Task 2: ReduceOps — collapse a dimension
# ---------------------------------------------------------------------------
def task2_reduce():
    print("\n=== Task 2: ReduceOps ===")
    from tinygrad import Tensor

    x = Tensor.randn(256, 512)

    print(f"Input shape: {x.shape}")

    print("\nSUM reductions:")
    r1 = x.sum()
    r2 = x.sum(axis=0)
    r3 = x.sum(axis=1)
    r4 = x.sum(axis=0, keepdim=True)
    print(f"  sum():            shape={r1.numpy().shape}")
    print(f"  sum(axis=0):      shape={r2.numpy().shape}")
    print(f"  sum(axis=1):      shape={r3.numpy().shape}")
    print(f"  sum(axis=0,kd):   shape={r4.numpy().shape}")

    print("\nMAX reduction:")
    m1 = x.max(axis=1)
    print(f"  max(axis=1):      shape={m1.numpy().shape}")

    print("\nCommon patterns built from ReduceOps:")
    # mean = sum / count
    mean = x.mean(axis=1)
    print(f"  mean(axis=1):     shape={mean.numpy().shape}")

    # variance = mean((x - mean)^2)
    xm = x - x.mean(axis=1, keepdim=True)
    var = (xm * xm).mean(axis=1)
    print(f"  var(axis=1):      shape={var.numpy().shape}")

    # softmax = exp(x) / sum(exp(x))
    sm = x.softmax(axis=1)
    print(f"  softmax(axis=1):  shape={sm.numpy().shape}")
    print(f"  softmax row sums: {sm.sum(axis=1).numpy()[:3]}  (should be ~1.0)")

    print("\nKEY: Reductions can't fuse with subsequent elementwise ops easily.")
    print("Each reduction typically becomes a separate kernel.")
    print("Run with DEBUG=1 and count kernels for softmax (should be 2).")


# ---------------------------------------------------------------------------
# Task 3: MovementOps — zero-copy shape manipulation
# ---------------------------------------------------------------------------
def task3_movement():
    print("\n=== Task 3: MovementOps (zero-copy) ===")
    from tinygrad import Tensor

    x = Tensor.randn(4, 8, 16)
    print(f"Input shape: {x.shape}")

    print("\nReshape and Permute (no data copy):")
    r1 = x.reshape(32, 16)
    r2 = x.reshape(4, 128)
    r3 = x.permute(2, 0, 1)    # (16, 4, 8)
    r4 = x.permute(1, 0, 2)    # (8, 4, 16)
    print(f"  reshape(32,16):   shape={r1.shape}")
    print(f"  reshape(4,128):   shape={r2.shape}")
    print(f"  permute(2,0,1):   shape={r3.shape}")
    print(f"  permute(1,0,2):   shape={r4.shape}")

    print("\nExpand (broadcast without copy):")
    v = Tensor.randn(1, 8)       # shape (1, 8)
    e = v.expand(4, 8)           # shape (4, 8) — no data copy
    print(f"  expand (1,8)→(4,8): shape={e.shape}")
    # All 4 rows point to the same data until materialized
    e_np = e.numpy()
    print(f"  All rows equal? {np.allclose(e_np[0], e_np[1])}")

    print("\nSlice (creates a strided view):")
    x2d = Tensor.arange(0, 24).reshape(4, 6)
    s1 = x2d[1:3, :]             # rows 1 and 2
    s2 = x2d[:, ::2]             # every other column
    s3 = x2d[1:3, 2:5]          # submatrix
    print(f"  x2d shape:     {x2d.shape}")
    print(f"  x2d[1:3,:]:    {s1.shape}")
    print(f"  x2d[:,::2]:    {s2.shape}")
    print(f"  x2d[1:3,2:5]:  {s3.shape}")

    print("\nKEY: Movement ops alone produce ZERO kernels.")
    print("They only change the ShapeTracker (strides + offsets).")
    print("The data move is fused into the first compute kernel that uses them.")

    # Demonstrate: chain of movement ops = still 0 kernels until compute
    chain = x.permute(2, 0, 1).reshape(16, 32).expand(2, 16, 32)
    # Force realization — the movement ops get baked into the elementwise kernel
    final = (chain + 1.0).realize()
    print(f"\n  10 movement ops + 1 elementwise → realize → still 1 kernel")
    print(f"  (run with DEBUG=1 to verify)")


# ---------------------------------------------------------------------------
# Task 4: Implement matmul manually from primitives
# ---------------------------------------------------------------------------
def task4_manual_matmul():
    print("\n=== Task 4: Manual Matmul ===")
    from tinygrad import Tensor

    # We'll implement C = A @ B using only expand, *, and sum
    # Standard matmul: C[i,j] = sum_k A[i,k] * B[k,j]
    # Steps:
    #   1. A: (M,K) → expand to (M,K,N)
    #   2. B: (K,N) → permute to (K,N) → expand to (M,K,N)  [via transpose + expand]
    #   3. element-wise multiply: (M,K,N)
    #   4. sum over K axis: (M,N)

    M, K, N = 8, 16, 12
    A = Tensor.randn(M, K)
    B = Tensor.randn(K, N)

    # Reference: built-in matmul
    ref = (A @ B).numpy()

    # Manual: expand-multiply-reduce
    # A:  (M, K)    → (M, K, 1)  → (M, K, N)
    # B:  (K, N)    → (1, K, N)  → (M, K, N)
    A_exp = A.reshape(M, K, 1).expand(M, K, N)
    B_exp = B.reshape(1, K, N).expand(M, K, N)
    manual = (A_exp * B_exp).sum(axis=1)   # sum over K

    result = manual.numpy()
    max_err = np.abs(result - ref).max()
    print(f"  A: {A.shape}, B: {B.shape} → C: {result.shape}")
    print(f"  Max error vs A@B: {max_err:.2e}")
    assert max_err < 1e-4, f"Matmul mismatch! max_error={max_err}"
    print("  ✓ Manual matmul matches A@B")

    print("\nKEY: @ operator decomposes to: reshape + expand + multiply + sum.")
    print("This shows how ALL operations in tinygrad reduce to the 3 primitives.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    task1_elementwise()
    task2_reduce()
    task3_movement()
    task4_manual_matmul()

    print("\n" + "="*60)
    print("Suggested next runs:")
    print("  DEBUG=1 python3 02_op_types.py   — count kernels per task")
    print("  DEBUG=3 python3 02_op_types.py   — read generated kernels")
    print("  VIZ=1   python3 02_op_types.py   — open graph visualizer")
