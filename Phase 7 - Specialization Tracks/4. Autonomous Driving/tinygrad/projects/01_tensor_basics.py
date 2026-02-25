"""
Project 1: Tensor Basics and Lazy Evaluation
=============================================
Goal: Understand that tinygrad doesn't compute until .realize(), and see
op fusion in action.

Run with:
  python3 01_tensor_basics.py
  DEBUG=1 python3 01_tensor_basics.py       # see kernel count
  DEBUG=3 python3 01_tensor_basics.py       # see generated C code
"""

import os
import time
import numpy as np


# ---------------------------------------------------------------------------
# Task 1: Creating tensors with all factory methods
# ---------------------------------------------------------------------------
def task1_create_tensors():
    print("\n=== Task 1: Creating Tensors ===")
    from tinygrad import Tensor

    # From Python list
    t1 = Tensor([1, 2, 3, 4, 5])
    print(f"from list:    shape={t1.shape}  dtype={t1.dtype}  values={t1.numpy()}")

    # From numpy
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    t2 = Tensor(arr)
    print(f"from numpy:   shape={t2.shape}  dtype={t2.dtype}")
    print(t2.numpy())

    # Factory methods
    print(f"zeros(2,3):   \n{Tensor.zeros(2, 3).numpy()}")
    print(f"ones(2,3):    \n{Tensor.ones(2, 3).numpy()}")
    print(f"eye(3):       \n{Tensor.eye(3).numpy()}")
    print(f"arange(0,10,2): {Tensor.arange(0, 10, 2).numpy()}")

    t_rand = Tensor.randn(3, 3)
    print(f"randn(3,3):   shape={t_rand.shape}  mean≈{t_rand.mean().item():.3f}")


# ---------------------------------------------------------------------------
# Task 2: Observe op fusion — how many kernels for a + relu + sum?
# ---------------------------------------------------------------------------
def task2_op_fusion():
    print("\n=== Task 2: Op Fusion ===")
    from tinygrad import Tensor

    a = Tensor.randn(1024, 1024)
    b = Tensor.randn(1024, 1024)

    print("\nFUSED (let the graph grow, one .realize() at the end):")
    print("  Expected: 1 kernel — all ops fused")
    t_start = time.perf_counter()
    result = (a + b).relu().sum()
    result.realize()
    print(f"  Time: {(time.perf_counter()-t_start)*1000:.2f}ms")

    print("\nUNFUSED (.realize() after each op):")
    print("  Expected: 3 kernels — one per op")
    t_start = time.perf_counter()
    c = (a + b).realize()
    d = c.relu().realize()
    e = d.sum().realize()
    print(f"  Time: {(time.perf_counter()-t_start)*1000:.2f}ms")

    print("\nKEY INSIGHT: Lazy evaluation allows tinygrad to fuse ops")
    print("  into a single kernel with NO intermediate buffers.")


# ---------------------------------------------------------------------------
# Task 3: Read the schedule before execution
# ---------------------------------------------------------------------------
def task3_schedule():
    print("\n=== Task 3: The Schedule ===")
    from tinygrad import Tensor

    a = Tensor.randn(4, 4)
    b = Tensor.randn(4, 4)

    # Build a graph: matmul → relu → sum
    result = (a @ b).relu().sum(axis=0)

    # Get the schedule WITHOUT executing
    sched = result.schedule()
    print(f"Number of scheduled kernels: {len(sched)}")
    for i, item in enumerate(sched):
        print(f"\n--- Kernel {i} ---")
        print(f"  AST: {item.ast}")
        # number of output buffers
        print(f"  Output bufs: {[b.shape for b in item.outputs]}")

    print("\nNow actually executing...")
    out = result.numpy()
    print(f"Result shape: {out.shape}")


# ---------------------------------------------------------------------------
# Task 4: Manual vs automatic fusion comparison
# ---------------------------------------------------------------------------
def task4_fusion_comparison():
    print("\n=== Task 4: Fusion Comparison (check DEBUG=1 output) ===")
    from tinygrad import Tensor

    a = Tensor.randn(512, 512)

    print("\nComputation: a.sqrt() + a.exp() + a.relu()")

    print("\n[Fused] — single realize at end:")
    fused = a.sqrt() + a.exp() + a.relu()
    fused.realize()

    print("\n[Unfused] — realize after each op:")
    s = a.sqrt().realize()
    e = a.exp().realize()
    r = a.relu().realize()
    result = (s + e + r).realize()

    print("\nRun with DEBUG=1 to see kernel counts for each approach.")


# ---------------------------------------------------------------------------
# Task 5: Verify output correctness vs numpy
# ---------------------------------------------------------------------------
def task5_verify_correctness():
    print("\n=== Task 5: Verify Against NumPy ===")
    from tinygrad import Tensor

    np.random.seed(42)
    data = np.random.randn(4, 4).astype(np.float32)

    # tinygrad
    t = Tensor(data)
    tg_result = (t + 1.0).relu().sum(axis=1).numpy()

    # numpy reference
    np_result = np.maximum(data + 1.0, 0).sum(axis=1)

    print(f"tinygrad: {tg_result}")
    print(f"numpy:    {np_result}")
    print(f"Max abs error: {np.abs(tg_result - np_result).max():.2e}")
    assert np.allclose(tg_result, np_result, atol=1e-5), "MISMATCH!"
    print("✓ Results match!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    task1_create_tensors()
    task2_op_fusion()
    task3_schedule()
    task4_fusion_comparison()
    task5_verify_correctness()

    print("\n" + "="*60)
    print("Done! Now try running with:")
    print("  DEBUG=1 python3 01_tensor_basics.py   — see kernel counts")
    print("  DEBUG=3 python3 01_tensor_basics.py   — see generated C code")
