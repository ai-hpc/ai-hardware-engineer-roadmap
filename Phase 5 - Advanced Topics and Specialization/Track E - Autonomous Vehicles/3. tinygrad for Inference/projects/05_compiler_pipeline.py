"""
Project 5: Inspecting the Compiler Pipeline
=============================================
Goal: Understand every stage from Python tensor ops to GPU kernel:
  Tensor graph → Schedule → UOp IR → Kernel code → Execution

Run with:
  python3 05_compiler_pipeline.py
  DEBUG=4 python3 05_compiler_pipeline.py     # full IR dump
  VIZ=1   python3 05_compiler_pipeline.py     # graph visualizer (opens browser)
  NOOPT=1 python3 05_compiler_pipeline.py     # disable all optimizations
"""

import os
import numpy as np


# ---------------------------------------------------------------------------
# Task 1: Schedule inspection — what gets fused, what doesn't?
# ---------------------------------------------------------------------------
def task1_schedule():
    print("\n=== Task 1: Schedule Inspection ===")
    from tinygrad import Tensor

    a = Tensor.randn(64, 64)
    b = Tensor.randn(64, 64)

    print("\n[Example 1] Elementwise chain — should fuse to 1 kernel:")
    z = (a + b).relu().exp().neg()
    sched = z.schedule()
    print(f"  {len(sched)} kernel(s)")
    for i, item in enumerate(sched):
        print(f"  Kernel {i}: output_shape={item.outputs[0].shape if item.outputs else '?'}")

    print("\n[Example 2] Matmul + elementwise — matmul is a separate kernel:")
    z2 = (a @ b).relu()
    sched2 = z2.schedule()
    print(f"  {len(sched2)} kernel(s)")
    for i, item in enumerate(sched2):
        print(f"  Kernel {i}: AST op={item.ast.op if hasattr(item.ast, 'op') else type(item.ast).__name__}")

    print("\n[Example 3] Two independent reductions — should be 2 kernels:")
    s1 = a.sum(axis=0)
    s2 = a.sum(axis=1)
    both = s1 + s2[:1]  # force dependency
    sched3 = both.schedule()
    print(f"  {len(sched3)} kernel(s)")

    print("\n[Example 4] Shared computation — tinygrad deduplicates buffers:")
    x = Tensor.randn(32, 32)
    # Both paths share `x` — will tinygrad compute x only once?
    y = x.relu()
    z_shared = y.sum() + y.max()
    sched4 = z_shared.schedule()
    print(f"  Shared intermediate: {len(sched4)} kernel(s)")

    print("\nKEY: Elementwise ops fuse freely. Reductions break fusion boundaries.")
    print("Between a reduce and the next elementwise, a new kernel starts.")


# ---------------------------------------------------------------------------
# Task 2: UOp tree structure
# ---------------------------------------------------------------------------
def task2_uop_tree():
    print("\n=== Task 2: UOp Tree ===")
    from tinygrad import Tensor

    x = Tensor([1.0, 2.0, 3.0, 4.0])
    y = x + 1.0

    # Access the UOp (lazy computation node)
    uop = y.lazydata
    print(f"  UOp type:  {type(uop).__name__}")
    print(f"  UOp op:    {uop.op if hasattr(uop, 'op') else 'N/A'}")
    print(f"  UOp shape: {uop.shape}")

    # Print a simple tree for a small computation
    def print_uop_tree(node, depth=0, max_depth=4):
        if depth > max_depth:
            print("  " * depth + "...")
            return
        indent = "  " * depth
        op_name = node.op.name if hasattr(node.op, 'name') else str(type(node).__name__)
        print(f"{indent}{op_name}  shape={getattr(node, 'shape', '?')}")
        for child in (getattr(node, 'src', None) or []):
            print_uop_tree(child, depth+1, max_depth)

    print(f"\n  UOp tree for (x + 1.0).relu().sum():")
    chain = (x + 1.0).relu().sum()
    try:
        print_uop_tree(chain.lazydata, depth=2)
    except Exception as e:
        print(f"    (Tree printing skipped: {e})")
        print("    Use VIZ=1 to see the graph in the browser instead.")


# ---------------------------------------------------------------------------
# Task 3: Observe algebraic rewrites with NOOPT
# ---------------------------------------------------------------------------
def task3_algebraic_rewrites():
    print("\n=== Task 3: Algebraic Rewrites ===")
    print("  Set NOOPT=1 in your environment to disable and compare:")

    from tinygrad import Tensor

    # These should be rewritten to simpler forms:
    x = Tensor.randn(4, 4)

    print("\n  Testing constant-folding patterns:")
    tests = [
        ("x * 1.0",    lambda t: t * 1.0),
        ("x + 0.0",    lambda t: t + 0.0),
        ("x - 0.0",    lambda t: t - 0.0),
        ("x / 1.0",    lambda t: t / 1.0),
        ("x * 0.0",    lambda t: t * 0.0),
        ("x ** 1.0",   lambda t: t.pow(1.0)),
        ("x ** 2.0",   lambda t: t.pow(2.0)),
        ("log(exp(x))", lambda t: t.exp().log()),
    ]

    for name, fn in tests:
        result = fn(x)
        sched = result.schedule()
        # Count non-trivial kernels (exclude constant-zero outputs)
        print(f"  {name:20s}: {len(sched)} scheduled kernel(s)")

    print("\n  With NOOPT=1: these should produce more kernels (no rewrites).")
    print("  Run: NOOPT=1 DEBUG=3 python3 05_compiler_pipeline.py")


# ---------------------------------------------------------------------------
# Task 4: Kernel count analysis for common ML patterns
# ---------------------------------------------------------------------------
def task4_kernel_counts():
    print("\n=== Task 4: Kernel Count Analysis ===")
    from tinygrad import Tensor

    print("  Counting scheduled kernels for common ML ops:")
    print("  (Predicted vs actual — try to predict before running)\n")

    N, D = 32, 128

    patterns = {
        "Linear (matmul)":           lambda: (Tensor.randn(N, D) @ Tensor.randn(D, D)),
        "Linear + ReLU":             lambda: (Tensor.randn(N, D) @ Tensor.randn(D, D)).relu(),
        "Softmax":                   lambda: Tensor.randn(N, D).softmax(axis=1),
        "LayerNorm":                 lambda: Tensor.randn(N, D).layernorm(),
        "Attention (QK^T)":          lambda: (Tensor.randn(N, D) @ Tensor.randn(N, D).T).softmax(axis=1),
        "Conv2d (as matmul)":        lambda: Tensor.randn(N, 1, 8, 8).conv2d(Tensor.randn(4, 1, 3, 3)),
    }

    for name, fn in patterns.items():
        result = fn()
        n_kernels = len(result.schedule())
        print(f"  {name:28s}: {n_kernels} kernel(s)")


# ---------------------------------------------------------------------------
# Task 5: See generated kernel code
# ---------------------------------------------------------------------------
def task5_kernel_code():
    print("\n=== Task 5: Generated Kernel Code ===")
    print("  To see the kernel source code, run with DEBUG=3")
    print("  To use readable C output: CLANG=1 DEBUG=3 python3 05_compiler_pipeline.py")

    from tinygrad import Tensor, Device

    print(f"\n  Current device: {Device.DEFAULT}")

    # Annotated example: what does the generated kernel for relu look like?
    print("\n  ReLU kernel structure (when using CLANG backend):")
    print("""
  void kernel_relu(float* x, float* out, int n) {
      for (int i = 0; i < n; i++) {
          out[i] = max(0.0f, x[i]);    // elementwise, 1 thread per element
      }
  }
  """)

    print("  Reduction kernel structure:")
    print("""
  void kernel_sum(float* x, float* out, int rows, int cols) {
      for (int r = 0; r < rows; r++) {
          float acc = 0.0f;
          for (int c = 0; c < cols; c++) {    // reduction loop
              acc += x[r * cols + c];
          }
          out[r] = acc;
      }
  }
  """)

    print("  Run with DEBUG=3 to see the actual generated code for your device.")
    print("  With CUDA: you'll see PTX. With METAL: MSL. With CLANG: C.")

    # Execute something to trigger actual codegen output (visible with DEBUG≥3)
    x = Tensor.randn(8, 8)
    result = x.relu().sum(axis=1)
    result.realize()
    print("\n  Executed relu().sum() — check debug output above if DEBUG>=3.")


# ---------------------------------------------------------------------------
# Task 6: BEAM search — automatic kernel tuning
# ---------------------------------------------------------------------------
def task6_beam_search():
    print("\n=== Task 6: BEAM Search ===")
    import time
    from tinygrad import Tensor

    print("  BEAM search tries multiple loop orderings and picks the fastest.")
    print("  Enable with: BEAM=2 python3 05_compiler_pipeline.py")
    print(f"  Current BEAM value: {os.environ.get('BEAM', '0')} (0=disabled)")

    # Run a matmul — this is where BEAM search has the most impact
    N = 512
    a = Tensor.randn(N, N)
    b = Tensor.randn(N, N)

    # Warmup
    (a @ b).realize()

    # Timed run
    t0 = time.perf_counter()
    for _ in range(5):
        (a @ b).realize()
    elapsed = (time.perf_counter() - t0) / 5

    flops = 2 * N * N * N  # FLOPs for matmul
    gflops = flops / elapsed / 1e9
    print(f"\n  {N}x{N} matmul: {elapsed*1000:.2f}ms  ({gflops:.1f} GFLOPS)")
    print(f"  Run with BEAM=2 to compare: BEAM=2 python3 05_compiler_pipeline.py")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    task1_schedule()
    task2_uop_tree()
    task3_algebraic_rewrites()
    task4_kernel_counts()
    task5_kernel_code()
    task6_beam_search()

    print("\n" + "="*60)
    print("EXPLORATION COMMANDS:")
    print("  VIZ=1   python3 05_compiler_pipeline.py  — graph browser")
    print("  DEBUG=3 python3 05_compiler_pipeline.py  — kernel source")
    print("  DEBUG=4 python3 05_compiler_pipeline.py  — full IR dump")
    print("  NOOPT=1 python3 05_compiler_pipeline.py  — no rewrites")
    print("  BEAM=4  python3 05_compiler_pipeline.py  — tune kernels")
