"""
Intro: Tinygrad Internals Demo
================================
Run this first to see lazy evaluation, fusion, and ShapeTracker in action.

Usage:
  python3 00_intro.py
  DEBUG=1 python3 00_intro.py   # see kernel counts
  DEBUG=3 python3 00_intro.py   # see generated kernel code
"""

# Prefer local tinygrad-source submodule over system installation
import sys
from pathlib import Path
_project_tinygrad = Path(__file__).resolve().parent.parent.parent / "tinygrad-source"
if _project_tinygrad.exists():
    sys.path.insert(0, str(_project_tinygrad))

import os
os.environ.setdefault('DEBUG', '2')

from tinygrad import Tensor, Device

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Note: numpy not installed. Install with: pip install numpy\n")

def show(tensor, name="Result"):
    if HAS_NUMPY:
        print(f"{name}: {tensor.numpy()}")
    else:
        print(f"{name}: {tensor.tolist()}")

print("=" * 60)
print("TINYGRAD INTERNALS DEMO")
print("=" * 60)


# 1. Lazy Evaluation
print("\n1. LAZY EVALUATION")
print("-" * 40)

a = Tensor([1.0, 2.0, 3.0, 4.0])
b = Tensor([5.0, 6.0, 7.0, 8.0])

c = a + b
d = c * 2
e = d.relu()

print("Operations defined but NOT executed (lazy graph).")
print(f"  Shape: {e.shape}, Dtype: {e.dtype}")
print("Calling .realize()...")
show(e.realize(), "a+b * 2 relu'd")


# 2. Operation Fusion
print("\n2. OPERATION FUSION")
print("-" * 40)

x = Tensor.randn(1000)
# All four ops → fused into ONE kernel
y = ((x + 1) * 2 - 0.5).relu()
print("(x+1)*2-0.5 then relu — should be 1 kernel with DEBUG=1")
y.realize()
print(f"Output shape: {y.shape}")


# 3. Schedule Inspection
print("\n3. SCHEDULE (KERNEL PLAN)")
print("-" * 40)

a = Tensor.randn(16, 16)
b = Tensor.randn(16, 16)
c = (a @ b).relu().sum()

sched = c.schedule()
print(f"(a @ b).relu().sum() → {len(sched)} scheduled kernel(s)")
print("  Matmul + reduce each force a new kernel boundary.")
c.realize()


# 4. Custom Activations
print("\n4. CUSTOM ACTIVATIONS FROM PRIMITIVES")
print("-" * 40)

def swish(x: Tensor) -> Tensor:
    return x * x.sigmoid()

def gelu(x: Tensor) -> Tensor:
    return 0.5 * x * (1 + (x * 0.7978845608 * (1 + 0.044715 * x * x)).tanh())

x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
show(x, "Input")
show(swish(x), "Swish")
show(gelu(x), "GELU")


# 5. ShapeTracker — Zero-Copy Ops
print("\n5. SHAPETRACKER: ZERO-COPY MOVEMENT OPS")
print("-" * 40)

x = Tensor.randn(4, 8, 16)
print(f"Original:      {x.shape}")

y = x.permute(2, 0, 1)
print(f"After permute: {y.shape}  ← no data copied")

z = y.reshape(16, 32)
print(f"After reshape: {z.shape}  ← no data copied")

z_exp = z.reshape(16, 32, 1).expand(16, 32, 4)
print(f"After expand:  {z_exp.shape}  ← no data copied")

result = z_exp.realize()  # data moves here, once
print(f"After realize: {result.shape}  ← data moved exactly once")


# 6. Device Info
print("\n6. DEVICE")
print("-" * 40)
print(f"Default backend: {Device.DEFAULT}")
print("Set CLANG=1 for CPU/C output, NV=1 for CUDA, METAL=1 for Apple GPU")


print("\n" + "=" * 60)
print("DONE. Next: work through projects 01–07 in order.")
print("=" * 60)
print("""
Key insights:
  • All ops are lazy until .realize() or .numpy()
  • Elementwise ops fuse automatically into single kernels
  • Movement ops (reshape/permute/expand) cost zero — metadata only
  • ReduceOps break fusion: each reduce starts a new kernel
  • Use DEBUG=1..4 and VIZ=1 to inspect every stage
""")
