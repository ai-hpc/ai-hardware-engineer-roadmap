"""
Hands-on Tinygrad Hacking Example
Run this to see tinygrad's internals in action!

Requirements:
    pip install tinygrad numpy
"""

# Prefer project's tinygrad-source over other installations (e.g. openpilot fork)
import sys
from pathlib import Path
_project_tinygrad = Path(__file__).resolve().parent.parent / "tinygrad-source"
if _project_tinygrad.exists():
    sys.path.insert(0, str(_project_tinygrad))

import os
# Enable debug output to see what's happening
os.environ['DEBUG'] = '2'

from tinygrad import Tensor, Device

# Check if numpy is available
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Note: numpy not installed. Install with: pip install numpy")
    print("Continuing without numpy visualization...\n")

def print_result(tensor, name="Result"):
    """Print tensor result, with or without numpy"""
    if HAS_NUMPY:
        print(f"{name}: {tensor.numpy()}")
    else:
        print(f"{name}: {tensor.tolist()}")

print("=" * 60)
print("TINYGRAD HACKING DEMO")
print("=" * 60)

# 1. Basic Operations with Lazy Evaluation
print("\n1. LAZY EVALUATION DEMO")
print("-" * 60)

a = Tensor([1.0, 2.0, 3.0, 4.0])
b = Tensor([5.0, 6.0, 7.0, 8.0])

# These operations are NOT executed yet!
c = a + b
d = c * 2
e = d.relu()

print("Operations defined but not executed yet (lazy!)")
print(f"Tensor shape: {e.shape}")
print(f"Tensor dtype: {e.dtype}")
print("Operation tree exists in memory, waiting for .realize()")

# Now execute
print("\nCalling .realize() to execute...")
result = e.realize()
print_result(result, "Result")

# 2. Inspecting the Computation Graph
print("\n2. COMPUTATION GRAPH INSPECTION")
print("-" * 60)

x = Tensor([[1.0, 2.0], [3.0, 4.0]])
y = Tensor([[5.0, 6.0], [7.0, 8.0]])

# Matrix multiply + activation
z = (x @ y).relu()

print(f"Tensor shape: {z.shape}")
print(f"Tensor dtype: {z.dtype}")
print("Computation graph built, waiting for execution...")

result = z.realize()
print(f"After realize - Result:")
print_result(result)

# 3. Viewing the Schedule (Execution Plan)
print("\n3. EXECUTION SCHEDULE")
print("-" * 60)

# Create a multi-step computation
a = Tensor.randn(16, 16)
b = Tensor.randn(16, 16)
c = (a @ b).relu().sum()

print(f"Created computation: (a @ b).relu().sum()")
print(f"Input shapes: a={a.shape}, b={b.shape}")
print("Tinygrad will optimize and schedule the execution...")

# Execute
result = c.realize()
print_result(result, "\nFinal result")
print("All operations were fused and optimized automatically!")

# 4. Custom Activation Function
print("\n4. CUSTOM OPERATIONS")
print("-" * 60)

def swish(x: Tensor) -> Tensor:
    """Swish activation: x * sigmoid(x)"""
    return x * x.sigmoid()

def gelu_approx(x: Tensor) -> Tensor:
    """Approximate GELU activation"""
    return 0.5 * x * (1 + (x * 0.7978845608 * (1 + 0.044715 * x * x)).tanh())

x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

print("Input:")
print_result(x)
print("Swish:")
print_result(swish(x))
print("GELU:")
print_result(gelu_approx(x))

# 5. ShapeTracker Magic (Zero-Copy Operations)
print("\n5. SHAPETRACKER - ZERO-COPY OPERATIONS")
print("-" * 60)

x = Tensor.randn(4, 8, 16)
print(f"Original shape: {x.shape}")

# These operations don't copy data!
y = x.permute(2, 0, 1)
print(f"After permute: {y.shape}")

z = y.reshape(16, 32)
print(f"After reshape: {z.shape}")

# Expand needs compatible dimensions (size 1 can expand)
z_expanded = z.reshape(16, 32, 1)
w = z_expanded.expand(16, 32, 4)
print(f"After expand: {w.shape}")

print("\nNo data was copied - just metadata changes!")
print("This is the magic of ShapeTracker!")

# Realize to actually compute
result = w.realize()
print(f"Final shape after realize: {result.shape}")

# 6. Operation Fusion
print("\n6. OPERATION FUSION")
print("-" * 60)

# Multiple operations that can be fused
x = Tensor.randn(1000)
y = ((x + 1) * 2 - 0.5).relu()

print("Multiple operations defined: add, mul, sub, relu")
print("Tinygrad will fuse these into a single kernel!")
print("This is much faster than executing each operation separately")

result = y.realize()
print(f"Result computed efficiently with automatic fusion")
print(f"Output shape: {result.shape}")

# 7. Device Information
print("\n7. DEVICE INFORMATION")
print("-" * 60)

print(f"Default device: {Device.DEFAULT}")
print(f"Tinygrad is using: {Device.DEFAULT} backend")

# Try to get more device info
try:
    import os
    if 'CL' in Device.DEFAULT or 'CUDA' in Device.DEFAULT or 'NV' in Device.DEFAULT:
        print("GPU acceleration is enabled!")
    else:
        print(f"Running on: {Device.DEFAULT}")
except:
    print("Device information available through Device.DEFAULT")

print("\n" + "=" * 60)
print("DEMO COMPLETE!")
print("=" * 60)
print("\nKey Insights:")
print("- All operations are lazy until .realize()")
print("- You can inspect the computation graph at any time")
print("- Operations are automatically fused for efficiency")
print("- ShapeTracker enables zero-copy transformations")
print("- Everything is visible and hackable!")
