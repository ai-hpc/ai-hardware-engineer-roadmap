# Hacking Tinygrad: Exploring the Compiler and IR

## What Makes Tinygrad Hackable?

Unlike PyTorch where the compiler and intermediate representation (IR) are hidden in C++/CUDA, tinygrad exposes everything in Python. You can:
- See the computation graph
- Inspect the IR at every stage
- Modify operations before execution
- Add custom backends
- Debug kernel generation

## Setup

```bash
pip install tinygrad
```

Or use the source in `../tinygrad-source/` for development.

## 1. Inspecting the Computation Graph

```python
from tinygrad import Tensor, Device

# Enable debug mode to see what's happening
import os
os.environ['DEBUG'] = '2'

# Create some tensors
a = Tensor([1, 2, 3, 4])
b = Tensor([5, 6, 7, 8])

# Perform operations (lazy evaluation)
c = a + b
d = c * 2

# Before .realize(), nothing has been computed yet
# The operations are stored in a graph (UOp)

# Inspect the UOp (graph node)
print("UOp:", d.uop)
print("Shape:", d.shape)
print("Dtype:", d.dtype)

# Now execute
result = d.realize()
print("Result:", result.numpy())
```

## 2. Viewing the IR (Intermediate Representation)

```python
from tinygrad import Tensor
import os

# Set debug level to see IR
os.environ['DEBUG'] = '4'  # Higher number = more verbose

# Simple matrix multiplication
a = Tensor.randn(32, 32)
b = Tensor.randn(32, 32)
c = a @ b  # Matrix multiply

# This will print the IR stages:
# - Initial ops
# - After optimization
# - Kernel code generation
c.realize()
```

## 3. Exploring the Scheduler

```python
from tinygrad import Tensor

# Create a computation
x = Tensor([1, 2, 3, 4])
y = Tensor([5, 6, 7, 8])
z = (x + y) * 2

# Get the schedule (execution plan)
schedule = z.schedule()

print(f"Number of kernels to execute: {len(schedule)}")
for i, si in enumerate(schedule):
    print(f"\nKernel {i}:")
    print(f"  AST: {si.ast}")
    print(f"  Buffers: {si.bufs}")
```

## 4. Custom Operations

```python
from tinygrad import Tensor

# Create custom operations by composing primitives
def custom_activation(x: Tensor) -> Tensor:
    """Custom activation: x^2 + sin(x)"""
    return x * x + x.sin()

# Use it
x = Tensor([0.5, 1.0, 1.5, 2.0])
y = custom_activation(x)
print(y.numpy())

# The operations are decomposed into tinygrad primitives
```

## 5. Inspecting Generated Kernels

```python
from tinygrad import Tensor, Device
import os

# Enable kernel printing
os.environ['DEBUG'] = '3'

# Force a specific device to see its kernel code
Device.DEFAULT = "CLANG"  # or "CUDA", "METAL", etc.

# Simple operation
a = Tensor.randn(1024)
b = a.relu().exp()
b.realize()

# This prints the actual C/CUDA/Metal code generated!
```

## 6. Custom Backend Example

```python
# tinygrad makes it easy to add new backends
# See tinygrad-source/tinygrad/device for the structure

from tinygrad.device import Compiled, Allocator

class MyCustomDevice(Compiled):
    def __init__(self):
        super().__init__(
            allocator=MyAllocator(),
            compiler=MyCompiler(),
            runtime=MyRuntime()
        )

# Register your device
# Device.register("MYDEVICE", MyCustomDevice)
```

## 7. Debugging with Graph Visualization

```python
from tinygrad import Tensor

# Create a more complex computation
x = Tensor.randn(10, 10)
y = Tensor.randn(10, 10)

# Multi-step computation
z = (x @ y).relu()
w = z.sum(axis=0)
result = w.softmax()

# Get the execution plan
schedule = result.schedule()
print(f"Execution plan has {len(schedule)} steps")

# Each step shows the operation
for i, si in enumerate(schedule):
    print(f"\nStep {i}: {si.ast}")
```

## 8. Understanding the Optimization Pipeline

```python
from tinygrad import Tensor
import os

# See optimization stages
os.environ['DEBUG'] = '4'

# Create inefficient computation
x = Tensor.randn(100, 100)
y = x + 1
z = y * 2
w = z - 1
result = w / 2

# Tinygrad will optimize this into fewer operations
result.realize()

# Watch how operations get fused!
```

## 9. Understanding Movement Ops (Zero-Copy)

```python
from tinygrad import Tensor

# ShapeTracker makes reshapes/permutes zero-copy
x = Tensor.randn(4, 8, 16)

# These operations don't move data!
y = x.permute(2, 0, 1)  # Reorder dimensions
z = y.reshape(16, 32)   # Reshape
w = z.expand(16, 32, 10) # Broadcast

# Inspect shape and dtype
print("Shape:", w.shape)
print("Dtype:", w.dtype)
print("No data copied yet!")

# Data only moves when necessary
result = w.realize()
```

## Key Takeaways

1. **Everything is visible** - No hidden C++ magic
2. **Lazy evaluation** - Build graphs, optimize, then execute
3. **Hackable at every level** - From high-level ops to kernel code
4. **Great for learning** - See exactly how deep learning works
5. **Easy to extend** - Add new ops, backends, or optimizations

## Next Steps

- Study the source: `../tinygrad-source/`
- Read the code (it's surprisingly readable!)
- Try modifying operations
- Implement a custom backend
- Contribute optimizations

## Resources

- [tinygrad GitHub](https://github.com/tinygrad/tinygrad)
- [Architecture docs](https://github.com/tinygrad/tinygrad/blob/master/docs/abstractions.py)
- [Discord community](https://discord.gg/tinygrad)
