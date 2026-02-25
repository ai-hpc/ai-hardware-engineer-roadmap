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
import os

os.environ['DEBUG'] = '2'

a = Tensor([1, 2, 3, 4])
b = Tensor([5, 6, 7, 8])

# Operations are lazy — nothing computed yet
c = a + b
d = c * 2

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

os.environ['DEBUG'] = '4'  # Higher = more verbose

a = Tensor.randn(32, 32)
b = Tensor.randn(32, 32)
c = a @ b

# Prints IR stages: initial ops → after optimization → kernel codegen
c.realize()
```

## 3. Exploring the Scheduler

```python
from tinygrad import Tensor

x = Tensor([1, 2, 3, 4])
y = Tensor([5, 6, 7, 8])
z = (x + y) * 2

schedule = z.schedule()
print(f"Number of kernels: {len(schedule)}")
for i, si in enumerate(schedule):
    print(f"\nKernel {i}:")
    print(f"  AST: {si.ast}")
    print(f"  Buffers: {si.bufs}")
```

## 4. Custom Operations

```python
from tinygrad import Tensor

def custom_activation(x: Tensor) -> Tensor:
    """Custom activation: x^2 + sin(x)"""
    return x * x + x.sin()

x = Tensor([0.5, 1.0, 1.5, 2.0])
y = custom_activation(x)
print(y.numpy())
# Decomposed into tinygrad primitives automatically
```

## 5. Inspecting Generated Kernels

```python
from tinygrad import Tensor, Device
import os

os.environ['DEBUG'] = '3'
Device.DEFAULT = "CLANG"  # readable C output

a = Tensor.randn(1024)
b = a.relu().exp()
b.realize()
# Prints the actual C/CUDA/Metal code generated
```

## 6. Custom Backend Skeleton

```python
from tinygrad.device import Compiled, Allocator

class MyCustomDevice(Compiled):
    def __init__(self):
        super().__init__(
            allocator=MyAllocator(),
            compiler=MyCompiler(),
            runtime=MyRuntime()
        )

# See projects/07_custom_backend.py for a complete working implementation
```

## 7. Debugging with the Execution Schedule

```python
from tinygrad import Tensor

x = Tensor.randn(10, 10)
y = Tensor.randn(10, 10)

z = (x @ y).relu()
w = z.sum(axis=0)
result = w.softmax()

schedule = result.schedule()
print(f"Execution plan: {len(schedule)} steps")
for i, si in enumerate(schedule):
    print(f"\nStep {i}: {si.ast}")
```

## 8. Watching Operation Fusion

```python
from tinygrad import Tensor
import os

os.environ['DEBUG'] = '4'

x = Tensor.randn(100, 100)
y = x + 1
z = y * 2
w = z - 1
result = w / 2

# Tinygrad fuses all of these into fewer kernels
result.realize()
```

## 9. Zero-Copy Movement Ops

```python
from tinygrad import Tensor

x = Tensor.randn(4, 8, 16)

# These don't move data — just change ShapeTracker metadata
y = x.permute(2, 0, 1)
z = y.reshape(16, 32)
z_exp = z.reshape(16, 32, 1)
w = z_exp.expand(16, 32, 4)

print("Shape:", w.shape)
print("No data copied yet!")

result = w.realize()  # Data moves here
```

## Debug Environment Variables

| Variable | Effect |
|----------|--------|
| `DEBUG=1` | Kernel count and timing per `.realize()` |
| `DEBUG=2` | Kernel names and output shapes |
| `DEBUG=3` | Generated kernel source code (C/CUDA/MSL) |
| `DEBUG=4` | Full UOp IR at every optimization stage |
| `VIZ=1` | Open browser-based computation graph visualizer |
| `BEAM=2` | Enable BEAM search kernel auto-tuning |
| `NOOPT=1` | Disable all algebraic rewrites (baseline) |
| `CLANG=1` | Force CPU/Clang backend (readable C output) |

## Key Takeaways

1. **Everything is visible** — No hidden C++ magic
2. **Lazy evaluation** — Build graphs, optimize, then execute
3. **Hackable at every level** — From high-level ops to kernel code
4. **Great for learning** — See exactly how deep learning works
5. **Easy to extend** — Add new ops, backends, or optimizations

## Resources

- [tinygrad GitHub](https://github.com/tinygrad/tinygrad)
- [Architecture walkthrough](https://github.com/tinygrad/tinygrad/blob/master/docs/abstractions.py)
- [Discord community](https://discord.gg/tinygrad)
- See [projects/07_custom_backend.py](../projects/07_custom_backend.py) for a full custom backend
