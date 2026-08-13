"""
Project 7: Custom Backend — CPU via ctypes
===========================================
Implements a minimal tinygrad backend targeting the CPU.
Reuses tinygrad's C code generator but provides its own
Allocator, Compiler, and Runner.

Goal: Understand the three parts every tinygrad backend must implement:
  1. Allocator  — allocate/free device memory, host↔device copies
  2. Compiler   — compile generated C source to a shared library
  3. Runner     — load and execute the compiled library

Usage:
  python3 07_custom_backend.py              — run self-tests
  python3 07_custom_backend.py benchmark    — compare with CLANG backend
"""

import ctypes
import ctypes.util
import subprocess
import tempfile
import os
import sys
import numpy as np


# ============================================================================
# Part A: Allocator
# ============================================================================
class MyAllocator:
    """
    Manages memory for our custom device.
    We use Python ctypes arrays as 'device' memory — simple and visible.
    """

    def alloc(self, size: int) -> ctypes.Array:
        """Allocate `size` bytes. Returns a ctypes char array."""
        return (ctypes.c_uint8 * size)()

    def free(self, buf: ctypes.Array):
        """Free the buffer (Python GC handles it, but we make it explicit)."""
        del buf

    def copyin(self, dst: ctypes.Array, src: memoryview):
        """Copy from host (numpy memoryview) into device buffer."""
        nbytes = len(src)
        ctypes.memmove(dst, (ctypes.c_uint8 * nbytes).from_buffer(bytearray(src)), nbytes)

    def copyout(self, dst: memoryview, src: ctypes.Array):
        """Copy from device buffer to host (numpy memoryview)."""
        nbytes = len(dst)
        ctypes.memmove((ctypes.c_uint8 * nbytes).from_buffer(dst), src, nbytes)


def test_allocator():
    print("\n--- Part A: Allocator ---")
    alloc = MyAllocator()

    buf = alloc.alloc(16)
    print(f"  Allocated {ctypes.sizeof(buf)} bytes")

    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    alloc.copyin(buf, memoryview(data))

    out = np.zeros(4, dtype=np.float32)
    alloc.copyout(memoryview(out), buf)

    print(f"  Written:   {data}")
    print(f"  Read back: {out}")
    assert np.allclose(data, out), "Allocator round-trip failed!"
    print("  ✓ Allocator round-trip test passed")

    alloc.free(buf)


# ============================================================================
# Part B: Compiler
# ============================================================================
C_KERNEL_HEADER = """
#include <math.h>
#include <stdlib.h>
#include <string.h>
#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))
"""

class MyCompiler:
    """
    Compiles a C source string to a shared library (.so) using clang.
    Returns the compiled binary as bytes.
    """

    def compile(self, src: str) -> bytes:
        """Compile C source string → .so bytes."""
        full_src = C_KERNEL_HEADER + src
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "kernel.c")
            lib_path = os.path.join(tmpdir, "kernel.so")

            with open(src_path, "w") as f:
                f.write(full_src)

            result = subprocess.run(
                ["clang", "-O2", "-march=native", "-shared", "-fPIC",
                 "-o", lib_path, src_path, "-lm"],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                raise RuntimeError(f"Compilation failed:\n{result.stderr}\nSource:\n{full_src[:500]}")

            with open(lib_path, "rb") as f:
                return f.read()


def test_compiler():
    print("\n--- Part B: Compiler ---")
    compiler = MyCompiler()

    trivial_c = """
void set_value(float* buf) {
    buf[0] = 42.0f;
    buf[1] = 99.0f;
}
"""
    lib_bytes = compiler.compile(trivial_c)
    print(f"  Compiled {len(trivial_c)} bytes C → {len(lib_bytes)} bytes .so")
    print("  ✓ Compiler test passed")
    return lib_bytes


# ============================================================================
# Part C: Runner
# ============================================================================
class MyRunner:
    """
    Loads a compiled .so from bytes and executes its kernel function.
    """

    def __init__(self, lib_bytes: bytes, fn_name: str, n_args: int):
        """
        lib_bytes: compiled shared library binary
        fn_name:   name of the kernel function to call
        n_args:    number of pointer arguments the kernel expects
        """
        self._tmpfile = tempfile.NamedTemporaryFile(suffix=".so", delete=False)
        self._tmpfile.write(lib_bytes)
        self._tmpfile.flush()

        self.lib = ctypes.CDLL(self._tmpfile.name)
        self.fn = getattr(self.lib, fn_name)
        self.fn.restype = None
        self.fn.argtypes = [ctypes.c_void_p] * n_args

    def __call__(self, *bufs: ctypes.Array):
        """Execute the kernel with the given buffers."""
        void_ptrs = [ctypes.cast(buf, ctypes.c_void_p) for buf in bufs]
        self.fn(*void_ptrs)

    def __del__(self):
        try:
            os.unlink(self._tmpfile.name)
        except Exception:
            pass


def test_runner(lib_bytes: bytes):
    print("\n--- Part C: Runner ---")

    runner = MyRunner(lib_bytes, "set_value", n_args=1)
    alloc = MyAllocator()

    buf = alloc.alloc(8)  # 2 float32s
    runner(buf)

    out = np.zeros(2, dtype=np.float32)
    alloc.copyout(memoryview(out), buf)
    print(f"  After kernel execution: {out}")
    assert np.allclose(out, [42.0, 99.0]), f"Expected [42, 99], got {out}"
    print("  ✓ Runner test passed")


# ============================================================================
# Part D: Integration — element-wise and reduce kernels
# ============================================================================
ELEMENTWISE_ADD_C = """
void elementwise_add(float* a, float* b, float* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

void elementwise_relu(float* a, float* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] > 0.0f ? a[i] : 0.0f;
    }
}

void reduce_sum(float* a, float* out, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; i++) acc += a[i];
    out[0] = acc;
}
"""


def test_custom_kernels():
    print("\n--- Part D: Custom Kernels ---")
    compiler = MyCompiler()
    alloc = MyAllocator()

    lib_bytes = compiler.compile(ELEMENTWISE_ADD_C)

    # elementwise_add
    a_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b_np = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    n = len(a_np)

    a_buf = alloc.alloc(n * 4)
    b_buf = alloc.alloc(n * 4)
    out_buf = alloc.alloc(n * 4)

    alloc.copyin(a_buf, memoryview(a_np))
    alloc.copyin(b_buf, memoryview(b_np))

    lib = ctypes.CDLL(MyRunner(lib_bytes, "elementwise_add", 3)._tmpfile.name)
    lib.elementwise_add.restype = None
    lib.elementwise_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.elementwise_add(
        ctypes.cast(a_buf, ctypes.c_void_p),
        ctypes.cast(b_buf, ctypes.c_void_p),
        ctypes.cast(out_buf, ctypes.c_void_p),
        ctypes.c_int(n)
    )

    out_np = np.zeros(n, dtype=np.float32)
    alloc.copyout(memoryview(out_np), out_buf)

    expected = a_np + b_np
    err = np.abs(out_np - expected).max()
    print(f"  elementwise_add: {a_np} + {b_np}")
    print(f"  Result:   {out_np}")
    print(f"  Error: {err:.2e}  {'✓' if err < 1e-6 else '✗'}")

    # reduce_sum
    sum_lib = ctypes.CDLL(MyRunner(lib_bytes, "reduce_sum", 3)._tmpfile.name)
    sum_lib.reduce_sum.restype = None
    sum_lib.reduce_sum.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    sum_buf = alloc.alloc(4)
    sum_lib.reduce_sum(ctypes.cast(a_buf, ctypes.c_void_p), ctypes.cast(sum_buf, ctypes.c_void_p), ctypes.c_int(n))
    sum_np = np.zeros(1, dtype=np.float32)
    alloc.copyout(memoryview(sum_np), sum_buf)
    expected_sum = a_np.sum()
    print(f"\n  reduce_sum({a_np}): got={sum_np[0]:.1f}, expected={expected_sum:.1f}  {'✓' if abs(sum_np[0]-expected_sum)<1e-5 else '✗'}")


# ============================================================================
# Part E: Benchmark vs CLANG backend
# ============================================================================
def benchmark():
    print("\n--- Part E: Benchmark vs CLANG ---")
    import time

    compiler = MyCompiler()
    alloc = MyAllocator()

    N = 256 * 256

    MATMUL_C = f"""
void add_large(float* a, float* b, float* out) {{
    for (int i = 0; i < {N}; i++) {{
        out[i] = a[i] + b[i];
    }}
}}
"""
    lib_bytes = compiler.compile(MATMUL_C)
    add_lib = ctypes.CDLL(MyRunner(lib_bytes, "add_large", 3)._tmpfile.name)
    add_lib.add_large.restype = None
    add_lib.add_large.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

    a_np = np.random.randn(N).astype(np.float32)
    b_np = np.random.randn(N).astype(np.float32)
    a_buf = alloc.alloc(N * 4); alloc.copyin(a_buf, memoryview(a_np))
    b_buf = alloc.alloc(N * 4); alloc.copyin(b_buf, memoryview(b_np))
    out_buf = alloc.alloc(N * 4)

    # Warmup
    add_lib.add_large(ctypes.cast(a_buf,ctypes.c_void_p), ctypes.cast(b_buf,ctypes.c_void_p), ctypes.cast(out_buf,ctypes.c_void_p))

    REPS = 20
    t0 = time.perf_counter()
    for _ in range(REPS):
        add_lib.add_large(ctypes.cast(a_buf,ctypes.c_void_p), ctypes.cast(b_buf,ctypes.c_void_p), ctypes.cast(out_buf,ctypes.c_void_p))
    custom_ms = (time.perf_counter() - t0) * 1000 / REPS

    try:
        import os as _os
        _os.environ.setdefault('CLANG', '1')
        from tinygrad import Tensor, Device
        a_tg = Tensor(a_np)
        b_tg = Tensor(b_np)
        (a_tg + b_tg).realize()  # warmup
        t0 = time.perf_counter()
        for _ in range(REPS):
            (a_tg + b_tg).realize()
        tg_ms = (time.perf_counter() - t0) * 1000 / REPS
        print(f"  {N} float32 add ({N*4//1024}KB each):")
        print(f"  Custom backend:  {custom_ms:.3f}ms")
        print(f"  tinygrad CLANG:  {tg_ms:.3f}ms")
        print(f"  Overhead ratio:  {custom_ms/tg_ms:.1f}x")
    except Exception as e:
        print(f"  Custom backend:  {custom_ms:.3f}ms")
        print(f"  (tinygrad comparison skipped: {e})")

    print("\nNOTE: tinygrad's backend has overhead from:")
    print("  - Kernel compilation caching")
    print("  - Multi-dimensional index computation")
    print("  - Support for strides, offsets, and shapes")
    print("  - Global/local work size threading")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("=== Project 7: Custom Backend ===")

    test_allocator()
    lib_bytes = test_compiler()
    test_runner(lib_bytes)
    test_custom_kernels()

    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        benchmark()
    else:
        print("\nRun with 'benchmark' argument for performance comparison:")
        print("  python3 07_custom_backend.py benchmark")

    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("  1. Add multi-threading (OpenMP) to the element-wise loop")
    print("  2. Add SIMD vectorization hints (__m256 or auto-vectorize)")
    print("  3. Implement a real reduce kernel with parallel tree reduction")
    print("  4. Integrate as a proper tinygrad Device class (see Guide.md Part 8)")
    print("  5. Target a different backend: RISC-V emulator, custom simulator")
