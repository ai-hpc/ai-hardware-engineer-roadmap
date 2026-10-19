"""Actually executed with apache-tvm==0.27.0.post1, LLVM CPU only."""
import json
from pathlib import Path
import time
import numpy as np
import tvm
from tvm import relax, s_tir
from tvm.relax.frontend import nn
from tvm.script import tirx as T


@T.prim_func(s_tir=True)
def affine_tail(A: T.Buffer((37,), 'float32'), B: T.Buffer((37,), 'float32')):
    T.func_attr({'global_symbol': 'affine_tail', 'tir.noalias': True})
    for i in range(37):
        with T.sblock('affine'):
            vi = T.axis.spatial(37, i)
            B[vi] = A[vi] * T.float32(2) + T.float32(1)


def schedule_affine(tile=8):
    if isinstance(tile, bool) or not isinstance(tile, int) or tile < 1:
        raise ValueError('tile must be a positive integer')
    schedule = s_tir.Schedule(affine_tail)
    block = schedule.get_sblock('affine')
    loop, = schedule.get_loops(block)
    outer, inner = schedule.split(loop, factors=[None, tile])
    schedule.vectorize(inner)
    return schedule


class AffineRelu(nn.Module):
    def forward(self, x, w, bias):
        return nn.relu(nn.matmul(x, w) + bias)


def relax_module(rows=3):
    module, parameters = AffineRelu().export_tvm(spec={'forward': {
        'x': nn.spec.Tensor((rows, 7), 'float32'),
        'w': nn.spec.Tensor((7, 5), 'float32'),
        'bias': nn.spec.Tensor((5,), 'float32')}})
    assert not parameters
    return module


def compile_relax(rows=3):
    original = relax_module(rows)
    lowered = relax.get_pipeline('zero')(original)
    executable = tvm.compile(lowered, tvm.target.Target('llvm'))
    return original, lowered, executable


def run_relax(executable, x, w, bias):
    vm = relax.VirtualMachine(executable, tvm.cpu())
    arrays = [tvm.runtime.tensor(a, tvm.cpu()) for a in (x, w, bias)]
    return vm['forward'](*arrays).numpy()


def measure_tiles(repeats=100):
    """Wall clock with synchronous CPU calls; timings are illustrative, not CI gates."""
    x = np.linspace(-2, 2, 37, dtype='float32')
    a = tvm.runtime.tensor(x, tvm.cpu())
    b = tvm.runtime.empty((37,), 'float32', tvm.cpu())
    measurements = []
    for tile in (1, 4, 8, 16):
        start = time.perf_counter()
        executable = tvm.compile(schedule_affine(tile).mod, tvm.target.Target('llvm'))
        build_seconds = time.perf_counter() - start
        function = executable['affine_tail']
        function(a, b)
        np.testing.assert_allclose(b.numpy(), 2*x+1, rtol=1e-6, atol=1e-6)
        samples = []
        for _ in range(5):
            start = time.perf_counter_ns()
            for _ in range(repeats):
                function(a, b)
            samples.append((time.perf_counter_ns()-start)/repeats)
        measurements.append(dict(tile=tile, compile_seconds=build_seconds,
                                 median_call_ns=float(np.median(samples))))
    return measurements


if __name__ == '__main__':
    print(json.dumps({'tvm_version': tvm.__version__, 'target': 'llvm',
                      'warning': '37 elements: Python/FFI overhead can dominate; no speedup gate',
                      'measurements': measure_tiles()}, indent=2))
