import importlib.util
from pathlib import Path
import tempfile
import unittest

TVM = importlib.util.find_spec('tvm') is not None
if TVM:
    import numpy as np
    import tvm
    from tvm import relax
    from compiler_lab import affine_tail, schedule_affine, compile_relax, run_relax


@unittest.skipUnless(TVM, 'optional pinned TVM CPU environment required')
class CompilerContract(unittest.TestCase):
    def test_pinned_api_version(self):
        self.assertEqual(tvm.__version__, '0.27.0.post1')

    def test_unscheduled_llvm_reference(self):
        x = np.linspace(-10, 10, 37, dtype='float32')
        a = tvm.runtime.tensor(x, tvm.cpu())
        b = tvm.runtime.empty((37,), 'float32', tvm.cpu())
        tvm.compile(affine_tail, tvm.target.Target('llvm'))['affine_tail'](a, b)
        np.testing.assert_allclose(b.numpy(), 2*x+1, rtol=1e-6, atol=1e-6)

    def test_vector_tails_all_positions_and_canaries(self):
        # Output is a 37-value view into a larger allocation. Sentinels expose
        # stores beyond logical extent, independently of TVM buffer shape checks.
        for tile in [1, 4, 8, 16, 64]:
            with self.subTest(tile=tile):
                x = np.arange(37, dtype='float32') - 18
                output = np.full(64, -999, dtype='float32')
                a = tvm.runtime.tensor(x, tvm.cpu())
                # Runtime validates shape, so a DLPack view supplies logical shape
                # while retaining a larger physical backing buffer.
                import tvm_ffi
                view = tvm_ffi.from_dlpack(output[:37])
                tvm.compile(schedule_affine(tile).mod, tvm.target.Target('llvm'))['affine_tail'](a, view)
                np.testing.assert_array_equal(output[:37], 2*x+1)
                np.testing.assert_array_equal(output[37:], -999)

    def test_schedule_records_predicated_tail(self):
        script = schedule_affine(8).mod.script()
        self.assertIn('37', script)
        self.assertIn('T.where', script)
        self.assertIn('vectorized', script)

    def test_relax_matmul_bias_relu_odd_shapes(self):
        rng = np.random.default_rng(4)
        for rows in [1, 3, 9]:
            x = rng.normal(size=(rows, 7)).astype('float32')
            w = rng.normal(size=(7, 5)).astype('float32')
            b = rng.normal(size=(5,)).astype('float32')
            original, lowered, executable = compile_relax(rows)
            self.assertIn('matmul', original.script())
            self.assertIn('call_tir', lowered.script())
            actual = run_relax(executable, x, w, b)
            np.testing.assert_allclose(actual, np.maximum(x@w+b, 0), rtol=2e-5, atol=2e-5)

    def test_export_reload_real_shared_library(self):
        _, _, executable = compile_relax(3)
        x = np.arange(21, dtype='float32').reshape(3, 7) / 7
        w = np.arange(35, dtype='float32').reshape(7, 5) / 5 - 3
        bias = np.array([1, -2, 3, -4, 5], dtype='float32')
        with tempfile.TemporaryDirectory() as directory:
            filename = str(Path(directory) / 'model.so')
            executable.export_library(filename)
            loaded = tvm.runtime.load_module(filename)
            actual = run_relax(loaded, x, w, bias)
            np.testing.assert_allclose(actual, np.maximum(x@w+bias, 0), rtol=2e-5, atol=2e-5)

    def test_wrong_runtime_shape_rejected(self):
        _, _, executable = compile_relax(3)
        with self.assertRaisesRegex(RuntimeError, "shape.*mismatch"):
            run_relax(executable, np.zeros((2, 7), 'float32'),
                      np.zeros((7, 5), 'float32'), np.zeros((5,), 'float32'))

    def test_invalid_schedule_factor(self):
        for tile in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                schedule_affine(tile)


if __name__ == '__main__':
    unittest.main()
