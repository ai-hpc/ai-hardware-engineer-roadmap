"""Compile actual cache pointer logic and execute a CPU attention math oracle.

This suite does not execute the CUDA attention kernel; use the optional harness.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class AttentionContracts(unittest.TestCase):
    def test_native_oracle_layout_and_negative_score_counterexample(self):
        compiler=shutil.which('c++')
        if not compiler:self.skipTest('C++17 compiler required for native attention oracle')
        with tempfile.TemporaryDirectory(prefix='course-attention-') as directory:
            binary=Path(directory)/'test-attention-reference'
            build=subprocess.run([compiler,'-std=c++17','-O2','-Wall','-Wextra','-Werror',
                                  '-I',str(ROOT/'tests/cuda_stub'),'-I',str(ROOT/'include'),
                                  str(ROOT/'tests/test_attention_reference.cpp'),
                                  str(ROOT/'src/memory/kv_cache.cpp'),'-o',str(binary)],
                                 text=True,capture_output=True,timeout=60)
            self.assertEqual(build.returncode,0,build.stdout+build.stderr)
            run=subprocess.run([str(binary)],text=True,capture_output=True,timeout=10)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            self.assertIn('13 fixtures x 5 tile sizes passed',run.stdout)
            self.assertIn('CUDA execution: not performed',run.stdout)

    def test_native_decode_scratch_reservation_reuse_and_failures(self):
        compiler=shutil.which('c++')
        if not compiler:self.skipTest('C++17 compiler required for native scratch contracts')
        with tempfile.TemporaryDirectory(prefix='course-scratch-') as directory:
            binary=Path(directory)/'test-decode-scratch'
            build=subprocess.run([compiler,'-std=c++17','-O2','-Wall','-Wextra','-Werror',
                                  '-I',str(ROOT/'tests/cuda_stub'),'-I',str(ROOT/'include'),
                                  str(ROOT/'tests/test_decode_scratch.cpp'),
                                  str(ROOT/'src/memory/pool.cpp'),'-o',str(binary)],
                                 text=True,capture_output=True,timeout=60)
            self.assertEqual(build.returncode,0,build.stdout+build.stderr)
            run=subprocess.run([str(binary)],text=True,capture_output=True,timeout=10)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            self.assertIn('decode scratch checks passed',run.stdout)
            self.assertIn('CUDA stream execution: not performed',run.stdout)


if __name__=='__main__':unittest.main(verbosity=2)
