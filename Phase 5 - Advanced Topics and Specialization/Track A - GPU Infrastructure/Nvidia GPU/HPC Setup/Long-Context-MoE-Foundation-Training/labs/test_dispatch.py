"""Real process/collective checks, including zero-work source and expert ranks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
try:
    import torch.distributed as dist
except ImportError:
    dist = None

LAB = Path(__file__).resolve().parent


@unittest.skipIf(dist is None or not dist.is_available() or not dist.is_gloo_available(),
                 'optional: CPU PyTorch with Gloo is required')
class DispatchTests(unittest.TestCase):
    def test_serial_parity_with_uneven_and_empty_ranks(self):
        for mode in ([], ['--skew'], ['--empty-expert'], ['--empty-source']):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(prefix='course-dispatch-') as tmp:
                output = Path(tmp) / 'report.json'
                environment = dict(os.environ, OMP_NUM_THREADS='1', GLOO_SOCKET_IFNAME='lo')
                result = subprocess.run(
                    [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                     '--nproc-per-node=2', str(LAB / 'dispatch_gloo.py'),
                     '--output', str(output), *mode],
                    env=environment, cwd=tmp, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                rows = json.loads(output.read_text())['ranks']
                self.assertEqual([row['rank'] for row in rows], [0, 1])
                for row in rows:
                    self.assertEqual(row['output'], row['serial'])
                    self.assertEqual(row['max_abs_error'], 0.)
                    self.assertEqual(sum(row['receive_splits']), row['received_assignments'])
                for source in range(2):
                    for destination in range(2):
                        self.assertEqual(rows[source]['send_splits'][destination],
                                         rows[destination]['receive_splits'][source])
                if mode == ['--empty-expert']:
                    self.assertEqual(rows[1]['received_assignments'], 0)
                    self.assertGreater(rows[1]['source_tokens'], 0)
                if mode == ['--empty-source']:
                    self.assertEqual(rows[0]['source_tokens'], 0)
                    self.assertEqual(rows[0]['output'], [])
                    self.assertGreater(rows[0]['received_assignments'], 0)


if __name__ == '__main__':
    unittest.main()
