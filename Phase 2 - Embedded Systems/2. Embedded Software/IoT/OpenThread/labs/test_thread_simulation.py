"""Optional real stack integration test; set OPENTHREAD_CLI to the pinned build."""
import importlib.util
import os
from pathlib import Path
import unittest

from run_thread_simulation import run


class ThreadSimulationTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("OPENTHREAD_CLI"),
                         "Set OPENTHREAD_CLI to enable the actual OpenThread simulation")
    def test_attach_ping_udp_offline_and_reattach(self):
        self.assertIsNotNone(importlib.util.find_spec("pexpect"),
                             "Install this lab's requirements.txt")
        result = run(Path(os.environ["OPENTHREAD_CLI"]))
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["checks"]), 6)


if __name__ == "__main__":
    unittest.main()
