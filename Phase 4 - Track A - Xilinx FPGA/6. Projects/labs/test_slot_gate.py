"""Actual Icarus RTL simulation; does not establish distributed clock sync."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
IVERILOG = shutil.which("iverilog")
VVP = shutil.which("vvp")


@unittest.skipUnless(IVERILOG and VVP, "Icarus Verilog required")
class SlotGateTests(unittest.TestCase):
    def simulate(self, cycles, slots, own, guard, mutation=False):
        with tempfile.TemporaryDirectory(prefix="course-slot-gate-") as tmp:
            output = Path(tmp) / "simulation"
            cmd = [IVERILOG, "-g2012", "-s", "tb_slot_gate", "-o", str(output)]
            for key, value in [("SLOT_CYCLES",cycles),("NUM_SLOTS",slots),
                               ("OWN_SLOT",own),("GUARD_CYCLES",guard)]:
                cmd.append(f"-Ptb_slot_gate.{key}={value}")
            if mutation:
                cmd.append("-DCOURSE_MUTATE_LONG_SLOT")
            cmd.extend([str(HERE / "slot_gate.sv"), str(HERE / "tb_slot_gate.sv")])
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=20)
            return subprocess.run([VVP, str(output)], capture_output=True, text=True, timeout=10)

    def test_non_power_of_two_singleton_guards_and_reset(self):
        for parameters in [(10,4,1,2),(5,3,2,1),(1,1,0,0),(4,2,0,0)]:
            with self.subTest(parameters=parameters):
                result=self.simulate(*parameters)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertIn("PASS:", result.stdout)

    def test_invalid_configuration_and_off_by_one_are_detected(self):
        for parameters in [(0,2,0,0),(4,0,0,0),(4,2,2,0),(4,2,0,2),(4,2,0,-1)]:
            with self.subTest(parameters=parameters):
                result=self.simulate(*parameters)
                self.assertNotEqual(result.returncode,0)
                self.assertIn("invalid slot configuration",result.stdout+result.stderr)
        result=self.simulate(5,3,2,1,mutation=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn("phase mismatch",result.stdout+result.stderr)


if __name__ == "__main__":
    unittest.main()
