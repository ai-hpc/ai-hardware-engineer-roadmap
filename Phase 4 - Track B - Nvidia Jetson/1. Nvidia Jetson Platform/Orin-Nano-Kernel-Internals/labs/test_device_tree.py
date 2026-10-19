from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from device_tree_lab import HERE, REQUIRED, SENSOR, build, compile_tree, get, validate_candidate


@unittest.skipUnless(all(shutil.which(tool) for tool in REQUIRED),
                     "optional native DT lab: install device-tree-compiler")
class DeviceTreeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="course-dt-test-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def overlay(self, text):
        path = self.directory / "variation.dts"
        path.write_text(text)
        return path

    def test_actual_overlay_changes_only_expected_properties(self):
        merged = build(self.directory)
        self.assertEqual(get(self.directory/"board.dtb", SENSOR, "status"), "disabled")
        self.assertEqual(get(self.directory/"board.dtb", SENSOR, "sample-rate-hz", "u"), "100")
        result = validate_candidate(merged)
        self.assertEqual(result["sample_rate_hz"], 200)
        self.assertEqual(get(merged, "/", "compatible"), "course,simulated-board")
        self.assertEqual(get(merged, "/i2c@1000", "reg", "x"), "1000 100")

    def test_missing_base_symbols_prevent_label_overlay(self):
        with self.assertRaises(subprocess.CalledProcessError):
            build(self.directory, symbols=False)

    def test_unresolved_external_label_fails_at_merge(self):
        overlay = self.overlay((HERE/"sensor-overlay.dts").read_text().replace("&sensor0", "&missing_sensor"))
        # A plugin compiles with an external fixup; applying it needs the target.
        compile_tree(overlay, self.directory/"syntactically-valid.dtbo")
        with self.assertRaises(subprocess.CalledProcessError):
            build(self.directory, overlay=overlay)

    def test_valid_binary_can_violate_device_contract(self):
        overlay = self.overlay((HERE/"sensor-overlay.dts").read_text().replace("<200>", "<3000>"))
        merged = build(self.directory, overlay=overlay)
        self.assertEqual(get(merged, SENSOR, "sample-rate-hz", "u"), "3000")
        with self.assertRaises(ValueError):
            validate_candidate(merged)

    def test_syntax_error_fails_compilation(self):
        overlay = self.overlay('/dts-v1/; / { broken = <1>;')
        with self.assertRaises(subprocess.CalledProcessError):
            compile_tree(overlay, self.directory/"invalid.dtb")

    def test_different_driver_compatible_rejected(self):
        text = (HERE/"sensor-overlay.dts").read_text().replace('sample-rate-hz = <200>;',
                'sample-rate-hz = <200>; compatible = "course,different-sensor";')
        merged = build(self.directory, overlay=self.overlay(text))
        with self.assertRaises(ValueError):
            validate_candidate(merged)


if __name__ == "__main__":
    unittest.main()
