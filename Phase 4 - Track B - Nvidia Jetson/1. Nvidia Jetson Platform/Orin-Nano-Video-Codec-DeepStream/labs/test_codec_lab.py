import copy
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from codec_lab import FRAMES, fixture, run, validate


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.pixels = fixture()
        self.probe = {"streams": [{"width":64, "height":48, "nb_read_frames":str(FRAMES), "time_base":"1/30"}],
                      "frames":[{"best_effort_timestamp":i} for i in range(FRAMES)]}

    def test_exact_reference_has_zero_error(self):
        self.assertTrue(validate(self.pixels, self.pixels, self.probe)["lossless"])

    def test_removed_frame_and_changed_timestamp_are_rejected(self):
        with self.assertRaises(ValueError): validate(self.pixels, self.pixels[:-64*48], self.probe)
        wrong = copy.deepcopy(self.probe)
        wrong["frames"][3]["best_effort_timestamp"] = 4
        with self.assertRaises(ValueError): validate(self.pixels, self.pixels, wrong)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "optional FFmpeg tools unavailable")
class FFmpegTests(unittest.TestCase):
    def test_actual_encode_decode_preserves_timing_and_bounded_fixture_error(self):
        encoders = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], text=True, stderr=subprocess.DEVNULL)
        if "libx264" not in encoders: self.skipTest("FFmpeg build has no libx264 encoder")
        with tempfile.TemporaryDirectory() as directory:
            report = run(Path(directory))
            self.assertEqual(report["frames"], 12)
            # Regression tolerance for this specific synthetic fixture, not a perception-quality criterion.
            self.assertLess(report["mse_gray_8bit"], 50)
            self.assertGreater(report["encoded_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
