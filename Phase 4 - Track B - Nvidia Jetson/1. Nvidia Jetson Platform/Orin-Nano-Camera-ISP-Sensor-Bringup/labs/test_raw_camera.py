import math
import struct
import unittest

from raw_camera import bayer_color, cropped_pattern, pack_raw10, timing, unpack_expanded10, unpack_raw10


class RawTests(unittest.TestCase):
    def test_independent_byte_fixture(self):
        known = bytes.fromhex("00 00 80 ff c4")
        self.assertEqual(unpack_raw10(known, 4, 1, 5), [[0, 1, 512, 1023]])
        self.assertEqual(pack_raw10([0, 1, 512, 1023]), known)

    def test_every_ten_bit_value_in_every_group_position(self):
        for value in range(1024):
            for index in range(4):
                inputs = [0, 1, 1022, 1023]
                inputs[index] = value
                self.assertEqual(unpack_raw10(pack_raw10(inputs), 4, 1, 5)[0], inputs)

    def test_row_padding_is_skipped(self):
        packed = bytes.fromhex("00 00 80 ff c4 aa bb ff 80 00 00 13 cc dd")
        self.assertEqual(unpack_raw10(packed, 4, 2, 7), [[0, 1, 512, 1023], [1023, 512, 1, 0]])

    def test_expanded_format_is_different(self):
        expanded = struct.pack("<4H", 0, 1, 512, 1023)
        self.assertEqual(unpack_expanded10(expanded, 4, 1, 8), [[0,1,512,1023]])
        # Wrong packed interpretation can be in range but semantically wrong.
        self.assertNotEqual(unpack_raw10(expanded, 4, 1, 8)[0], [0,1,512,1023])
        with self.assertRaises(ValueError): unpack_expanded10(b"\x00\x80", 1, 1, 2)

    def test_invalid_format_inputs(self):
        for values in ([0], [-1,0,0,0], [1024,0,0,0], [0.0,0,0,0]):
            with self.assertRaises(ValueError): pack_raw10(values)
        for geometry in ((3,1,5), (4,1,4), (4,2,5), (0,1,5)):
            with self.assertRaises(ValueError): unpack_raw10(bytes(5), *geometry)

    def test_crop_changes_bayer_phase(self):
        self.assertEqual(cropped_pattern("RGGB", 1, 0), "GRBG")
        self.assertEqual(cropped_pattern("RGGB", 0, 1), "GBRG")
        self.assertEqual(cropped_pattern("RGGB", 1, 1), "BGGR")
        self.assertEqual(cropped_pattern("RGGB", 2, 2), "RGGB")
        for args in (("wrong",0,0), ("RGGB",-1,0)):
            with self.assertRaises(ValueError): bayer_color(*args)

    def test_crop_rejects_boolean_and_noninteger_coordinates(self):
        for left, top in ((True, 0), (0, False), (0.0, 0), (0, -1)):
            with self.subTest(crop=(left, top)), self.assertRaises(ValueError):
                cropped_pattern("RGGB", left, top)


class TimingTests(unittest.TestCase):
    def test_known_mode_and_exposure_quantization(self):
        result = timing()
        self.assertEqual(result["frame_rate_hz"], 30)
        self.assertEqual(result["exposure_lines"], 338)
        self.assertAlmostEqual(result["actual_exposure_s"], .010014814814814815)
        self.assertFalse(result["clamped"])

    def test_long_exposure_clamps_and_frame_length_changes_rate(self):
        result = timing(requested_exposure_s=1)
        self.assertEqual(result["exposure_lines"], 1121)
        self.assertTrue(result["clamped"])
        self.assertEqual(timing(frame_length=2250)["frame_rate_hz"], 15)

    def test_invalid_timing_rejected(self):
        for kwargs in ({"pixel_clock_hz":0}, {"line_length":-1}, {"margin_lines":1125},
                       {"requested_exposure_s":math.nan}, {"requested_exposure_s":0},
                       {"requested_exposure_s":True}, {"requested_exposure_s":"0.01"}):
            with self.assertRaises(ValueError): timing(**kwargs)

    def test_extreme_finite_exposure_saturates_before_integer_conversion(self):
        result = timing(requested_exposure_s=1e308)
        self.assertEqual(result["exposure_lines"], 1121)
        self.assertTrue(result["clamped"])


if __name__ == "__main__":
    unittest.main()
