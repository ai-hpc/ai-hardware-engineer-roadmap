import hashlib
import unittest

from video_lab import gst_modules, rgb_rows, run

try:
    gst_modules()
    HAVE_GST = True
except RuntimeError:
    HAVE_GST = False


class LayoutTests(unittest.TestCase):
    def test_row_padding_and_plane_offset_are_not_pixels(self):
        self.assertEqual(rgb_rows(b"XXabcdefPPghijklYY", 2, 2, 8, 2), b"abcdefghijkl")

    def test_invalid_geometry_and_short_buffers(self):
        for dimensions in ((2, 2, 5, 0), (0, 2, 6, 0), (2, 2, 6, -1),
                           (2, 3, 6, 0), (2.0, 2, 6, 0)):
            with self.assertRaises(ValueError): rgb_rows(b"0" * 12, *dimensions)

    def test_runtime_limits_reject_invalid_arguments_before_dependency_load(self):
        for kwargs in ({"frames": 0}, {"frames": 301}, {"consumer_ms": float("nan")},
                       {"consumer_ms": -1}, {"drain_after_eos": True}):
            with self.assertRaises(ValueError): run(**kwargs)


@unittest.skipUnless(HAVE_GST, "optional GStreamer GI/base-plugin dependencies unavailable")
class GStreamerTests(unittest.TestCase):
    def test_actual_lossless_stream_and_padded_black_pixels(self):
        result = run(frames=12, consumer_ms=0, drop_oldest=False, live=False)
        self.assertEqual([s["frame_id"] for s in result["samples"]], list(range(12)))
        self.assertEqual([s["pts_ns"] for s in result["samples"]], [i * 20_000_000 for i in range(12)])
        for sample in result["samples"]:
            self.assertEqual(sample["logical_bytes"], 65 * 48 * 3)
            self.assertGreaterEqual(sample["stride"], 65 * 3)
            self.assertEqual(sample["logical_sha256"], hashlib.sha256(bytes(65 * 48 * 3)).hexdigest())
            self.assertTrue(sample["all_black"])

    def test_actual_bounded_sink_keeps_newest_two_frames(self):
        result = run(frames=10, consumer_ms=0, live=False, drain_after_eos=True)
        self.assertEqual([s["frame_id"] for s in result["samples"]], [8, 9])
        self.assertEqual(result["omitted_frames"], 8)

    def test_actual_live_pipeline_has_running_time_age(self):
        result = run(frames=4, consumer_ms=0, drop_oldest=False)
        self.assertEqual(result["consumed_frames"], 4)
        self.assertTrue(all(s["age_at_pull_ns"] >= 0 for s in result["samples"]))
        # No fixed host-dependent latency assertion.


if __name__ == "__main__":
    unittest.main()
