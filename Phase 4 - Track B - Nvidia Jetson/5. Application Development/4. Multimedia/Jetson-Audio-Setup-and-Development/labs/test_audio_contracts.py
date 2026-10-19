import unittest
from audio_contracts import stream_budget, clock_drift


class AudioContractTests(unittest.TestCase):
    def test_memory_frames_differ_from_wire_slots(self):
        d=stream_budget(48000,4,2,480,4,4,32)
        self.assertEqual(d,{'frame_bytes':8,'bytes_per_second':384000,'period_ms':10,
                            'buffer_ms':40,'bit_clock_hz':6144000})
    def test_clock_drift_accumulates(self):
        self.assertAlmostEqual(clock_drift(48000,50,600)['sample_difference'],1440)
        self.assertAlmostEqual(clock_drift(48000,50,600)['time_difference_ms'],30)
    def test_invalid_contracts(self):
        for args in [(48000,4,2,480,4,2,32),(48000,4,4,480,4,4,16),(0,4,2,480,4,4,32)]:
            with self.assertRaises(ValueError):stream_budget(*args)
        with self.assertRaises(ValueError):clock_drift(48000,float('nan'),2)


if __name__=='__main__':unittest.main()
