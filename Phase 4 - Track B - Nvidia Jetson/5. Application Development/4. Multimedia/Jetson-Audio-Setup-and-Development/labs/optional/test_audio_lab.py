from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audio_lab import FIRDecimator, lowpass_taps, pcm16_decode, pcm16_encode, read_wave, run, write_wave


class AudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.report=run(cls.tmp.name)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_independent_little_endian_pcm_and_clipping(self):
        data=bytes([0,128,0,0,255,127,0,64])
        x=pcm16_decode(data,2)
        np.testing.assert_allclose(x,[[-1,0],[32767/32768,.5]])
        self.assertEqual(pcm16_encode(np.array([[-2,2]])),bytes([0,128,255,127]))
        with self.assertRaises(ValueError):pcm16_decode(b'123',2)

    def test_actual_wav_metadata_and_roundtrip(self):
        d=self.report;self.assertEqual((d['input_rate'],d['input_frames'],d['channels']),(48000,48000,4))
        self.assertLessEqual(d['pcm_roundtrip_max_error'],.5/32768+1e-12)
        wrong,rate=read_wave(Path(self.tmp.name)/'wrong-header-16k.wav')
        self.assertEqual(len(wrong)/rate,3)

    def test_causal_fir_preserves_state_and_decimation_phase(self):
        rng=np.random.default_rng(7);x=rng.normal(size=1003);h=lowpass_taps(31)
        ref=np.convolve(x,h,mode='full')[:len(x)][::3]
        dec=FIRDecimator(h);observed=np.concatenate([dec.process(c) for c in np.array_split(x,37)])
        np.testing.assert_allclose(observed,ref,atol=1e-12,rtol=1e-12)
        self.assertLess(self.report['stream_vs_single_block_max_error'],1e-12)

    def test_filter_rejects_alias_and_preserves_wanted_tone(self):
        d=self.report
        self.assertAlmostEqual(d['naive_4khz_alias_amplitude'],.2,delta=.002)
        self.assertAlmostEqual(d['passband_1khz_amplitude'],.4,delta=.005)
        self.assertGreater(d['alias_reduction_db'],50)
        self.assertEqual(d['output_frames'],16000)
        self.assertEqual(d['causal_group_delay_ms'],5.3125)

    def test_short_filter_has_correct_delay_and_streaming_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=run(tmp,63)
        self.assertEqual(d['filter_taps'],63)
        self.assertAlmostEqual(d['causal_group_delay_ms'],31/48)
        self.assertEqual(d['output_frames'],16000)
        self.assertLess(d['stream_vs_single_block_max_error'],1e-12)

    def test_channel_cancellation_and_invalid_samples(self):
        self.assertEqual(self.report['opposite_channel_average_rms'],0)
        with self.assertRaises(ValueError):pcm16_encode(np.array([[float('nan')]]))
        with self.assertRaises(ValueError):FIRDecimator([1]).process([float('inf')])
        with self.assertRaises(ValueError):lowpass_taps(30)


if __name__=='__main__':unittest.main()
