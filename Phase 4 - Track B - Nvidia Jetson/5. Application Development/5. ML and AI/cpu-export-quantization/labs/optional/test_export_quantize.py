import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export_quantize as lab


class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.tmp.name)
        cls.report = lab.run(cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_export_parity_at_distinct_batch_sizes(self):
        self.assertEqual(set(self.report['fp32_export_max_abs_errors']), {'1', '7', '32'})
        self.assertLess(max(self.report['fp32_export_max_abs_errors'].values()), 1e-5)
        self.assertGreater(self.report['swapped_feature_max_abs_error'], 1)

    def test_calibration_does_not_touch_validation_or_test(self):
        splits = self.report['split_episode_ids']
        for case in self.report['quantization_cases'].values():
            ids = set(case['calibration_episode_ids'])
            self.assertTrue(ids <= set(splits['training']))
            self.assertFalse(ids & set(splits['validation'] + splits['test']))

    def test_quantization_reload_and_degenerate_calibration(self):
        good, bad = (self.report['quantization_cases'][k] for k in ('representative', 'repeated_one_row'))
        self.assertEqual(good['unique_rows'], 256)
        self.assertEqual(bad['unique_rows'], 1)
        self.assertGreater(good['operator_counts']['QuantizeLinear'], 0)
        self.assertEqual(good['fresh_session_reload_max_abs_error'], 0)
        self.assertGreater(good['test']['stop_recall'], .95)
        self.assertLess(bad['test']['stop_recall'], good['test']['stop_recall'] - .15)
        self.assertLess(good['distance_bias_0_8m']['stop_recall'], good['test']['stop_recall'] - .1)

    def test_input_contract_and_direct_runtime_shape_failure(self):
        runtime = lab.session(self.output / 'representative.onnx')
        for count in (1, 7, 32):
            self.assertEqual(lab.predict(runtime, np.zeros((count, 4), dtype=np.float32)).shape, (count, 2))
        for x in (np.zeros((2, 5), dtype=np.float32), np.zeros((2, 4), dtype=np.float64),
                  np.full((2, 4), np.nan, dtype=np.float32), np.zeros((1025, 4), dtype=np.float32)):
            with self.assertRaises(ValueError):
                lab.predict(runtime, x)
        with self.assertRaises(Exception):
            runtime.run(['logits'], {'observations': np.zeros((2, 5), dtype=np.float32)})


if __name__ == '__main__':
    unittest.main()
