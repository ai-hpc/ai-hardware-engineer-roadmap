"""Run explicitly inside the pinned PyTorch environment; not a stdlib suite."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from train_stop_model import StopModel, choose_threshold, export_and_check, metrics, run


class PyTorchLabTests(unittest.TestCase):
    def test_known_confusion_and_cost(self):
        result = metrics([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual((result["tp"], result["fp"], result["fn"], result["tn"]), (1, 1, 1, 1))
        self.assertEqual(result["cost_5fn_plus_fp_per_example"], 1.5)

    def test_threshold_respects_declared_miss_cost(self):
        threshold, _ = choose_threshold([1, 0], [.4, .3])
        self.assertGreater(threshold, .3)
        self.assertLessEqual(threshold, .4)

    def test_scaler_is_serialized_as_model_state(self):
        model = StopModel([1, 2, 3, 4], [2, 3, 4, 5])
        state = model.state_dict()
        self.assertTrue(torch.equal(state["mean"], torch.tensor([1, 2, 3, 4])))
        self.assertIn("std", state)

    def test_end_to_end_quality_artifacts_and_shift(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run(Path(directory))
            self.assertGreater(result["test"]["accuracy"], .93)
            self.assertGreater(result["test"]["stop_recall"], .93)
            self.assertLess(result["test"]["cross_entropy_nats"], .2)
            self.assertEqual(result["baselines"]["analytic_rule_clean"]["accuracy"], 1)
            self.assertGreater(result["shifted_test"]["fn"], result["test"]["fn"])
            self.assertEqual(result["reload_max_abs_logit_error"], 0)
            self.assertTrue(result["export"]["wrong_feature_shape_rejected"])
            for name in ("state_dict.pt", "stop_model.pt2", "report.json"):
                self.assertTrue((Path(directory) / name).is_file())


if __name__ == "__main__":
    unittest.main()
