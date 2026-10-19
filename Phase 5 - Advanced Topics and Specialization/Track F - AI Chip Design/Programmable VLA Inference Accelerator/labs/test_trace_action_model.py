import importlib.util
from pathlib import Path
import tempfile
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from trace_action_model import (ActionNet, LABELS, SIGMA, build_report,
                                    export_model, inputs, load_model,
                                    tile_fraction, trace_linear_calls)


@unittest.skipUnless(HAS_TORCH, "optional CPU PyTorch export/profiler environment is not installed")
class MeasuredTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_actual_invocations_match_hand_count_and_hooks_are_removed(self):
        torch.manual_seed(7)
        model = ActionNet().eval()
        rows, output = trace_linear_calls(model, 7, 3)
        self.assertEqual(output.shape, (7, 1))
        self.assertEqual(len(rows), 9)
        self.assertEqual([r["mnk"] for r in rows[:3]], [[7, 64, 13], [7, 64, 64], [7, 1, 64]])
        self.assertEqual(sum(r["derived_multiply_add_ops"] for r in rows), 3 * 7 * 9984)
        self.assertTrue(all(not m._forward_hooks for m in model.modules()))

    def test_export_and_serialized_reload_preserve_changed_inputs(self):
        torch.manual_seed(7)
        with tempfile.TemporaryDirectory() as directory:
            for batch in (1, 7):
                report = export_model(ActionNet().eval(), batch, Path(directory))
                self.assertLessEqual(report["max_export_probe_error"], 1e-6)
                self.assertLessEqual(report["max_serialized_reload_probe_error"], 1e-6)
                self.assertEqual(report["export_probe_count"], 12)
                self.assertEqual(report["serialized_reload_probe_count"], 12)
                self.assertTrue((Path(directory) / f"flow-batch-{batch}.pt2").is_file())
        targets = [node["target"] for node in report["graph"]]
        self.assertIn("aten.sin.default", targets)
        self.assertIn("aten.silu.default", targets)
        self.assertEqual(targets.count("aten.linear.default"), 3)

    def test_profiler_observes_real_addmm_and_storage_is_not_repeated(self):
        report = build_report(ActionNet().eval(), 1, 2)
        self.assertEqual(report["observed_linear_calls"], 6)
        self.assertEqual(report["distinct_logical_parameter_bytes"], 5121 * 4)
        self.assertEqual(report["hypothetical_parameter_bytes_if_streamed_each_call"], 5121 * 4 * 2)
        self.assertEqual(report["derived_multiply_add_ops"], 2 * 9984)
        self.assertEqual(len(report["profiler"]["warm_wall_ms"]), 11)

    def test_padding_counts_all_three_dimensions_and_input_validation(self):
        rows = [{"mnk": [1, 1, 13]}]
        self.assertEqual(tile_fraction(rows, 16, 16, 16), 13 / 4096)
        self.assertEqual(tile_fraction(rows, 1, 16, 16), 13 / 256)
        for arguments in ((rows, 0, 1, 1), ([], 1, 1, 1)):
            with self.assertRaises(ValueError):
                tile_fraction(*arguments)
        with self.assertRaises(ValueError):
            inputs(True)

    def test_checkpoint_contract_rejects_wrong_objective_labels_and_nonfinite_weights(self):
        model = ActionNet().eval()
        saved = {"schema": "scalar-action-v1", "method": "flow", "labels": LABELS,
                 "sigma": SIGMA, "state_dict": model.state_dict()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(saved, path)
            loaded = load_model(path)
            torch.testing.assert_close(loaded(*inputs(2)), model(*inputs(2)), atol=0, rtol=0)
            for changed in ({"method": "ddim"}, {"labels": ("right", "left", "either")}):
                torch.save({**saved, **changed}, path)
                with self.assertRaises(ValueError):
                    load_model(path)
            saved["state_dict"]["layers.0.weight"][0, 0] = float("nan")
            torch.save(saved, path)
            with self.assertRaises(ValueError):
                load_model(path)


if __name__ == "__main__":
    unittest.main()
