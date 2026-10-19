import importlib.util
from pathlib import Path
import tempfile
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from action_generation import ActionNet, fit, reload_verified, sample, training_batch


@unittest.skipUnless(HAS_TORCH, "optional CPU PyTorch is not installed")
class TrainedActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_training_reduces_independent_validation_loss_for_both_objectives(self):
        for method in ("flow", "ddim"):
            model, metrics = fit(method, 7, 200)
            self.assertLess(metrics["history"][-1]["validation_loss"],
                            0.6 * metrics["initial_validation_loss"])
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_zero_field_preserves_noise_and_fixed_noise_is_reproducible(self):
        model = ActionNet().eval()
        for parameter in model.parameters():
            parameter.data.zero_()
        noise = torch.tensor([[-1.3], [0.0], [1.3]])
        labels = torch.tensor([0, 1, 2])
        torch.testing.assert_close(sample(model, noise, labels, "flow", 8), noise)
        torch.testing.assert_close(sample(model, noise, labels, "flow", 32), noise)
        with self.assertRaises(ValueError):
            sample(model, noise, labels, "ddim", 65)

    def test_sampler_uses_condition_and_time_and_preserves_input(self):
        class KnownField(torch.nn.Module):
            def forward(self, x, t, labels):
                return labels.unsqueeze(1) + t
        noise = torch.zeros(3, 1)
        labels = torch.tensor([0, 1, 2])
        out = sample(KnownField(), noise, labels, "flow", 4)
        torch.testing.assert_close(out, torch.tensor([[0.375], [1.375], [2.375]]))
        torch.testing.assert_close(noise, torch.zeros_like(noise))

    def test_checkpoint_reloads_exactly_and_wrong_shape_is_rejected(self):
        torch.manual_seed(7)
        model = ActionNet().eval()
        probe = training_batch(12, torch.Generator().manual_seed(8), "flow")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(model.state_dict(), path)
            restored = ActionNet().eval()
            restored.load_state_dict(torch.load(path, weights_only=True))
            torch.testing.assert_close(model(*probe[:3]), restored(*probe[:3]), atol=0, rtol=0)
        with self.assertRaises(ValueError):
            sample(model, torch.zeros(2, 2), torch.zeros(2, dtype=torch.long), "flow", 8)

    def test_corrupted_reload_is_rejected_before_sampling(self):
        model = ActionNet().eval()
        probe = training_batch(12, torch.Generator().manual_seed(8), "flow")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            state = {key: value.clone() for key, value in model.state_dict().items()}
            torch.save({"state_dict": state}, path)
            _, error = reload_verified(model, path, probe)
            self.assertEqual(error, 0.0)
            state["layers.4.bias"] += 1.0
            torch.save({"state_dict": state}, path)
            with self.assertRaisesRegex(ValueError, "parity failed"):
                reload_verified(model, path, probe)
            state["layers.4.bias"].fill_(float("nan"))
            torch.save({"state_dict": state}, path)
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                reload_verified(model, path, probe)

    def test_tensor_ddim_uses_selected_endpoints_and_declared_clean_clipping(self):
        # Known per-example noise gives exact clean reconstruction independently
        # of the scalar helper; this executes the actual tensor sampling path.
        class KnownNoise(torch.nn.Module):
            def forward(self, x, t, labels):
                return torch.tensor([[0.1], [-1.0]], dtype=x.dtype)
        clean = torch.tensor([[0.4], [-0.6]], dtype=torch.float64)
        epsilon = torch.tensor([[0.1], [-1.0]], dtype=torch.float64)
        start = 0.01 * clean + (1 - 1e-4)**0.5 * epsilon
        labels = torch.tensor([0, 1])
        for steps in (1, 3, 8, 64):
            torch.testing.assert_close(sample(KnownNoise(), start, labels, "ddim", steps),
                                       clean, atol=1e-12, rtol=0)
        class ZeroNoise(torch.nn.Module):
            def forward(self, x, t, labels):
                return torch.zeros_like(x)
        start = torch.tensor([[-0.03], [0.005], [0.03]], dtype=torch.float64)
        expected = torch.tensor([[-2.0], [0.5], [2.0]], dtype=torch.float64)
        for steps in (1, 3, 8, 64):
            torch.testing.assert_close(sample(ZeroNoise(), start, torch.tensor([0, 1, 2]),
                                               "ddim", steps), expected, atol=1e-12, rtol=0)


if __name__ == "__main__":
    unittest.main()
