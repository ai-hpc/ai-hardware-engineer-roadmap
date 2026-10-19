import math
import unittest

from receiver import IQWindows, awgn_bits, fit, loss_and_gradient, metrics, nr_slot_us, receiver_budget


class ReceiverTests(unittest.TestCase):
    def test_analytic_llr_sign(self):
        # With x=2b-1, positive LLR means bit 1; a different mapper can reverse it.
        self.assertEqual(metrics([(-1 + .6, 0), (1 + .6, 1)], 4, -2.4)["errors"], 0)

    def test_gradient_matches_finite_difference(self):
        rows = [(-1.2, 0), (.1, 1), (1.4, 1)]
        w, c, eps = .8, -.3, 1e-5
        _, dw, dc = loss_and_gradient(rows, w, c)
        numerical_w = (loss_and_gradient(rows, w + eps, c)[0]
                       - loss_and_gradient(rows, w - eps, c)[0]) / (2 * eps)
        numerical_c = (loss_and_gradient(rows, w, c + eps)[0]
                       - loss_and_gradient(rows, w, c - eps)[0]) / (2 * eps)
        self.assertAlmostEqual(dw, numerical_w, places=9)
        self.assertAlmostEqual(dc, numerical_c, places=9)

    def test_large_logits_stay_finite(self):
        self.assertTrue(all(map(math.isfinite, loss_and_gradient([(-1, 1), (1, 0)], 10000, 0))))

    def test_monte_carlo_matches_awgn_theory(self):
        rows = awgn_bits(50000, 2, .6, 44)
        measured = metrics(rows, 1, -.6)["ber"]
        expected = .5 * math.erfc(math.sqrt(10 ** .2))
        standard_error = math.sqrt(expected * (1 - expected) / len(rows))
        self.assertLess(abs(measured - expected), 5 * standard_error)

    def test_training_generalizes_but_channel_shift_fails(self):
        training = awgn_bits(2000, 2, .6, 11)
        w, c = fit(training)
        matched, shifted = awgn_bits(20000, 2, .6, 22), awgn_bits(20000, 2, -.6, 33)
        learned = metrics(matched, w, c)["ber"]
        self.assertLess(learned, metrics(matched, 1, 0)["ber"] / 2)
        self.assertLess(abs(learned - metrics(matched, 1, -.6)["ber"]), .005)
        self.assertGreater(metrics(shifted, w, c)["ber"], learned * 4)

    def test_irregular_chunks_match_single_capture(self):
        samples = [complex(i, -i) for i in range(103)]
        one, fragmented = IQWindows(16), IQWindows(16)
        expected = one.push(samples)
        observed = []
        for start, end in ((0, 3), (3, 3), (3, 19), (19, 20), (20, 89), (89, 103)):
            observed += fragmented.push(samples[start:end])
        self.assertEqual(observed, expected)
        self.assertEqual([w.start_sample for w in observed], [0, 16, 32, 48, 64, 80])
        self.assertEqual(fragmented.pending, samples[96:])
        self.assertEqual(fragmented.finish(), 7)
        with self.assertRaises(ValueError):
            fragmented.push([1j])

    def test_invalid_batch_is_atomic(self):
        receiver = IQWindows(4)
        receiver.push([1j, 2j])
        with self.assertRaises(ValueError):
            receiver.push([3j, complex(float("nan"), 0)])
        self.assertEqual(receiver.pending, [1j, 2j])
        self.assertEqual(receiver.next_start, 0)

    def test_timing_and_cost_units(self):
        budget = receiver_budget()
        self.assertEqual(nr_slot_us(1), 500)
        self.assertEqual(nr_slot_us(3), 125)
        self.assertEqual(budget["total_macs_per_slot"], 642829824)
        self.assertEqual(budget["first_activation_fp16_bytes"], 5870592)
        self.assertEqual(budget["one_antenna_complex_int16_bytes_per_second"], 122880000)
        self.assertEqual(budget["window_1024_at_2msps_us"], 512)

    def test_invalid_configuration(self):
        for bad in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                IQWindows(bad)
        for bad in (-1, 1.5, True, 7):
            with self.assertRaises(ValueError):
                nr_slot_us(bad)


if __name__ == "__main__":
    unittest.main()
