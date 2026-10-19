import json
from pathlib import Path
import unittest
from fractions import Fraction
from inference_contracts import (gqa_bytes, mla_bytes, grouped_weight_bytes,
    roofline_seconds, causal_gqa, dispatch, expert_reference,
    speculative_one_step_distribution, quantize_int4_group)

HERE = Path(__file__).parent
BASE = dict(num_hidden_layers=80, num_attention_heads=64,
            num_key_value_heads=8, hidden_size=8192)


class Accounting(unittest.TestCase):
    def test_gqa_known_geometry_and_units(self):
        self.assertEqual(gqa_bytes(BASE, [1]), 327680)
        self.assertEqual(gqa_bytes(BASE, [131072]), 40 * 1024**3)

    def test_request_rounding_separate_and_empty(self):
        self.assertEqual(gqa_bytes(BASE, [1, 17], block_tokens=16), 48 * 327680)
        self.assertEqual(gqa_bytes(BASE, []), 0)
        self.assertEqual(gqa_bytes(BASE, [0]), 0)

    def test_explicit_head_dimension_not_hidden_div_heads(self):
        config = json.loads((HERE / 'qwen3_moe_config.json').read_text())
        self.assertEqual(config['head_dim'], 128)
        self.assertEqual(gqa_bytes(config, [1]), 192512)

    def test_mla_latent_layout(self):
        config = json.loads((HERE / 'deepseek_v31_config.json').read_text())
        self.assertEqual(mla_bytes(config, [1]), 70272)
        self.assertEqual(mla_bytes(config, [131072]), 9210691584)

    def test_invalid_shapes_do_not_silently_truncate(self):
        for config in [dict(BASE, num_key_value_heads=3), dict(BASE, hidden_size=8193),
                       dict(BASE, num_hidden_layers=0), dict(BASE, head_dim=True)]:
            with self.assertRaises(ValueError):
                gqa_bytes(config, [1])
        for lengths in [[-1], [1.5], [True]]:
            with self.assertRaises(ValueError):
                gqa_bytes(BASE, lengths)

    def test_packing_group_overhead_and_tail(self):
        self.assertEqual(grouped_weight_bytes(1, 128), 66)
        self.assertEqual(grouped_weight_bytes(2, 129), 138)
        self.assertEqual(grouped_weight_bytes(1, 1), 3)

    def test_roofline_units_and_compute_branch(self):
        self.assertAlmostEqual(float(roofline_seconds(140*10**9, 140*10**9,
                             989*10**12, 3350*10**9)), 0.0417910447761194)
        self.assertEqual(roofline_seconds(100, 1, 10, 100), 10)
        with self.assertRaises(ValueError):
            roofline_seconds(1, 1, 1, 0)


class Attention(unittest.TestCase):
    def test_cache_matches_full_prefix_for_multiple_chunk_sizes(self):
        q = [[[0.1*(t+1)*(h+1), -0.2*h] for h in range(4)] for t in range(7)]
        k = [[[0.3*t, 0.1*(h+1)] for h in range(2)] for t in range(7)]
        v = [[[t+h, t-h] for h in range(2)] for t in range(7)]
        full = causal_gqa(q, k, v)
        for chunk in [1, 2, 3, 7]:
            incremental = []
            for start in range(0, len(q), chunk):
                end = min(start+chunk, len(q))
                incremental += causal_gqa(q[start:end], k[:end], v[:end], start)
            self.assertEqual(full, incremental)

    def test_single_decode_query_needs_absolute_offset(self):
        q = [[[0.0]]]
        k = [[[0.0]], [[0.0]], [[0.0]]]
        v = [[[1.0]], [[3.0]], [[8.0]]]
        self.assertEqual(causal_gqa(q, k, v, 2), [[[4.0]]])
        self.assertEqual(causal_gqa(q, k, v, 0), [[[1.0]]])  # wrong decode mask

    def test_future_values_cannot_affect_past(self):
        q = [[[1.0]], [[2.0]]]
        self.assertEqual(causal_gqa(q, q, [[[3.0]], [[999.0]]])[0], [[3.0]])

    def test_head_groups_and_stable_softmax(self):
        q = [[[100.0], [100.0], [100.0], [100.0]]]
        self.assertEqual(causal_gqa(q, [[[100.0], [100.0]]], [[[2.0], [7.0]]]),
                         [[[2.0], [2.0], [7.0], [7.0]]])

    def test_reject_ragged_nonfinite_or_invalid_offset(self):
        for q, k, v, start in [([[[1.0]]], [[[1.0]]], [[[1.0]]], 1),
                ([[[float('nan')]]], [[[1.0]]], [[[1.0]]], 0),
                ([[[1.0, 2.0]]], [[[1.0]]], [[[1.0]]], 0)]:
            with self.assertRaises(ValueError):
                causal_gqa(q, k, v, start)


class Quantization(unittest.TestCase):
    def test_worked_group_and_zero(self):
        result = quantize_int4_group([-1, 0, 0.3, 1])
        self.assertEqual(result['codes'], [-7, 0, 2, 7])
        self.assertAlmostEqual(result['reconstructed'][2], 2/7)
        self.assertEqual(quantize_int4_group([0, 0])['reconstructed'], [0, 0])

    def test_round_even_and_saturate_calibration_mismatch(self):
        result = quantize_int4_group([-8, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 8], scale=1)
        self.assertEqual(result['codes'], [-7, -2, -2, 0, 0, 2, 2, 7])
        calibration = quantize_int4_group([-1, 1])['scale']
        self.assertEqual(quantize_int4_group([3], scale=calibration)['reconstructed'], [1])

    def test_invalid_and_extreme_values(self):
        for values, scale in [([], None), ([float('nan')], None), ([1], 0),
                              ([1], float('inf')), ([1], True)]:
            with self.assertRaises(ValueError):
                quantize_int4_group(values, scale)
        self.assertEqual(quantize_int4_group([1e300, -1e300], scale=1e-300)['codes'], [7, -7])


class Routing(unittest.TestCase):
    def test_destination_dedup_local_and_remote(self):
        d = dispatch([[0, 2, 3], [2, 3]], [0, 1], [0, 0, 1, 1], 2, 8)
        self.assertEqual(d['assignment_bytes'], 32)
        self.assertEqual(d['destination_bytes'], 16)
        self.assertEqual(d['matrix'], [[0, 1], [0, 0]])
        self.assertEqual(d['rank_load'], [1, 4])

    def test_batch_union_not_active_count_per_token(self):
        d = dispatch([[0, 1], [2, 3]], [0, 0], [0, 0, 1, 1], 2, 8)
        self.assertEqual(d['active_experts'], 4)
        self.assertEqual(d['rank_load'], [2, 2])

    def test_weighted_combine_and_order(self):
        result = expert_reference([2, 3], [[1, 0], [0, 1]], [[0.75, 0.25], [0.5, 0.5]],
                                  [lambda x: x+1, lambda x: 2*x])
        self.assertEqual(result, [3.75, 5.0])

    def test_bad_routes_and_weights_rejected(self):
        for routes in [[[0, 0]], [[2]], [[]]]:
            with self.assertRaises(ValueError):
                dispatch(routes, [0], [0, 1], 2, 1)
        with self.assertRaises(ValueError):
            expert_reference([1], [[0]], [[0.5]], [lambda x: x])


class Speculation(unittest.TestCase):
    def test_rejection_restores_target_not_draft(self):
        p = [Fraction(4, 5), Fraction(1, 5)]
        self.assertEqual(speculative_one_step_distribution(p, [Fraction(1, 2)]*2), p)

    def test_disjoint_support_and_all_accepted(self):
        self.assertEqual(speculative_one_step_distribution([1, 0], [0, 1]), [1, 0])
        self.assertEqual(speculative_one_step_distribution([0, 1], [0, 1]), [0, 1])

    def test_enumerated_probability_grid(self):
        for a in range(11):
            for b in range(11):
                p = [Fraction(a, 10), Fraction(10-a, 10)]
                q = [Fraction(b, 10), Fraction(10-b, 10)]
                self.assertEqual(speculative_one_step_distribution(p, q), p)


if __name__ == '__main__':
    unittest.main()
