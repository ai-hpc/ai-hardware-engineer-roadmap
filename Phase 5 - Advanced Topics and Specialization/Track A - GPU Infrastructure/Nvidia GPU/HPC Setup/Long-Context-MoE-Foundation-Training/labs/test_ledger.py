import unittest
from ledger import capacity, dense_mesh, layer_costs, routing_summary


class LedgerTests(unittest.TestCase):
    def test_hand_count_forward(self):
        r = layer_costs(3, 4, 2, 1, 2, 8)
        self.assertEqual(r['attention_pair_flops'], 96)  # six causal pairs, four channels
        self.assertEqual(r['qkv_and_output_flops'], 288)  # Q48 + K24 + V24 + O48 multiplies, x2
        self.assertEqual(r['nongated_mlp_flops'], 384)

    def test_byte_units_once(self):
        r = layer_costs(3, 4, 2, 1, 2, 8)
        self.assertEqual(r['q_k_v_o_tensor_bytes'], 72)
        self.assertEqual(r['one_mlp_intermediate_bytes'], 48)
        self.assertEqual(r['one_dense_score_matrix_bytes'], 36)
        self.assertEqual(r['inference_kv_bytes'], 24)

    def test_causal_and_dense_pairs(self):
        self.assertEqual(layer_costs(3, 4, 2, 1, 2, 8, causal=False)['attention_pair_flops'], 144)

    def test_gqa_changes_projection_and_kv_not_query_pair_work(self):
        a, b = (layer_costs(5, 4, 2, kv, 2, 8) for kv in (1, 2))
        self.assertEqual(a['attention_pair_flops'], b['attention_pair_flops'])
        self.assertEqual(2 * a['inference_kv_bytes'], b['inference_kv_bytes'])
        self.assertLess(a['qkv_and_output_flops'], b['qkv_and_output_flops'])

    def test_exact_integer_capacity(self):
        self.assertEqual(capacity(4, 2, 4, 1.0), 2)  # old int(x+1) incorrectly returned 3
        self.assertEqual(capacity(5, 2, 4, 1.0), 3)

    def test_capacity_preserves_counts_and_positive_decimal_factors(self):
        self.assertEqual(capacity(2**53 + 1, 1, 1, 1.), 2**53 + 1)
        self.assertEqual(capacity(1, 1, 4, 5e-324), 1)
        self.assertEqual(capacity(10, 1, 1, .1), 1)
        self.assertEqual(capacity(10, 1, 1, .3), 3)
        for factor in (None, "1.0", True, float('inf')):
            with self.assertRaises(ValueError): capacity(4, 2, 4, factor)

    def test_drop_denominators_and_uniform_shares(self):
        r = routing_summary([[0, 1]] * 4, 4, 2)
        self.assertEqual(r['dropped_assignment_fraction'], .5)
        self.assertEqual(r['tokens_with_no_expert_fraction'], .5)
        self.assertEqual(r['assignment_shares'], [.5, .5, 0, 0])
        r = routing_summary([[0, 1], [1, 2], [2, 3], [3, 0]], 4, 2)
        self.assertEqual(r['assignment_shares'], [.25] * 4)
        self.assertEqual(r['dropped_assignment_fraction'], 0)

    def test_any_drop_and_zero_coverage_differ(self):
        r = routing_summary([[0, 1], [0, 2]], 3, 1)
        self.assertEqual(r['dropped_assignment_fraction'], .25)
        self.assertEqual(r['tokens_with_any_drop_fraction'], .5)
        self.assertEqual(r['tokens_with_no_expert_fraction'], 0)

    def test_expert_axis_not_multiplied_twice(self):
        r = dense_mesh(64, tp=2, pp=2, cp=2, ep=4, experts=8)
        self.assertEqual(r['dense_dp'], 8)
        self.assertEqual(r['expert_dp'], 4)
        self.assertEqual(r['experts_per_ep_rank'], 2)

    def test_invalid_shapes(self):
        for args in [(0,4,2,1,2,8), (3,5,2,1,2,8), (3,4,2,3,2,8), (3,4,2,1,2,True)]:
            with self.assertRaises(ValueError): layer_costs(*args)
        for args in [(4,2,4,float('nan')), (4,5,4,1), (4,2,4,-1)]:
            with self.assertRaises(ValueError): capacity(*args)
        for rows in ([], [[0,0]], [[4]], [[0],[0,1]], [[True]]):
            with self.assertRaises(ValueError): routing_summary(rows,4,2)
        for kwargs in ({'tp':3}, {'ep':3,'experts':6}, {'ep':4,'experts':6}):
            with self.assertRaises(ValueError): dense_mesh(8,**kwargs)


if __name__ == '__main__': unittest.main()
