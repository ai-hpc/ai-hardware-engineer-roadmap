import copy
import json
import math
from pathlib import Path
import random
import unittest

import hybrid_reference as ref


class HybridContracts(unittest.TestCase):
    def assertMatrixClose(self, actual, expected, tolerance=1e-11):
        self.assertEqual(len(actual), len(expected))
        for row, target in zip(actual, expected):
            self.assertEqual(len(row), len(target))
            for x, y in zip(row, target):
                self.assertAlmostEqual(x, y, delta=tolerance)

    def test_worked_example_and_input_ownership(self):
        before = [[2., 0.], [0., 4.]]
        output, state = ref.kda_step(before, [1., 0.], [1., 0.], [6., 1.], [1., 1.], .5)
        self.assertEqual(state, [[4., .5], [0., 4.]])
        self.assertEqual(output, [4., .5])
        self.assertEqual(before, [[2., 0.], [0., 4.]])

    def test_zero_write_is_decay_only(self):
        output, state = ref.kda_step([[2.], [4.]], [1., 1.], [.6, .8], [100.], [.5, .25], 0.)
        self.assertEqual(state, [[1.], [1.]])
        self.assertEqual(output, [2.])

    def test_normalized_key_full_write_replaces_association(self):
        output, _ = ref.kda_step([[2.], [4.]], [.6, .8], [.6, .8], [7.], [1., 1.], 1.)
        self.assertAlmostEqual(output[0], 7.)

    def test_decay_correction_order_mutant_is_detected(self):
        state, key, value, decay, beta = [[2.], [4.]], [.6, .8], [3.], [.25, .8], .7
        _, correct = ref.kda_step(state, key, key, value, decay, beta)
        a, b = ref.affine_update(key, value, decay, beta)
        self.assertMatrixClose(ref.apply_affine((a, b), state), correct)
        # Mutant D(I-beta kk^T) instead of (I-beta kk^T)D.
        swapped = [[decay[i] * ((i == j) - beta * key[i] * key[j]) for j in range(2)] for i in range(2)]
        wrong = ref.apply_affine((swapped, b), state)
        self.assertGreater(max(abs(x[0] - y[0]) for x, y in zip(wrong, correct)), .1)

    def test_affine_chunk_composition_matches_serial_for_rectangular_states(self):
        rng = random.Random(27)
        for length in (1, 3, 4, 5, 7, 8, 9, 17):
            initial = [[rng.uniform(-1, 1) for _ in range(5)] for _ in range(3)]
            serial = copy.deepcopy(initial)
            total = None
            for _ in range(length):
                key = [rng.uniform(-.5, .5) for _ in range(3)]
                value = [rng.uniform(-1, 1) for _ in range(5)]
                decay = [rng.uniform(.2, 1) for _ in range(3)]
                beta = rng.random()
                _, serial = ref.kda_step(serial, key, key, value, decay, beta)
                update = ref.affine_update(key, value, decay, beta)
                total = update if total is None else ref.compose(update, total)
            self.assertMatrixClose(ref.apply_affine(total, initial), serial)

    def test_composition_order_and_associativity(self):
        updates = [ref.affine_update([.6, .8], [v], [a, .7], .4) for v, a in ((1., .2), (2., .4), (3., .9))]
        first, second, third = updates
        left = ref.compose(third, ref.compose(second, first))
        right = ref.compose(ref.compose(third, second), first)
        for x, y in zip(left, right):
            self.assertMatrixClose(x, y)
        state = [[1.], [-2.]]
        self.assertNotEqual(ref.apply_affine(ref.compose(second, first), state),
                            ref.apply_affine(ref.compose(first, second), state))

    def test_output_agreement_can_hide_state_corruption(self):
        correct, corrupt = [[1.], [2.]], [[1.], [99.]]
        for query, expected_equal in (([1., 0.], True), ([0., 1.], False)):
            x, _ = ref.kda_step(correct, query, [1., 0.], [0.], [1., 1.], 0.)
            y, _ = ref.kda_step(corrupt, query, [1., 0.], [0.], [1., 1.], 0.)
            self.assertEqual(x == y, expected_equal)

    def test_mla_absorption_preserves_softmax_and_outputs(self):
        rng = random.Random(9)
        for tokens in (1, 3, 9):
            latents = [[rng.uniform(-2, 2) for _ in range(2)] for _ in range(tokens)]
            key_up = [[rng.random() for _ in range(2)] for _ in range(3)]
            value_up = [[rng.random() for _ in range(2)] for _ in range(4)]
            query = [rng.uniform(-3, 3) for _ in range(3)]
            self.assertMatrixClose([ref.mla_expanded(latents, key_up, value_up, query, 1 / math.sqrt(3))],
                                   [ref.mla_absorbed(latents, key_up, value_up, query, 1 / math.sqrt(3))])

    def test_mla_wrong_scaling_is_detected(self):
        latents, key_up, value_up, query = [[1., 0.], [0., 2.]], [[2., 0.], [0., 1.]], [[1., 2.]], [1., 1.]
        good = ref.mla_absorbed(latents, key_up, value_up, query, .5)
        # Deliberately avoid a tied score so dropping scale changes probabilities.
        query[1] = 2.
        good = ref.mla_absorbed(latents, key_up, value_up, query, .5)
        bad = ref.mla_absorbed(latents, key_up, value_up, query, 1.)
        self.assertGreater(abs(good[0] - bad[0]), .1)

    def test_sinkhorn_convergence_and_finite_iteration_error(self):
        matrix = [[3., 1., .5], [.5, 2., 1.], [1., .5, 3.]]
        early, late = ref.sinkhorn(matrix, 2), ref.sinkhorn(matrix, 20)
        early_error = max(abs(sum(row) - 1) for row in early)
        late_error = max(abs(sum(row) - 1) for row in late)
        self.assertGreater(early_error, .001)
        self.assertLess(late_error, 1e-10)
        for column in zip(*late):
            self.assertAlmostEqual(sum(column), 1.)

    def test_doubly_stochastic_mix_can_cancel_signed_activations(self):
        self.assertEqual(ref.matmul([[.5, .5], [.5, .5]], [[1.], [-1.]]), [[0.], [0.]])

    def test_restore_every_field_reproduces_uninterrupted_continuation(self):
        clean = ref.HybridState([[0.], [0.]], [], [])
        for scalar in (1., -2., 3.):
            clean.advance(scalar)
        checkpoint = clean.snapshot()
        expected = clean.advance(4.)
        dirty = checkpoint.snapshot()
        for rejected in (40., 50., 60.):
            dirty.advance(rejected)
        restored = checkpoint.snapshot()
        self.assertEqual(restored.advance(4.), expected)
        self.assertEqual(restored, clean)
        self.assertEqual(checkpoint.prefix_length, 3)
        self.assertEqual(len(checkpoint.latent_history), 3)

    def test_partial_rollback_mutants_fail_even_with_correct_prefix_length(self):
        base = ref.HybridState([[0.], [0.]], [], [])
        for scalar in (1., -2., 3.):
            base.advance(scalar)
        expected = base.snapshot().advance(4.)
        dirty = base.snapshot()
        for scalar in (40., 50., 60.):
            dirty.advance(scalar)
        for omitted in ("recurrent", "convolution", "latent_history"):
            wrong = base.snapshot()
            setattr(wrong, omitted, copy.deepcopy(getattr(dirty, omitted)))
            self.assertEqual(wrong.prefix_length, 3)
            self.assertGreater(abs(wrong.advance(4.) - expected), .01, omitted)

    def test_fork_does_not_alias_parent_or_other_branch(self):
        parent = ref.HybridState([[0.], [0.]], [], [])
        parent.advance(1.)
        before, left, right = parent.snapshot(), parent.snapshot(), parent.snapshot()
        left.advance(100.)
        right.advance(-100.)
        self.assertEqual(parent, before)
        self.assertNotEqual(left.recurrent, right.recurrent)

    def test_pinned_config_layer_schedules(self):
        cfg = json.loads(Path(ref.__file__).with_name("checkpoint-config.json").read_text())["text_config"]
        expected = ["linear_attention" if i % 4 != 3 else "deepseek_sparse_attention" for i in range(45)]
        self.assertEqual(cfg["layer_types"], expected)
        self.assertEqual(len(cfg["linear_attn_config"]["kda_layers"]), 34)
        self.assertEqual(cfg["first_k_dense_replace"], 3)

    def test_memory_replication_concurrency_and_units(self):
        one = ref.memory_ledger(1048576, 1, 8)
        two = ref.memory_ledger(1048576, 2, 8)
        self.assertEqual(one["kda_all_heads_bytes_per_request"], 136 * 1024 ** 2)
        self.assertEqual(one["kda_bytes_per_gpu"], 17 * 1024 ** 2)
        self.assertEqual(one["mla_bytes_per_gpu_if_replicated"], 11 * 1024 ** 3)
        for field in ("kda_bytes_per_gpu", "mla_bytes_per_gpu_if_replicated"):
            self.assertEqual(two[field], 2 * one[field])
        self.assertEqual(ref.memory_ledger(1048576, 1, 4)["mla_bytes_per_gpu_if_replicated"], 11 * 1024 ** 3)

    def test_invalid_contracts(self):
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                ref.kda_step([[0.]], [1.], [1.], [bad], [1.], .5)
        with self.assertRaises(ValueError):
            ref.kda_step([[0.]], [1.], [1.], [1.], [1.], 2.)
        for args in ((0, 1, 8), (10, 0, 8), (10, 1, 3), (10.5, 1, 8)):
            with self.assertRaises(ValueError):
                ref.memory_ledger(*args)

    def test_ragged_matrix_and_mismatched_addition_never_truncate(self):
        # zip used to silently return [[14], [12]] for the invalid product.
        with self.assertRaises(ValueError):
            ref.matmul([[1, 2], [3]], [[4], [5]])
        with self.assertRaises(ValueError):
            ref.transpose([[1, 2], [3]])
        with self.assertRaises(ValueError):
            ref.add([[1, 2]], [[3]])
        with self.assertRaises(ValueError):
            ref.matmul([[1e308]], [[2]])

    def test_both_mla_paths_reject_query_and_projection_dimension_errors(self):
        for implementation in (ref.mla_expanded, ref.mla_absorbed):
            for query in ([1.], [1., 2., 3.]):
                with self.subTest(implementation=implementation.__name__, query=query):
                    with self.assertRaises(ValueError):
                        implementation([[1., 2.]], [[1., 0.], [0., 1.]], [[1., 1.]], query)
            with self.assertRaises(ValueError):
                implementation([[1., 2.]], [[1., 0.], [0., 1.]], [[1.]], [1., 2.])
            with self.assertRaises(ValueError):
                implementation([[1., 2.]], [[1., 0.], [0., 1.]], [[1., 1.]], [1., 2.], math.nan)

    def test_sinkhorn_rejects_nonfinite_epsilon_ragged_rows_and_overflow(self):
        for epsilon in (math.nan, math.inf, -1.):
            with self.assertRaises(ValueError):
                ref.sinkhorn([[1., 2.], [3., 4.]], epsilon=epsilon)
        with self.assertRaises(ValueError):
            ref.sinkhorn([[1., 2.], [3.]])
        with self.assertRaises(ValueError):
            ref.sinkhorn([[1e308, 1e308], [1., 1.]])
        for rounds in (True, 1.5, 0):
            with self.assertRaises(ValueError):
                ref.sinkhorn([[1.]], rounds=rounds)

    def test_affine_path_rejects_invalid_decay_and_gate(self):
        for decay, beta in (([1.], .5), ([1., math.nan], .5), ([1., 1.], 2.)):
            with self.assertRaises(ValueError):
                ref.affine_update([.6, .8], [1.], decay, beta)


if __name__ == "__main__":
    unittest.main()
