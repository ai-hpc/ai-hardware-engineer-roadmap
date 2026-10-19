"""Boundary, accounting, and token-alignment tests with independent oracles."""
import unittest

from tiling_lab import Matrix, fixture, ideal_cycle_estimates, reference, systolic, tiled


class TilingTests(unittest.TestCase):
    def test_hand_computed_rectangular_product(self):
        a = Matrix(2, 3, (1, 2, 3, 4, 5, 6))
        b = Matrix(3, 2, (7, 8, 9, 10, 11, 12))
        expected = Matrix(2, 2, (58, 64, 139, 154))
        self.assertEqual(reference(a, b), expected)
        self.assertEqual(tiled(a, b, 1, 2, 2)[0], expected)
        self.assertEqual(systolic(a, b, 2, 2).output, expected)

    def test_ragged_and_empty_shapes_against_dot_products(self):
        for m in range(6):
            for n in range(6):
                for k in range(5):
                    a, b = fixture(m, k, 2), fixture(k, n, 5)
                    for tm, tn, tk in ((1, 1, 1), (2, 3, 2), (4, 4, 3)):
                        with self.subTest(shape=(m, n, k), tile=(tm, tn, tk)):
                            out, traffic, trace = tiled(a, b, tm, tn, tk)
                            self.assertEqual(out, reference(a, b))
                            self.assertEqual(traffic.useful_macs, m * n * k)
                            self.assertEqual(traffic.a_read_bytes,
                                             2 * m * k * ((n + tn - 1) // tn))
                            self.assertEqual(traffic.b_read_bytes,
                                             2 * k * n * ((m + tm - 1) // tm))
                            self.assertEqual(traffic.c_write_bytes, 4 * m * n)
                            self.assertEqual(sum(t.rows * t.cols * t.depth for t in trace),
                                             m * n * k)

    def test_reduction_tile_changes_storage_not_logical_reads(self):
        a, b = fixture(5, 3), fixture(3, 7)
        _, small, _ = tiled(a, b, 2, 4, 1)
        _, large, _ = tiled(a, b, 2, 4, 3)
        self.assertEqual(small.total_bytes, large.total_bytes)
        self.assertEqual(small.total_bytes, 326)
        self.assertEqual(small.peak_resident_bytes, 44)
        self.assertEqual(large.peak_resident_bytes, 68)

    def test_capacity_exact_boundary_and_zero_reduction(self):
        a, b = fixture(5, 3), fixture(3, 7)
        tiled(a, b, 2, 4, 2, capacity_bytes=56)
        with self.assertRaisesRegex(ValueError, "needs 56 bytes"):
            tiled(a, b, 2, 4, 2, capacity_bytes=55)
        out, traffic, trace = tiled(Matrix(2, 0, ()), Matrix(0, 3, ()),
                                    2, 3, 4, capacity_bytes=24)
        self.assertEqual(out.data, (0,) * 6)
        self.assertEqual(traffic.total_bytes, 24)
        self.assertEqual(traffic.peak_resident_bytes, 24)
        self.assertEqual(trace, [])
        _, empty, _ = tiled(Matrix(0, 2, ()), fixture(2, 3),
                            2, 3, 1, capacity_bytes=0)
        self.assertEqual(empty.total_bytes, 0)
        self.assertEqual(empty.peak_resident_bytes, 0)

    def test_token_pipeline_fill_drain_and_padding(self):
        result = systolic(fixture(2, 3), fixture(3, 2), 2, 2)
        self.assertEqual(result.active_macs_per_cycle, (1, 3, 4, 3, 1))
        self.assertEqual(result.cycles, 5)
        self.assertEqual(result.issued_macs, 12)
        padded = systolic(fixture(1, 3), fixture(3, 1), 2, 2)
        self.assertEqual(padded.output, reference(fixture(1, 3), fixture(3, 1)))
        self.assertEqual(padded.issued_macs, 12)
        self.assertEqual(padded.useful_macs, 3)
        self.assertEqual(padded.cycles, 5)

    def test_systolic_small_arrays_against_reference(self):
        for rows in range(1, 5):
            for cols in range(1, 5):
                for m in range(rows + 1):
                    for n in range(cols + 1):
                        for k in range(5):
                            a, b = fixture(m, k, 1), fixture(k, n, 3)
                            got = systolic(a, b, rows, cols)
                            self.assertEqual(got.output, reference(a, b))
                            active = bool(m and n and k)
                            self.assertEqual(got.cycles, k + rows + cols - 2 if active else 0)
                            self.assertEqual(got.issued_macs, rows * cols * k if active else 0)
                            self.assertTrue(all(v <= rows * cols for v in got.active_macs_per_cycle))

    def test_ideal_estimate_has_explicit_overlap_assumption(self):
        _, traffic, trace = tiled(fixture(5, 3), fixture(3, 7), 2, 4, 2)
        result = ideal_cycle_estimates(traffic, trace, 2, 4, 16)
        self.assertEqual(result, {"compute_cycles": 66, "transfer_cycles": 21,
                                 "perfect_overlap_lower_bound": 66,
                                 "serialized_ideal_estimate": 87})

    def test_invalid_shapes_parameters_and_coordinates(self):
        with self.assertRaises(ValueError):
            Matrix(2, 2, (1, 2))
        with self.assertRaises(ValueError):
            Matrix(True, 1, (1,))
        with self.assertRaises(ValueError):
            reference(fixture(2, 3), fixture(2, 3))
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                tiled(fixture(1, 1), fixture(1, 1), value, 1, 1)
        with self.assertRaises(ValueError):
            systolic(fixture(3, 2), fixture(2, 1), 2, 2)
        with self.assertRaises(IndexError):
            fixture(2, 2).at(-1, 0)


if __name__ == "__main__":
    unittest.main()
