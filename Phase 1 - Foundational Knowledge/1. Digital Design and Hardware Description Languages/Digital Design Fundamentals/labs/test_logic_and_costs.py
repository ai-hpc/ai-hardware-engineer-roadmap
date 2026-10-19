from fractions import Fraction
from itertools import combinations
import unittest

from logic_and_costs import (cache_address, decode_secded, encode_secded,
                             lfsr_step, lru_order_bits, peak_bandwidth, pipeline)


class LogicAndCostTests(unittest.TestCase):
    def test_secded_independent_fixture_and_minimum_distance(self):
        # data 1010 -> positions d1,d2,d3,d4 = 0,1,0,1;
        # p1=0, p2=1, p4=0, overall=1 -> bits 7..0 = 11010010.
        self.assertEqual(encode_secded(10), 0xD2)
        words = [encode_secded(x) for x in range(16)]
        self.assertEqual(min((a ^ b).bit_count() for a, b in combinations(words, 2)), 4)
        for message, word in enumerate(words):
            self.assertEqual(decode_secded(word), (message, "clean", None))

    def test_every_single_bit_error_is_corrected(self):
        for message in range(16):
            for position in range(1, 9):
                with self.subTest(message=message, position=position):
                    damaged = encode_secded(message) ^ (1 << (position - 1))
                    self.assertEqual(decode_secded(damaged), (message, "corrected", position))

    def test_every_double_bit_error_is_rejected(self):
        for message in range(16):
            for a, b in combinations(range(8), 2):
                self.assertEqual(decode_secded(encode_secded(message) ^ (1 << a) ^ (1 << b)),
                                 (None, "double_error", None))

    def test_four_errors_can_be_another_valid_word(self):
        # Do not imply protection against arbitrary corruption.
        self.assertEqual(encode_secded(0), 0)
        self.assertEqual((encode_secded(0) ^ encode_secded(1)).bit_count(), 4)
        self.assertEqual(decode_secded(encode_secded(1)), (1, "clean", None))

    def test_lfsr_orbit_from_every_nonzero_seed_and_zero_lockup(self):
        self.assertEqual(lfsr_step(0), 0)
        self.assertEqual([lfsr_step(x) for x in (15, 14, 12, 8)], [14, 12, 8, 1])
        for seed in range(1, 16):
            seen, state = set(), seed
            for _ in range(15):
                self.assertNotIn(state, seen)
                self.assertNotEqual(state, 0)
                seen.add(state); state = lfsr_step(state)
            self.assertEqual(state, seed)
            self.assertEqual(seen, set(range(1, 16)))

    def test_bandwidth_uses_transfers_without_extra_burst_factor(self):
        self.assertEqual(peak_bandwidth(64, 6_400_000_000), 51_200_000_000)
        self.assertEqual(peak_bandwidth(32, 6_400_000_000), 25_600_000_000)
        self.assertEqual(peak_bandwidth(1024, 6_400_000_000), 819_200_000_000)

    def test_pipeline_fill_latency_and_overhead(self):
        p = pipeline([Fraction(3), Fraction(4), Fraction(2)], Fraction(1, 5), 10)
        self.assertEqual(p, {"period_ns": Fraction(21, 5), "latency_ns": Fraction(63, 5),
                             "batch_ns": Fraction(252, 5)})
        single = pipeline([Fraction(9)], Fraction(1, 5), 10)
        self.assertEqual(single["batch_ns"], 92)
        self.assertEqual(pipeline([Fraction(3), Fraction(4), Fraction(2)], Fraction(1, 5), 1)
                         ["batch_ns"], p["latency_ns"])

    def test_cache_mapping_and_lru_information(self):
        self.assertEqual(cache_address(0x140), (0, 5, 0))
        self.assertEqual(cache_address(0x540), (1, 5, 0))
        self.assertEqual(cache_address(0x57F), (1, 5, 63))
        self.assertEqual(cache_address(1023), (0, 15, 63))
        self.assertEqual(cache_address(1024), (1, 0, 0))
        self.assertEqual([lru_order_bits(w) for w in (1, 2, 4, 8)], [0, 1, 5, 16])

    def test_invalid_contract_inputs(self):
        for bad in (True, -1, 16, 1.5):
            with self.assertRaises(ValueError): encode_secded(bad)
            with self.assertRaises(ValueError): lfsr_step(bad)
        for bad in (True, -1, 256, "1"):
            with self.assertRaises(ValueError): decode_secded(bad)
        with self.assertRaises(ValueError): peak_bandwidth(0, 6400)
        with self.assertRaises(ValueError): cache_address(0, 1024, 64, 3)
        with self.assertRaises(ValueError): cache_address(True)
        with self.assertRaises(ValueError): pipeline([], Fraction(0), 1)
        with self.assertRaises(ValueError): pipeline([Fraction(1)], Fraction(-1), 1)
        with self.assertRaises(ValueError): pipeline([Fraction(1)], Fraction(0), 0)


if __name__ == "__main__":
    unittest.main()
