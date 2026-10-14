"""Checks boundaries that would silently change an architecture decision."""
import math
import unittest

from design_lab import Attempt, Job, MIB, memory_fixture, schedule, storage_bytes, summarize, tile_utilization, traffic_bytes


class StorageAndTrafficTests(unittest.TestCase):
    def test_proposal_units_and_repeated_subgraph(self):
        result = memory_fixture()
        self.assertEqual(result["weight_capacity_bytes"], 1_800_000_000)
        self.assertEqual(result["weight_traffic_bytes_per_evaluation"], 4_500_000_000)
        self.assertAlmostEqual(result["weight_bandwidth_floor_ms"], 37.5)
        self.assertEqual(result["required_weight_GB_per_s_at_20_Hz"], 90)

    def test_packing_and_partial_final_block(self):
        self.assertEqual(storage_bytes(1, 4), 1)
        self.assertEqual(storage_bytes(4096, 4, 16, 8, 4), 2308)
        self.assertEqual(storage_bytes(17, 4, 16, 8, 4), 22)
        self.assertEqual(memory_fixture(nvfp4_metadata=True)["backbone_weight_bytes"], 1_687_500_004)

    def test_cold_fill_is_not_free(self):
        self.assertEqual(traffic_bytes(100, 1, 100), 100)
        self.assertEqual(traffic_bytes(100, 10, 100), 100)
        self.assertEqual(traffic_bytes(100, 10, 100, False), 0)
        self.assertEqual(traffic_bytes(100, 10, 30), 730)
        self.assertEqual(traffic_bytes(100, 10, 30, False), 700)

    def test_activation_reserve_reduces_weight_residency(self):
        a = memory_fixture(64, 8)
        self.assertEqual(a["expert_resident_bytes"], 56 * MIB)
        self.assertEqual(a["weight_traffic_bytes_per_evaluation"], 4_500_000_000 - 9 * 56 * MIB)
        self.assertLess(a["expert_resident_bytes"], a["expert_weight_bytes"])

    def test_invalid_memory_inputs_rejected(self):
        for options in ({"bandwidth_gbps": 0}, {"bandwidth_gbps": math.nan}, {"bandwidth_gbps": math.inf}, {"sram_mib": 16, "reserve_mib": 32}, {"sram_mib": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                memory_fixture(**options)
        with self.assertRaises(ValueError): traffic_bytes(100, 1, 101)
        with self.assertRaises(ValueError): storage_bytes(10, 4, scale_bits=8)

    def test_tiles_include_all_three_tail_dimensions(self):
        self.assertEqual(tile_utilization(8, 256, 256, 64, 64, 64), .125)
        self.assertEqual(tile_utilization(16, 16, 16, 16, 16, 16), 1)
        self.assertAlmostEqual(tile_utilization(17, 19, 23, 16, 16, 16), 17*19*23/(32**3))
        with self.assertRaises(ValueError): tile_utilization(0, 1, 1, 1, 1, 1)


class SchedulingTests(unittest.TestCase):
    def setUp(self):
        self.jobs = [Job("vision", 0, 18, 40), Job("action", 2, 3, 8)]

    def test_priority_cannot_interrupt_whole_nonpreemptive_job(self):
        fifo = schedule(self.jobs)
        self.assertEqual(fifo["jobs"]["action"]["finish_ms"], 21)
        self.assertTrue(fifo["jobs"]["action"]["missed"])

    def test_legal_tile_boundaries_allow_fresh_action(self):
        tiled = schedule(self.jobs, 3)
        self.assertEqual(tiled["jobs"]["action"]["finish_ms"], 6)
        self.assertFalse(tiled["jobs"]["action"]["missed"])
        self.assertEqual(tiled["jobs"]["vision"]["finish_ms"], 21)

    def test_dispatch_cost_can_destroy_apparent_guarantee(self):
        result = schedule(self.jobs, 3, dispatch_ms=2)
        self.assertTrue(result["jobs"]["action"]["missed"])

    def test_decimal_release_is_not_delayed_one_quantum(self):
        r = schedule([Job("vision", 0, 1.2, 5), Job("action", 1, .05, 1.05)], .1)
        self.assertEqual(r["jobs"]["action"]["finish_ms"], 1.05)
        self.assertFalse(r["jobs"]["action"]["missed"])

    def test_decimal_deadline_equality_is_not_a_miss(self):
        r = schedule([Job("job", 0, 1, 1.4)], .3, .1)
        self.assertEqual(r["jobs"]["job"]["finish_ms"], 1.4)
        self.assertFalse(r["jobs"]["job"]["missed"])
        self.assertTrue(schedule([Job("job", 0, 1, 1.399)], .3, .1)["jobs"]["job"]["missed"])

    def test_idle_release_and_invalid_requests(self):
        self.assertEqual(schedule([Job("later", 10, 2, 12)], 1)["jobs"]["later"]["finish_ms"], 12)
        for jobs, quantum in ((self.jobs, 0), ([Job("a", 2, 1, 1)], 1), ([self.jobs[0], self.jobs[0]], 1)):
            with self.subTest(jobs=jobs), self.assertRaises(ValueError): schedule(jobs, quantum)


class EvaluationTests(unittest.TestCase):
    def test_failed_attempt_energy_and_observation_age_count(self):
        rows = [Attempt("a", True, 10, 0, 35, 40), Attempt("b", False, 12, 100, 160, 190), Attempt("c", True, 8, 200, 245, 260)]
        r = summarize(rows, 50, 70)
        self.assertEqual(r["energy_per_success_j"], 15)
        self.assertEqual(r["energy_per_attempt_j"], 10)
        self.assertEqual(r["deadline_misses"], 1)
        self.assertEqual(r["stale_at_consumption"], 1)
        self.assertEqual(r["latency_ms"]["p99"], 60)

    def test_no_success_is_undefined_not_zero_energy(self):
        r = summarize([Attempt("x", False, 2, 0, 1, 2)], 5, 5)
        self.assertIsNone(r["energy_per_success_j"])

    def test_decimal_latency_and_age_equality_are_valid(self):
        rows = [Attempt("a", True, 1, .1, .8, .8)]
        r = summarize(rows, .7, .7)
        self.assertEqual(r["deadline_misses"], 0)
        self.assertEqual(r["stale_at_consumption"], 0)
        r = summarize(rows, .699, .699)
        self.assertEqual(r["deadline_misses"], 1)
        self.assertEqual(r["stale_at_consumption"], 1)

    def test_bad_clocks_and_duplicate_attempts_rejected(self):
        a = Attempt("a", True, 10, 0, 1, 2)
        for rows in ([a, a], [Attempt("b", True, 2, 4, 3, 5)], [Attempt("c", True, math.nan, 0, 1, 2)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError): summarize(rows, 5, 5)


if __name__ == "__main__":
    unittest.main()
