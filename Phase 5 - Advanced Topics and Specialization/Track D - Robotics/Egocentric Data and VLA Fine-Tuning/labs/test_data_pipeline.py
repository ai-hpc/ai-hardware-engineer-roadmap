import math
import random
import unittest

from budget_eval import (epochs_seen, fits, lora_params, mcnemar_exact_p, mixture_batches, paired_counts,
                         training_state_bytes, wilson_interval)
from dataset_contract import (aggregate_stats, check_timestamps, feature_stats, normalize, stats_parity,
                              unnormalize, validate_frame)
from ego_audit import (cohens_kappa, duration_stats, find_duplicates, group_split, label_distribution,
                       split_leakage, timestamp_report)


def make_episodes(n=60, seed=1):
    rng = random.Random(seed)
    verbs = ["pick"] * 20 + ["place"] * 12 + ["pour"] * 6 + ["open", "close", "wipe", "stir"]
    eps = []
    for i in range(n):
        frames = rng.randint(60, 300)
        eps.append({
            "episode_id": f"e{i:03d}", "participant": f"p{i % 12:02d}", "location": f"kitchen_{i % 5}",
            "timestamps_s": [k / 30.0 for k in range(frames)], "verb": rng.choice(verbs),
            "content_hash": f"h{i:03d}",
        })
    eps[7]["content_hash"] = eps[3]["content_hash"]  # a re-uploaded clip
    return eps


class TestAudit(unittest.TestCase):
    def test_timestamps_ok_and_broken(self):
        good = timestamp_report([k / 30.0 for k in range(100)], 30.0)
        self.assertTrue(good["ok"])
        ts = [k / 30.0 for k in range(100)]
        ts[50] = ts[48]                       # host-clock jitter: time goes backwards
        del ts[70]                            # a dropped frame
        bad = timestamp_report(ts, 30.0)
        self.assertFalse(bad["ok"])
        self.assertGreaterEqual(bad["non_monotonic"], 1)
        self.assertGreaterEqual(bad["estimated_dropped_frames"], 1)

    def test_duplicates_and_durations(self):
        eps = make_episodes()
        self.assertEqual(find_duplicates(eps), [["e003", "e007"]])
        s = duration_stats(eps, 30.0)
        self.assertEqual(s["count"], 60)
        self.assertLessEqual(s["min_s"], s["median_s"])

    def test_random_split_leaks_and_group_split_does_not(self):
        eps = make_episodes()
        naive = {"train": eps[:45], "test": eps[45:]}
        self.assertTrue(split_leakage(naive, "participant"))       # same people in both splits
        grouped = group_split(eps, "participant", {"train": 0.75, "test": 0.25}, seed=0)
        self.assertEqual(split_leakage(grouped, "participant"), {})
        self.assertEqual(sum(len(v) for v in grouped.values()), len(eps))
        self.assertGreater(len(grouped["test"]), 0)

    def test_label_long_tail(self):
        d = label_distribution(make_episodes(), "verb")
        self.assertLess(d["entropy_bits"], d["max_entropy_bits"])
        self.assertGreater(d["top10pct_share"], 0.2)

    def test_kappa(self):
        a = ["pick", "place", "pick", "pour", "pick", "place"]
        self.assertAlmostEqual(cohens_kappa(a, a), 1.0)
        b = ["pick", "place", "place", "pour", "pick", "pick"]
        self.assertLess(cohens_kappa(a, b), 1.0)


class TestDatasetContract(unittest.TestCase):
    schema = {"observation.state": {"dtype": "float32", "shape": [7]},
              "action": {"dtype": "float32", "shape": [7]},
              "task": {"dtype": "string", "shape": [1]}}

    def test_validate_frame(self):
        good = {"observation.state": [0.0] * 7, "action": [0.1] * 7, "task": "put the cup in the bowl"}
        self.assertEqual(validate_frame(good, self.schema), [])
        bad = {"observation.state": [0.0] * 6, "action": [0.1] * 6 + [float("nan")], "extra": 1}
        problems = validate_frame(bad, self.schema)
        self.assertTrue(any("missing feature 'task'" in p for p in problems))
        self.assertTrue(any("unexpected feature 'extra'" in p for p in problems))
        self.assertTrue(any("shape" in p for p in problems))
        self.assertTrue(any("non-finite" in p for p in problems))

    def test_timestamps(self):
        self.assertTrue(check_timestamps([k / 10 for k in range(20)], 10.0))
        self.assertFalse(check_timestamps([0.0, 0.1, 0.25], 10.0))

    def test_stats_merge_matches_full_data(self):
        rng = random.Random(3)
        episodes = [[[rng.gauss(0.3 * d, 1 + d) for d in range(3)] for _ in range(rng.randint(20, 80))]
                    for _ in range(6)]
        full = feature_stats([v for ep in episodes for v in ep])
        merged = aggregate_stats([feature_stats(ep) for ep in episodes])
        for key in ("mean", "std", "min", "max"):
            for a, b in zip(full[key], merged[key]):
                self.assertAlmostEqual(a, b, places=9)
        self.assertTrue(merged["quantiles_approximate"])

    def test_normalization_roundtrip_all_modes(self):
        data = [[float(i), 10.0 - i, 0.5 * i * i] for i in range(50)]
        st = feature_stats(data)
        x = [12.0, -3.0, 40.0]
        for mode in ("MEAN_STD", "MIN_MAX", "QUANTILES", "IDENTITY"):
            back = unnormalize(normalize(x, st, mode), st, mode)
            for a, b in zip(x, back):
                self.assertAlmostEqual(a, b, places=6)

    def test_scalar_and_bool_features(self):
        schema = {"timestamp": {"dtype": "float32", "shape": [1]},
                  "frame_index": {"dtype": "int64", "shape": [1]},
                  "action": {"dtype": "float32", "shape": [2]}}
        self.assertEqual(validate_frame({"timestamp": 0.5, "frame_index": 3, "action": [0.1, 0.2]}, schema), [])
        bad = validate_frame({"timestamp": 0.5, "frame_index": True, "action": [0.1, True]}, schema)
        self.assertTrue(any("frame_index: expected int" in p for p in bad))
        self.assertTrue(any("action: non-finite or non-numeric" in p for p in bad))

    def test_quantile_mode_requires_quantiles(self):
        st = feature_stats([[0.0], [1.0]])
        del st["q01"]
        with self.assertRaisesRegex(ValueError, "QUANTILES normalization mode requires q01 and q99 stats"):
            normalize([0.5], st, "QUANTILES")
        with self.assertRaises(ValueError):
            unnormalize([0.5], st, "QUANTILES")

    def test_stats_parity_catches_wrong_dataset(self):
        a = feature_stats([[float(i), 2.0 * i] for i in range(20)])
        b = feature_stats([[2.0 * i, float(i)] for i in range(20)])      # dimensions swapped
        self.assertEqual(stats_parity(a, a), [])
        self.assertTrue(stats_parity(a, b))
        # Same mean/std/min/max but different q99: a quantile-normalized policy would see other actions
        c = {k: list(v) if isinstance(v, list) else v for k, v in a.items()}
        c["q99"] = [x * 1.1 for x in a["q99"]]
        self.assertTrue(any(issue.startswith("q99") for issue in stats_parity(a, c)))


class TestBudgetAndEval(unittest.TestCase):
    def test_training_state_arithmetic(self):
        full = training_state_bytes(450e6, 450e6)
        # bf16 weights 0.9 + bf16 grads 0.9 + fp32 master 1.8 + two fp32 moments 3.6 = 7.2 GB
        self.assertAlmostEqual(full["total_gb"], 7.2, places=6)
        frozen = training_state_bytes(450e6, 100e6)
        self.assertLess(frozen["total_gb"], full["total_gb"])
        self.assertFalse(fits(full["total_gb"], 8.0, activation_gb=2.0))
        self.assertTrue(fits(frozen["total_gb"], 8.0, activation_gb=2.0))

    def test_lora_params(self):
        self.assertEqual(lora_params(2048, 2048, 16), 16 * 4096)

    def test_mixture_follows_weights_not_sizes(self):
        counts = mixture_batches({"human": 1_000_000, "robot": 5_000}, {"human": 0.5, "robot": 0.5},
                                 batch_size=64, num_batches=200, seed=0)
        frac = counts["robot"] / (counts["robot"] + counts["human"])
        self.assertAlmostEqual(frac, 0.5, delta=0.03)
        self.assertGreater(epochs_seen(5_000, counts["robot"]), 1.0)   # robot data repeats
        with self.assertRaises(ValueError):
            mixture_batches({"human": 10, "robot": 10}, {"human": 0.0, "robot": 0.0}, 4, 1)

    def test_wilson_and_mcnemar(self):
        lo, hi = wilson_interval(45, 50)
        self.assertTrue(0.78 < lo < 0.81 and 0.95 < hi < 0.97)
        a = [True] * 30 + [False] * 20
        b = [True] * 25 + [False] * 5 + [True] * 2 + [False] * 18
        bc = paired_counts(a, b)
        self.assertEqual(bc, (5, 2))
        self.assertGreater(mcnemar_exact_p(*bc), 0.05)       # 5 vs 2 discordant pairs: not significant
        self.assertLess(mcnemar_exact_p(15, 1), 0.01)
        with self.assertRaises(ValueError):
            paired_counts(a, b[:-1])


if __name__ == "__main__":
    unittest.main()
