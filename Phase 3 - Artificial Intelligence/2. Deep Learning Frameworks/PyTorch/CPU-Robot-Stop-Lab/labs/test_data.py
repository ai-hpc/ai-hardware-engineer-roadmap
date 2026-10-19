import unittest

from data import Sample, dataset_sha256, fit_scaler, make_dataset, sensor_shift, split_episodes, stopping_distance, validate_splits


class DataTests(unittest.TestCase):
    def test_known_physics_and_boundary(self):
        self.assertAlmostEqual(stopping_distance(2, 2, .1), 1.4)
        with self.assertRaises(ValueError):
            stopping_distance(2, 0, .1)

    def test_episode_split_is_disjoint_and_complete(self):
        rows = make_dataset()
        parts = split_episodes(rows)
        self.assertEqual([len({r.episode for r in p}) for p in parts], [96, 24, 24])
        self.assertEqual(sum(map(len, parts)), len(rows))
        parts[1].append(parts[0][0])
        with self.assertRaises(ValueError):
            validate_splits(parts)

    def test_fixed_dataset_and_scaler(self):
        self.assertEqual(dataset_sha256(make_dataset()), dataset_sha256(make_dataset()))
        self.assertNotEqual(dataset_sha256(make_dataset()), dataset_sha256(make_dataset(18)))
        rows = [Sample(0, 1, 2, 3, .1, 1), Sample(1, 3, 2, 3, .1, 0)]
        mean, std = fit_scaler(rows)
        self.assertEqual(mean[:3], [2, 2, 3])
        self.assertEqual(std[:3], [1, 1, 1])

    def test_shift_changes_observation_not_ground_truth(self):
        rows = make_dataset()
        shifted = sensor_shift(rows, .8)
        for before, after in zip(rows, shifted):
            self.assertAlmostEqual(after.distance_m - before.distance_m, .8)
            self.assertEqual(before.episode, after.episode)
            self.assertEqual(before.stop_required, after.stop_required)


if __name__ == "__main__":
    unittest.main()
