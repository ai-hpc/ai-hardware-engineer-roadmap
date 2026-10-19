"""Independent oracles and bounded failure cases for the teaching models."""

from collections import deque
import math
import unittest

from foundations import (astar, compose_pose, control_rollout, drive_step, fk_2r,
                         ik_2r, inflate_cells, inverse_pose, jacobian_2r, project,
                         scalar_update, transform_point, unproject, validate_age)


class GeometryTests(unittest.TestCase):
    def assertVectorClose(self, actual, expected, places=10):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b, places=places)

    def test_hand_computed_transform(self):
        self.assertVectorClose(transform_point((1, 2, math.pi / 2), (1, 0)), (1, 3))

    def test_inverse_rotates_translation(self):
        for pose in ((1, 2, .4), (-2, .5, -2), (0, 0, math.pi)):
            point = (.3, -.7)
            self.assertVectorClose(transform_point(inverse_pose(pose), transform_point(pose, point)), point)

    def test_composition_and_noncommutativity(self):
        ab, bc, p = (1, 2, math.pi / 2), (.2, 0, 0), (1, 0)
        self.assertVectorClose(transform_point(compose_pose(ab, bc), p), (1, 3.2))
        self.assertNotEqual(transform_point(compose_pose(bc, ab), p), (1, 3.2))

    def test_timestamp_failures(self):
        self.assertAlmostEqual(validate_age(9.96, 10, .05), .04)
        for stamp, now in ((9.9, 10), (10.1, 10), (math.nan, 10)):
            with self.assertRaises(ValueError):
                validate_age(stamp, now, .05)

    def test_forward_known_pose(self):
        self.assertVectorClose(fk_2r(0, math.pi / 2), (1, 1))

    def test_two_ik_branches(self):
        for target in ((1, 1), (.2, 1.5), (-1, -.5), (0, 0), (2, 0)):
            for elbow in (-1, 1):
                self.assertVectorClose(fk_2r(*ik_2r(*target, elbow=elbow)), target)

    def test_ik_rejects_unreachable(self):
        for x, y, l1, l2 in ((3, 0, 1, 1), (0, 0, 2, 1)):
            with self.assertRaises(ValueError):
                ik_2r(x, y, l1, l2)

    def test_jacobian_by_central_difference(self):
        epsilon = 1e-6
        for q1, q2 in ((.3, -.7), (0, 0), (1, 1.5)):
            jacobian = jacobian_2r(q1, q2, .7, .4)
            for joint in range(2):
                plus, minus = [q1, q2], [q1, q2]
                plus[joint] += epsilon
                minus[joint] -= epsilon
                p, m = fk_2r(*plus, .7, .4), fk_2r(*minus, .7, .4)
                numerical = [(p[row] - m[row]) / (2 * epsilon) for row in range(2)]
                self.assertVectorClose([jacobian[row][joint] for row in range(2)], numerical, 8)

    def test_singularity(self):
        j = jacobian_2r(0, 0)
        self.assertVectorClose(j[0], (0, 0))
        self.assertAlmostEqual(j[0][0] * j[1][1] - j[0][1] * j[1][0], 0)

    def test_drive_straight_spin_arc(self):
        self.assertVectorClose(drive_step((0, 0, 0), .2, .2, .4, 1), (.2, 0, 0))
        self.assertVectorClose(drive_step((0, 0, 0), -.2, .2, .4, 1), (0, 0, 1))
        # v=.2, omega=.5, radius=.4. Compare with circle equation.
        self.assertVectorClose(drive_step((0, 0, 0), .1, .3, .4, 1),
                               (.4 * math.sin(.5), .4 * (1 - math.cos(.5)), .5))

    def test_camera_roundtrip_and_axis_order(self):
        self.assertVectorClose(project((.1, .05, 1), 500, 500, 320, 240), (370, 265))
        for point in ((.1, .05, 1), (-.2, .1, 2)):
            pixel = project(point, 500, 480, 320, 240)
            self.assertVectorClose(unproject(pixel, point[2], 500, 480, 320, 240), point)

    def test_invalid_depth_and_nan(self):
        for depth in (0, -1, math.nan):
            with self.assertRaises(ValueError):
                unproject((370, 265), depth, 500, 500, 320, 240)


class ControlAndEstimationTests(unittest.TestCase):
    def test_control_matches_closed_form(self):
        positions = control_rollout(2, .1, steps=10)
        for step, position in enumerate(positions):
            self.assertAlmostEqual(1 - position, .8 ** step)

    def test_delay_can_destroy_stability(self):
        self.assertAlmostEqual(control_rollout(12, .1)[-1], 1)
        self.assertGreater(max(abs(x - 1) for x in control_rollout(12, .1, delay=1)), 100)

    def test_velocity_limit_bounds_each_step(self):
        positions = control_rollout(12, .1, delay=1, velocity_limit=1)
        self.assertTrue(all(abs(b - a) <= .1 + 1e-12 for a, b in zip(positions, positions[1:])))

    def test_scalar_posterior_hand_calculation(self):
        mean, variance, gain = scalar_update(1, .04, 1.3, .01)
        self.assertAlmostEqual(gain, .8)
        self.assertAlmostEqual(mean, 1.24)
        self.assertAlmostEqual(variance, .008)

    def test_noisy_measurement_has_less_weight(self):
        precise = scalar_update(1, .04, 1.3, .01)
        noisy = scalar_update(1, .04, 1.3, 1)
        self.assertLess(noisy[2], precise[2])
        self.assertLess(abs(noisy[0] - 1), abs(precise[0] - 1))

    def test_estimation_rejects_invalid_variance(self):
        for prior, sensor in ((-.1, .1), (.1, 0), (.1, math.nan)):
            with self.assertRaises(ValueError):
                scalar_update(0, prior, 1, sensor)


class PlanningTests(unittest.TestCase):
    @staticmethod
    def bfs_distance(width, height, obstacles, start, goal):
        """Independent shortest-path oracle: no heuristic or priority queue."""
        queue, seen = deque([(start, 0)]), {start}
        while queue:
            (x, y), distance = queue.popleft()
            if (x, y) == goal:
                return distance
            for cell in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if (0 <= cell[0] < width and 0 <= cell[1] < height
                        and cell not in obstacles and cell not in seen):
                    seen.add(cell)
                    queue.append((cell, distance + 1))
        return None

    def test_astar_matches_bfs_with_and_without_detour(self):
        for obstacles in (set(), {(2, y) for y in range(4)}, {(2, y) for y in range(5)}):
            path = astar(5, 5, obstacles, (0, 0), (4, 0))
            expected = self.bfs_distance(5, 5, obstacles, (0, 0), (4, 0))
            if expected is None:
                self.assertIsNone(path)
            else:
                self.assertEqual(len(path) - 1, expected)
                self.assertEqual(path[0], (0, 0))
                self.assertEqual(path[-1], (4, 0))
                self.assertTrue(all(p not in obstacles for p in path))
                self.assertTrue(all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
                                    for a, b in zip(path, path[1:])))

    def test_footprint_closes_narrow_passage(self):
        obstacles = {(3, y) for y in range(7) if y != 3}
        self.assertEqual(len(astar(7, 7, obstacles, (1, 3), (5, 3))) - 1, 4)
        inflated = inflate_cells(obstacles, 7, 7, 1)
        self.assertIsNone(astar(7, 7, inflated, (1, 3), (5, 3)))

    def test_footprint_respects_map_boundary(self):
        inflated = inflate_cells(set(), 5, 5, 1)
        self.assertIn((0, 2), inflated)
        self.assertNotIn((2, 2), inflated)

    def test_start_equals_goal(self):
        self.assertEqual(astar(3, 3, set(), (1, 1), (1, 1)), [(1, 1)])

    def test_invalid_start_is_not_a_no_path_result(self):
        with self.assertRaises(ValueError):
            astar(3, 3, {(0, 0)}, (0, 0), (2, 2))


if __name__ == "__main__":
    unittest.main()
