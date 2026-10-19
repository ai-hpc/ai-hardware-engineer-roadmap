import unittest

from reference import fixture, fold_batchnorm, folded, max_error, reference


class RewriteTests(unittest.TestCase):
    def test_scalar_worked_example(self):
        # sqrt(variance + epsilon) = 2, so W'=4 and b'=1.
        w, b = fold_batchnorm([[2.0]], [1.0], [3.0], [3.0], [4.0], [5.0], epsilon=1.0)
        self.assertEqual(w, [[4.0]])
        self.assertEqual(b, [1.0])
        self.assertEqual(folded([[2.0]], w, b), [[9.0]])

    def test_irregular_shapes_and_negative_gamma(self):
        for shape in [(1, 3, 4), (3, 5, 7), (8, 16, 9)]:
            with self.subTest(shape=shape):
                case = fixture(*shape)
                w, b = fold_batchnorm(**{k: v for k, v in case.items() if k != "inputs"})
                self.assertLess(max_error(folded(case["inputs"], w, b), reference(**case)), 1e-12)

    def test_training_mode_rejected(self):
        case = fixture()
        with self.assertRaisesRegex(ValueError, "training-mode"):
            fold_batchnorm(**{k: v for k, v in case.items() if k != "inputs"}, training=True)

    def test_activation_order_counterexample(self):
        # z=-1, gamma=-1, mean=beta=0, sqrt(v+eps)=1.
        correct = reference([[-1.0]], [[1.0]], [0.0], [0.0], [0.5], [-1.0], [0.0], epsilon=0.5)
        illegal = -max(0.0, -1.0)
        self.assertEqual(correct, [[1.0]])
        self.assertNotEqual(correct[0][0], illegal)

    def test_bad_variance_and_epsilon_rejected(self):
        case = fixture()
        for name, value in [("variance", [-1.0] * 7), ("epsilon", 0.0), ("epsilon", float("nan"))]:
            with self.subTest(name=name, value=value):
                bad = dict(case, **{name: value})
                with self.assertRaises(ValueError):
                    reference(**bad)

    def test_invalid_shapes_and_nonfinite_input_rejected(self):
        for inputs in [[], [[1.0]], [[1.0, 2.0], [1.0]], [[float("inf")] * 5]]:
            with self.subTest(inputs=inputs):
                with self.assertRaises(ValueError):
                    reference(**dict(fixture(), inputs=inputs))

    def test_changed_running_mean_requires_new_fold(self):
        case = fixture()
        w, b = fold_batchnorm(**{k: v for k, v in case.items() if k != "inputs"})
        changed = dict(case, mean=[m + 2.0 for m in case["mean"]])
        self.assertGreater(max_error(folded(case["inputs"], w, b), reference(**changed)), 0.1)


if __name__ == "__main__":
    unittest.main()
