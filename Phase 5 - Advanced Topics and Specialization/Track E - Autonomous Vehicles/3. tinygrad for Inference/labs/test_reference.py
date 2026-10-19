import math
import unittest
from reference import matmul, softmax, causal_attention, rmsnorm, swiglu, finite_gradient


class OperatorReference(unittest.TestCase):
    def test_nonsquare_matmul(self):
        self.assertEqual(matmul([[1,2,3],[-1,0,2]], [[1,0],[2,1],[-1,3]]), [[2,11],[-3,6]])

    def test_large_softmax_is_finite(self):
        p = softmax([1000,1000])
        self.assertEqual(p, [.5,.5])

    def test_causal_first_row_cannot_read_future(self):
        q = [[1,0],[0,1]];k = [[1,0],[0,1]];v = [[2,3],[100,200]]
        a = causal_attention(q,k,v)
        self.assertEqual(a[0], v[0])
        p = 1/(1+math.exp(-1/math.sqrt(2)))
        self.assertAlmostEqual(a[1][0], (1-p)*2+p*100)

    def test_rmsnorm_does_not_center(self):
        self.assertEqual(rmsnorm([3,3],7), [.75,.75])

    def test_swiglu_zero_and_negative_gate(self):
        self.assertEqual(swiglu([5],[0]), [0])
        self.assertAlmostEqual(swiglu([2],[-1])[0], -2/(1+math.e))

    def test_finite_difference_matches_hand_derivative(self):
        result = finite_gradient(lambda x: sum(y*y for y in x), [2,-3])
        for a,b in zip(result,[4,-6]): self.assertAlmostEqual(a,b,places=6)

    def test_invalid_shapes_and_values(self):
        for a,b in [([],[[1]]), ([[1,2]],[[1]]), ([[math.nan]],[[1]])]:
            with self.assertRaises(ValueError): matmul(a,b)
        with self.assertRaises(ValueError): softmax([math.inf])
        with self.assertRaises(ValueError): rmsnorm([1],0)
        with self.assertRaises(ValueError): swiglu([1],[])
        with self.assertRaises(ValueError): causal_attention([[1],[2]],[[1],[2,3]],[[1],[2]])
        with self.assertRaises(ValueError): causal_attention([[1]],[[1]],[[math.nan]])


if __name__ == '__main__': unittest.main()
