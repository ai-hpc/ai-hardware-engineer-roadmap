"""Scalar oracle for Linear -> fixed BatchNorm -> ReLU, using Python floats."""

import math
import random


def check_matrix(matrix, name):
    if not matrix or not matrix[0]:
        raise ValueError(f"{name} must be nonempty")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError(f"{name} must be rectangular")
    if not all(math.isfinite(value) for row in matrix for value in row):
        raise ValueError(f"{name} must contain finite values")
    return len(matrix), width


def check_parameters(weights, bias, mean, variance, gamma, beta, epsilon):
    _, outputs = check_matrix(weights, "weights")
    for name, vector in [("bias", bias), ("mean", mean), ("variance", variance),
                         ("gamma", gamma), ("beta", beta)]:
        if len(vector) != outputs or not all(math.isfinite(x) for x in vector):
            raise ValueError(f"{name} must have {outputs} finite entries")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if any(v < 0 for v in variance):
        raise ValueError("variance must be nonnegative")


def linear(inputs, weights, bias):
    _, features = check_matrix(inputs, "inputs")
    rows, outputs = check_matrix(weights, "weights")
    if features != rows or len(bias) != outputs:
        raise ValueError("Linear dimensions do not agree")
    return [[sum(x[k] * weights[k][j] for k in range(features)) + bias[j]
             for j in range(outputs)] for x in inputs]


def reference(inputs, weights, bias, mean, variance, gamma, beta, epsilon=1e-5):
    check_parameters(weights, bias, mean, variance, gamma, beta, epsilon)
    z = linear(inputs, weights, bias)
    return [[max(0.0, gamma[j] * (value - mean[j]) / math.sqrt(variance[j] + epsilon) + beta[j])
             for j, value in enumerate(row)] for row in z]


def fold_batchnorm(weights, bias, mean, variance, gamma, beta, epsilon=1e-5, training=False):
    if training:
        raise ValueError("Fixed-statistics folding is invalid for training-mode BatchNorm")
    check_parameters(weights, bias, mean, variance, gamma, beta, epsilon)
    scales = [g / math.sqrt(v + epsilon) for g, v in zip(gamma, variance)]
    folded_weights = [[w * scales[j] for j, w in enumerate(row)] for row in weights]
    folded_bias = [a * (b - m) + shift for a, b, m, shift in zip(scales, bias, mean, beta)]
    return folded_weights, folded_bias


def folded(inputs, weights, bias):
    return [[max(0.0, value) for value in row] for row in linear(inputs, weights, bias)]


def max_error(actual, expected):
    if check_matrix(actual, "actual") != check_matrix(expected, "expected"):
        raise ValueError("Output shapes differ")
    return max(abs(a - e) for ar, er in zip(actual, expected) for a, e in zip(ar, er))


def fixture(batch=3, features=5, outputs=7, seed=13):
    if min(batch, features, outputs) <= 0:
        raise ValueError("Fixture dimensions must be positive")
    rng = random.Random(seed)
    matrix = lambda r, c: [[rng.uniform(-1, 1) for _ in range(c)] for _ in range(r)]
    return {"inputs": matrix(batch, features), "weights": matrix(features, outputs),
            "bias": [rng.uniform(-0.3, 0.3) for _ in range(outputs)],
            "mean": [rng.uniform(-0.4, 0.4) for _ in range(outputs)],
            "variance": [rng.uniform(0.2, 2) for _ in range(outputs)],
            "gamma": [(-1 if j % 2 else 1) * rng.uniform(0.3, 1.5) for j in range(outputs)],
            "beta": [rng.uniform(-0.3, 0.3) for _ in range(outputs)], "epsilon": 1e-5}


if __name__ == "__main__":
    case = fixture()
    weights, bias = fold_batchnorm(**{k: v for k, v in case.items() if k != "inputs"})
    print({"fixture": "[3,5] @ [5,7]", "max_absolute_error": max_error(
        folded(case["inputs"], weights, bias), reference(**case))})
