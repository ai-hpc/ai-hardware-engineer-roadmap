"""Portable experiments for RoPE, quantization, parallelism and serving math."""
import json
import math
import random


def matvec(weight, vector):
    return [sum(a * b for a, b in zip(row, vector, strict=True)) for row in weight]


def rope(vector, position, base=1_000_000):
    """Split-half RoPE; educational scalar reference, no scaling extension."""
    if len(vector) == 0 or len(vector) % 2 or base <= 0:
        raise ValueError("positive even head dimension and positive base required")
    half = len(vector) // 2
    out = [0.] * len(vector)
    for i in range(half):
        angle = position * base ** (-2 * i / len(vector))
        c, s = math.cos(angle), math.sin(angle)
        out[i] = vector[i] * c - vector[i + half] * s
        out[i + half] = vector[i + half] * c + vector[i] * s
    return out


def quantize_rows(weight):
    """Symmetric [-7,7], one scale per row. Not a GGUF/AWQ/GPTQ encoder."""
    quantized, scales = [], []
    for row in weight:
        if not row or any(not math.isfinite(x) for x in row):
            raise ValueError("finite nonempty rows required")
        scale = max(map(abs, row)) / 7 or 1.0
        quantized.append([max(-7, min(7, round(value / scale))) for value in row])
        scales.append(scale)
    return quantized, scales


def dequantize_rows(quantized, scales):
    return [[value * scale for value in row] for row, scale in zip(quantized, scales, strict=True)]


def rescale_equivalent(weight, vector, scales):
    """Column-vector convention: W x = (W diag(s)) (diag(s)^-1 x)."""
    if any(not math.isfinite(s) or s <= 0 for s in scales):
        raise ValueError("finite positive channel scales required")
    return matvec([[w * s for w, s in zip(row, scales, strict=True)] for row in weight],
                  [x / s for x, s in zip(vector, scales, strict=True)])


def row_parallel(weight, vector, ranks):
    """Split input/reduction dimension, then SUM partial outputs (AllReduce idea)."""
    if ranks <= 0 or len(vector) % ranks:
        raise ValueError("input width must divide ranks")
    width = len(vector) // ranks
    partial = [matvec([row[r * width:(r + 1) * width] for row in weight],
                      vector[r * width:(r + 1) * width]) for r in range(ranks)]
    return [sum(values) for values in zip(*partial)], partial


def speculative_expected_tokens(acceptance, draft_length):
    """IID constant conditional acceptance; includes corrected/bonus token."""
    if not 0 <= acceptance <= 1 or type(draft_length) is not int or draft_length < 0:
        raise ValueError("acceptance in [0,1] and nonnegative integer length required")
    return sum(acceptance ** i for i in range(draft_length + 1))


def speculative_monte_carlo(acceptance, length, rounds=50000, seed=42):
    speculative_expected_tokens(acceptance, length)
    rng, total = random.Random(seed), 0
    for _ in range(rounds):
        accepted = 0
        for _ in range(length):
            if rng.random() >= acceptance:
                break
            accepted += 1
        total += accepted + 1
    return total / rounds


def speculative_rate(acceptance, length, draft_ms, verify_ms, overhead_ms=0):
    if draft_ms < 0 or verify_ms <= 0 or overhead_ms < 0:
        raise ValueError("invalid execution times")
    return 1000 * speculative_expected_tokens(acceptance, length) / (length * draft_ms + verify_ms + overhead_ms)


def required_replicas(peak_output_tokens_per_second, measured_replica_rate, utilization_limit=.7):
    if peak_output_tokens_per_second < 0 or measured_replica_rate <= 0 or not 0 < utilization_limit <= 1:
        raise ValueError("invalid capacity inputs")
    return math.ceil(peak_output_tokens_per_second / (measured_replica_rate * utilization_limit))


def column_major_gemm(a, b, c, m, n, k, lda, ldb, ldc, alpha=1., beta=0.):
    """NN column-major reference. Reads C only when beta != 0, as BLAS permits."""
    if min(m, n, k) <= 0 or lda < m or ldb < k or ldc < m:
        raise ValueError("invalid dimensions/leading dimensions")
    if len(a) < lda * k or len(b) < ldb * n or len(c) < ldc * n:
        raise ValueError("insufficient storage including column padding")
    result = list(c)
    for col in range(n):
        for row in range(m):
            product = sum(a[row + reduction * lda] * b[reduction + col * ldb]
                          for reduction in range(k))
            result[row + col * ldc] = alpha * product + (beta * c[row + col * ldc] if beta else 0)
    return result


def report():
    weight, vector = [[0., -1., 2., .1], [.02, -.03, .04, 0.]], [1., -2., .5, 3.]
    q, scales = quantize_rows(weight)
    reference, candidate = matvec(weight, vector), matvec(dequantize_rows(q, scales), vector)
    peak = 1000 * 50 * .30 * 400 / 86400 * 5
    return {"scope": "scalar numerical experiments; no model or GPU benchmark",
            "quantized_ints": q, "row_scales": scales,
            "reference_output": reference, "quantized_output": candidate,
            "maximum_output_error": max(abs(a - b) for a, b in zip(reference, candidate)),
            "rope_norm_before": sum(x * x for x in vector),
            "rope_norm_after": sum(x * x for x in rope(vector, 12345, 5_000_000)),
            "tp_reference": matvec(weight, vector), "tp_sum_and_partials": row_parallel(weight, vector, 2),
            "speculation_expected_tokens_alpha_055_k5": speculative_expected_tokens(.55, 5),
            "speculation_simulated_tokens_alpha_055_k5": speculative_monte_carlo(.55, 5),
            "illustrative_speculation_tokens_per_second": speculative_rate(.55, 5, 5, 32.5),
            "peak_output_tokens_per_second": peak,
            "replicas_if_measured_400_tps_at_70percent_limit": required_replicas(peak, 400)}


if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
