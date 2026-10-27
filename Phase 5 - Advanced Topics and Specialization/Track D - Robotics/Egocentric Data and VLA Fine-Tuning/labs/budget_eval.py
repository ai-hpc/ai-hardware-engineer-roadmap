"""Fine-tuning budgets, co-training mixtures and evaluation statistics (pure Python, CPU).

The memory model is arithmetic for *weights, gradients and optimizer state* only. It ignores
activations, the CUDA context, allocator fragmentation and framework workspace, which often
dominate. Use it to rule configurations out, never to promise that one fits: compare with the
stacks' documented requirements and a measured run.
"""
from __future__ import annotations

import math
import random

GB = 1e9


def training_state_bytes(total_params: float, trainable_params: float, weight_bytes: int = 2,
                         grad_bytes: int = 2, master_bytes: int = 4, optimizer_moments: int = 2,
                         moment_bytes: int = 4) -> dict:
    """Bytes for frozen + trainable weights, gradients, fp32 master copy and AdamW moments.
    Defaults: bf16 weights/grads, fp32 master weights and two fp32 Adam moments for the
    trainable subset. Frozen parameters cost only their weights."""
    weights = total_params * weight_bytes
    grads = trainable_params * grad_bytes
    master = trainable_params * master_bytes
    moments = trainable_params * optimizer_moments * moment_bytes
    total = weights + grads + master + moments
    return {"weights_gb": weights / GB, "grads_gb": grads / GB, "master_gb": master / GB,
            "optimizer_gb": moments / GB, "total_gb": total / GB}


def lora_params(d_in: int, d_out: int, rank: int) -> int:
    """Parameters LoRA adds to one d_in x d_out projection: A (r x d_in) + B (d_out x r)."""
    return rank * (d_in + d_out)


def fits(state_gb: float, device_gb: float, activation_gb: float, reserve_gb: float = 1.0) -> bool:
    """Necessary (not sufficient) condition: training state + activations + a reserve for the
    CUDA context and fragmentation must stay under device memory."""
    return state_gb + activation_gb + reserve_gb <= device_gb


# --------------------------------------------------------------------------- co-training mixtures
def mixture_batches(sources: dict[str, int], weights: dict[str, float], batch_size: int,
                    num_batches: int, seed: int = 0) -> dict[str, int]:
    """Sample which source each training example comes from, with replacement, according to
    `weights` (not dataset size). Returns how many examples each source contributed.
    Weighting is how a few hours of robot data can stay visible next to thousands of hours
    of human video."""
    rng = random.Random(seed)
    names = [n for n in sources if weights.get(n, 0.0) > 0.0 and sources[n] > 0]
    if not names:
        raise ValueError("no source has both a positive weight and at least one example")
    total_w = sum(weights[n] for n in names)
    cdf, acc = [], 0.0
    for n in names:
        acc += weights[n] / total_w
        cdf.append(acc)
    counts = {n: 0 for n in sources}
    for _ in range(batch_size * num_batches):
        r = rng.random()
        for n, c in zip(names, cdf):
            if r <= c:
                counts[n] += 1
                break
        else:
            counts[names[-1]] += 1
    return counts


def epochs_seen(source_size: int, examples_drawn: int) -> float:
    """How many times each example of a source is seen on average (repetition risk)."""
    return examples_drawn / source_size if source_size else float("inf")


# --------------------------------------------------------------------------- evaluation
def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar test for paired episodes (same seeds/layouts for both policies).
    b = A succeeded and B failed, c = A failed and B succeeded. Ties carry no information."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = math.fsum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2.0 * tail)


def paired_counts(results_a: list[bool], results_b: list[bool]) -> tuple[int, int]:
    if len(results_a) != len(results_b):
        raise ValueError("paired evaluation needs the same episodes for both policies")
    b = sum(1 for x, y in zip(results_a, results_b) if x and not y)
    c = sum(1 for x, y in zip(results_a, results_b) if y and not x)
    return b, c
