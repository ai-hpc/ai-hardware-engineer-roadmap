"""Small, inspectable scalar diffusion/flow oracles; Python standard library only.

The action distribution is synthetic: left/right select N(-1,.1²)/N(1,.1²),
and either selects their equal mixture. Values are normalized coordinates,
not physical motor commands. An oracle knows this distribution, not a future
sample or the hidden component that generated a noisy observation.
"""
from __future__ import annotations

import math
import random

SIGMA = 0.1
LABELS = ("left", "right", "either")


def means(label: str) -> tuple[float, ...]:
    if label not in LABELS:
        raise ValueError("label must be left, right, or either")
    return {"left": (-1.0,), "right": (1.0,), "either": (-1.0, 1.0)}[label]


def clean_sample(label: str, rng: random.Random) -> float:
    return rng.choice(means(label)) + rng.gauss(0.0, SIGMA)


def target_cdf(x: float, label: str) -> float:
    return sum(0.5 * (1 + math.erf((x - m) / (SIGMA * math.sqrt(2))))
               for m in means(label)) / len(means(label))


def posterior_means(x: float, scale: float, variance: float,
                    label: str) -> tuple[tuple[float, ...], list[float]]:
    components = means(label)
    logs = [-(x - scale * m) ** 2 / (2 * variance) for m in components]
    peak = max(logs)
    weights = [math.exp(logp - peak) for logp in logs]
    total = sum(weights)
    return components, [w / total for w in weights]


def diffusion_epsilon(x: float, alpha_bar: float, label: str) -> float:
    """E[epsilon | sqrt(alpha_bar)*A + sqrt(1-alpha_bar)*epsilon=x]."""
    if not math.isfinite(x) or not 0 < alpha_bar < 1:
        raise ValueError("finite x and 0 < alpha_bar < 1 required")
    signal, noise = math.sqrt(alpha_bar), math.sqrt(1 - alpha_bar)
    variance = alpha_bar * SIGMA**2 + 1 - alpha_bar
    components, weights = posterior_means(x, signal, variance, label)
    return sum(w * noise * (x - signal * m) / variance
               for m, w in zip(components, weights))


def flow_velocity(x: float, t: float, label: str) -> float:
    """E[A-Z | (1-t)*Z+t*A=x], with t=0 noise and t=1 data."""
    if not math.isfinite(x) or not math.isfinite(t) or not 0 <= t <= 1:
        raise ValueError("finite x and 0 <= t <= 1 required")
    variance = (1 - t) ** 2 + t**2 * SIGMA**2
    covariance = t * SIGMA**2 - (1 - t)
    components, weights = posterior_means(x, t, variance, label)
    return sum(w * (m + covariance * (x - t * m) / variance)
               for m, w in zip(components, weights))


def alpha_schedule(steps: int = 64) -> list[float]:
    """Clean index 0, then cosine cumulative signal power; terminal=1e-4.

    The terminal distribution is close to, but not exactly, N(0,1).
    Sampling starts from N(0,1); this approximation is part of the experiment.
    Reject schedules whose fixed floor duplicates a noise level. Increasing
    this fixture beyond 157 levels needs a different schedule construction.
    """
    if type(steps) is not int or steps < 2:
        raise ValueError("at least two integer noise levels required")
    schedule = [1.0] + [max(1e-4, math.cos(i / steps * math.pi / 2)**2)
                        for i in range(1, steps + 1)]
    if any(left <= right for left, right in zip(schedule, schedule[1:])):
        raise ValueError("fixed terminal floor duplicates levels; use at most 157 steps")
    return schedule


def ddim_indices(total: int, evaluations: int) -> list[int]:
    if type(total) is not int or type(evaluations) is not int or not 1 <= evaluations <= total:
        raise ValueError("require 1 <= evaluations <= total, both integers")
    return [round(total * (1 - i / evaluations)) for i in range(evaluations)] + [0]


def ddim_step(x: float, epsilon: float, alpha_now: float, alpha_next: float) -> float:
    """DDIM eta=0 jump to a LESS noisy level, including the clean endpoint."""
    if not all(math.isfinite(v) for v in (x, epsilon, alpha_now, alpha_next)):
        raise ValueError("finite DDIM values required")
    if not 0 < alpha_now < alpha_next <= 1:
        raise ValueError("DDIM must move toward larger cumulative signal power")
    clean = (x - math.sqrt(1 - alpha_now) * epsilon) / math.sqrt(alpha_now)
    return math.sqrt(alpha_next) * clean + math.sqrt(1 - alpha_next) * epsilon


def oracle_sample(label: str, noise: float, evaluations: int = 32,
                  method: str = "flow", reverse_sign: bool = False) -> float:
    means(label)
    if type(evaluations) is not int or evaluations < 1 or not math.isfinite(noise):
        raise ValueError("positive integer evaluations and finite noise required")
    x = noise
    if method == "flow":
        for i in range(evaluations):
            x += (-1 if reverse_sign else 1) * flow_velocity(x, i / evaluations, label) / evaluations
    elif method == "ddim":
        schedule = alpha_schedule()
        indices = ddim_indices(len(schedule) - 1, evaluations)
        for current, following in zip(indices, indices[1:]):
            eps = diffusion_epsilon(x, schedule[current], label)
            x = ddim_step(x, eps, schedule[current], schedule[following])
    else:
        raise ValueError("method must be flow or ddim")
    return x


def distribution_metrics(values: list[float], label: str) -> dict:
    """CDF distance uses both sides of each empirical jump (one-sample KS D).

    This reports a descriptive distance; no iid significance test is claimed
    when the values were constructed using paired noise or a fixed noise grid.
    """
    means(label)
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("nonempty finite sample required")
    ordered = sorted(values)
    count = len(ordered)
    distance = max(max(abs(target_cdf(x, label) - i / count),
                       abs((i + 1) / count - target_cdf(x, label)))
                   for i, x in enumerate(ordered))
    return {"count": count, "mean": sum(values) / count,
            "positive_fraction": sum(x > 0 for x in values) / count,
            "between_modes_fraction": sum(abs(x) < 0.5 for x in values) / count,
            "outside_1_5_fraction": sum(abs(x) > 1.5 for x in values) / count,
            "cdf_distance": distance}
