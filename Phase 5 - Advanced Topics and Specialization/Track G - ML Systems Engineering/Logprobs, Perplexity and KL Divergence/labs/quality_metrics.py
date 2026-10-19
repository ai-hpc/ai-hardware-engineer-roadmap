"""Small full-vocabulary oracle; no model downloads or optional dependencies."""
import json
import math


def log_softmax(logits):
    if not logits or not all(math.isfinite(x) for x in logits):
        raise ValueError("logits must be a nonempty finite vector")
    peak = max(logits)
    normalizer = math.log(math.fsum(math.exp(x - peak) for x in logits))
    return [x - peak - normalizer for x in logits]


def masked_nll(logit_rows, targets):
    """Rows already predict corresponding targets; None excludes that position."""
    if len(logit_rows) != len(targets):
        raise ValueError("one target or None is required per prediction row")
    losses = []
    for row, target in zip(logit_rows, targets):
        if target is None:
            continue
        if not isinstance(target, int) or isinstance(target, bool) or not 0 <= target < len(row):
            raise ValueError("target is outside vocabulary")
        losses.append(-log_softmax(row)[target])
    return math.fsum(losses), len(losses)


def aggregate(parts):
    parts = list(parts)
    for total, count in parts:
        if not math.isfinite(total) or total < 0 or not isinstance(count, int) or count < 0:
            raise ValueError("expected nonnegative finite NLL sum and integer count")
        if count == 0 and total != 0:
            raise ValueError("empty batch cannot have a nonzero NLL sum")
    count = sum(n for _, n in parts)
    if count == 0:
        raise ValueError("no scored tokens")
    total = math.fsum(value for value, _ in parts)
    try:
        perplexity = math.exp(total / count)
    except OverflowError:
        perplexity = math.inf
    return {"total_nll_nats": total, "scored_tokens": count, "mean_nll_nats": total / count,
            "perplexity": perplexity}


def distribution(values):
    if not values or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("probabilities must be finite and nonnegative")
    if not math.isclose(math.fsum(values), 1.0, abs_tol=1e-12, rel_tol=1e-12):
        raise ValueError("use the full normalized distribution, not unnormalized top-k")


def forward_kl(p, q):
    if len(p) != len(q):
        raise ValueError("matching vocabularies required")
    distribution(p)
    distribution(q)
    if any(a > 0 and b == 0 for a, b in zip(p, q)):
        return math.inf
    return math.fsum(a * (math.log(a) - math.log(b)) for a, b in zip(p, q) if a > 0)


def sliding_targets(length, window, stride):
    """Return absolute target indices, after causal shift, in each window.

    The first token is unscored (no BOS). Overlap supplies context. This models
    one unpadded stream; document boundaries require an explicit policy.
    """
    if length < 2 or window < 2 or not 1 <= stride < window:
        raise ValueError("need length/window >=2 and 1 <= stride < window")
    previous_end = 0
    result = []
    for begin in range(0, length, stride):
        end = min(begin + window, length)
        result.append(list(range(max(begin + 1, previous_end), end)))
        previous_end = end
        if end == length:
            break
    return result


def demo():
    p, q = [0.8, 0.2], [0.9, 0.1]
    return {"unequal_batches": aggregate([(2.0, 2), (4.0, 1)]),
            "sliding_targets_10_6_3": sliding_targets(10, 6, 3),
            "teacher_kl": forward_kl(p, q),
            "data_nll_delta_target_zero": -math.log(q[0]) + math.log(p[0]),
            "confident_wrong_ppl": aggregate([(-math.log(0.001), 1)])["perplexity"]}


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2))
