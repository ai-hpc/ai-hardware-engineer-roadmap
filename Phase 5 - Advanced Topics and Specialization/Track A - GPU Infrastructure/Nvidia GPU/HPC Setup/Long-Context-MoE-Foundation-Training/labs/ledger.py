"""Explicit teaching ledgers, not a GPU memory/performance predictor."""
import json
import math
from fractions import Fraction


def positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def layer_costs(tokens, hidden, heads, kv_heads, head_dim, ffn_dim, dtype_bytes=2, causal=True):
    for name, value in locals().copy().items():
        if name != "causal":
            positive_int(name, value)
    if not isinstance(causal, bool) or hidden != heads * head_dim or heads % kv_heads:
        raise ValueError("require boolean causal, hidden=heads*head_dim, and integral GQA groups")
    pairs = tokens * (tokens + 1) // 2 if causal else tokens * tokens
    return {
        "tokens": tokens,
        "attention_pair_flops": 4 * pairs * hidden,
        "qkv_and_output_flops": 4 * tokens * hidden * (hidden + kv_heads * head_dim),
        "nongated_mlp_flops": 4 * tokens * hidden * ffn_dim,
        "q_k_v_o_tensor_bytes": (2 * tokens * hidden + 2 * tokens * kv_heads * head_dim) * dtype_bytes,
        "one_mlp_intermediate_bytes": tokens * ffn_dim * dtype_bytes,
        "one_dense_score_matrix_bytes": heads * tokens * tokens * dtype_bytes,
        "inference_kv_bytes": 2 * tokens * kv_heads * head_dim * dtype_bytes,
    }


def capacity(tokens, top_k, experts, factor):
    for name, value in (("tokens", tokens), ("top_k", top_k), ("experts", experts)):
        positive_int(name, value)
    if (top_k > experts or isinstance(factor, bool)
            or not isinstance(factor, (int, float))
            or (isinstance(factor, float) and not math.isfinite(factor)) or factor <= 0):
        raise ValueError("invalid top-k or capacity factor")
    # Use the supplied number's decimal representation exactly. This avoids
    # float count rounding/underflow and makes factor=.1 with 10 tokens yield 1.
    amount = Fraction(str(factor)) * tokens * top_k / experts
    return (amount.numerator + amount.denominator - 1) // amount.denominator



def routing_summary(assignments, experts, cap):
    """Keep first assignments per expert; this policy is deliberately explicit."""
    positive_int("experts", experts)
    positive_int("cap", cap)
    if not assignments or not assignments[0]:
        raise ValueError("need nonempty token assignments")
    k = len(assignments[0])
    counts, kept = [0] * experts, []
    for row in assignments:
        if len(row) != k or len(set(row)) != k or any(type(e) is not int or not 0 <= e < experts for e in row):
            raise ValueError("invalid or duplicate expert assignment")
        kept_row = []
        for expert in row:
            kept_row.append(counts[expert] < cap)
            counts[expert] += 1
        kept.append(kept_row)
    total = len(assignments) * k
    dropped = sum(not item for row in kept for item in row)
    return {"assignment_counts": counts, "assignment_shares": [n / total for n in counts],
            "dropped_assignment_fraction": dropped / total,
            "tokens_with_any_drop_fraction": sum(not all(row) for row in kept) / len(kept),
            "tokens_with_no_expert_fraction": sum(not any(row) for row in kept) / len(kept)}


def dense_mesh(world, tp=1, pp=1, cp=1, ep=1, experts=1, expert_tp=None):
    """Megatron dense/expert grid cardinalities; GTP/EGTP=1, PP-last order.

    Cardinalities alone do not construct rank placement or prove compatibility.
    """
    expert_tp = tp if expert_tp is None else expert_tp
    for name, value in locals().copy().items():
        positive_int(name, value)
    dense_group = tp * pp * cp
    expert_group = expert_tp * pp * ep
    if world % dense_group or world % expert_group or experts % ep:
        raise ValueError("invalid dense grid, expert grid, or expert division")
    return {"world": world, "dense_dp": world // dense_group,
            "expert_dp": world // expert_group, "expert_tp": expert_tp,
            "tp": tp, "pp": pp, "cp": cp, "ep": ep, "experts_per_ep_rank": experts // ep}


def main():
    rows = [layer_costs(n, 4096, 32, 8, 128, 16384) for n in (4096, 32768, 131072, 1000000)]
    report = {"scope": "one layer; forward FLOPs; listed tensor bytes are not peak allocation",
              "rows": rows, "mesh": dense_mesh(64, tp=2, pp=2, cp=2, ep=4, experts=8),
              "skew": routing_summary([[0, 1]] * 4, 4, capacity(4, 2, 4, 1.0))}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
