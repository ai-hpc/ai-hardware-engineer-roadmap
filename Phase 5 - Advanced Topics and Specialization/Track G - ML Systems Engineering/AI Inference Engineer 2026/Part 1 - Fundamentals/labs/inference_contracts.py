"""Small CPU references: accounting, causal GQA, and expert dispatch.

These functions measure no GPU. Bytes exclude allocator, kernel workspace, and
network protocol overhead unless explicitly included by an argument.
"""
from collections import Counter
from fractions import Fraction
import math


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def gqa_bytes(config, lengths, element_bytes=2, block_tokens=1):
    """Allocated KV bytes; each request rounds separately to a token block.

    head_dim can differ from hidden_size / num_attention_heads (Qwen3 MoE).
    No sharing or KV sharding is assumed.
    """
    layers = integer(config["num_hidden_layers"], "layers", 1)
    heads = integer(config["num_attention_heads"], "query heads", 1)
    kv_heads = integer(config["num_key_value_heads"], "KV heads", 1)
    if heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    dim = config.get("head_dim")
    if dim is None:
        hidden = integer(config["hidden_size"], "hidden size", 1)
        if hidden % heads:
            raise ValueError("head_dim missing and hidden size not divisible by heads")
        dim = hidden // heads
    dim = integer(dim, "head dimension", 1)
    element_bytes = integer(element_bytes, "element bytes", 1)
    block_tokens = integer(block_tokens, "block tokens", 1)
    allocated_tokens = sum((integer(n, "length") + block_tokens - 1)
                           // block_tokens * block_tokens for n in lengths)
    return 2 * layers * kv_heads * dim * element_bytes * allocated_tokens


def mla_bytes(config, lengths, element_bytes=2):
    """Compressed latent + shared RoPE key layout; no expanded K/V stored."""
    fields = [integer(config[k], k, 1) for k in
              ("num_hidden_layers", "kv_lora_rank", "qk_rope_head_dim")]
    layers, latent, rope = fields
    return layers * (latent + rope) * integer(element_bytes, "element bytes", 1) * sum(
        integer(n, "length") for n in lengths)


def grouped_weight_bytes(rows, cols, bits=4, group_size=128, scale_bytes=2):
    """Row-local groups, packed row payload, one scale/group, no zero points.

    Final groups need scales even if partially full. A real format may pad each
    group, add zero points, use mixed precision, or align tensor allocations.
    """
    for v, n in ((rows, "rows"), (cols, "cols"), (bits, "bits"),
                 (group_size, "group size"), (scale_bytes, "scale bytes")):
        integer(v, n, 1)
    return rows * ((cols * bits + 7) // 8
                   + (cols + group_size - 1) // group_size * scale_bytes)


def quantize_int4_group(values, scale=None):
    """Toy symmetric [-7,7] group with ties-to-even, no binary packing.

    A supplied scale demonstrates calibration mismatch/clipping on new data.
    All-zero groups choose scale 1 so dequantization stays well defined.
    """
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError('nonempty finite values required')
    if scale is None:
        maximum = max(abs(x) for x in values)
        scale = maximum / 7 if maximum else 1.0
    if isinstance(scale, bool) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('finite positive scale required')
    # Clamp before round so a finite input / tiny scale overflowing to inf
    # still saturates rather than passing infinity to round().
    codes = [round(min(7.0, max(-7.0, x / scale))) for x in values]
    return {'scale': scale, 'codes': codes, 'reconstructed': [q*scale for q in codes]}


def roofline_seconds(flops, transferred_bytes, flops_per_second, bytes_per_second):
    """Optimistic lower bound at one stated memory boundary, not a latency fit."""
    for v, n in ((flops, "FLOPs"), (transferred_bytes, "bytes")):
        integer(v, n)
    for v, n in ((flops_per_second, "compute rate"), (bytes_per_second, "byte rate")):
        integer(v, n, 1)
    return max(Fraction(flops, flops_per_second),
               Fraction(transferred_bytes, bytes_per_second))


def causal_gqa(q, k, v, query_start=0):
    """Reference on lists [token][head][dimension], one sequence, no dropout.

    Query i has absolute position query_start+i. K/V start at position zero.
    Group h uses KV head h // (query_heads // kv_heads). This intentionally
    materializes scores; it is not FlashAttention or a serving implementation.
    """
    integer(query_start, "query start")
    if not q or not k or not v or len(k) != len(v):
        raise ValueError("nonempty Q and equally long nonempty K/V required")
    hq, hk, d = len(q[0]), len(k[0]), len(q[0][0]) if q[0] else 0
    if not hq or not hk or not d or hq % hk or query_start + len(q) > len(k):
        raise ValueError("invalid heads/dimensions or query positions")
    for tensor, heads in ((q, hq), (k, hk), (v, hk)):
        if any(len(row) != heads or any(len(vec) != d or
               any(not math.isfinite(x) for x in vec) for vec in row) for row in tensor):
            raise ValueError("ragged or nonfinite tensor")
    out = []
    for i, query in enumerate(q):
        row = []
        end = query_start + i + 1
        for head, vec in enumerate(query):
            kh = head // (hq // hk)
            scores = [sum(a*b for a, b in zip(vec, k[j][kh])) / math.sqrt(d)
                      for j in range(end)]
            maximum = max(scores)
            weights = [math.exp(s - maximum) for s in scores]
            total = sum(weights)
            row.append([sum(weights[j] * v[j][kh][c] for j in range(end)) / total
                        for c in range(d)])
        out.append(row)
    return out


def dispatch(routes, origins, expert_owner, ranks, hidden, element_bytes=2):
    """Count one-way activation payload: per assignment versus per destination.

    Destination deduplication sends a token once to a rank owning several selected
    experts. This needs local fan-out plus routing metadata, excluded here.
    Returned matrix counts remote token vectors, not bytes or collective time.
    """
    for n, label in ((ranks, "ranks"), (hidden, "hidden"), (element_bytes, "element bytes")):
        integer(n, label, 1)
    if len(routes) != len(origins) or not expert_owner:
        raise ValueError("one origin per token and nonempty expert map required")
    if any(integer(r, "owner") >= ranks for r in expert_owner):
        raise ValueError("owner outside communicator")
    matrix = [[0] * ranks for _ in range(ranks)]
    expert_load = Counter()
    remote_assignments = 0
    for selected, origin in zip(routes, origins):
        if integer(origin, "origin") >= ranks or not selected or len(set(selected)) != len(selected):
            raise ValueError("invalid origin, empty selection, or duplicate expert")
        destinations = set()
        for expert in selected:
            if integer(expert, "expert") >= len(expert_owner):
                raise ValueError("unknown expert")
            expert_load[expert] += 1
            owner = expert_owner[expert]
            if owner != origin:
                remote_assignments += 1
                destinations.add(owner)
        for owner in destinations:
            matrix[origin][owner] += 1
    rank_load = [sum(n for e, n in expert_load.items() if expert_owner[e] == r)
                 for r in range(ranks)]
    vector_bytes = hidden * element_bytes
    return {"assignment_bytes": remote_assignments * vector_bytes,
            "destination_bytes": sum(map(sum, matrix)) * vector_bytes,
            "matrix": matrix, "expert_load": dict(expert_load),
            "rank_load": rank_load, "active_experts": len(expert_load)}


def expert_reference(tokens, routes, weights, experts):
    """Weighted output of scalar toy experts, preserving token order."""
    if not (len(tokens) == len(routes) == len(weights)):
        raise ValueError("token, route, and weight lengths differ")
    result = []
    for token, chosen, scales in zip(tokens, routes, weights):
        if not chosen or len(chosen) != len(scales) or len(set(chosen)) != len(chosen):
            raise ValueError("invalid top-k selection")
        if any(not math.isfinite(s) or s < 0 for s in scales) or not math.isclose(sum(scales), 1):
            raise ValueError("nonnegative normalized weights required")
        if any(integer(e, "expert") >= len(experts) for e in chosen):
            raise ValueError("unknown expert")
        result.append(sum(scale * experts[e](token) for e, scale in zip(chosen, scales)))
    return result


def speculative_one_step_distribution(target, draft):
    """Enumerate the exact one-proposal rejection-sampling distribution.

    This proves a small probability identity; it is not a multi-token draft
    engine. Zero-probability draft tokens cannot be proposed, so no p/q divide
    is needed. Both vectors must already reflect the intended sampling policy.
    """
    if not target or len(target) != len(draft):
        raise ValueError('nonempty equally sized distributions required')
    p, q = [Fraction(x) for x in target], [Fraction(x) for x in draft]
    if any(x < 0 for x in p+q) or sum(p) != 1 or sum(q) != 1:
        raise ValueError('nonnegative normalized distributions required')
    accepted = [min(a, b) for a, b in zip(p, q)]
    residual = [max(a-b, 0) for a, b in zip(p, q)]
    rejected = 1-sum(accepted)
    if not rejected:
        return accepted
    residual_total = sum(residual)
    return [a + rejected*r/residual_total for a, r in zip(accepted, residual)]
