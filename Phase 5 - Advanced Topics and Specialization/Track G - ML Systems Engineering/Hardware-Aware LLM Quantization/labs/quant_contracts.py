"""Portable exact arithmetic contracts; no ML dependency or hardware claims."""
import itertools
import math


def kv_payload_bytes(batch, length, full_layers, kv_heads, head_dim, bytes_per_element):
    values = (batch, length, full_layers, kv_heads, head_dim, bytes_per_element)
    if any(type(v) is not int or v < 0 for v in values) or any(v == 0 for v in (batch, kv_heads, head_dim, bytes_per_element)):
        raise ValueError("nonnegative integer dimensions, positive batch/heads/dtype required")
    return 2 * math.prod(values)


def fp4_payload_bytes(rows, cols, block=16):
    if any(type(v) is not int or v <= 0 for v in (rows, cols, block)) or block % 2:
        raise ValueError("positive dimensions and even block required")
    blocks = rows * ((cols + block - 1) // block)
    return blocks * (block // 2 + 1) + 4  # nibbles, byte scales, one float32 scale


def validate_distribution(p):
    if not p or any(not math.isfinite(x) or x < 0 for x in p) or not math.isclose(sum(p), 1, abs_tol=1e-12, rel_tol=0):
        raise ValueError("finite normalized probability vector required")


def speculative_mass(p, q):
    """Enumerate accepted and corrected mass, including q=0; no stochastic error."""
    validate_distribution(p); validate_distribution(q)
    if len(p) != len(q):
        raise ValueError("same vocabulary required")
    accepted = [min(a, b) for a, b in zip(p, q)]
    rejection = max(0., 1 - sum(accepted))
    residual = [max(a-b, 0.) for a,b in zip(p,q)]
    total = sum(residual)
    correction = [r/total for r in residual] if total else [0.]*len(p)
    output = [a + rejection*r for a,r in zip(accepted,correction)]
    return {'acceptance':sum(accepted), 'accepted_mass':accepted, 'rejection':rejection,
            'correction':correction, 'output':output}


def expected_emitted(prefix_survival):
    last = 1.
    for probability in prefix_survival:
        if not math.isfinite(probability) or not 0 <= probability <= last:
            raise ValueError("prefix survival probabilities must be nonincreasing in [0,1]")
        last = probability
    return 1 + sum(prefix_survival)


def exhaustive_savings(items, budget):
    """Each item is (saved_bytes, surrogate_cost); retain deterministic tie rule."""
    if not math.isfinite(budget) or budget < 0 or any(not math.isfinite(v) or v < 0 for item in items for v in item):
        raise ValueError("finite nonnegative budget and items required")
    feasible = []
    for mask in itertools.product((False, True), repeat=len(items)):
        saved = sum(item[0] for item,chosen in zip(items,mask) if chosen)
        cost = sum(item[1] for item,chosen in zip(items,mask) if chosen)
        if cost <= budget:
            feasible.append((saved, -cost, mask))
    saved, minus_cost, mask = max(feasible)
    return {'saved':saved, 'cost':-minus_cost, 'selected':mask}
