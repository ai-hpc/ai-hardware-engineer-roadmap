"""Portable arithmetic used before allocating memory or choosing quantization."""
import math


def affine_quantize(value, scale, zero_point=0, qmin=-127, qmax=127):
    if not math.isfinite(value) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("finite value and positive scale required")
    if not qmin <= zero_point <= qmax:
        raise ValueError("zero point outside integer range")
    q = min(qmax, max(qmin, round(value / scale) + zero_point))
    return q, scale * (q - zero_point)


def kv_bytes(layers, kv_heads, head_dim, tokens, sequences=1, bytes_per_element=2):
    dims = (layers, kv_heads, head_dim, tokens, sequences, bytes_per_element)
    if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in dims):
        raise ValueError("positive integer dimensions required")
    return 2 * math.prod(dims)


def chunk_deadline_seconds(remaining_actions, control_hz, guard_seconds):
    if remaining_actions < 0 or control_hz <= 0 or guard_seconds < 0:
        raise ValueError("invalid queue, frequency or guard")
    if not all(math.isfinite(x) for x in (remaining_actions, control_hz, guard_seconds)):
        raise ValueError("finite timing values required")
    return max(0, remaining_actions / control_hz - guard_seconds)
