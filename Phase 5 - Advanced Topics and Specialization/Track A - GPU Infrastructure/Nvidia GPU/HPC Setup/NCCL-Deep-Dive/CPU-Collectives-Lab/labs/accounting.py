"""Collective accounting, using decimal GB/s and explicit per-operation sizes."""
import math


def bandwidth(size_bytes, seconds, ranks, collective="all_reduce"):
    if isinstance(ranks, bool) or not isinstance(ranks, int) or ranks < 1:
        raise ValueError("ranks must be a positive integer")
    if not math.isfinite(size_bytes) or size_bytes <= 0:
        raise ValueError("size_bytes must be positive and finite")
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be positive and finite")
    factors = {"all_reduce": 2 * (ranks - 1) / ranks,
               "all_gather": (ranks - 1) / ranks,
               "reduce_scatter": (ranks - 1) / ranks}
    if collective not in factors:
        raise ValueError("unsupported collective")
    # AllGather: size is the TOTAL output; ReduceScatter: TOTAL input per rank.
    algbw = size_bytes / seconds / 1e9
    return {"algbw_GBps": algbw, "busbw_GBps": algbw * factors[collective]}


def validate_signatures(signatures):
    """Preflight for this lab's same-shape, same-dtype SUM AllReduce only."""
    if not signatures or any(s != signatures[0] for s in signatures[1:]):
        raise ValueError("all_reduce ranks must agree on shape, dtype, and operation")
    return True
