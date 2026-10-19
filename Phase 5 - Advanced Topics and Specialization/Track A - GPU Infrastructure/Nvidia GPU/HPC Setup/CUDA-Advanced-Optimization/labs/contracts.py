"""CPU mathematical checks for CUDA lessons; no GPU performance simulation."""
import argparse
import json
import math


def layernorm(values, block_size=None, epsilon=1e-5):
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError("a nonempty finite row is required")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    n = len(values)
    block_size = n if block_size is None else block_size
    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size < n:
        raise ValueError("block must contain the full row")
    padded = list(values) + [0.] * (block_size - n)
    mean = sum(padded) / n
    # Inactive lanes contribute zero AFTER centering, not (0 - mean)^2.
    variance = sum((x - mean) ** 2 for x in padded[:n]) / n
    return [(x - mean) / math.sqrt(variance + epsilon) for x in values]


def faulty_padded_layernorm(values, block_size, epsilon=1e-5):
    """Deliberate historical bug: invalid padded lanes affect the variance."""
    padded = list(values) + [0.] * (block_size - len(values))
    mean = sum(padded) / len(values)
    variance = sum((x - mean) ** 2 for x in padded) / len(values)
    return [(x - mean) / math.sqrt(variance + epsilon) for x in values]


def pointwise(values, residual, keep, probability):
    if not (len(values) == len(residual) == len(keep)):
        raise ValueError("shapes differ")
    if not math.isfinite(probability) or not 0 <= probability < 1:
        raise ValueError("drop probability must be finite in [0,1)")
    if any(not isinstance(k, bool) for k in keep):
        raise ValueError("keep decisions must be boolean")
    if any(not math.isfinite(x) for x in [*values, *residual]):
        raise ValueError("inputs must be finite")
    # Exact same externally supplied decisions isolate fusion from RNG changes.
    def gelu(x):
        return .5 * x * (1 + math.erf(x / math.sqrt(2)))
    intermediate = [gelu(x) for x in values]
    dropped = [y / (1 - probability) if k else 0. for y, k in zip(intermediate, keep)]
    separate = [y + r for y, r in zip(dropped, residual)]
    fused = [(gelu(x) / (1 - probability) if k else 0.) + r
             for x, r, k in zip(values, residual, keep)]
    return separate, fused


def traffic(n, bytes_per_element=4):
    """Array traffic estimate only; excludes masks/RNG, caches and parameters."""
    if any(not isinstance(x, int) or isinstance(x, bool) for x in (n, bytes_per_element)):
        raise ValueError("integer sizes required")
    if n < 0 or bytes_per_element <= 0:
        raise ValueError("invalid size")
    return {"separate_bytes": 7 * n * bytes_per_element,
            "fused_bytes": 3 * n * bytes_per_element}


def claimed_tiles(n, block_size, workers):
    """Serial model of finite atomic tile claims, not a CUDA memory-model test."""
    if any(not isinstance(x, int) or isinstance(x, bool) for x in (n, block_size, workers)):
        raise ValueError("integer sizes required")
    if n < 0 or block_size <= 0 or workers <= 0:
        raise ValueError("invalid schedule size")
    tile_count = (n + block_size - 1) // block_size
    assignments = [[] for _ in range(workers)]
    for tile in range(tile_count):
        assignments[tile % workers].extend(range(tile * block_size, min(n, (tile + 1) * block_size)))
    return assignments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements", type=int, default=1003)
    args = parser.parse_args()
    rows = [1., 2., 4.]
    report = {"status": "CPU mathematical reference; no CUDA execution",
              "layernorm_row": rows, "padded_block": 4,
              "correct": layernorm(rows, 4), "faulty": faulty_padded_layernorm(rows, 4),
              "traffic": traffic(args.elements),
              "tile_worker_element_counts": [len(x) for x in claimed_tiles(args.elements, 256, 3)]}
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
