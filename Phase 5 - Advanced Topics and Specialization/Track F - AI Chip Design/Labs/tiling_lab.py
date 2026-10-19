#!/usr/bin/env python3
"""Exact integer matmul, scratchpad traffic, and an ideal systolic token model.

These are functional and analytical models. No hardware timing is measured.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json


def integer(name: str, value: int, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


@dataclass(frozen=True)
class Matrix:
    rows: int
    cols: int
    data: tuple[int, ...]

    def __post_init__(self) -> None:
        integer("rows", self.rows)
        integer("cols", self.cols)
        if len(self.data) != self.rows * self.cols:
            raise ValueError("data length does not match matrix shape")
        if any(isinstance(x, bool) or not isinstance(x, int) for x in self.data):
            raise ValueError("this exact arithmetic lab uses integer values")

    def at(self, row: int, col: int) -> int:
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise IndexError("matrix coordinate out of range")
        return self.data[row * self.cols + col]


def fixture(rows: int, cols: int, salt: int = 0) -> Matrix:
    integer("rows", rows)
    integer("cols", cols)
    return Matrix(rows, cols, tuple(((i * 7 + salt) % 13) - 6
                                   for i in range(rows * cols)))


def compatible(a: Matrix, b: Matrix) -> None:
    if a.cols != b.rows:
        raise ValueError("matmul reduction dimensions must match")


def reference(a: Matrix, b: Matrix) -> Matrix:
    """Independent dot-product oracle; arbitrary-size Python integer arithmetic."""
    compatible(a, b)
    return Matrix(a.rows, b.cols, tuple(
        sum(x * y for x, y in zip(
            a.data[i * a.cols:(i + 1) * a.cols],
            b.data[j::b.cols] if b.cols else ()))
        for i in range(a.rows) for j in range(b.cols)))


@dataclass(frozen=True)
class Tile:
    row: int
    col: int
    reduction: int
    rows: int
    cols: int
    depth: int
    input_read_bytes: int
    resident_bytes: int


@dataclass(frozen=True)
class Traffic:
    a_read_bytes: int
    b_read_bytes: int
    c_write_bytes: int
    peak_resident_bytes: int
    useful_macs: int
    output_tiles: int
    reduction_tiles: int

    @property
    def total_bytes(self) -> int:
        return self.a_read_bytes + self.b_read_bytes + self.c_write_bytes


def tiled(a: Matrix, b: Matrix, tile_m: int, tile_n: int, tile_k: int,
          input_bytes: int = 2, accumulator_bytes: int = 4,
          capacity_bytes: int | None = None) -> tuple[Matrix, Traffic, list[Tile]]:
    """Output-stationary C=AB, beta=0; each A/B tile loaded once per C tile.

    No reuse across output tiles, no double buffer, no C spill between K tiles.
    Logical bytes exclude cache lines, bursts, alignment, instructions, and DMA.
    """
    compatible(a, b)
    for name, value in (("tile_m", tile_m), ("tile_n", tile_n),
                        ("tile_k", tile_k), ("input_bytes", input_bytes),
                        ("accumulator_bytes", accumulator_bytes)):
        integer(name, value, 1)
    if capacity_bytes is not None:
        integer("capacity_bytes", capacity_bytes)
    m, n, k = a.rows, b.cols, a.cols
    result = [0] * (m * n)
    trace: list[Tile] = []
    a_bytes = b_bytes = peak = output_tiles = 0
    for i0 in range(0, m, tile_m):
        mi = min(tile_m, m - i0)
        for j0 in range(0, n, tile_n):
            nj = min(tile_n, n - j0)
            output_tiles += 1
            # Local C is zero-initialized and retained across every K tile.
            local_c = [[0] * nj for _ in range(mi)]
            peak = max(peak, mi * nj * accumulator_bytes)
            for k0 in range(0, k, tile_k):
                kk = min(tile_k, k - k0)
                local_a = [[a.at(i0 + i, k0 + t) for t in range(kk)]
                           for i in range(mi)]
                local_b = [[b.at(k0 + t, j0 + j) for j in range(nj)]
                           for t in range(kk)]
                read_a = mi * kk * input_bytes
                read_b = kk * nj * input_bytes
                resident = read_a + read_b + mi * nj * accumulator_bytes
                peak = max(peak, resident)
                a_bytes += read_a
                b_bytes += read_b
                trace.append(Tile(i0, j0, k0, mi, nj, kk,
                                  read_a + read_b, resident))
                for t in range(kk):
                    for i in range(mi):
                        for j in range(nj):
                            local_c[i][j] += local_a[i][t] * local_b[t][j]
            for i in range(mi):
                for j in range(nj):
                    result[(i0 + i) * n + j0 + j] = local_c[i][j]
    if capacity_bytes is not None and peak > capacity_bytes:
        raise ValueError(f"tile needs {peak} bytes, capacity is {capacity_bytes}")
    traffic = Traffic(a_bytes, b_bytes, m * n * accumulator_bytes, peak,
                      m * n * k, output_tiles, len(trace))
    return Matrix(m, n, tuple(result)), traffic, trace


@dataclass(frozen=True)
class SystolicResult:
    output: Matrix
    cycles: int
    issued_macs: int
    useful_macs: int
    active_macs_per_cycle: tuple[int, ...]


def systolic(a: Matrix, b: Matrix, array_rows: int,
             array_cols: int) -> SystolicResult:
    """One padded tile, one MAC/PE/cycle, registered neighbor links, no stalls.

    A enters row i at cycle k+i; B enters column j at cycle k+j. Both meet
    at PE(i,j) on cycle k+i+j. Cycle zero is the first compute cycle.
    """
    compatible(a, b)
    integer("array_rows", array_rows, 1)
    integer("array_cols", array_cols, 1)
    m, n, k = a.rows, b.cols, a.cols
    if m > array_rows or n > array_cols:
        raise ValueError("one output tile must fit in the physical array")
    if not m or not n or not k:
        return SystolicResult(Matrix(m, n, (0,) * (m * n)), 0, 0, 0, ())
    accum = [[0] * array_cols for _ in range(array_rows)]
    prev_a = [[None] * array_cols for _ in range(array_rows)]
    prev_b = [[None] * array_cols for _ in range(array_rows)]
    activity = []
    for cycle in range(k + array_rows + array_cols - 2):
        next_a = [[None] * array_cols for _ in range(array_rows)]
        next_b = [[None] * array_cols for _ in range(array_rows)]
        active = 0
        for i in range(array_rows):
            for j in range(array_cols):
                ka, kb = cycle - i, cycle - j
                av = ((ka, a.at(i, ka) if i < m else 0) if 0 <= ka < k
                      else None) if j == 0 else prev_a[i][j - 1]
                bv = ((kb, b.at(kb, j) if j < n else 0) if 0 <= kb < k
                      else None) if i == 0 else prev_b[i - 1][j]
                next_a[i][j], next_b[i][j] = av, bv
                if (av is None) != (bv is None):
                    raise AssertionError("unmatched operand token")
                if av is not None and bv is not None:
                    if av[0] != bv[0]:
                        raise AssertionError("reduction indices are misaligned")
                    accum[i][j] += av[1] * bv[1]
                    active += 1
        prev_a, prev_b = next_a, next_b
        activity.append(active)
    out = Matrix(m, n, tuple(accum[i][j] for i in range(m) for j in range(n)))
    return SystolicResult(out, len(activity), sum(activity), m * n * k,
                          tuple(activity))


def ideal_cycle_estimates(traffic: Traffic, trace: list[Tile], array_rows: int,
                          array_cols: int, bytes_per_cycle: int) -> dict[str, int]:
    """Compare ideal compute/transfer totals; no measured time or frequency."""
    integer("array_rows", array_rows, 1)
    integer("array_cols", array_cols, 1)
    integer("bytes_per_cycle", bytes_per_cycle, 1)
    if any(t.rows > array_rows or t.cols > array_cols for t in trace):
        raise ValueError("tile does not fit array")
    compute = sum(t.depth + array_rows + array_cols - 2 for t in trace)
    transfer = ceil_div(traffic.total_bytes, bytes_per_cycle)
    return {"compute_cycles": compute, "transfer_cycles": transfer,
            "perfect_overlap_lower_bound": max(compute, transfer),
            "serialized_ideal_estimate": compute + transfer}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", nargs=3, type=int, default=[5, 7, 3],
                        metavar=("M", "N", "K"))
    parser.add_argument("--tile", nargs=3, type=int, default=[2, 4, 2],
                        metavar=("TM", "TN", "TK"))
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--bytes-per-cycle", type=int, default=16)
    args = parser.parse_args()
    try:
        m, n, k = args.shape
        if any(x < 0 or x > 128 for x in args.shape):
            raise ValueError("CLI dimensions must be in 0..128")
        a, b = fixture(m, k, 1), fixture(k, n, 3)
        out, traffic, trace = tiled(a, b, *args.tile, capacity_bytes=args.capacity)
        if out != reference(a, b):
            raise AssertionError("tiled result differs from independent oracle")
        report = asdict(traffic)
        report["total_bytes"] = traffic.total_bytes
        report.update(ideal_cycle_estimates(traffic, trace, args.tile[0],
                                           args.tile[1], args.bytes_per_cycle))
        report["shape_mnk"] = args.shape
        report["tile_mnk"] = args.tile
        report["checksum"] = sum(out.data)
        report["validated"] = True
        report["model"] = "analytical logical bytes and ideal cycles; no hardware measurement"
        print(json.dumps(report, indent=2))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
