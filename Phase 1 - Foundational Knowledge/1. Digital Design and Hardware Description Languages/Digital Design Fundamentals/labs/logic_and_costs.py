"""Small CPU references: SECDED, an explicit LFSR, cache mapping and timing.

No synthesis, device timing, DRAM transfer, or physical error injection occurs.
"""
from __future__ import annotations

from fractions import Fraction
import json
import math


def integer(value, minimum, maximum, name):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def encode_secded(nibble: int) -> int:
    """Hamming(8,4): positions 1..8 are bits 0..7; even parity."""
    integer(nibble, 0, 15, "nibble")
    word = 0
    for bit, position in enumerate((3, 5, 6, 7)):
        word |= ((nibble >> bit) & 1) << (position - 1)
    for parity in (1, 2, 4):
        check = sum((word >> (position - 1)) & 1
                    for position in range(1, 8) if position & parity) % 2
        word |= check << (parity - 1)
    return word | ((word.bit_count() % 2) << 7)


def decode_secded(word: int) -> tuple[int | None, str, int | None]:
    """Return data, classification, corrected position (1..8).

    The classification guarantee covers zero, one, or two flipped bits only.
    """
    integer(word, 0, 255, "codeword")
    syndrome = 0
    for position in range(1, 8):
        if word & (1 << (position - 1)):
            syndrome ^= position
    odd = word.bit_count() % 2
    if syndrome and not odd:
        return None, "double_error", None
    corrected = (syndrome or 8) if odd else None
    if corrected:
        word ^= 1 << (corrected - 1)
    data = sum(((word >> (position - 1)) & 1) << bit
               for bit, position in enumerate((3, 5, 6, 7)))
    return data, "corrected" if corrected else "clean", corrected


def lfsr_step(state: int) -> int:
    """q_next = {q[2:0], q[3] XOR q[2]}, including the zero lock-up state."""
    integer(state, 0, 15, "state")
    return ((state << 1) & 15) | (((state >> 3) ^ (state >> 2)) & 1)


def peak_bandwidth(data_bits: int, transfers_per_second: int) -> Fraction:
    integer(data_bits, 1, 1 << 20, "data_bits")
    integer(transfers_per_second, 1, 10**15, "transfers_per_second")
    return Fraction(data_bits * transfers_per_second, 8)


def pipeline(stage_delays_ns: list[Fraction], overhead_ns: Fraction, items: int) -> dict:
    if not stage_delays_ns or any(type(x) is not Fraction or x <= 0 for x in stage_delays_ns):
        raise ValueError("positive Fraction stage delays required")
    if type(overhead_ns) is not Fraction or overhead_ns < 0:
        raise ValueError("nonnegative Fraction overhead required")
    integer(items, 1, 10**9, "items")
    period = max(stage_delays_ns) + overhead_ns
    return {"period_ns": period, "latency_ns": len(stage_delays_ns) * period,
            "batch_ns": (items + len(stage_delays_ns) - 1) * period}


def cache_address(address: int, capacity_bytes=1024, line_bytes=64, ways=1) -> tuple[int, int, int]:
    integer(address, 0, (1 << 64) - 1, "address")
    for value, name in ((capacity_bytes, "capacity"), (line_bytes, "line"), (ways, "ways")):
        integer(value, 1, 1 << 40, name)
    if capacity_bytes % (line_bytes * ways):
        raise ValueError("capacity must be a multiple of line bytes times ways")
    sets = capacity_bytes // (line_bytes * ways)
    block, offset = divmod(address, line_bytes)
    tag, index = divmod(block, sets)
    return tag, index, offset


def lru_order_bits(ways: int) -> int:
    """Information lower bound, not a proposed practical LRU circuit."""
    integer(ways, 1, 64, "ways")
    return (math.factorial(ways) - 1).bit_length()


def demo() -> dict:
    states, state = [], 15
    while state not in states:
        states.append(state)
        state = lfsr_step(state)
    word = encode_secded(0b1010)
    return {"secded_data": "1010", "secded_codeword_bit7_to_bit0": f"{word:08b}",
            "single_error_position_5": decode_secded(word ^ (1 << 4)),
            "double_error_positions_3_5": decode_secded(word ^ (1 << 2) ^ (1 << 4)),
            "lfsr_states": [f"{s:04b}" for s in states],
            "ddr5_6400_64_data_bits_GB_s": str(peak_bandwidth(64, 6_400_000_000) / 10**9),
            "pipeline": {k: str(v) for k, v in
                         pipeline([Fraction(3), Fraction(4), Fraction(2)], Fraction(1, 5), 10).items()},
            "cache_0x140": cache_address(0x140), "cache_0x540": cache_address(0x540),
            "exact_lru_minimum_bits_4ways": lru_order_bits(4),
            "boundary": "CPU logic and exact arithmetic; no hardware timing or memory measurement"}


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2))
