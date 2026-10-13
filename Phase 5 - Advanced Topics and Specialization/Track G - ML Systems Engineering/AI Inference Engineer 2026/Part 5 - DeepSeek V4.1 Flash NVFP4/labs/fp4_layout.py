#!/usr/bin/env python3
"""E2M1 packing exercise. Not a production NVFP4 quantizer or scale-layout converter."""
import math

MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def decode(code):
    if type(code) is not int or not 0 <= code <= 15:
        raise ValueError("E2M1 code must be an integer nibble")
    return (-1.0 if code & 8 else 1.0) * MAGNITUDES[code & 7]


def pack(codes):
    codes = list(codes)
    if len(codes) % 2:
        raise ValueError("fixture requires an even number of codes")
    for code in codes:
        decode(code)
    # This exercise explicitly defines low nibble first; verify real loader order.
    return bytes(codes[i] | (codes[i + 1] << 4) for i in range(0, len(codes), 2))


def unpack(payload):
    return [code for byte in payload for code in (byte & 15, byte >> 4)]


def reconstruct(codes, scales, block_size, tensor_scale=1.0):
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if len(scales) != math.ceil(len(codes) / block_size):
        raise ValueError("one explicit scale required for each block")
    if any(not math.isfinite(s) or s <= 0 for s in [tensor_scale, *scales]):
        raise ValueError("fixture scales must be finite and positive")
    return [decode(c) * scales[i // block_size] * tensor_scale for i, c in enumerate(codes)]


def main():
    codes = [1, 10, 7, 0, 15, 2, 4, 9]
    payload = pack(codes)
    assert unpack(payload) == codes
    values = reconstruct(codes, [2.0, 0.5], 4)
    assert values == [1.0, -2.0, 12.0, 0.0, -3.0, 0.5, 1.0, -0.25]
    print("Synthetic packing fixture: PASS")
    print("Packed bytes:", payload.hex())
    print("Reconstructed values:", values)
    print("Asymptotic bits/value: block-16 =", 4 + 8/16, "; block-32 =", 4 + 8/32)
    print("Real NVFP4 also requires scale encoding, layout and activation contracts.")


if __name__ == "__main__":
    main()
