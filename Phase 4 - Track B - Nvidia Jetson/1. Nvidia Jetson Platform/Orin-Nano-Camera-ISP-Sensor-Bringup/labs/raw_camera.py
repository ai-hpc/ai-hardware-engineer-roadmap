"""CPU contracts for explicitly packed RAW10 and a simple sensor timing model."""
from __future__ import annotations

import json
import math


def pack_raw10(values: list[int]) -> bytes:
    """V4L2 *10P packing: four high bytes, then four 2-bit low fields."""
    if len(values) % 4 or any(type(v) is not int or not 0 <= v <= 1023 for v in values):
        raise ValueError("need a multiple of four integer 10-bit samples")
    packed = bytearray()
    for start in range(0, len(values), 4):
        group = values[start:start+4]
        packed.extend(v >> 2 for v in group)
        packed.append(sum((v & 3) << (2 * i) for i, v in enumerate(group)))
    return bytes(packed)


def unpack_raw10(data: bytes, width: int, height: int, stride: int) -> list[list[int]]:
    if any(type(v) is not int for v in (width, height, stride)):
        raise ValueError("integer geometry required")
    if width <= 0 or height <= 0 or width % 4:
        raise ValueError("this packed-row contract needs positive width divisible by four")
    payload = width // 4 * 5
    if stride < payload or len(data) != stride * height:
        raise ValueError("wrong stride or buffer length")
    rows = []
    for row in range(height):
        values = []
        for offset in range(row * stride, row * stride + payload, 5):
            group = data[offset:offset+5]
            values.extend((group[i] << 2) | ((group[4] >> (2 * i)) & 3) for i in range(4))
        rows.append(values)
    return rows


def unpack_expanded10(data: bytes, width: int, height: int, stride: int) -> list[list[int]]:
    """V4L2 RG10-style little-endian 16-bit container with zero upper six bits."""
    if any(type(v) is not int for v in (width, height, stride)) or width <= 0 or height <= 0:
        raise ValueError("positive integer geometry required")
    if stride < width * 2 or len(data) != height * stride:
        raise ValueError("wrong stride or buffer length")
    rows = [[int.from_bytes(data[row*stride+2*x:row*stride+2*x+2], "little")
             for x in range(width)] for row in range(height)]
    if any(value > 1023 for row in rows for value in row):
        raise ValueError("nonzero upper bits violate this expanded10 format")
    return rows


def bayer_color(pattern: str, x: int, y: int) -> str:
    if pattern not in ("RGGB", "BGGR", "GRBG", "GBRG"):
        raise ValueError("unknown Bayer pattern")
    if type(x) is not int or type(y) is not int or x < 0 or y < 0:
        raise ValueError("nonnegative integer coordinates required")
    return pattern[(y % 2) * 2 + x % 2]


def cropped_pattern(pattern: str, left: int, top: int) -> str:
    if any(type(value) is not int or value < 0 for value in (left, top)):
        raise ValueError("nonnegative integer crop coordinates required")
    return "".join(bayer_color(pattern, left+x, top+y) for y in range(2) for x in range(2))


def timing(pixel_clock_hz: int = 74_250_000, line_length: int = 2200,
           frame_length: int = 1125, requested_exposure_s: float = .010,
           margin_lines: int = 4) -> dict:
    for value in (pixel_clock_hz, line_length, frame_length):
        if type(value) is not int or value <= 0: raise ValueError("positive integer timing required")
    if type(margin_lines) is not int or not 0 <= margin_lines < frame_length:
        raise ValueError("invalid line margin")
    if (type(requested_exposure_s) not in (int, float) or
            not math.isfinite(requested_exposure_s) or requested_exposure_s <= 0):
        raise ValueError("positive finite exposure required")
    line_s = line_length / pixel_clock_hz
    # Explicit half-up quantization, then clamp to this model's legal range.
    requested_lines = requested_exposure_s / line_s
    maximum_lines = frame_length - margin_lines
    # Clamp before integer conversion: a finite exposure can overflow the
    # floating-point quotient, but still has an unambiguous saturated result.
    if requested_lines >= maximum_lines + .5:
        lines, clamped = maximum_lines, True
    else:
        raw_lines = math.floor(requested_lines + .5)
        lines = max(1, min(raw_lines, maximum_lines))
        clamped = raw_lines != lines
    return {"line_time_us": line_s * 1e6,
            "frame_rate_hz": pixel_clock_hz / (line_length * frame_length),
            "requested_exposure_s": requested_exposure_s,
            "exposure_lines": lines, "actual_exposure_s": lines * line_s,
            "clamped": clamped}


if __name__ == "__main__":
    values = [0, 1, 512, 1023]
    packed = pack_raw10(values)
    print(json.dumps({"samples": values, "packed_hex": packed.hex(" "),
                      "decoded": unpack_raw10(packed, 4, 1, 5)[0],
                      "rggb_crop_one_column": cropped_pattern("RGGB", 1, 0),
                      "timing_model": timing(),
                      "boundary": "CPU format/timing model; no sensor or ISP execution"}, indent=2))
