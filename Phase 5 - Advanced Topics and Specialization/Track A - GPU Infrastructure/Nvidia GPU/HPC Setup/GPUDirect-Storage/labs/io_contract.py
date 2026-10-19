"""CPU file-I/O contracts for the GDS lessons; this does not execute GPU I/O.

Run with Python 3.12: python3 io_contract.py
The temporary file is removed automatically. No storage benchmark is reported.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tempfile
from typing import BinaryIO


def nonnegative_int(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def chunks(total_bytes: int, chunk_bytes: int) -> list[tuple[int, int]]:
    """Return (file offset, logical length); never round a read beyond EOF."""
    nonnegative_int(total_bytes, "total_bytes")
    nonnegative_int(chunk_bytes, "chunk_bytes")
    if chunk_bytes == 0:
        raise ValueError("chunk_bytes must be positive")
    return [(offset, min(chunk_bytes, total_bytes - offset))
            for offset in range(0, total_bytes, chunk_bytes)]


def aligned_cover(offset: int, length: int, alignment: int) -> dict[str, int]:
    """Model an aligned cover; caller still needs real file/buffer bounds.

    Alignment is an explicit exercise input, not a universal GDS requirement.
    The covered span can extend beyond EOF: that does NOT authorize reading it.
    """
    for value, name in ((offset, "offset"), (length, "length"), (alignment, "alignment")):
        nonnegative_int(value, name)
    if alignment == 0 or length == 0:
        raise ValueError("alignment and length must be positive")
    start = offset // alignment * alignment
    end = ((offset + length + alignment - 1) // alignment) * alignment
    return {"start": start, "length": end - start, "prefix": offset - start,
            "logical_length": length}


def read_exact(stream: BinaryIO, offset: int, length: int) -> bytes:
    """POSIX-like teaching contract: handle short chunks; reject unexpected EOF."""
    nonnegative_int(offset, "offset")
    nonnegative_int(length, "length")
    stream.seek(offset)
    result = bytearray()
    while len(result) < length:
        part = stream.read(length - len(result))
        if not part:
            raise EOFError(f"expected {length} bytes; received {len(result)}")
        result.extend(part)
    return bytes(result)


class BufferState:
    """One-buffer ownership model. Completion calls stand in for real events."""
    def __init__(self) -> None:
        self.state = "free"

    def transition(self, expected: str, new: str) -> None:
        if self.state != expected:
            raise RuntimeError(f"expected {expected}; buffer is {self.state}")
        self.state = new

    def submit_read(self) -> None:
        self.transition("free", "reading")

    def finish_read(self) -> None:
        self.transition("reading", "ready")

    def submit_compute(self) -> None:
        self.transition("ready", "computing")

    def finish_compute(self) -> None:
        self.transition("computing", "free")


def experiment() -> dict:
    payload = bytes((i * 17 + i // 4096) % 256 for i in range(12345))
    plan = chunks(len(payload), 4096)
    with tempfile.TemporaryDirectory(prefix="course-io-") as directory:
        path = Path(directory) / "fixture.bin"
        path.write_bytes(payload)
        with path.open("rb") as stream:
            reconstructed = b"".join(read_exact(stream, offset, length) for offset, length in plan)
    if reconstructed != payload:
        raise AssertionError("chunked reads changed the payload")
    try:
        read_exact(io.BytesIO(payload[:-1]), 0, len(payload))
    except EOFError:
        truncation_rejected = True
    else:
        raise AssertionError("truncated input was accepted")
    buffer = BufferState()
    buffer.submit_read()
    buffer.finish_read()
    buffer.submit_compute()
    try:
        buffer.submit_read()
    except RuntimeError:
        early_reuse_rejected = True
    else:
        raise AssertionError("buffer was reused before compute completed")
    buffer.finish_compute()
    return {"execution_status": "CPU file I/O and synthetic ownership model; no GDS",
            "logical_bytes": len(payload), "chunks": plan,
            "sha256": hashlib.sha256(reconstructed).hexdigest(),
            "aligned_cover_example": aligned_cover(100, 5000, 4096),
            "truncation_rejected": truncation_rejected,
            "early_reuse_rejected": early_reuse_rejected,
            "final_buffer_state": buffer.state}


if __name__ == "__main__":
    print(json.dumps(experiment(), indent=2))
