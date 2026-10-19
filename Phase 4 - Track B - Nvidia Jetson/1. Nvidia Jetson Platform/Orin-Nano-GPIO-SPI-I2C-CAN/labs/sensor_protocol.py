"""Teaching Classic CAN ABI/payload codec; no physical device access."""
from __future__ import annotations

from dataclasses import dataclass
import json
import socket
import struct

CAN_ID = 0x321
CAN_EFF_FLAG, CAN_RTR_FLAG, CAN_ERR_FLAG = 0x80000000, 0x40000000, 0x20000000
CAN_SFF_MASK = 0x7FF
FRAME = struct.Struct('=IB3x8s')  # Linux native endian can_id, len, padding, data
PAYLOAD = struct.Struct('<BBhHH')  # Explicit application wire byte order


def integer(value: int, low: int, high: int, name: str) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in [{low}, {high}]')
    return value


@dataclass(frozen=True)
class Sample:
    sequence: int
    speed_mm_s: int
    distance_mm: int
    sender_age_ms: int

    def __post_init__(self):
        integer(self.sequence, 0, 255, 'sequence')
        integer(self.speed_mm_s, -32768, 32767, 'speed_mm_s')
        integer(self.distance_mm, 0, 65535, 'distance_mm')
        integer(self.sender_age_ms, 0, 65535, 'sender_age_ms')


def pack(sample: Sample) -> bytes:
    if not isinstance(sample, Sample):
        raise ValueError('expected Sample')
    return FRAME.pack(CAN_ID, 8, PAYLOAD.pack(1, sample.sequence, sample.speed_mm_s,
                                           sample.distance_mm, sample.sender_age_ms))


def unpack(frame: bytes) -> Sample:
    if not isinstance(frame, bytes) or len(frame) != FRAME.size:
        raise ValueError('expected exactly one 16-byte Classic CAN ABI record')
    can_id, length, payload = FRAME.unpack(frame)
    if can_id != CAN_ID:
        raise ValueError('expected standard data frame ID 0x321 without EFF/RTR/ERR flags')
    if length != 8:
        raise ValueError('schema 1 requires exactly eight data bytes')
    version, sequence, speed, distance, age = PAYLOAD.unpack(payload)
    if version != 1:
        raise ValueError('unsupported payload schema')
    return Sample(sequence, speed, distance, age)


def filter_matches(received_id: int) -> bool:
    """Equivalent masked comparison; kernel error-frame subscription is separate."""
    integer(received_id, 0, 0xFFFFFFFF, 'received_id')
    mask = CAN_SFF_MASK | CAN_EFF_FLAG | CAN_RTR_FLAG | CAN_ERR_FLAG
    return (received_id & mask) == CAN_ID


class SequenceTracker:
    """Classify modulo-256 sequence progress within a known session.

    Assumes at most 126 missed frames between accepted observations. A new
    sender session needs an explicit tracker reset; no authentication is implied.
    """
    def __init__(self):
        self.last = None

    def observe(self, sequence: int) -> tuple[str, int]:
        integer(sequence, 0, 255, 'sequence')
        if self.last is None:
            self.last = sequence
            return 'first', 0
        delta = (sequence - self.last) & 255
        if delta == 0:
            return 'duplicate', 0
        if delta >= 128:
            return 'old_or_ambiguous', 0
        self.last = sequence
        return ('next' if delta == 1 else 'gap'), delta - 1


def demo() -> dict:
    sample = Sample(127, -250, 1200, 15)
    frame = pack(sample)
    # Real OS datagram transport, deliberately not labeled a SocketCAN test.
    sender, receiver = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    with sender, receiver:
        receiver.settimeout(1)
        sender.send(frame)
        received = unpack(receiver.recv(64))
    tracker = SequenceTracker()
    progress = [dict(sequence=s, classification=(r := tracker.observe(s))[0],
                     missed=r[1]) for s in (254, 255, 0, 0, 3, 2)]
    return {'transport': 'actual AF_UNIX datagram socketpair; no SocketCAN interface or wire',
            'payload_hex': frame[8:].hex(), 'abi_bytes': len(frame),
            'received': received.__dict__, 'sequence_progress': progress,
            'sender_age_limit': 'sender-declared age excludes unknown transport delay; not capture-age proof'}


if __name__ == '__main__':
    print(json.dumps(demo(), indent=2))
