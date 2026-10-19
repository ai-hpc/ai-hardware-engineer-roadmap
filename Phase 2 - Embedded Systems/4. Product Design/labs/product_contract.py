"""Virtual-time local timer contract. No microphone, model or actuator access."""
from __future__ import annotations
import argparse
from collections import deque
from dataclasses import dataclass
import json
import uuid


def integer(value: int, name: str, *, minimum: int = 0, maximum: int = 2**63-1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f'{name} must be an integer in [{minimum}, {maximum}]')
    return value


@dataclass(frozen=True)
class Token:
    session: str
    sequence: int


@dataclass(frozen=True)
class Pending:
    origin: str
    expires_ms: int


class Appliance:
    def __init__(self, *, timeout_ms: int = 500, session: str | None = None):
        self.timeout_ms = integer(timeout_ms, 'timeout', minimum=1, maximum=60_000)
        self.session = session or str(uuid.uuid4())
        self.sequence = 0
        self.now_ms = 0
        self.services_ready = False
        self.network_connected = False
        self.muted = False
        self.frames: deque[bytes] = deque(maxlen=4)
        self.dropped_frames = 0
        self.pending: dict[Token, Pending] = {}
        self.timer_due_ms: int | None = None
        self.expiry_events = 0
        self.events: list[dict] = []

    def advance(self, now_ms: int) -> None:
        integer(now_ms, 'time')
        if now_ms < self.now_ms:
            raise ValueError('virtual monotonic clock moved backwards')
        self.now_ms = now_ms
        for token, pending in list(self.pending.items()):
            if now_ms >= pending.expires_ms:
                del self.pending[token]
                self._log('request_expired', sequence=token.sequence)
        if self.timer_due_ms is not None and now_ms >= self.timer_due_ms:
            self.timer_due_ms = None
            self.expiry_events += 1
            self._log('timer_expired')

    def _log(self, event: str, **fields) -> None:
        self.events.append({'time_ms': self.now_ms, 'event': event, **fields})

    @property
    def state(self) -> str:
        if not self.services_ready:
            return 'starting'
        if self.timer_due_ms is not None:
            return 'timer_active'
        if self.pending:
            return 'processing'
        return 'ready'

    @property
    def capture_permitted(self) -> bool:
        return self.services_ready and not self.muted

    def ready(self, now_ms: int) -> None:
        self.advance(now_ms)
        self.services_ready = True
        self._log('services_ready')

    def frame(self, data: bytes) -> bool:
        if not isinstance(data, bytes) or len(data) > 1024:
            raise ValueError('fixture frame must be bytes with at most 1024 bytes')
        if not self.capture_permitted:
            return False
        if len(self.frames) == self.frames.maxlen:
            self.dropped_frames += 1
        self.frames.append(data)
        return True

    def set_mute(self, enabled: bool, now_ms: int) -> None:
        if not isinstance(enabled, bool):
            raise ValueError('mute must be boolean')
        self.advance(now_ms)
        self.muted = enabled
        if enabled:
            self.frames.clear()
            self.pending = {token: p for token, p in self.pending.items() if p.origin != 'voice'}
        self._log('mute_changed', enabled=enabled)

    def request(self, origin: str, now_ms: int) -> Token:
        self.advance(now_ms)
        if origin not in ('voice', 'typed'):
            raise ValueError('unsupported input origin')
        if not self.services_ready or (origin == 'voice' and self.muted):
            raise ValueError('input unavailable')
        if self.pending or self.timer_due_ms is not None:
            raise ValueError('one request or timer is already active')
        self.sequence += 1
        token = Token(self.session, self.sequence)
        self.pending[token] = Pending(origin, now_ms + self.timeout_ms)
        self._log('request_started', sequence=token.sequence, origin=origin)
        return token

    def result(self, token: Token, duration_s: int, now_ms: int) -> bool:
        self.advance(now_ms)
        integer(duration_s, 'timer duration', minimum=1, maximum=3600)
        if token.session != self.session or token not in self.pending:
            self._log('result_rejected', reason='not_pending_in_this_session')
            return False
        del self.pending[token]
        self.timer_due_ms = now_ms + duration_s * 1000
        self._log('timer_started', sequence=token.sequence, duration_s=duration_s)
        return True

    def cancel(self, now_ms: int) -> None:
        self.advance(now_ms)
        self.pending.clear()
        self.timer_due_ms = None
        self.frames.clear()
        self._log('cancelled')

    def snapshot(self) -> dict:
        return {'time_ms': self.now_ms, 'state': self.state, 'muted': self.muted,
                'capture_permitted': self.capture_permitted,
                'network_connected': self.network_connected,
                'queued_frames': len(self.frames), 'pending': len(self.pending),
                'timer_due_ms': self.timer_due_ms, 'expiry_events': self.expiry_events}


def demonstration() -> dict:
    device = Appliance(session='fictional-demo-session')
    device.network_connected = True
    connected = device.snapshot()
    device.ready(100)
    token = device.request('voice', 120)
    device.frame(b'fixture audio bytes')
    device.set_mute(True, 150)
    late_accepted = device.result(token, 10, 200)
    device.network_connected = False
    local = device.request('typed', 220)
    device.result(local, 1, 250)
    muted_active = device.snapshot()
    device.advance(1250)
    return {'network_only': connected, 'late_voice_result_accepted': late_accepted,
            'muted_local_timer': muted_active, 'final': device.snapshot(), 'events': device.events}


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(demonstration(), indent=2))
