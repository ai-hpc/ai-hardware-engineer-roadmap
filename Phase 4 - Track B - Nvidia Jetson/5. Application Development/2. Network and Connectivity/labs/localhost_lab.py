"""Real loopback sockets; synthetic work and faults, never a robot command server."""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import platform
import socket
import struct
import threading
import time

MAX_FRAME = 4096


def encode_frame(message):
    payload = json.dumps(message, separators=(',', ':'), allow_nan=False).encode('utf8')
    if not 0 < len(payload) <= MAX_FRAME:
        raise ValueError('frame must have 1..4096 payload bytes')
    return struct.pack('!I', len(payload)) + payload


def read_exact(sock, count, deadline):
    result = bytearray()
    while len(result) < count:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('whole-frame deadline expired')
        sock.settimeout(remaining)
        # Small cap deliberately exercises multiple reads, even on loopback.
        part = sock.recv(min(7, count - len(result)))
        if not part:
            raise EOFError('connection ended before frame completed')
        result.extend(part)
    return bytes(result)


def receive_frame(sock, deadline):
    length = struct.unpack('!I', read_exact(sock, 4, deadline))[0]
    if not 0 < length <= MAX_FRAME:
        raise ValueError('invalid or oversized frame length')
    return json.loads(read_exact(sock, length, deadline).decode('utf8'))


def percentile(values, p):
    """Nearest-rank percentile, stated explicitly for the report."""
    if not values or not 0 < p <= 100:
        raise ValueError('nonempty values and 0 < percentile <= 100 required')
    return sorted(values)[math.ceil(p / 100 * len(values)) - 1]


@contextmanager
def tcp_peer(worker):
    """One real TCP connection on loopback; propagate server errors and clean up."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(('127.0.0.1', 0))
    listener.listen(1)
    listener.settimeout(3)
    errors = []
    def serve():
        try:
            with listener.accept()[0] as connection:
                connection.settimeout(3)
                worker(connection)
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        with socket.create_connection(listener.getsockname(), timeout=3) as client:
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            yield client
    finally:
        thread.join(timeout=4)
        listener.close()
        if thread.is_alive():
            raise RuntimeError('server thread did not finish')
        if errors:
            raise errors[0]


def tcp_experiment(samples=30, slow_ms=15):
    if not 1 <= samples <= 1000 or not 0 <= slow_ms <= 100:
        raise ValueError('samples 1..1000 and slow_ms 0..100 required')
    timings = []
    delays = [slow_ms if i % 5 == 4 else 0 for i in range(samples)]
    def server(connection):
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        for delay in delays:
            message = receive_frame(connection, time.monotonic() + 3)
            time.sleep(delay / 1000)
            response = encode_frame({'sequence': message['sequence'], 'work_ms': delay})
            # Neither two sends nor this split promises matching TCP packets/reads.
            connection.sendall(response[:2])
            connection.sendall(response[2:])
    with tcp_peer(server) as client:
        for i in range(samples):
            start = time.monotonic()
            client.sendall(encode_frame({'sequence': i, 'payload': 'robot telemetry'}))
            response = receive_frame(client, start + 3)
            if response['sequence'] != i:
                raise AssertionError('response associated with the wrong request')
            timings.append((time.monotonic() - start) * 1000)
    return {'samples': samples, 'rtt_ms': timings,
            'p50_ms': percentile(timings, 50), 'p95_ms': percentile(timings, 95),
            'max_ms': max(timings), 'injected_server_work_ms': delays,
            'timing_meaning': 'client send to complete response, includes server work; not radio latency'}


class CommandGate:
    """Bounded, single-session simulated counter with duplicate and expiry checks.

    Monotonic timestamps are valid ONLY because both endpoints share one host.
    No authentication, persistence or reconnect protocol is implemented.
    """
    def __init__(self, session='fixture-session', limit=32):
        self.session, self.limit = session, limit
        self.highest = -1
        self.value = 0
        self.cache = {}

    def accept(self, message, now_ns=None):
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        if not isinstance(message, dict) or set(message) != {'session', 'sequence', 'delta', 'deadline_ns'}:
            return {'status': 'invalid'}
        session, sequence, delta, deadline = (message[k] for k in ('session', 'sequence', 'delta', 'deadline_ns'))
        if session != self.session:
            return {'status': 'wrong_session'}
        if any(type(x) is not int for x in (sequence, delta, deadline)) or sequence < 0 or abs(delta) > 100:
            return {'status': 'invalid'}
        identity = (delta, deadline)
        if sequence in self.cache:
            old_identity, result = self.cache[sequence]
            if identity != old_identity:
                return {'status': 'id_conflict', 'sequence': sequence}
            return {**result, 'duplicate': True}
        if sequence <= self.highest:
            return {'status': 'out_of_order', 'sequence': sequence}
        if deadline <= now_ns:
            return {'status': 'expired', 'sequence': sequence}
        if deadline > now_ns + 5_000_000_000:
            return {'status': 'invalid_deadline', 'sequence': sequence}
        if len(self.cache) >= self.limit:
            return {'status': 'session_full', 'sequence': sequence}
        self.value += delta
        self.highest = sequence
        result = {'status': 'applied', 'sequence': sequence, 'value_after_application': self.value,
                  'duplicate': False}
        self.cache[sequence] = identity, result
        return result


def udp_experiment():
    gate = CommandGate()
    stop = threading.Event()
    received = Counter()
    errors = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server:
        server.bind(('127.0.0.1', 0))
        server.settimeout(.02)
        def serve():
            try:
                while not stop.is_set():
                    try:
                        payload, sender = server.recvfrom(MAX_FRAME + 1)
                    except socket.timeout:
                        continue
                    if len(payload) > MAX_FRAME:
                        continue
                    message = json.loads(payload)
                    sequence = message['sequence']
                    received[sequence] += 1
                    if sequence == 0 and received[sequence] == 1:
                        continue  # controlled loss before application
                    reply = gate.accept(message)
                    if sequence == 1 and received[sequence] == 1:
                        continue  # controlled ACK loss AFTER application
                    server.sendto(json.dumps(reply).encode(), sender)
            except BaseException as error:
                errors.append(error)
        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        rows = []
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
                client.connect(server.getsockname())
                client.settimeout(.03)
                for sequence, delta, expired in [(0, 2, False), (1, 3, False), (3, 4, False),
                                                  (2, 100, False), (4, 100, True), (5, 1, False)]:
                    message = {'session': gate.session, 'sequence': sequence, 'delta': delta,
                               'deadline_ns': time.monotonic_ns() + (-1 if expired else 2_000_000_000)}
                    packet = json.dumps(message).encode()
                    start = time.monotonic()
                    for attempt in range(1, 4):
                        client.send(packet)  # the SAME message ID and deadline on retry
                        try:
                            reply = json.loads(client.recv(MAX_FRAME))
                        except socket.timeout:
                            continue
                        if reply.get('sequence') != sequence:
                            continue
                        rows.append({'sequence': sequence, 'attempts': attempt, 'reply': reply,
                                     'completion_ms': (time.monotonic() - start) * 1000})
                        break
                    else:
                        raise TimeoutError('three bounded attempts exhausted')
        finally:
            stop.set()
            thread.join(timeout=2)
        if thread.is_alive():
            raise RuntimeError('UDP server did not stop')
        if errors:
            raise errors[0]
    return {'requests': rows, 'received_datagrams': dict(received), 'final_counter': gate.value,
            'unique_applied_commands': len(gate.cache),
            'faults': 'one discarded request; one discarded ACK; reordered sequence 2; expired sequence 4',
            'limit': 'deduplication lasts only within this in-memory session; not exactly-once delivery'}


def run(samples=30, slow_ms=15):
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'transport': 'real TCP and UDP sockets on 127.0.0.1, OS-assigned ephemeral ports',
            'tcp': tcp_experiment(samples, slow_ms), 'udp': udp_experiment(),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--slow-ms', type=float, default=15)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run(args.samples, args.slow_ms)
    body = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(body)
    print(body)
