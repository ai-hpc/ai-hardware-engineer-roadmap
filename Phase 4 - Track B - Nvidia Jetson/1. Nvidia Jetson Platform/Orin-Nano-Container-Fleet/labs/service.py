#!/usr/bin/env python3
"""Local teaching HTTP service for a deterministic stop rule, not a learned policy."""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
from pathlib import Path
import signal
import threading


MAX_BODY = 1024


def number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    # Compare the range first: Python can parse integers too large to convert
    # to a float, and those must be rejected rather than raising OverflowError.
    if not low <= value <= high or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and in [{low}, {high}]")
    return float(value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def parse_json(raw):
    return json.loads(raw, object_pairs_hook=unique_object,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


class Application:
    def __init__(self, model_bytes):
        self.model_sha256 = hashlib.sha256(model_bytes).hexdigest()
        self.ready = False
        self.error = None
        self.draining = False
        try:
            self.model = parse_json(model_bytes)
            if not isinstance(self.model, dict) or set(self.model) != {"schema", "horizon_s", "margin_m"}:
                raise ValueError("unexpected model schema")
            if type(self.model["schema"]) is not int or self.model["schema"] != 1:
                raise ValueError("unsupported schema")
            self.horizon = number(self.model["horizon_s"], "horizon_s", .1, 5)
            self.margin = number(self.model["margin_m"], "margin_m", 0, 10)
            expected = [(.3, 0, True), (2, 0, False), (1, 1, True), (3, 1, False)]
            if any(self.stop_rule(distance, speed) != stop for distance, speed, stop in expected):
                raise ValueError("known-input self-test failed")
            self.ready = True
        except (ValueError, TypeError, UnicodeError) as error:
            self.error = str(error)

    def stop_rule(self, distance, speed):
        return distance <= max(speed, 0) * self.horizon + self.margin

    def infer(self, body):
        if not isinstance(body, dict) or set(body) != {"distance_m", "closing_speed_mps", "sensor_age_ms"}:
            raise ValueError("require distance_m, closing_speed_mps and sensor_age_ms only")
        distance = number(body["distance_m"], "distance_m", 0, 100)
        speed = number(body["closing_speed_mps"], "closing_speed_mps", -10, 10)
        age = number(body["sensor_age_ms"], "sensor_age_ms", 0, 100)
        return {"stop": self.stop_rule(distance, speed), "model_sha256": self.model_sha256,
                "declared_sensor_age_ms": age,
                "evidence_kind": "deterministic teaching rule; no actuator command"}


def make_server(host, port, application):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def setup(self):
            super().setup()
            self.connection.settimeout(2)

        def log_message(self, *_):
            pass

        def send_json(self, status, payload):
            raw = json.dumps(payload, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path == "/live":
                self.send_json(200, {"live": True})
            elif self.path == "/ready":
                ready = application.ready and not application.draining
                self.send_json(200 if ready else 503,
                               {"ready": ready, "model_sha256": application.model_sha256,
                                "error": application.error, "draining": application.draining})
            else:
                self.send_json(404, {"error": "unknown endpoint"})

        def do_POST(self):
            if self.path != "/infer":
                self.send_json(404, {"error": "unknown endpoint"}); return
            if not application.ready or application.draining:
                self.send_json(503, {"error": "application is not ready"}); return
            if self.headers.get("Transfer-Encoding") is not None:
                self.send_json(400, {"error": "transfer encoding unsupported"}); return
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit():
                self.send_json(400, {"error": "one numeric Content-Length required"}); return
            # Bound the value before integer conversion, including on Python
            # versions with a limit on the number of parsed integer digits.
            digits = lengths[0].lstrip("0") or "0"
            if len(digits) > len(str(MAX_BODY)):
                self.send_json(413, {"error": "body must be 1..1024 bytes"}); return
            length = int(digits)
            if not 0 < length <= MAX_BODY:
                self.send_json(413, {"error": "body must be 1..1024 bytes"}); return
            if self.headers.get("Content-Type", "").split(";", 1)[0].strip() != "application/json":
                self.send_json(415, {"error": "application/json required"}); return
            try:
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("truncated body")
                output = application.infer(parse_json(raw))
            except TimeoutError:
                self.send_json(408, {"error": "request body timed out"})
            except (ValueError, TypeError, UnicodeError) as error:
                self.send_json(422, {"error": str(error)})
            else:
                self.send_json(200, output)

    server = HTTPServer((host, port), Handler)
    server.timeout = .2
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("model.json"))
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("port must be in [0, 65535]")
    app = Application(args.model.read_bytes())
    server = make_server(args.host, args.port, app)
    stop = threading.Event()

    def request_stop(*_):
        app.draining = True
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    print(json.dumps({"host": server.server_address[0], "port": server.server_address[1],
                      "model_ready": app.ready}), flush=True)
    try:
        while not stop.is_set():
            server.handle_request()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
