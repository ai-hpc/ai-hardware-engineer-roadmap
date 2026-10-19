import http.client
import json
from pathlib import Path
import select
import subprocess
import sys
import threading
import unittest

from service import Application, make_server


MODEL = Path(__file__).with_name("model.json").read_bytes()


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.app = Application(MODEL)
        self.server = make_server("127.0.0.1", 0, self.app)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .02})
        self.thread.start()
        self.addCleanup(self.close)

    def close(self):
        self.server.shutdown(); self.thread.join(timeout=3); self.server.server_close()
        self.assertFalse(self.thread.is_alive())

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection(*self.server.server_address, timeout=3)
        try:
            conn.request(method, path, body, headers or {})
            response = conn.getresponse()
            return response.status, json.loads(response.read())
        finally:
            conn.close()

    def infer(self, body):
        return self.request("POST", "/infer", json.dumps(body), {"Content-Type": "application/json"})

    def test_live_ready_and_known_input(self):
        self.assertEqual(self.request("GET", "/live")[0], 200)
        self.assertTrue(self.request("GET", "/ready")[1]["ready"])
        for distance, speed, stop in ((.3, 0, True), (2, 0, False), (1, 1, True), (3, 1, False)):
            status, result = self.infer(dict(distance_m=distance, closing_speed_mps=speed, sensor_age_ms=20))
            self.assertEqual(status, 200); self.assertEqual(result["stop"], stop)
            self.assertEqual(result["model_sha256"], self.app.model_sha256)

    def test_wrong_model_is_live_but_unready(self):
        bad = Application(b'{"schema":1,"horizon_s":1,"margin_m":9}')
        self.assertFalse(bad.ready)
        self.app.ready = bad.ready; self.app.error = bad.error
        self.assertEqual(self.request("GET", "/live")[0], 200)
        self.assertEqual(self.request("GET", "/ready")[0], 503)
        self.assertEqual(self.infer({})[0], 503)

    def test_invalid_schema_values_and_stale_observation(self):
        good = dict(distance_m=1, closing_speed_mps=1, sensor_age_ms=20)
        for body in ({}, [], {**good, "extra": 1}, {**good, "sensor_age_ms": 101},
                     {**good, "sensor_age_ms": -1}, {**good, "distance_m": True},
                     {**good, "closing_speed_mps": float("nan")}, {**good, "distance_m": "1"}):
            with self.subTest(body=body):
                self.assertEqual(self.infer(body)[0], 422)

    def test_body_limits_content_type_duplicate_keys(self):
        self.assertEqual(self.request("POST", "/infer", b"x"*1025,
                                      {"Content-Type": "application/json"})[0], 413)
        self.assertEqual(self.request("POST", "/infer", b"{}")[0], 415)
        self.assertEqual(self.request("POST", "/infer", b'{"distance_m":1,"distance_m":2}',
                                      {"Content-Type": "application/json"})[0], 422)
        self.assertEqual(self.request("POST", "/infer", b"not json",
                                      {"Content-Type": "application/json"})[0], 422)

    def test_large_json_integer_is_rejected_without_crashing_handler(self):
        for value in (10**400, -(10**400)):
            with self.subTest(sign=value > 0):
                self.assertEqual(self.infer(dict(distance_m=value, closing_speed_mps=1,
                                                sensor_age_ms=20))[0], 422)
        raw = json.dumps({"schema": 1, "horizon_s": 10**400, "margin_m": .4}).encode()
        self.assertFalse(Application(raw).ready)
        self.assertEqual(self.request("GET", "/live")[0], 200)

    def test_content_length_and_transfer_encoding_contract(self):
        cases = [([("Content-Length", "²")], 400),
                 ([("Content-Length", "9" * 5000)], 413),
                 ([("Content-Length", "2"), ("Content-Length", "2")], 400),
                 ([("Content-Length", "-1")], 400),
                 ([("Content-Length", "2.0")], 400),
                 ([("Content-Length", "")], 400),
                 ([("Content-Length", "2"), ("Transfer-Encoding", "chunked")], 400),
                 ([("Content-Length", "0002")], 422)]
        for headers, expected in cases:
            with self.subTest(headers=[(key, value[:20]) for key, value in headers]):
                conn = http.client.HTTPConnection(*self.server.server_address, timeout=3)
                try:
                    conn.putrequest("POST", "/infer")
                    conn.putheader("Content-Type", "application/json")
                    for key, value in headers:
                        conn.putheader(key, value)
                    conn.endheaders(b"{}")
                    response = conn.getresponse()
                    self.assertEqual(response.status, expected)
                    self.assertIsInstance(json.loads(response.read()), dict)
                finally:
                    conn.close()

    def test_draining_rejects_new_work(self):
        self.app.draining = True
        self.assertEqual(self.request("GET", "/live")[0], 200)
        self.assertEqual(self.request("GET", "/ready")[0], 503)
        self.assertEqual(self.infer({})[0], 503)

    def test_unknown_endpoint(self):
        self.assertEqual(self.request("GET", "/unknown")[0], 404)
        self.assertEqual(self.request("POST", "/unknown")[0], 404)

    def test_invalid_model_contract(self):
        for raw in (b"{}", b"[]", b'{"schema":true,"horizon_s":1,"margin_m":0.4}',
                    b'{"schema":1,"horizon_s":1,"margin_m":0.4,"margin_m":0.5}'):
            with self.subTest(raw=raw):
                self.assertFalse(Application(raw).ready)

    def test_real_process_sigterm_exit(self):
        process = subprocess.Popen([sys.executable, str(Path(__file__).with_name("service.py")),
                                    "--port", "0"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True)
        try:
            self.assertTrue(select.select([process.stdout], [], [], 5)[0], "startup timed out")
            address = json.loads(process.stdout.readline())
            self.assertTrue(address["model_ready"])
            process.terminate()
            self.assertEqual(process.wait(timeout=5), 0)
        finally:
            if process.poll() is None:
                process.kill(); process.wait(timeout=3)
            process.stdout.close(); process.stderr.close()


if __name__ == "__main__":
    unittest.main()
