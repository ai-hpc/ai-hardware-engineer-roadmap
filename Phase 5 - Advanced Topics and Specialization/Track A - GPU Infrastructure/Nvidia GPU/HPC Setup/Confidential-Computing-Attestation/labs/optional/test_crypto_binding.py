import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binding import canonical, strict_load
from binding_lab import Fixture, Rejected, Ed25519PrivateKey, prove, sign_claim, authenticate, MODEL


class CryptoBindingTests(unittest.TestCase):
    def reject(self, fixture, reason):
        with self.assertRaisesRegex(Rejected, reason):
            fixture.verifier.accept(fixture.bundle)

    def test_round_trip_and_replay(self):
        f = Fixture()
        bundle = strict_load(canonical(f.bundle))
        self.assertTrue(f.verifier.accept(bundle)["accepted"])
        self.reject(f, "unknown-or-consumed")

    def test_tamper_signed_payload(self):
        f = Fixture(); f.bundle["cpu"]["payload"]["measurement"] = "evil"
        self.reject(f, "signature")

    def test_attacker_controlled_signing_key(self):
        f = Fixture()
        f.bundle["cpu"] = sign_claim(Ed25519PrivateKey.generate(), f.bundle["cpu"]["payload"])
        prove(f.bundle, f.session_key)
        self.reject(f, "signature")

    def test_authentic_bad_policy(self):
        for field, value in [("measurement", "unapproved"), ("tcb", "OutOfDate"), ("debug", True), ("debug", 0)]:
            f = Fixture(); f.alter_signed_claim("cpu", field, value)
            self.reject(f, "policy")

    def test_valid_signed_wrong_binding(self):
        f = Fixture(); f.alter_signed_claim("gpu", "binding", "00" * 32)
        self.reject(f, "binding")

    def test_full_cpu_binding_checked(self):
        f = Fixture(); original = f.bundle["cpu"]["payload"]["binding"]
        f.alter_signed_claim("cpu", "binding", original[:64] + "00" * 32)
        self.reject(f, "binding")

    def test_mixed_sessions_same_authority(self):
        f = Fixture(); old = copy.deepcopy(f.bundle["gpus"][0])
        f.transcript = f.verifier.issue(f.transcript["session_public"])
        f.bundle = f.make_bundle(); f.bundle["gpus"][0] = old
        prove(f.bundle, f.session_key)
        self.reject(f, "binding")

    def test_transcript_substitution(self):
        for field, value in [("model", "f" * 64), ("audience", "other"), ("session_public", "a" * 64)]:
            f = Fixture(); f.bundle["transcript"][field] = value
            self.reject(f, "transcript")

    def test_expiry_boundary_and_future(self):
        for offset in (-1, 60):
            f = Fixture(); f.now += offset
            self.reject(f, "challenge-time")
        f = Fixture(); f.now += 59
        self.assertTrue(f.verifier.accept(f.bundle)["accepted"])

    def test_claim_time_and_schema(self):
        for field, value, reason in [("expires_at", 1800000000, "time"), ("issued_at", True, "time"), ("version", True, "schema")]:
            f = Fixture(); f.alter_signed_claim("gpu", field, value)
            self.reject(f, reason)

    def test_claim_expiry_during_verification_does_not_consume_challenge(self):
        for role in ("cpu", "gpu"):
            with self.subTest(role=role):
                f = Fixture()
                f.alter_signed_claim(role, "expires_at", f.now + 5)
                clock = iter((f.now, f.now + 5))
                f.verifier.clock = lambda: next(clock)
                self.reject(f, "^time$")
                self.assertIn(f.transcript["challenge"], f.verifier.pending)
                # A newly signed, still-valid claim can complete the same request.
                f.now += 5
                f.verifier.clock = lambda: f.now
                f.alter_signed_claim(role, "expires_at", f.transcript["expires_at"])
                self.assertTrue(f.verifier.accept(f.bundle)["accepted"])

    def test_all_device_expiries_rechecked_at_authorization(self):
        f = Fixture(("gpu-demo-0", "gpu-demo-1"))
        payload = f.bundle["gpus"][1]["payload"]
        payload["expires_at"] = f.now + 5
        f.bundle["gpus"][1] = sign_claim(f.gpu_key, payload)
        prove(f.bundle, f.session_key)
        clock = iter((f.now, f.now + 5))
        f.verifier.clock = lambda: next(clock)
        self.reject(f, "^time$")
        self.assertIn(f.transcript["challenge"], f.verifier.pending)

    def test_claim_not_before_rechecked_after_clock_moves_backward(self):
        f = Fixture()
        f.alter_signed_claim("cpu", "issued_at", f.now + 2)
        clock = iter((f.now + 3, f.now + 1))
        f.verifier.clock = lambda: next(clock)
        self.reject(f, "^time$")
        self.assertIn(f.transcript["challenge"], f.verifier.pending)

    def test_caller_mutation_cannot_change_authenticated_authorization(self):
        f = Fixture()
        calls = 0
        def clock():
            nonlocal calls
            calls += 1
            if calls == 2:
                # Deterministic stand-in for another thread owning the input.
                f.bundle["transcript"]["model"] = "f" * 64
                f.bundle["transcript"]["devices"] = ["unrelated-gpu"]
            return f.now
        f.verifier.clock = clock
        result = f.verifier.accept(f.bundle)
        self.assertEqual(result["model"], MODEL)
        self.assertEqual(result["devices"], ["gpu-demo-0"])
        self.reject(f, "unknown-or-consumed")

    def test_possession_failure(self):
        f = Fixture(); prove(f.bundle, Ed25519PrivateKey.generate())
        self.reject(f, "possession")

    def test_multi_device_inventory(self):
        f = Fixture(("gpu-demo-0", "gpu-demo-1"))
        self.assertTrue(f.verifier.accept(f.bundle)["accepted"])
        for mode in ("empty", "duplicate", "unrelated"):
            f = Fixture(("gpu-demo-0", "gpu-demo-1"))
            if mode == "empty": f.bundle["gpus"] = []
            elif mode == "duplicate": f.bundle["gpus"][1] = copy.deepcopy(f.bundle["gpus"][0])
            else: f.alter_signed_claim("gpu", "device", "unrelated")
            prove(f.bundle, f.session_key)
            self.reject(f, "device-inventory")

    def test_failure_does_not_consume_pending_challenge(self):
        f = Fixture(); good = copy.deepcopy(f.bundle)
        f.bundle["gpus"] = []
        self.reject(f, "device-inventory")
        self.assertTrue(f.verifier.accept(good)["accepted"])

    def test_signature_is_not_computation_truth(self):
        key = Ed25519PrivateKey.generate()
        claim = {"left": 2, "right": 2, "sum": 5}
        verified = authenticate(key.public_key(), sign_claim(key, claim))
        self.assertNotEqual(verified["left"] + verified["right"], verified["sum"])

    def test_concurrent_single_authorization(self):
        f = Fixture()
        def attempt(_):
            try: return f.verifier.accept(f.bundle)["accepted"]
            except Rejected: return False
        with ThreadPoolExecutor(max_workers=8) as pool:
            self.assertEqual(sum(pool.map(attempt, range(16))), 1)


if __name__ == "__main__":
    unittest.main()
