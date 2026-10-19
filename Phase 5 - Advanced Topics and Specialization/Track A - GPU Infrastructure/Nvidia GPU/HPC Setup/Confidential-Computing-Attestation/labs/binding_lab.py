"""Real signatures over locally issued fixtures. NO hardware attestation.

The verifier's authorities are local teaching Ed25519 keys. The session key is
also Ed25519 for possession proof; it is not an encryption/key-exchange key.
"""
import argparse
import copy
import hashlib
import json
import secrets
import threading
import time
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from binding import bindings, canonical, strict_load

SIGN_DOMAIN = b"course/fixture-signature/v1\0"
PROOF_DOMAIN = b"course/session-possession/v1\0"
MODEL = hashlib.sha256(b"fictional model + tokenizer + configuration v1").hexdigest()
AUDIENCE = "course-local-relying-party"


class Rejected(ValueError):
    pass


def public_hex(key):
    return key.public_key().public_bytes_raw().hex()


def sign_claim(key, payload):
    return {"payload": copy.deepcopy(payload),
            "signature": key.sign(SIGN_DOMAIN + canonical(payload)).hex()}


def authenticate(public_key, token):
    if type(token) is not dict or set(token) != {"payload", "signature"}:
        raise Rejected("schema")
    try:
        public_key.verify(bytes.fromhex(token["signature"]),
                          SIGN_DOMAIN + canonical(token["payload"]))
    except (InvalidSignature, ValueError, TypeError):
        raise Rejected("signature") from None
    return token["payload"]


def proof_message(bundle):
    unsigned = {k: bundle[k] for k in ("transcript", "cpu", "gpus")}
    return PROOF_DOMAIN + hashlib.sha256(canonical(unsigned)).digest()


def prove(bundle, session_key):
    bundle["proof"] = session_key.sign(proof_message(bundle)).hex()
    return bundle


class Verifier:
    """Single-process, in-memory authorization example with atomic consume."""
    def __init__(self, cpu_public, gpu_public, clock=None):
        self.roots = {"cpu": cpu_public, "gpu": gpu_public}
        self.clock = clock or (lambda: int(time.time()))
        self.pending = {}
        self.lock = threading.Lock()

    def issue(self, session_public, devices=("gpu-demo-0",)):
        if not devices or len(set(devices)) != len(devices):
            raise ValueError("expected distinct non-empty devices")
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(session_public))
        now = self.clock()
        transcript = {"version": 1, "audience": AUDIENCE,
                      "challenge": secrets.token_hex(32), "issued_at": now,
                      "expires_at": now + 60, "session_public": session_public,
                      "model": MODEL, "devices": sorted(devices)}
        with self.lock:
            self.pending[transcript["challenge"]] = canonical(transcript)
        return transcript

    def _claim(self, token, role, transcript, now):
        claim = authenticate(self.roots[role], token)
        fields = {"version", "role", "binding", "measurement", "tcb",
                  "debug", "issued_at", "expires_at", "device"}
        if type(claim) is not dict or set(claim) != fields:
            raise Rejected("schema")
        if type(claim["version"]) is not int or claim["version"] != 1 or claim["role"] != role:
            raise Rejected("schema")
        if claim["binding"] != bindings(transcript)[role]:
            raise Rejected("binding")
        if any(type(claim[k]) is not int for k in ("issued_at", "expires_at")):
            raise Rejected("time")
        if not (transcript["issued_at"] <= claim["issued_at"] <= now < claim["expires_at"] <= transcript["expires_at"]):
            raise Rejected("time")
        expected = "approved-app-v1" if role == "cpu" else "approved-firmware-v1"
        if claim["measurement"] != expected or claim["tcb"] != "UpToDate" or claim["debug"] is not False:
            raise Rejected("policy")
        if type(claim["device"]) is not str:
            raise Rejected("schema")
        return claim

    def accept(self, bundle):
        if type(bundle) is not dict or set(bundle) != {"transcript", "cpu", "gpus", "proof"}:
            raise Rejected("schema")
        # Own one bounded snapshot throughout appraisal and the final decision.
        # Caller mutation must not change which model/device values we authorize
        # after their signed binding has already been checked.
        try:
            bundle = strict_load(canonical(bundle))
        except (ValueError, TypeError, RecursionError):
            raise Rejected("schema") from None
        transcript = bundle["transcript"]
        if type(transcript) is not dict or type(transcript.get("challenge")) is not str:
            raise Rejected("schema")
        challenge = transcript["challenge"]
        with self.lock:
            expected = self.pending.get(challenge)
            if expected is None:
                raise Rejected("unknown-or-consumed-challenge")
            if canonical(transcript) != expected:
                raise Rejected("transcript")
        now = self.clock()
        if not transcript["issued_at"] <= now < transcript["expires_at"]:
            raise Rejected("challenge-time")
        cpu_claim = self._claim(bundle["cpu"], "cpu", transcript, now)
        if type(bundle["gpus"]) is not list or len(bundle["gpus"]) != len(transcript["devices"]):
            raise Rejected("device-inventory")
        gpu_claims = [self._claim(g, "gpu", transcript, now) for g in bundle["gpus"]]
        devices = [claim["device"] for claim in gpu_claims]
        if len(set(devices)) != len(devices) or sorted(devices) != transcript["devices"]:
            raise Rejected("device-inventory")
        try:
            key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(transcript["session_public"]))
            key.verify(bytes.fromhex(bundle["proof"]), proof_message(bundle))
        except (InvalidSignature, ValueError, TypeError):
            raise Rejected("possession") from None
        with self.lock:
            if self.pending.get(challenge) != expected:
                raise Rejected("unknown-or-consumed-challenge")
            final_now = self.clock()
            if not transcript["issued_at"] <= final_now < transcript["expires_at"]:
                raise Rejected("challenge-time")
            # Signature verification can outlast a shorter claim validity window.
            # The authorization instant must satisfy every authenticated window,
            # including not-before if the relying party's clock moved backward.
            if any(not claim["issued_at"] <= final_now < claim["expires_at"]
                   for claim in (cpu_claim, *gpu_claims)):
                raise Rejected("time")
            del self.pending[challenge]
        return {"accepted": True, "model": transcript["model"], "devices": devices,
                "hardware_attestation": False, "scope": "locally signed teaching fixtures"}


class Fixture:
    """Authorities are deliberately controllable for negative tests."""
    def __init__(self, devices=("gpu-demo-0",)):
        self.now = 1800000000
        self.cpu_key = Ed25519PrivateKey.generate()
        self.gpu_key = Ed25519PrivateKey.generate()
        self.session_key = Ed25519PrivateKey.generate()
        self.verifier = Verifier(self.cpu_key.public_key(), self.gpu_key.public_key(), lambda: self.now)
        self.transcript = self.verifier.issue(public_hex(self.session_key), devices)
        self.bundle = self.make_bundle()

    def make_claim(self, role, device):
        t = self.transcript
        payload = {"version": 1, "role": role, "binding": bindings(t)[role],
                   "measurement": "approved-app-v1" if role == "cpu" else "approved-firmware-v1",
                   "tcb": "UpToDate", "debug": False, "issued_at": t["issued_at"],
                   "expires_at": t["expires_at"], "device": device}
        key = self.cpu_key if role == "cpu" else self.gpu_key
        return sign_claim(key, payload)

    def make_bundle(self):
        bundle = {"transcript": copy.deepcopy(self.transcript),
                  "cpu": self.make_claim("cpu", "cpu-demo"),
                  "gpus": [self.make_claim("gpu", d) for d in self.transcript["devices"]]}
        return prove(bundle, self.session_key)

    def alter_signed_claim(self, role, field, value):
        token = self.bundle["cpu"] if role == "cpu" else self.bundle["gpus"][0]
        token["payload"][field] = value
        key = self.cpu_key if role == "cpu" else self.gpu_key
        replacement = sign_claim(key, token["payload"])
        token.update(replacement)
        prove(self.bundle, self.session_key)


def run():
    report = {"hardware_attestation": False, "cases": []}
    f = Fixture()
    decoded = strict_load(canonical(f.bundle))
    report["cases"].append({"case": "valid", "result": f.verifier.accept(decoded)})
    trials = [("replay", f, decoded)]
    g = Fixture(); g.bundle["cpu"]["payload"]["measurement"] = "changed"
    trials.append(("tampering", g, g.bundle))
    g = Fixture(); g.alter_signed_claim("cpu", "debug", True)
    trials.append(("authentic-disallowed-debug", g, g.bundle))
    g = Fixture(); g.alter_signed_claim("gpu", "binding", "00" * 32)
    trials.append(("authentic-wrong-session", g, g.bundle))
    g = Fixture(); g.now += 60
    trials.append(("expired-challenge", g, g.bundle))
    g = Fixture(); prove(g.bundle, Ed25519PrivateKey.generate())
    trials.append(("wrong-session-key", g, g.bundle))
    g = Fixture(); g.bundle["gpus"] = []; prove(g.bundle, g.session_key)
    trials.append(("missing-gpu", g, g.bundle))
    for name, fixture, bundle in trials:
        try:
            fixture.verifier.accept(bundle)
            raise AssertionError(f"attack accepted: {name}")
        except Rejected as error:
            report["cases"].append({"case": name, "rejected": str(error)})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run()
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
