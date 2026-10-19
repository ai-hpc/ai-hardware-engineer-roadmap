# CPU Lab — Authenticate, Bind and Authorize a Session

**Time:** 30–45 minutes. **Hardware:** ordinary CPU. **Prerequisites:** lessons 01–03, basic Python dictionaries and public-key signatures.

You will issue a challenge, authenticate two signed fixture claims, compare their session bindings, apply an explicit policy and verify session-key possession. Then you will change one property at a time and locate the first rejecting check.

This lab uses real Ed25519 signing/verification through `cryptography`. All authorities and devices are fictional; all keys are generated locally. It neither collects nor verifies Intel/NVIDIA hardware evidence. It does not implement JWT, certificate validation, TDX, NRAS or encrypted inference. These boundaries matter: a successful CPU exercise cannot certify a GPU deployment.

## 1. Reproduce the actual run

From the repository root:

```bash
cd "Phase 5 - Advanced Topics and Specialization/Track A - GPU Infrastructure/Nvidia GPU/HPC Setup/Confidential-Computing-Attestation/labs"
python3 -m unittest discover -s . -p 'test_*.py' -v
python3 -m venv /tmp/course-attestation-venv
/tmp/course-attestation-venv/bin/python -m pip install --require-hashes -r requirements-cpu.lock
/tmp/course-attestation-venv/bin/python -m unittest discover -s optional -p 'test_*.py' -v
/tmp/course-attestation-venv/bin/python binding_lab.py --output /tmp/attestation-binding-report.json
```

The lock targets **CPython 3.12 on Linux x86_64** and includes the three downloaded wheel hashes: `cryptography==46.0.5`, `cffi==2.0.0`, `pycparser==2.23`. Total wheel download was under 5 MB. On another platform, use a separate environment and resolve compatible wheels for those versions; record a new full hash lock and rerun tests instead of disabling hash verification. The portable transcript tests need no packages.

Expected baseline: **4 portable tests and 19 cryptographic tests pass**. The report has one accepted fictional session and seven rejected attacks. Keys/challenges change every run; classifications stay the same. The fixture clock is a synthetic integer; expiry tests also inject clock changes during verification, with no sleeping. [Actual recorded run](evidence/cpu-run-2026-10-18.json), [validation record](evidence/validation.json).

## 2. Trace the successful session

Read [binding.py](labs/binding.py), then `Verifier.issue`, `Verifier.accept` and `Fixture` in [binding_lab.py](labs/binding_lab.py).

1. The relying party stores an exact transcript containing a random 32-byte challenge, audience, model manifest digest, session public key, device inventory and a 60-second deadline.
2. Two local teaching authorities sign fixture claims. CPU and GPU binding fields use SHA-512 and SHA-256 of the same domain-separated transcript respectively.
3. The verifier authenticates each signature using its preconfigured public keys, compares all binding bytes, and checks measurement, debug state, TCB label, time and the expected GPU identities. Its CPU policy checks the approved application measurement; it does not pin a particular CPU device name.
4. The endpoint proves possession of its session Ed25519 key by signing a digest of the whole bundle under a different domain separator.
5. Only after all checks succeed does the verifier recheck the challenge and every claim's validity window at authorization time, then atomically consume the challenge.

`accept` takes a private, size-bounded snapshot of the supplied bundle before appraisal. A test changes the caller's original model/device fields while verification runs and requires the returned decision to retain the authenticated values. This addresses shared Python-object ownership; it is not a demonstration of forging a remote signature. Replay state and configured trust roots remain verifier-owned.

The authorities' private keys are available to the fixture generator because negative tests need authentic-but-unacceptable claims. In a real protocol those keys and checks belong to the actual evidence/verifier infrastructure. The session key in this lab signs a possession proof; **Ed25519 is not an encryption or key-exchange algorithm**. [Cryptography's Ed25519 API](https://cryptography.io/en/46.0.5/hazmat/primitives/asymmetric/ed25519/).

## 3. Predict before running the attacks

| Case | First rejection | Why |
|---|---|---|
| Reuse accepted bundle | `unknown-or-consumed-challenge` | Authorization consumed this pending request |
| Edit a signed measurement | `signature` | Original signature no longer authenticates the payload |
| Authority signs `debug=true` | `policy` | Authentic evidence can describe disallowed state |
| Authority signs a different session binding | `binding` | Signer is trusted but evidence belongs elsewhere |
| Advance fixture clock by 60 seconds | `challenge-time` | Valid interval is `[issued_at, expires_at)` |
| Prove possession with another key | `possession` | Endpoint does not own the key bound in the transcript |
| Remove required GPU | `device-inventory` | An empty list cannot satisfy the required component set |

The optional tests add attacker-controlled signers, mixed sessions with the **same legitimate authority**, substitution of audience/model/key, changes in the second half of the CPU binding, missing/duplicated/unrelated devices, invalid claim times and a 16-attempt concurrent replay. Exactly one concurrent attempt is authorized. A failed attempt leaves the valid pending session available; a real service would additionally bound retries and rate-limit abuse.

**A deadline can expire during verification.** A challenge valid until `t+60` does not extend a GPU claim valid only until `t+5`. Starting checks at `t` and reaching authorization at exactly `t+5` must reject with `time`, leaving the challenge unconsumed. The tests exercise CPU claims, both positions in a two-GPU inventory, and a backward clock change that violates a claim's issue time. This checks the authorization instant; it does not grant an indefinitely valid session afterward.

## 4. Guided variation: two GPUs

From `labs/`, run:

```bash
/tmp/course-attestation-venv/bin/python - <<'PYCODE'
from binding_lab import Fixture
fixture = Fixture(("gpu-demo-0", "gpu-demo-1"))
print(fixture.verifier.accept(fixture.bundle))
PYCODE
```

Expected: acceptance with both distinct device names. In a copy of this experiment, replace the second GPU token with the first, then recreate the session possession proof using `prove(fixture.bundle, fixture.session_key)`. Predict the outcome before calling `accept`.

**Answer:** `device-inventory`, even though both tokens and the new proof are authentic. The same GPU cannot fill two required positions. If you forget to recreate the proof, the inventory check still rejects first in this implementation; the test deliberately repairs the proof to isolate the inventory property.

## 5. Independent transfer and worked answers

**A. Expiry boundary.** Test clock offsets 59 and 60 seconds with fresh fixtures. **Answer:** 59 accepts, 60 rejects. A fresh fixture is necessary because acceptance consumes the challenge.

**B. Model substitution.** Change only `bundle["transcript"]["model"]` to a different digest. **Answer:** `transcript`; it differs from the client's stored request before any cryptographic appraisal. Giving the attacker a valid signing key for a different authority must still fail the configured trust-root check.

**C. Result correctness.** The signature primitive authenticates a fixture payload containing `2 + 2 = 5`. Does the signature fail? **No.** The included test demonstrates that authenticity and arithmetic truth are separate. The session verifier's schema does not accept that payload as platform evidence either.

**D. Extend the threat model.** Which passing result proves that these CPU/GPU identities share a protected physical connection? **None.** The fixture only signs names. A real deployment must verify authenticated endpoint/channel assignment and supported topology.

## 6. Submit a defensible conclusion

Submit the JSON report, package lock, test output and a short explanation of three distinct failures. Your conclusion should say “real signatures over local teaching fixtures,” with the measured test counts. State that replay state is process-local and lost on restart, no certificate/collateral chain was checked, no hardware report was collected and no encrypted inference traffic was exercised.

For a deployment extension, replace fixture authorities with versioned vendor evidence adapters and validated result parsing. Keep challenge storage, policy, inventory and channel checks. Do not map a vendor `passed` flag directly to key release. The actual NVAT command and its execution limits are in [lesson 02](02-NVIDIA-CC-Attestation-Chain.md).
