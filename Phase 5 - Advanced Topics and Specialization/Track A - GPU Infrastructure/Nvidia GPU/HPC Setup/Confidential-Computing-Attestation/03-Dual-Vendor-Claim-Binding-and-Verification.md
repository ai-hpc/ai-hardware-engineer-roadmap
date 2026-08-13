# 03 — Dual-Vendor Claim Binding & Verification Review

## 1. The Problem This Solves

A verifier for a remote or rented compute job wants two paths:

```text
   Slow path:  recipe + dataset ──▶ re-execute from source ──▶ compare to claim   (expensive, always correct)
   Fast path:  claim + attestation bundle ──▶ verify cryptographically             (cheap, only as strong as the attestation)
```

The fast path only works if the claim can't be forged. Define a single digest that commits to everything about the job that matters:

```text
claim_sha256 = sha256(result_scores ‖ job_metadata ‖ output_manifest)
```

`output_manifest` — a per-file SHA-256 of output artifacts (checkpoints, results) — lets the claim commit to output *identity* without the verifier downloading multi-gigabyte payloads. Files [01](./01-Intel-TDX-Attestation-Chain.md) and [02](./02-NVIDIA-CC-Attestation-Chain.md) each showed how to bind this hash into one hardware vendor's signed report (`REPORTDATA` for Intel TDX, `eat_nonce` for NVIDIA CC) and authenticate that report back to the vendor's root of trust. This file composes both and generalizes the underlying review skill.

## 2. Composing the Two Chains

One `claim_sha256` anchored independently into both hardware roots:

```text
                                    claim_sha256
                        (result_scores ‖ job_metadata ‖ output_manifest)
                                          │
                       ┌───────────────────┴───────────────────┐
                       ▼                                       ▼
              eat_nonce (NVIDIA CC)                REPORTDATA (Intel TDX, zero-padded)
                       │                                       │
        gpu_signature: NRAS JWKS,                  tdx_signature: DCAP —
        ES384, kid-resolved,                        ECDSA sig, PCK → Intel root CA,
        iss + exp checked                           QE identity, live TCB status
                       │                                       │
                       └───────────────────┬───────────────────┘
                                            ▼
                     claim is bound to a genuine GPU AND
                     a genuine, measured TDX VM — independently
```

A verifier's output for one claim should surface every check as a distinct field — never collapse them into one boolean:

```text
claim_bound:            true              # claim digest found in eat_nonce
gpu_signature:           true, 2 tokens    # NVIDIA JWKS, ES384, per-device + platform
tdx_bound:               true              # claim digest found in REPORTDATA
tdx_signature:           true, UpToDate    # Intel DCAP / PCS
output_hash_match:       true              # claim manifest matches submitted output files
```

Forging this now requires compromising **both** Intel's and NVIDIA's signing infrastructure simultaneously — a materially different bar than writing convincing JSON against a single-vendor check.

## 3. Binding vs. Authenticity — The Core Review Discipline

Every gap in a real dual-vendor attestation rollout tends to be the same pattern, repeated: **a field got checked before its signature did.** This generalizes past TDX and NVIDIA CC entirely — it's what to look for in *any* attestation code review (an SGX enclave, a TPM-backed boot chain, a different vendor's CC offering):

| Symptom in code | What it actually checks | What it's missing | Concrete forgery |
|---|---|---|---|
| `if quote["reportdata"] == claim_hash` | binding only | signature/chain validation | Hand-write JSON with the right bytes in `reportdata`; no real TDX Module ever ran |
| `if attestation["passed"] == True` | trusting a self-reported flag | who computed `passed` and how | Fabricate the JSON; nothing forces `passed` to reflect a real check |
| `jwt.decode(token, verify=False)` or checking only `HS256` | JWT well-formedness | the *right kind* of signature (asymmetric, vendor-held key) | Sign your own token with a key you generated; nothing ties it to NVIDIA/Intel |
| Boolean-only TCB/status output | pass/fail | the actual risk level (advisory IDs, staleness) | Not a forgery, but a validator loses the ability to apply its own policy — silent risk-laundering |

When reviewing or designing any attestation-gated system, ask the same three questions in order:

1. **What value is being bound in?** (a claim hash, a nonce, a public key)
2. **Who signed the container that value lives in?** (the prover itself, or a hardware vendor's key?)
3. **Does that signature chain to a root you actually trust, checked live rather than cached?**

If the answer to (2) or (3) is "nothing verifies that," the binding in (1) is decorative — it will pass a review that only reads the binding check and stops.

## 4. Threat Model — What's Proven, What Remains Trusted

| Threat | Defended by | Residual assumption |
|---|---|---|
| Fabricated claim JSON with no real hardware involved | binding checks alone | **none — this is the gap, not a defense; needs the authenticity checks in files 01 §4 / 02 §4 too** |
| Forged TD Quote with correct REPORTDATA bytes | DCAP signature + PCK chain ([file 01, §4](./01-Intel-TDX-Attestation-Chain.md#4-dcap-verification-chaining-the-quote-back-to-intels-root-ca)) | Intel's root CA and PCK infrastructure are uncompromised |
| Forged GPU EAT with correct eat_nonce | NRAS JWKS signature check ([file 02, §4](./02-NVIDIA-CC-Attestation-Chain.md#4-authenticity-nras-signature--jwks-key-resolution)) | NVIDIA's NRAS signing key and JWKS endpoint are uncompromised |
| Genuine hardware, wrong/tampered software stack | MRTD / RTMR measurement | the verifier's *golden* reference measurement is itself correct and current — a stale or wrong golden value defeats this silently |
| Genuine attestation, semantically wrong result (bad inputs, wrong parameters, quietly-broken logic) | **nothing here** | out of scope for hardware attestation entirely — this needs re-execution (the slow path) or a proof-of-computation approach (e.g. zk-STARK execution-trace proofs) |
| Node with a disclosed CPU vulnerability | TCB status field | this is a *reported signal*, not an automatic reject — the validator's policy has to actually consume it |

Three honest caveats:

- **MRTD is a fingerprint, not an audit.** It proves *which* image booted, bit-for-bit reproducibly. It says nothing about whether that image's code is correct, unless the verifier independently maintains a trusted mapping from "expected job harness" to "expected MRTD value." Attestation without a trustworthy golden-measurement registry just proves "something specific and reproducible ran" — not "the right thing ran."
- **Hardware attestation and computation-correctness proofs answer different questions.** TDX/NRAS prove *provenance*: genuine hardware, genuine measured stack. They prove nothing about whether the computation itself executed correctly step by step. For large training runs, full computation proofs (zk-STARK / ZKML) remain far more expensive than hardware attestation today — which is exactly why proof-of-training systems lean on the hardware path for their fast verification lane, reserving re-execution or proof-of-computation for spot checks.
- **"Attested" is not "audited."** A verifier that only checks `tdx_bound` and `gpu_bound` (binding, no signature chain) looks like the full picture in §2 but provides none of its guarantees. This is the failure mode worth remembering above every other one here.

## 5. Verification Review Checklist

Use this when standing up or reviewing any remote-compute claim verifier, not just this specific TDX + NVIDIA CC pairing:

- [ ] Every claimed value has a **binding check** (it appears in a specific field of a hardware report) *and* a separate **authenticity check** (that report's signature chains to a live-verified vendor root) — never one without the other.
- [ ] TCB/status fields are surfaced as their actual value (`UpToDate` / `OutOfDate` / advisory ID / `Revoked`), not flattened into a single pass/fail before the policy layer sees them.
- [ ] Any self-signed or symmetric-key-signed token from the same SDK/process generating the evidence is explicitly excluded from anything reported as a "signature verified" result.
- [ ] Multi-component claims (multi-GPU, multi-node) check **every** component's binding and signature, not one representative sample.
- [ ] Golden/expected measurement values (MRTD, RTMR, expected MRSIGNER) are sourced from a maintained, trusted registry — not hardcoded once and forgotten as hardware/firmware revisions ship.
- [ ] The design is explicit about what attestation does **not** prove — computational correctness and data/result quality — and states what mechanism (re-execution, sampling, proof-of-computation) covers that gap, if anything does.
