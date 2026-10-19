# 02 — NVIDIA Evidence and Attestation Results

**Learning target:** distinguish a GPU's signed evidence from a verifier's signed appraisal, and check an entire device bundle.

## 1. GPU protection depends on mode and topology

Supported confidential GPU configurations restrict host inspection and establish protected data paths between the confidential VM and devices. Do not summarize this as “all VRAM is encrypted” or “plaintext exists only in silicon”; those statements obscure the documented threat boundary and application-controlled plaintext inside the guest.

NVIDIA's August 2025 v1.3 white paper distinguishes encrypted CPU/GPU bounce buffers, compatible Blackwell trusted-I/O paths, Hopper Protected PCIe and Blackwell multi-GPU protection. Its Hopper Protected PCIe configuration leaves NVLink traffic unencrypted; its Blackwell multi-GPU configuration encrypts that path and describes up to eight GPUs per CVM. These are specific configurations, not proof that any NVL72 rack forms one confidential domain. Recheck current support for the exact platform. [NVIDIA Secure AI white paper](https://docs.nvidia.com/nvidia-secure-ai-with-blackwell-and-hopper-gpus-whitepaper.pdf).

A genuine GPU can be in a mode unsuitable for secrets. Development/debug permissions, firmware/driver compatibility, device assignment and fabric membership are policy inputs. Availability and arbitrary malicious behavior inside the trusted guest remain separate concerns.

## 2. Evidence is different from a result token

```text
GPU identity certificate + signed device report + challenge
             │
             ▼
Verifier checks certificates, report signature, nonce,
reference measurements, revocation and required device state
             │
             ▼
Signed attestation result → relying party applies its policy
```

The GPU does not itself emit an NRAS-signed JWT. NRAS is a remote verifier that appraises device evidence and issues signed results. A local verifier is another valid architecture when the relying party trusts that verifier and authenticates its output appropriately. [NRAS introduction](https://docs.nvidia.com/attestation/cloud-services/latest/nras/nras_introduction.html), [RATS architecture](https://www.rfc-editor.org/rfc/rfc9334).

The current NRAS detached-EAT format includes an overall JWT and device submodule JWTs. The overall token binds submodules through digests. Validate those digest references and the exact required device inventory; independent valid signatures on an arbitrary collection of tokens are insufficient. An empty list must never pass merely because `all([])` evaluates to true. [NVAT 2.0.0 verification API](https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/2.0.0/api/group__attestation__result__verification.html).

## 3. Current tooling and a source-checked command

NVIDIA's Python `nv_attestation_sdk` ended support on **2026-09-15**. NVAT replaces it with C/C++ APIs and `nvattest`; policy handling uses Rego. GPU ready-state management is not automatically performed by the new SDK. [Migration guide](https://docs.nvidia.com/attestation/attestation-client-tools-sdk/latest/migration_guide.html).

For a compatible deployment with NVAT installed, this is a **source-checked template, not a hardware run performed in this course**:

```bash
# EAT_FILE is the detached-EAT bundle received for this pending request.
# EXPECTED_NONCE_HEX is stored by the relying party before collection.
nvattest verify-token \
  --token-file "$EAT_FILE" \
  --nras-url https://nras.attestation.nvidia.com \
  --nonce "$EXPECTED_NONCE_HEX" \
  --relying-party-policy ./policy.rego \
  --format json
```

You must supply a policy written for your deployment and the returned claims schema; this course does not ship a pretend production policy. Exit status and authenticated claims must both be handled. The CLI and C API support an optional expected nonce: **omitting it disables that request-binding comparison**. The application still owns challenge expiry, audience/session binding and replay state. [Command reference](https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/2.0.0/sdk-cli/command-reference.html).

## 4. Trust roots and complete JWT checks

Configure the trusted issuer/JWKS origin in relying-party configuration. An attacker-supplied `iss`, `jku`, URL or public key must not choose the trust root. The current NRAS profile uses ES384; enforce the expected algorithm, key type, issuer, signature and required time claims. Resolve `kid` only within the trusted issuer's keys; key caching/rotation needs bounded refresh and failure behavior.

Require the profile's freshness claims and an application maximum age. Clock-skew tolerance is an explicit policy choice. Do not assume a library requires every time claim simply because it validates one when present. In particular, the current EAR verification API validates time claims when supplied but does not require them, and authentication alone does not validate EAR appraisal semantics. [NVAT verification API](https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/2.0.0/api/group__attestation__result__verification.html).

A local HS256 wrapper signed with an attester-controlled secret does not establish independent trust. But **“overall” does not mean HS256**: the current NRAS overall token is part of its asymmetrically signed result. Nor does “only NVIDIA may sign” describe every valid architecture; an explicitly trusted local appraisal service can sign its own results. Identify the signer, authority and validated claims, not merely the token's label.

## 5. Experimental formats need separate policies

NVAT 2.0.0 also documents a CoRIM/CMW/EAR workflow as experimental. It has different evidence/result schemas from the legacy TCG RIM detached-EAT path. An unsigned EAR is a test artifact, and an authentic signed EAR may report a warning or rejection. Pin the workflow and apply its result policy; do not feed an EAR into a detached-EAT parser or treat authentication as an affirmative appraisal. [Current workflow reference](https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/2.0.0/sdk-cli/command-reference.html).

## Worked review

Expected inventory is `{gpu-A, gpu-B}`. Both returned tokens have valid signatures and the current nonce, but both describe `gpu-A`. **Reject:** one device is duplicated and `gpu-B` is missing. If an overall token references a different digest for `gpu-B`, **reject before policy acceptance:** the bundle has been substituted. If every check passes but the application's CPU/GPU channel is not authenticated, **do not infer attachment from nonce equality**; lesson 03 explains why.

Next: [compose evidence around a session](03-Dual-Vendor-Claim-Binding-and-Verification.md).
