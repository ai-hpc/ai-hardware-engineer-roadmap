# Lab: verify a signed artifact and its deployment policy

A checksum can detect changed bytes when its expected value is trusted. A signature can authenticate a manifest under a trusted public key. Neither alone decides whether a signed model belongs on this device or whether it is an unacceptable rollback. This lab makes those checks separate and executable.

Allow 30–45 minutes. You need Python 3.10+ and the OpenSSL command-line tool for the signature exercise. The policy/serialization tests use only Python. The lab creates local files, generates a disposable test key, and performs no installation, flash operation, fuse programming, or network request.

From the repository root:

```bash
python3 'Phase 4 - Track B - Nvidia Jetson/1. Nvidia Jetson Platform/Orin-Nano-Security/labs/signed_update.py' --output-dir /tmp/course-signed-update
python3 -m unittest discover -s 'Phase 4 - Track B - Nvidia Jetson/1. Nvidia Jetson Platform/Orin-Nano-Security/labs' -v
```

Choose a new output directory when retaining comparisons. The example uses an ephemeral RSA-2048 key and SHA-256 signatures through OpenSSL. This is a generic local artifact exercise, **not a Jetson boot-key format or provisioning recipe**. The private key is generated inside a temporary directory and removed when the demo ends; only the public key and verification artifacts are retained.

## 1. Identify the trusted inputs

The verifier receives a trusted public key, expected target `course-simulator-v1`, and minimum allowed version 2. The incoming bundle contains an artifact, manifest, and signature. It cannot choose its own trust anchor or minimum acceptable version.

The manifest binds:

```json
{"schema":1,"target":"course-simulator-v1","version":3,"size_bytes":40,"sha256":"5540ee9fbcbc6562c0b707bb5aff85f52a2d0f1921a2798b78db46102cf6fb59"}
```

This display is for reading. The actual signature covers the exact bytes produced by the lab's `canonical` function, which sorts keys, removes optional whitespace, and encodes ASCII. The format is a deliberately narrow course schema, not a claim to implement a general JSON canonicalization standard.

The [verifier](signed_update.py) rejects duplicate/unknown fields, unsupported schema, invalid types, invalid digest spelling, excessive lengths, and noncanonical serialization. Python booleans are not accepted as integer versions even though `bool` is an `int` subclass.

## 2. Follow the verification order

1. Parse a bounded manifest under the strict schema.
2. Verify its signature using the trusted public key.
3. Check the expected device/application target.
4. Reject versions below the trusted minimum.
5. Check the artifact size and SHA-256 against the authenticated manifest.

The implementation checks captured byte strings so the content used for verification is the same content used for the hash/policy decision. A real installer must also preserve that relationship when writing to storage; verifying a path and later reopening a changed file introduces a time-of-check/time-of-use problem.

The native commands use OpenSSL's [key generation](https://docs.openssl.org/3.0/man1/openssl-genpkey/) and [digest signature verification](https://docs.openssl.org/3.0/man1/openssl-dgst/) interfaces. A nonzero verification result is a rejection, not a reason to continue with a warning.

## 3. Explain the negative tests

| Change | Expected reason for rejection |
|---|---|
| Modify signed metadata without a new signature | Signature mismatch |
| Modify payload but keep the signed manifest | Digest or size mismatch |
| Truncate the signature | Signature verification failure |
| Use a different public key | Wrong trust anchor |
| Supply a correctly signed artifact for another target | Target mismatch |
| Raise the trusted minimum to 4 while artifact version is 3 | Rollback policy |
| Include two `version` fields or a boolean version | Ambiguous/invalid schema |

The eleven tests executed successfully with Python 3.12.3 and OpenSSL 3.0.13 on **2026-10-18**. Six tests perform actual signing or verification. The saved [example result](example-result.json) and [validation record](validation.json) identify the run; signatures and test keys vary on each execution.

## 4. Understand what this lab does not prove

The minimum version is a trusted function argument. It is not stored in tamper-resistant hardware. An attacker who can replace both the verifier and its trusted inputs defeats this local demonstration. Boot-chain integrity, key provisioning, protected state, revocation, and recovery require additional mechanisms.

An accepted version is not committed automatically. A real update needs compatible images/configuration, an inactive-slot write or equivalent transaction, verification, a bounded trial boot, a meaningful health check, and a policy for when the version floor advances. Advancing it too early can prevent recovery to the previous working image.

A valid signature also says nothing about model quality. A correctly signed model can use the wrong normalization, violate a robot's action limits, or fail under a sensor shift. Keep the learning/evaluation and deployment checks together.

**Completion:** run all tests, save a successful report, demonstrate one rejected tamper and one correctly signed but policy-rejected artifact, and explain where the trust anchor and version floor would live in a real design. Continue with the [security architecture guide](../Guide.md).
