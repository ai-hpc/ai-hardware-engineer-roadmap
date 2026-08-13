# 04 — Confidential Inference: Model Identity & the End-to-End Protocol

Files [01](./01-Intel-TDX-Attestation-Chain.md)–[03](./03-Dual-Vendor-Claim-Binding-and-Verification.md) attest that **a genuine node ran a genuine, measured software stack**. This file applies that machinery to a specific, high-stakes case: a client calling a "confidential" or "sealed" LLM API and wanting to know, before sending a single prompt, that (1) the server is a real TEE, (2) the model is *exactly* the weights it claims to be, and (3) the prompt/response never exist in plaintext outside that TEE. Each of those is a separate property, and marketing copy that says "sealed" or "TEE-hosted" tends to bundle them into one word when they aren't.

## 1. What "Sealed" Actually Means

"Sealed" means the workload runs inside a TEE — nothing more, nothing less. It says nothing by itself about the network path or about which model is loaded.

```text
Normal VM                                   Sealed VM (Intel TDX Trust Domain)
┌────────────────────────┐                  ╔════════════════════════╗
│ Linux VM                │                  ║ Intel TDX Trust Domain ║
│  Model                  │                  ║  Linux                ║
│  Inference server       │                  ║  Model                ║
│  Memory                 │                  ║  Inference server     ║
└────────────────────────┘                  ║  Memory               ║
                                              ╚════════════════════════╝
Host OS / Hypervisor / Cloud operator        Host OS / Hypervisor / Cloud operator
  → can read guest memory                      → memory encrypted + integrity-protected
                                                → cannot read or modify guest memory
```

This is exactly the MRTD/RTMR isolation model from [file 01 §2](./01-Intel-TDX-Attestation-Chain.md#2-the-tdx-module-cpu-resident-root-of-trust-for-measurement) — "sealed" is the plain-English name for the same TD boundary. What it covers and doesn't:

| In scope for "sealed" (TEE isolation) | Out of scope — separate mechanisms required |
|---|---|
| Memory encrypted against the hypervisor/host OS | Network transport encryption |
| Isolation from co-tenants on the same host | API authentication of the calling client |
| Cloud operator cannot dump/inspect guest RAM | Verification of *which* model is loaded |
| Hardware-enforced (CPU package), not a policy promise | End-to-end encryption of prompts/responses tied to the attestation |

A CC-mode GPU or TDX host being genuinely sealed and a client being able to *verify* any of the right-hand column are independent claims. Conflating them is the single most common mistake in "confidential AI" product claims.

## 2. Two Encryption Layers That Don't Substitute for Each Other

A request to a sealed inference endpoint crosses two distinct protected boundaries, and each needs its own check:

```text
Layer 1 — TLS (transport)                  Layer 2 — TDX (execution)
  Client ──HTTPS──▶ Server                   Server ──▶ TDX Trust Domain ──▶ Model

  Protects the request in transit            Protects the request while it's being computed on
  over the public Internet.                  Standard for any HTTPS API — proves nothing about
  Says nothing about what happens             what runs inside the box once the TLS session
  once the bytes reach the server.            terminates.
```

TLS terminates at the server's edge — a plain-old reverse proxy in front of a non-confidential VM can offer TLS with zero TEE involvement. TDX protects what happens after that termination. A client that only checked "does this API use HTTPS" verified layer 1 and learned nothing about layer 2. Section 8 covers the protocol that actually chains the two together.

## 3. Does Attestation Prove *Which Model* Is Loaded?

Not by itself, and this is the same limitation file 03 §4 flags for the golden-measurement registry generally, now applied specifically to model weights.

TDX measures bytes, not semantics. Everything that loads into the TD — bootloader, kernel, initrd, filesystem, inference server, model weights — gets folded into RTMR via the same extend operation from [file 01 §2](./01-Intel-TDX-Attestation-Chain.md#2-the-tdx-module-cpu-resident-root-of-trust-for-measurement):

```text
RTMR_new = SHA-384( RTMR_old ‖ component )

  bootloader ─▶ kernel ─▶ initrd ─▶ filesystem ─▶ inference_server ─▶ weights.bin
       (each stage extends the running hash; changing any stage changes every hash after it)
```

The resulting quote proves: **"this exact measured stack, including this exact weights.bin, is what launched."** It does not inherently know that measurement corresponds to a model called "Qwen3-32B" — that mapping is external:

```text
Measurement  ABC123...  =  Qwen3-32B, FP16, published 2026-03-01     (maintained registry, not part of TDX)
```

Flip one byte of the weights and the chain changes shape entirely:

```text
weights.bin (unmodified)  ──▶  RTMR extend  ──▶  Measurement ABC123...  ──▶  known-good, attestation succeeds
weights.bin (1 byte swapped) ─▶ RTMR extend  ──▶  Measurement XYZ987...  ──▶  not in registry, attestation fails
```

So TDX **does** prove weight integrity and detects any substitution the instant it happens — but only relative to whatever measurement the verifier's registry says is "the real Qwen3-32B." The strength of "this is Qwen" is entirely a function of how that registry got its ABC123 → Qwen3-32B mapping in the first place, which is section 6's problem.

## 4. The Gap: A Modified Model Registered as Legitimate

Suppose the server operator — not an external attacker, the operator itself — builds a modified inference stack that quietly logs every prompt before serving it:

```text
Qwen3-32B (official)  ──modify──▶  Qwen3-32B + prompt exfiltration
```

If the operator computes this modified stack's measurement and registers *that* hash as the expected value, attestation succeeds every time: the quote correctly reports "the measured bytes match the registry," because the registry itself was poisoned at the source. This is precisely the caveat from [file 03 §4](./03-Dual-Vendor-Claim-Binding-and-Verification.md#4-threat-model--whats-proven-what-remains-trusted) — *"a stale or wrong golden value defeats this silently"* — except here the golden value was never right to begin with, because the party setting it and the party being verified are the same entity.

Closing this gap needs a trust root that is independent of the server operator:

```text
Attestation (proves: these exact bytes launched)
        +
Signed model manifest (weight_hash, model_name, version)
        +
Publisher's signature over that manifest (Alibaba/Qwen team's key, not the server operator's)
        │
        ▼
Now the client knows: not just "unmodified since launch," but "this is the
model the publisher actually released" — a claim the hosting operator cannot forge alone.
```

This is structurally the same pattern as DCAP chaining a TD Quote to Intel's root CA ([file 01 §4](./01-Intel-TDX-Attestation-Chain.md#4-dcap-verification-chaining-the-quote-back-to-intels-root-ca)) or NRAS chaining a GPU EAT to NVIDIA's signing key ([file 02](./02-NVIDIA-CC-Attestation-Chain.md)) — a third independent signer, so forging the claim requires compromising the publisher's key too, not just controlling the box the model runs on.

## 5. Three Independent Trust Roots, One Session

Composing everything from files 01–03 plus the publisher chain from section 4 gives three vendors that all have to agree, anchored to one client session:

```text
                              client session
                    ┌───────────────┼───────────────┐
                    ▼               ▼                ▼
            Intel root CA    NVIDIA root (NRAS)   Publisher signing key
         (CPU/TD genuine,   (GPU genuine, CC-On   (weight_hash matches
          stack measured)      mode, EAT valid)     the official release)
```

Each column defends a different forgery:

| Property | Provided by | What fails without it |
|---|---|---|
| Verified server (genuine TDX TD) | DCAP → Intel root CA ([file 01 §4](./01-Intel-TDX-Attestation-Chain.md#4-dcap-verification-chaining-the-quote-back-to-intels-root-ca)) | Attacker runs a normal VM and lies about it being a TD |
| Verified GPU (genuine CC-mode silicon) | NRAS → NVIDIA root ([file 02](./02-NVIDIA-CC-Attestation-Chain.md)) | Attacker runs on non-CC GPUs and self-signs an EAT |
| Encrypted transport | TLS / mTLS | Passive network eavesdropping of prompts/responses |
| Encrypted execution (RAM never plaintext to the host) | TDX memory encryption | A privileged host admin dumps guest RAM and reads prompts |
| Model identity (exact official weights) | Publisher-signed manifest (§4) | Operator swaps in a modified model and registers its own hash as "expected" |
| Session bound to all of the above | Protocol design (§8), not any single vendor | A valid attestation from *a* request gets replayed to authorize a *different*, unattested one |

No single row is sufficient alone — this table is the model-serving-specific instance of the "binding vs. authenticity" discipline from [file 03 §3](./03-Dual-Vendor-Claim-Binding-and-Verification.md#3-binding-vs-authenticity--the-core-review-discipline): each property needs both something bound in and an independent signature chain validating it.

## 6. The Confidential Inference Protocol

Putting all three trust roots and both encryption layers into one client-driven flow — the client verifies before it ever sends a prompt, not after:

```text
Client                                                      Server (TDX TD + CC-mode GPU)
  │
  │ 1. Request attestation bundle
  │───────────────────────────────────────────────────────▶│
  │                                                          │  generates TD Quote + GPU EAT,
  │                                                          │  includes signed model manifest
  │◀───────────────────────────────────────────────────────│
  │
  │ 2. Verify TD Quote → Intel root CA           (file 01 §4)
  │ 3. Verify GPU EAT → NVIDIA NRAS/JWKS          (file 02)
  │ 4. Verify weight_hash in measurement matches
  │    the signed model manifest                  (§4 above)
  │ 5. Verify manifest signature → publisher key   (§4 above)
  │
  │    ── only proceed past this line if 2–5 all pass ──
  │
  │ 6. Generate ephemeral session key, encrypt to a
  │    public key bound inside the attested TD
  │───────────────────────────────────────────────────────▶│
  │ 7. Encrypt prompt under the session key                 │
  │───────────────────────────────────────────────────────▶│  decrypts only inside the
  │                                                          │  TDX TD; plaintext prompt
  │                                                          │  never exists outside it
  │                                                          │  runs inference
  │ 8. Encrypted response                                    │  encrypts response under
  │◀───────────────────────────────────────────────────────│  the same session key
  │
  │ 9. Client decrypts locally
```

What this buys over "the API uses HTTPS and the box happens to be a TDX host":

- **Verification precedes trust, not the reverse.** Steps 2–5 gate step 6 — no prompt is ever encrypted toward a server that failed any check, unlike a bare TLS connection where the client has already committed data before learning anything about the execution environment.
- **The session key is bound to the attested identity**, not just to the TLS certificate. TLS alone proves "I'm talking to whoever holds this cert" — it does not prove that endpoint is a genuine TD running the genuine model. Encrypting to a key generated *inside* the attested TD (vs. a key the server process could generate anywhere) ties the two together.
- **Model substitution and prompt exfiltration are both covered**, not just one. §4's publisher-signature chain stops a silently-modified model from passing; TDX memory encryption stops the host operator from reading prompts even with root access.

## 7. What This Adds Beyond Files 01–03

Files 01–03 answer "did a genuine node, with a genuine GPU, run a genuine measured stack" — sufficient for proof-of-training or proof-of-execution claims where the verifier checks a claim *after the fact*. This file's protocol is stronger and narrower: it's for the case where a client needs to verify *before sending any data*, and where the thing being protected (a user's prompt) is exactly the payload flowing through the channel being attested. The publisher-signature chain in §4 is the one genuinely new trust root — it doesn't come from Intel or NVIDIA at all, and a design that skips it inherits every risk of "attested, but not audited" from [file 03 §4](./03-Dual-Vendor-Claim-Binding-and-Verification.md#4-threat-model--whats-proven-what-remains-trusted), just aimed at the model weights instead of the training claim.
