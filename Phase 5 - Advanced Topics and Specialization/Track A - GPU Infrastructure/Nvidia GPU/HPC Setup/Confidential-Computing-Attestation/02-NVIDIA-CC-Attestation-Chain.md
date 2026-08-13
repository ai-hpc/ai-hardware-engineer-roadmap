# 02 — NVIDIA CC Attestation Chain

## 1. CC-On Mode Recap

The GPUs in this HPC track — H100 ([8x H200 Training & Inference](../8x-H200-Training-Inference/README.md)'s GH100 die, [Blackwell B200](../Blackwell-B200-Qwen-Inference/README.md) — support a **Confidential Computing (CC-On)** mode that extends the CPU-side TEE from [file 01](./01-Intel-TDX-Attestation-Chain.md) all the way onto the accelerator. Three properties make this possible:

- **Encrypted VRAM and an encrypted bus.** The GPU's DMA engine encrypts data moving between CPU and GPU with AES-GCM-256; the confidential VM and GPU exchange data through encrypted bounce buffers. Plaintext exists only inside the protected silicon.
- **The operator is locked out.** With CC-On enabled, host-side inspection paths into GPU memory are disabled — a cloud admin with root on the box still cannot read model weights, activations, or training data resident in VRAM.
- **Measured firmware.** The GPU's firmware and CC state are measured at boot, which is what makes the rest of this file possible: those measurements feed into a signed attestation report.

Blackwell extends this from a single GPU to a pod: multi-GPU confidential computing across NVLink/NVSwitch (so a model sharded across many GPUs — exactly the [Multi-B200 NVL72](../Blackwell-B200-Qwen-Inference/04-Multi-B200-NVL72.md) topology — stays inside one trust domain), plus **Trusted I/O** with end-to-end CPU+GPU attestation built on Intel TDX. NVIDIA ships the open **nvTrust** toolkit to drive this.

## 2. The GPU Attestation Report: EAT

A CC-On GPU produces a signed **Entity Attestation Token (EAT)** — a JWT describing its identity, firmware measurements, and CC state. In practice a verification round produces **more than one token**: one per physical GPU device plus one covering the platform as a whole, since a multi-GPU node has multiple pieces of silicon each needing their own measurement.

```text
GPU 0 EAT (JWT)   ─┐
GPU 1 EAT (JWT)   ─┤── all signed independently by NRAS, verified independently
Platform EAT (JWT) ─┘
```

## 3. Binding a Claim: `eat_nonce`

Just as Intel TDX leaves REPORTDATA free for application data ([file 01, §5](./01-Intel-TDX-Attestation-Chain.md#5-binding-application-data-via-reportdata)), the NVIDIA EAT format has an equivalent slot: `eat_nonce`. Whatever hash the application wants attested goes there before requesting the token:

```text
application_hash ──▶ eat_nonce field ──▶ GPU attestation request ──▶ signed EAT (JWT)
```

A verifier's binding check mirrors the TDX one, and has the exact same limitation:

```python
def check_gpu_binding(claim_hash, gpu_eat):
    return gpu_eat.eat_nonce == claim_hash    # binding only — see file 03 for why this alone is forgeable
```

## 4. Authenticity: NRAS Signature + JWKS Key Resolution

`eat_bound: true` only proves *some* JWT carries the right nonce — nothing yet says who signed it. The **NVIDIA Remote Attestation Service (NRAS)** closes that gap: a verifier resolves the actual signing key from NVIDIA's published JWKS endpoint and checks the signature against it, not against anything the prover supplied.

```text
   EAT (JWT)
      │
      ├─ header.kid ──▶ resolve signing key from NVIDIA's published JWKS endpoint
      ├─ signature   ──▶ ES384 verify against the resolved key
      ├─ iss         ──▶ must equal https://nras.attestation.nvidia.com
      └─ exp         ──▶ not expired
```

```python
def verify_gpu_token(eat_jwt, jwks):
    header = jwt.get_unverified_header(eat_jwt)
    key = jwks.resolve(header["kid"])
    claims = jwt.decode(eat_jwt, key=key, algorithms=["ES384"])
    assert claims["iss"] == "https://nras.attestation.nvidia.com"
    assert claims["exp"] > now()
    return claims
```

The `kid` (key ID) in the JWT header is what lets the verifier fetch the *correct* key from a JWKS endpoint that rotates keys over time — never hardcode a single expected key; resolve it per-token.

## 5. The Symmetric-Key Self-Attestation Trap

GPU attestation SDKs commonly hand back an additional convenience field: an "overall" JWT that's **HS256-signed with a key the SDK itself holds**. This is the single most common mistake in GPU-attestation verifier code, because it's also the easiest field to reach for:

```python
# WRONG — this only proves the SDK agrees with itself
jwt.decode(overall_token, key=sdk_local_secret, algorithms=["HS256"])
```

A symmetric key controlled by the same process generating the evidence cannot serve as third-party proof — verifying it just confirms internal consistency, not that NVIDIA's hardware vouches for anything. The only tokens that count as evidence are the **NRAS-signed, ES384, asymmetric** per-device and platform EATs from §4. A correct verifier explicitly excludes the SDK-local HS256 token from anything it reports as a signature check, even though pulling `overall.passed` is the path of least resistance.

## 6. Composed Verifier Sketch

Putting §3–§5 together, a single GPU claim check looks like:

```python
def verify_gpu_claim(claim_hash, gpu_eat_tokens, jwks):
    results = []
    for eat_jwt in gpu_eat_tokens:          # per-device + platform, NRAS-signed only
        claims = verify_gpu_token(eat_jwt, jwks)          # §4 — authenticity
        bound = (claims["eat_nonce"] == claim_hash)        # §3 — binding
        results.append({"bound": bound, "authentic": True})
    return all(r["bound"] and r["authentic"] for r in results)
```

Note both checks are required on **every** token, not just one representative token — a multi-GPU claim is only as strong as its weakest verified device.
