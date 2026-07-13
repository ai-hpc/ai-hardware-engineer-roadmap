# 01 — Intel TDX Attestation Chain

## 1. What TDX Adds to a Multi-Tenant HPC Node

A regular VM on a shared host trusts the hypervisor completely — the host admin, or anyone who compromises the host, can read guest memory. **Intel Trust Domain Extensions (TDX)** removes that assumption: it creates a **Trust Domain (TD)**, a hardware-isolated VM whose memory is encrypted and integrity-protected such that the hypervisor, host OS, and cloud operator cannot read or modify it. On a rented or multi-tenant HPC node, this is the difference between "the operator promises not to look" and "the operator's root access is cryptographically prevented from looking."

TDX alone only gets you confidentiality, though. The other half — **proving to a remote party that a genuine, unmodified TD is what actually ran** — is **remote attestation**, and that's the subject of this file.

## 2. The TDX Module: CPU-Resident Root of Trust for Measurement

When a TD launches, the hardware-isolated **Intel TDX Module** acts as the **Root of Trust for Measurement (RTM)** — the immutable, CPU-package-resident component that computes and stores cryptographic hashes of everything that boots. It maintains:

| Register | Type | What it measures |
|---|---|---|
| **MRTD** | Static, SHA-384 | The TD's initial build: virtual firmware, initialization parameters, initial memory layout. A one-shot "identity fingerprint" of the freshly launched domain. |
| **RTMR[0..3]** | Dynamic, SHA-384, 4 registers | Runtime measurements extended during boot, exactly like TPM PCRs: `new_hash = SHA-384(old_hash ‖ payload)`. Tracks OS kernel, filesystem, and application payloads as they load — later stages layer on top of earlier ones, so tampering with any stage changes every subsequent hash. |
| **REPORTDATA** | Free-form, 64 bytes | Not a measurement — a scratch field the guest can fill with *anything* at report-generation time. This is the hook applications use to bind their own data (a nonce, a public key, a claim hash) into a hardware-signed report. |

## 3. From TDCALL to TD Quote

A tenant workload inside the TD requests a report with a single hardware instruction, and the report is transformed twice before it can leave the machine:

```text
[ Tenant Workload ]                [ Intel TDX Module ]                  [ Quoting Enclave (SGX) ]
        │                                   │                                      │
        │──TDCALL[TDG.MR.REPORT]──────────▶│                                      │
        │                                   │  captures MRTD + RTMR[0..3]         │
        │                                   │  + REPORTDATA (caller-supplied)     │
        │                                   │  authenticates locally with an      │
        │                                   │  HMAC only the CPU package can read │
        │                                   │──local HMAC'd TD Report────────────▶│
        │                                   │                       EVERIFYREPORT2 checks the HMAC
        │                                   │                       replaces it with an asymmetric
        │                                   │                       signature from an Attestation Key
        │                                   │                                      │
        │◀──────────────────────────────────────────── TD Quote ───────────────────┘
```

Two things to internalize here:

- **The TD Report's HMAC never leaves the CPU package** — it's only useful for the local Quoting Enclave to check. A remote verifier can't validate a raw TD Report at all.
- **The TD Quote is the re-signed, exportable object.** The Quoting Enclave (an SGX enclave on the same host) validates the local HMAC via `EVERIFYREPORT2`, then re-signs the report with a hardware-derived **Attestation Key** using ECDSA. This asymmetric signature is what a remote party can actually verify — and it's what the untrusted host routes out to a verifier.

## 4. DCAP Verification: Chaining the Quote Back to Intel's Root CA

A TD Quote's signature alone isn't enough — a verifier also needs to know the signing key itself is legitimate. **DCAP (Data Center Attestation Primitives)** provides that chain:

```text
TD Quote
   ├─ ECDSA-P256 signature over the quote body   ──verified against──▶  Attestation Key
   ├─ Attestation Key certified by                ──────────────────▶  PCK (Platform Certification Key) certificate
   ├─ PCK certificate chains to                     ────────────────▶  Intel PCK Platform CA → Intel Root CA
   ├─ Quoting Enclave identity                       ──────────────▶  MRSIGNER / ISVPRODID match expected values
   └─ Live TCB status from Intel PCS collateral       ────────────▶  "UpToDate" / "OutOfDate" / specific advisory IDs
```

```python
def verify_tdx_quote(quote_bytes):
    result = dcap_qvl.verify(quote_bytes)   # ECDSA sig, PCK chain -> Intel root CA, QE identity
    return {
        "signature_valid": result.signature_valid,
        "chain_valid":     result.chain_valid_to_intel_root,
        "tcb_status":      result.tcb_status,        # e.g. "UpToDate"
        "advisory_ids":    result.advisory_ids,       # e.g. []
    }
```

**Report the TCB status, don't collapse it to a boolean.** `UpToDate`, `OutOfDate`, and a specific advisory ID are different risk levels — an `OutOfDate` platform with a non-exploitable advisory for your threat model may be an acceptable accept, while a `Revoked` platform never should be. A verifier that flattens this into `passed: true/false` takes that policy decision away from whoever consumes the attestation, silently.

This is the same chain the [canonical/tdx](https://github.com/canonical/tdx) open-source reference stack drives via `trustauthority-cli evidence --tdx` / `trustauthority-cli token`, just routed through Intel Tiber Trust Services as an external verifier instead of a local DCAP library — same trust chain, different verifier endpoint. That project is also the fastest path to a working TDX host + guest + attestation setup if you want to reproduce any of this on real Sapphire Rapids / Emerald Rapids / Xeon 6 hardware: host setup, TD image creation, boot, and both local (DCAP) and cloud-brokered (Intel Tiber Trust Services) attestation flows are scripted end to end.

## 5. Binding Application Data via REPORTDATA

REPORTDATA is what turns "this hardware is genuine" into "this hardware genuinely produced *this specific claim*." Any 64-byte value the workload wants attested to gets zero-padded into the field before requesting the report:

```text
application_hash (32 bytes) ──zero-padded──▶ REPORTDATA (64 bytes) ──▶ TDCALL[TDG.MR.REPORT] ──▶ TD Report
```

For example, an HPC job that wants to prove *which exact recipe and output* ran on a rented node computes a digest of its own claim (job spec, environment measurement, output hash) and drops it straight into REPORTDATA before requesting the quote. A verifier then does two independent checks — not one:

```python
def check_tdx_binding(claim_hash, tdx_report):
    return tdx_report.reportdata[:32] == claim_hash    # binding only — see file 03 for why this is not enough alone
```

Binding without the DCAP signature/chain check from §4 is a decorative check — a hand-crafted JSON blob with the right 32 bytes in a field named `reportdata` passes `check_tdx_binding` with no TDX Module ever involved. [File 03](./03-Dual-Vendor-Claim-Binding-and-Verification.md) covers this failure mode and its generalization in depth; treat §4 and §5 here as a matched pair, never one without the other.

## 6. Supported Hardware Quick Reference

| Processor | Code Name | TDX Module Version |
|---|---|---|
| 4th Gen Intel Xeon Scalable | Sapphire Rapids | 1.5.x |
| 5th Gen Intel Xeon Scalable | Emerald Rapids | 1.5.x |
| Intel Xeon 6 (E-Cores) | Sierra Forest | 1.5.x |
| Intel Xeon 6 (P-Cores) | Granite Rapids | 2.0.x |

Check the currently loaded module version with `sudo dmesg | grep -i tdx` — the line reporting `virt/tdx: module initialized` alongside `major_version`/`minor_version`/`build_date` confirms both that TDX initialized and which module build is running.
