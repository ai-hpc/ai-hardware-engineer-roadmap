# Confidential Computing & Hardware Attestation — Deep Dive

Most of this HPC track assumes you control the whole stack: your cluster, your NVLink fabric, your Slurm queue. That assumption breaks the moment compute is **rented, shared, or run on a node you don't physically control** — GPU-cloud capacity, a multi-tenant HPC cluster, or a decentralized compute network where anonymous operators contribute nodes. In that world a new question shows up before any performance question does: **how do you know a remote node actually ran the workload you think it ran, on real hardware, without tampering?**

That's what this deep dive covers: **hardware-rooted remote attestation** for CPU (Intel TDX) and GPU (NVIDIA Confidential Computing), and — because a single-vendor check is a single point of forgery — how to **bind one claim into both hardware roots at once** so a verifier doesn't have to trust either vendor's silicon alone.

## Why This Belongs in HPC Infrastructure

Every other deep dive in this section (8x H200, Blackwell B200, GPUDirect Storage, NCCL) optimizes a cluster you trust. This one answers the orthogonal question that shows up as soon as a node leaves your physical custody:

```text
Trusted cluster (rest of HPC Setup):        Untrusted/rented/decentralized node (this deep dive):
  "how fast can this run?"                    "did this actually run, on real hardware,
                                                unmodified, and can I prove it without
                                                re-executing the whole job?"
```

Concretely, this matters whenever a training or inference job runs somewhere you can't just SSH in and check: **GPU-cloud rental** (you don't trust the operator's host), **multi-tenant on-prem clusters** (co-tenants shouldn't read your weights or data), and **decentralized/proof-of-training compute networks** (anonymous nodes submit training claims that must be verified without redoing the training).

## Reference Hardware

| Component | What's needed | Notes |
|---|---|---|
| **CPU** | 4th/5th-gen Intel Xeon Scalable (Sapphire Rapids, Emerald Rapids), Xeon 6 (Sierra Forest, Granite Rapids) with Intel TDX | TDX Module v1.5.x (SPR/EMR) or v2.0.x (Granite Rapids) enabled in BIOS |
| **GPU** | NVIDIA H100 (Hopper) or Blackwell (B200/HGX), CC-On mode | Same silicon covered in [8x H200 Training & Inference](../8x-H200-Training-Inference/README.md) and [Blackwell B200 Qwen Inference](../Blackwell-B200-Qwen-Inference/README.md) — this deep dive adds the security layer on top |
| **Attestation tooling** | Intel DCAP (`dcap-qvl` or equivalent) or Intel Tiber Trust Services; NVIDIA NRAS + nvTrust | See [canonical/tdx](https://github.com/canonical/tdx) for a full open-source Intel TDX host/guest reference stack |

## Topic Index

| # | Topic | Description |
|---|---|---|
| 01 | [Intel TDX Attestation Chain](./01-Intel-TDX-Attestation-Chain.md) | TDX Module as Root of Trust for Measurement, MRTD/RTMR/REPORTDATA, TD Report → Quoting Enclave → TD Quote, DCAP verification against Intel's root CA |
| 02 | [NVIDIA CC Attestation Chain](./02-NVIDIA-CC-Attestation-Chain.md) | CC-On mode GPU EAT tokens, `eat_nonce` binding, NRAS signature verification via JWKS, the symmetric-key self-attestation trap |
| 03 | [Dual-Vendor Claim Binding & Verification Review](./03-Dual-Vendor-Claim-Binding-and-Verification.md) | Composing both chains around one claim hash, binding-vs-authenticity as a generalizable code-review discipline, threat model, verification checklist |
| 04 | [Confidential Inference: Model Identity & the End-to-End Protocol](./04-Confidential-Inference-Protocol.md) | What "sealed" does and doesn't mean, proving *which model* (not just which bytes) is loaded, the publisher-signature gap, and a full client-verifies-before-sending confidential inference protocol |

## Quick Navigation

- **Setting up TDX on a host/guest for the first time?** → [01 — Intel TDX Attestation Chain](./01-Intel-TDX-Attestation-Chain.md)
- **Verifying a GPU's CC-mode attestation token?** → [02 — NVIDIA CC Attestation Chain](./02-NVIDIA-CC-Attestation-Chain.md)
- **Designing or reviewing a proof-of-training / remote-compute verifier?** → [03 — Dual-Vendor Claim Binding & Verification Review](./03-Dual-Vendor-Claim-Binding-and-Verification.md)
- **Building or evaluating a "confidential AI" / sealed-model inference API?** → [04 — Confidential Inference: Model Identity & the End-to-End Protocol](./04-Confidential-Inference-Protocol.md)
