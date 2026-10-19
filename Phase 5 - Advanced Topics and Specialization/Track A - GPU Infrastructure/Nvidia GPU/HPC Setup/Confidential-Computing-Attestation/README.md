# Confidential Computing and Attestation

**Outcome:** explain why a remote client should release a secret to a particular workload, and identify evidence that would make the client refuse. You will distinguish protected execution, authenticated evidence, freshness, application identity and a protected client channel.

A confidential VM protects private memory against specified privileged-host threats. Attestation helps a verifier assess the platform and measured software. Neither mechanism alone proves that an inference used the advertised model, that a training result is correct, or that an application never logs a prompt.

**Review date: 2026-10-18.** The current NVIDIA path in this course is NVAT C++ SDK documentation **2.0.0**, source release **2026.10.02**. NVIDIA's older Python SDK ended support on **2026-09-15**. The CPU exercise uses real Ed25519 signatures and locally generated teaching keys; it produces **no Intel or NVIDIA attestation**. [NVIDIA migration guide](https://docs.nvidia.com/attestation/attestation-client-tools-sdk/latest/migration_guide.html).

## Start here

You need hashes, public/private keys, basic TLS and the distinction between a VM and its host. No GPU or cloud account is needed for the exercise. Budget about two hours for the reading and CPU lab, then a separate deployment project if compatible hardware is available.

Before reading, answer: can a valid signature contain a false application claim? **Yes.** A signature authenticates bytes and a signer; its meaning depends on what the signer checks. Can an unexpired token be replayed? **Yes.** Time validity alone does not bind it to your pending request.

| Order | Lesson | Evidence you produce |
|---|---|---|
| 1 | [Intel TDX chain](01-Intel-TDX-Attestation-Chain.md) | Measurement/REPORTDATA distinction and quote-verification decision |
| 2 | [NVIDIA chain](02-NVIDIA-CC-Attestation-Chain.md) | Device evidence versus verifier result; complete bundle requirements |
| 3 | [Composition](03-Dual-Vendor-Claim-Binding-and-Verification.md) | Fresh session transcript, identity policy and attack analysis |
| 4 | [Confidential inference](04-Confidential-Inference-Protocol.md) | Key release and model lifecycle design |
| 5 | [CPU cryptographic binding lab](CPU-Binding-Lab.md) | Real signature checks, successful session and rejected attacks |
| Reference | [Sources and execution limits](Source-Evidence.md) | Versioned sources, hashes and actual test record |

## Hardware is a deployment contract

For a real system, record exact CPU SKU, BIOS, TDX module, guest kernel, quote service/DCAP, GPU SKU, firmware, driver, CC mode, topology and verifier versions. A processor generation name or a GPU architecture name is insufficient evidence of an enabled, supported confidential configuration. Use the [Intel enabling guide](https://cc-enabling.trustedservices.intel.com/intel-tdx-enabling-guide/) and [NVIDIA deployment documentation](https://docs.nvidia.com/confidential-computing/).

Do not extrapolate a single-GPU configuration to arbitrary NVL72 racks. A system can have many authentic devices while its channel assignment or fabric policy remains wrong. The [B200 inference course](../Blackwell-B200-Qwen-Inference/README.md) teaches capacity/topology; this course adds explicit trust boundaries.

## Completion evidence

Submit a diagram showing the client, attester, verifier, relying party, trust roots and plaintext boundaries. Include the lab report, the first rejecting check for each attack, and two limitations that remain after every test passes. For a real deployment, add raw evidence, authenticated verification results, collateral age and an exact policy revision; never substitute the CPU fixture for hardware evidence.

An independent transfer exercise is complete when a second allowed GPU can be added without accepting a missing, duplicated or unrelated device. Explain why a shared nonce alone cannot establish a physical CPU/GPU connection. Worked answers are in the lab and lessons.
