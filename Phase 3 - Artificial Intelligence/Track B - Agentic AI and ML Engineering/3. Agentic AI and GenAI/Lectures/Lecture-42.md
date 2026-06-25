# Lecture 42 - Confidential & Verifiable AI Agents: NVIDIA Confidential Computing, zk-STARK Attestation & the PearlChain Stack

**Course:** [AI Agent Development 2026](../Guide.md) | **Previous:** [Lecture 41](Lecture-41.md) | **Next:** [Lab 01](Lab-01-Research-Agent.md)

---

An enterprise agent is only useful if it can touch the data that matters — contracts, ledgers, patient records, supply-chain manifests. That is exactly the data an enterprise cannot afford to hand to a black-box cloud endpoint on trust alone. Today's default architecture asks the customer to believe four unprovable promises: that their data was not copied, that the advertised model actually ran, that the output was not altered, and that someone, somewhere, kept an honest log. For a regulated bank or hospital, "trust us" is not a control.

This lecture is about turning those four promises into **cryptographic guarantees**. The tools exist now: a **confidential computing** layer that runs the agent inside hardware the cloud operator itself cannot read; **attestation** that proves the code and model running are the exact ones you approved; **zero-knowledge proofs** (specifically **zk-STARK**) that prove a computation was performed correctly without revealing its inputs; and a **blockchain** audit layer that makes the record immutable and shareable across mutually-distrusting parties. Stacked together they produce a **Verifiable AI Agent** — one that can act autonomously on sensitive data while emitting proof of privacy, correctness, and auditability.

We build the stack layer by layer using NVIDIA's confidential GPUs and STARK-based verification as the concrete primitives, then assemble them into the reference architecture this lecture is modeled on — **PearlChain**, a confidential-and-verifiable agent runtime for enterprises. (PearlChain is treated here as the integrating platform; the layer technologies below are described from their public specifications.)

---

## Learning Objectives

By the end of this lecture you will be able to:

1. State the **four trust guarantees** an enterprise needs from an AI agent — confidentiality, computational integrity, output integrity, auditability — and map each to a specific cryptographic or hardware control.
2. Explain how **NVIDIA Confidential Computing** (H100/Blackwell) extends a Trusted Execution Environment to the GPU, and why "the cloud operator cannot read VRAM" is now a hardware property, not a policy.
3. Describe **remote attestation** (NRAS + a CPU TEE) and the principle that gates everything: *expected code == running code*, enforced by releasing data only after measurement verification.
4. Contrast **TEE-based** trust with **zk-STARK / ZKML** proof, know why STARKs (transparent, post-quantum) suit multi-party enterprise settings, and reason honestly about today's LLM-scale proving cost.
5. Explain what belongs **on-chain** (hashes, proofs, audit trail, permissions) versus **off-chain** (data, weights, full traces), and why blockchain is the audit layer — never the inference layer.
6. Assemble the layers into the **PearlChain** near-term architecture and the GPU-inference + STARK-attestation pipeline, and articulate the residual threat model.

---

## 1. The Trust Gap in Cloud AI Agents

The conventional enterprise agent is a one-way street into someone else's infrastructure:

```text
   Enterprise Data ──▶ Cloud LLM ──▶ Agent Decision ──▶ Action
        (plaintext)     (opaque)        (unlogged)       (unverified)
```

Four problems fall out of that picture, and each is a compliance blocker, not a nicety:

| Problem | What the enterprise cannot prove | Who is exposed |
|---|---|---|
| Trust the provider | that data in RAM/VRAM was never read or copied | data owner, DPO |
| No cryptographic proof | that the claimed model/code actually ran | risk, audit |
| Difficult auditing | what the agent did, when, on whose behalf | regulator, legal |
| Data privacy | that prompts, retrievals, and outputs stayed confidential | customers, partners |

The fix is to convert each unprovable promise into a checkable guarantee:

| Guarantee | Meaning | Primary mechanism |
|---|---|---|
| **Confidentiality** | nobody — including the operator — sees the data in use | confidential computing (TEE, incl. GPU) |
| **Computational integrity** | the agreed model/code ran, unmodified | attestation; zk proof of execution |
| **Output integrity** | the result was not tampered with in flight | signed/attested output; on-chain commitment |
| **Auditability** | every action is recorded and non-repudiable | blockchain audit log |

The verifiable-agent flow rearranges the one-way street into a loop that produces evidence:

```text
   Enterprise Data ──▶ Confidential Environment (TEE / Secure Enclave)
                              │
                              ▼
                         AI Agent  ──▶  Cryptographic Proof  ──▶  Blockchain
                                                                      │
   verify: which model · which code · when · output unaltered ◀───────┘
            …all without exposing the underlying data.
```

The rest of the lecture is the five layers that make that loop real.

---

## 2. Layer 1 — Confidential Computing with NVIDIA

A **Trusted Execution Environment (TEE)** is a hardware-isolated region whose memory is encrypted and integrity-protected such that *even privileged software* — the hypervisor, the host OS, the cloud operator — cannot read or modify it. CPUs have had this for years: **Intel TDX** and **AMD SEV-SNP** create confidential VMs; **Intel SGX** creates smaller enclaves; **Google Confidential Space** packages it as a service. The problem for AI was always the same: the interesting computation happens on the **GPU**, which historically sat *outside* the trust boundary.

NVIDIA closed that gap. Starting with the **Hopper H100**, the GPU runs in a **Confidential Computing (CC-On) mode** that makes it a first-class member of the TEE:

- **Encrypted VRAM and an encrypted bus.** The H100's DMA engine encrypts data moving between CPU and GPU with **AES-GCM-256**; the confidential VM and the GPU exchange data through encrypted bounce buffers. Data is in plaintext only inside the protected silicon.
- **The operator is locked out.** With CC-On, host-side inspection paths into GPU memory are disabled. A cloud admin with root on the box still cannot read model weights or activations.
- **Measured firmware.** The GPU's firmware and CC state are measured so they can be attested (Layer 2).

> **Performance note.** Because LLM inference is compute- and bandwidth-bound, the relative overhead of CC mode is small for realistic generation workloads (often low single-digit percent); the cost concentrates in many small host↔device transfers, which batched inference already minimizes. Confidential GPU inference is a **production primitive in 2026**, not a research demo.

The **Blackwell** generation extends this from one GPU to a pod: **multi-GPU confidential computing across NVLink/NVSwitch** (so a model sharded over many GPUs stays inside one trust domain) and **Trusted I/O**, with end-to-end CPU+GPU attestation built on Intel TDX and Intel Trust Authority. NVIDIA ships the open **nvTrust** toolkit and SDKs to drive all of this.

> **Hardware lens.** This is where the roadmap's GPU thread meets security: the same H100/Blackwell you tune for throughput (Phase 5 — [MLSys Deep Dives](../../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/7.%20ML%20Systems%20Engineering/MLSys%20Deep%20Dives/README.md)) is also the root of a confidentiality guarantee. A confidential **TensorRT-LLM** server is a normal high-performance server that additionally refuses to reveal what it processed.

---

## 3. Layer 2 — Attestation: "Expected Code == Running Code"

Confidentiality is worthless if you cannot be sure *what* is running inside the enclave. Attestation is the keystone of the whole stack: it lets a remote party verify, cryptographically, that the environment is genuine and runs exactly the code and model they approved — **before** any secret is handed over.

The check an enterprise runs before trusting the agent:

```text
   Enterprise
       │   1. request attestation
       ▼
   Verify enclave/firmware measurement  ─┐
   Verify model hash                      ├─▶  expected == measured ?
   Verify software/agent version          ─┘        │
       │                                      yes ──┴──▶ release data + keys to the enclave
       │                                      no  ─────▶ abort; provision nothing
```

On NVIDIA hardware this is the job of the **NVIDIA Remote Attestation Service (NRAS)**:

- The GPU produces a **signed attestation report** describing its identity, firmware measurements, and CC state.
- A verifier validates NVIDIA's signed token (a JWT) against the **NRAS JWKS** endpoint; a CPU-side attestation (TDX/SEV-SNP, often brokered by **Intel Trust Authority**) covers the confidential VM. Together they attest the *whole* stack — CPU enclave + GPU.
- Crucially, attestation is wired to **secret release**: a key-management/relying service hands the data-decryption key (or the data itself) to the workload **only if** the measurement matches the expected golden value. Substitute the model or tamper with the agent binary and the measurement changes, the check fails, and nothing is released.

```python
# Attestation-gated secret release (conceptual)
def provision_enterprise_data(enclave_quote, gpu_attestation):
    cpu_ok  = verify_tdx_quote(enclave_quote, expected_mrtd=GOLDEN_VM_MEASUREMENT)
    gpu_ok  = verify_nras_token(gpu_attestation, jwks=NRAS_JWKS, cc_required=True)
    model_ok = (gpu_attestation.bound_model_hash == APPROVED_MODEL_HASH)

    if not (cpu_ok and gpu_ok and model_ok):
        raise AttestationError("expected code/model != running code/model")
    return kms.release_data_key(audience=enclave_quote.identity)   # only now
```

This single principle — *bind the data key to a measurement of the exact code and model* — is what turns "trust the provider" into "verify the provider." It is the most important concept in confidential AI.

---

## 4. Layer 3 — Verifiable Execution with zk-STARK (ZKML)

Attestation proves *the right program ran in genuine hardware*. It still asks you to trust the **hardware vendor's root of trust**. **Zero-knowledge proofs** offer a complementary, hardware-independent guarantee: a mathematical proof that a computation was performed correctly, checkable by anyone, revealing nothing about the inputs.

The ZKML claim is exactly the enterprise's wish:

```text
   given   input X, model M  ──▶  output Y
   produce a proof π that "Y = M(X) was computed correctly"
   while revealing  none of:  X · the weights of M · any private data
```

A verifier checks `π` in milliseconds-to-seconds and learns only that the output is correct for the committed model. **zk-STARK** is a particularly good fit for the enterprise setting:

- **Transparent** — no trusted setup ceremony (unlike many SNARKs). For a consortium of mutually-distrusting parties (bank, insurer, regulator) this matters: nobody has to trust a setup nobody can audit.
- **Post-quantum** — security rests only on collision-resistant hashes, not on elliptic-curve assumptions.
- **Scalable** — verification is **polylogarithmic** in the size of the computation; proofs are larger than SNARKs but cheap to check.

The tooling landscape you will actually touch:

| System | Approach | Notes |
|---|---|---|
| **EZKL** | ONNX → Halo2 circuit | most-used ZKML toolchain; SNARK-based |
| **RISC Zero** | inference compiled to a RISC-V **zkVM** (STARK) | general-purpose; "prove arbitrary code"; Bonsai proving service |
| **Cairo / Starknet** (e.g. Orion, Giza) | STARK-native ML ops | on-chain verification on a STARK L2 |
| **Modulus Labs, Nexus, NANOZK** | ML-specialized / layerwise LLM proving | NANOZK (2026) targets layerwise proofs for LLM inference |

> **The honest cost reality (2026).** Proving is not free. For small-to-moderate models, proof generation runs **seconds to a few minutes**; for a **7B-parameter LLM**, an end-to-end proof can still take **hours**. Proving full frontier-LLM inference cryptographically is not yet practical. This is the single most important engineering fact in the space — and it dictates the architecture below.

Because of that cost, the realistic 2026 design is **hybrid TEE + ZK**, not pure ZK:

- Run the **heavy LLM forward pass inside the confidential GPU** (Layer 1–2) — fast, with a hardware attestation.
- Use **zk-STARK proofs for the parts you can afford and most need to prove publicly**: the agent's **execution trace** and control flow, **retrieval/RAG integrity** (the right documents were used), **policy/permission checks**, smaller classifier or routing models, and the **commitment that links** input, model, and output. Research lines like *optimistic TEE-rollups* and lightweight verifiable-inference frameworks formalize exactly this split.

> **Your background fits here.** A STARK-based **execution-trace proof** over a confidential GPU run — `Agent Request → TensorRT-LLM → GPU inference (CC) → execution trace → zk-STARK proof → on-chain verification` — is precisely the niche at the intersection of GPU inference, confidential computing, and STARK cryptography that this stack is built on.

---

## 5. Layer 4 — Blockchain as the Audit & Multi-Party Trust Anchor

A common misconception is that the inference runs "on-chain." It does not — that would be astronomically expensive and would expose the data. **Blockchain is the audit and trust-anchoring layer**, and it stores only small, non-sensitive commitments:

```text
   ON-CHAIN (small, public-ish)            OFF-CHAIN (large, private)
   ───────────────────────────            ──────────────────────────
   agent ID                                enterprise documents
   model hash                              vector DB / embeddings
   inference hash (input/output commit)    model weights
   attestation & zk-proof references       full execution traces
   audit trail (who, when, which policy)   raw prompts & outputs
   permissions / access grants
```

What the ledger buys you:

- **Immutable logs** — the audit record cannot be quietly rewritten after the fact.
- **Non-repudiation** — a signed, timestamped commitment means no party can later deny an action.
- **Multi-party trust** — when several organizations share one agent, **no single party controls the log**.

That last property is the real unlock. Picture a claims agent shared by a **bank, an insurer, and a regulator**: each can independently verify the model hash, the attestation, and the proof references for any decision, and none can tamper with the others' view. The chain is the neutral ground.

---

## 6. The PearlChain Stack — Assembling the Layers

**PearlChain** is the reference architecture that ties Layers 1–5 into one confidential, verifiable agent runtime. The mapping:

| Layer | Role | Concrete tech |
|---|---|---|
| L1 Confidential computing | run the agent where the operator can't look | NVIDIA H100/Blackwell CC, Intel TDX / AMD SEV-SNP |
| L2 Attestation | prove expected code/model == running | NRAS + CPU-TEE quote, attestation-gated key release |
| L3 Verifiable execution | prove correctness without revealing data | zk-STARK over the execution trace (hybrid with TEE) |
| L4 Blockchain | immutable, multi-party audit | on-chain hashes, proofs, audit trail, permissions |
| L5 Agent | plan, retrieve, act on enterprise data | planning · tool use · retrieval · memory · action |

The **realistic near-term** deployment keeps data local and the chain thin:

```text
   Enterprise Documents ─▶ Vector DB ─▶ Confidential Agent ─▶ TEE (NVIDIA CC GPU)
                                                                    │
                                                          execution trace
                                                                    │
                                                            zk-STARK proof
                                                                    │
                                                              Audit Log ─▶ Blockchain (anchor)
```

…explicitly **not** "documents → public blockchain → LLM," which would be both unaffordable and a privacy breach.

The performance-critical inner pipeline — the part that connects this roadmap's GPU work to the crypto — is:

```text
   Agent Request ─▶ TensorRT-LLM ─▶ GPU inference (CC-On) ─▶ execution trace ─▶ zk-STARK proof ─▶ blockchain verification
```

On this substrate the agent does ordinary enterprise work — **contract analysis, financial reporting, supply-chain optimization, internal knowledge search, customer support** — except that every run leaves behind a verifiable, privacy-preserving receipt.

What the enterprise can now **verify without ever exposing the data**: *which model was used, which code executed, when inference happened, and whether the output was altered.*

---

## 7. Threat Model — What Each Layer Defends, What Remains Trusted

Stacking the layers only helps if you can say precisely what each one stops:

| Threat | Defended by | Residual assumption |
|---|---|---|
| Operator / co-tenant reads data in use | L1 confidential computing (encrypted VRAM) | trust in the silicon root of trust |
| Code/model silently swapped | L2 attestation (measurement gate) | golden measurements are correct & current |
| Provider lies about what it computed | L3 zk-STARK proof of execution | proven scope (hybrid: only proven parts) |
| Output altered in transit / log rewritten | L4 on-chain commitment + signatures | chain liveness & key custody |
| One consortium member forges the record | L4 multi-party ledger | quorum / consensus integrity |

Two honest caveats keep this from becoming security theater:

- **TEEs reduce trust; they do not eliminate it.** You still trust the hardware vendor's root of trust and the absence of side channels. zk-STARK removes that assumption **for the parts you actually prove** — which today is not the whole LLM. Be explicit about the boundary.
- **Confidentiality ≠ agent safety.** None of these layers stop a **prompt injection via tool results** or the **lethal trifecta** (private data + untrusted content + an exfiltration channel). A confidential, attested, on-chain-logged agent can still be socially engineered into misusing its access. The agent-level controls from [Lecture 24 — Runtime Discipline](Lecture-24.md), [Lecture 40 — OpenClaw Threat Model](Lecture-40.md), and [MCP Security (MCP course · Lecture 07)](../MCP%20for%20AI%20Agents/Lecture-07.md) are still mandatory. Confidential computing protects the data *from the operator*; it does not protect the agent *from the inputs*.

---

## Key Takeaways

- Enterprises need four **provable** properties from an AI agent — confidentiality, computational integrity, output integrity, auditability — and each maps to a specific control, not a promise.
- **NVIDIA Confidential Computing** (H100 encrypted VRAM + AES-GCM-256 DMA; Blackwell multi-GPU CC) makes "the cloud operator cannot read your data or weights" a hardware guarantee, at small overhead for LLM inference.
- **Attestation is the keystone:** bind the data-decryption key to a measurement of the exact code and model (NRAS + CPU-TEE), so secrets are released only when *expected == running*.
- **zk-STARK / ZKML** adds a hardware-independent proof of correct execution — transparent and post-quantum, ideal for multi-party trust — but **full-LLM proving still costs hours**, so the practical 2026 design is **hybrid TEE + ZK**.
- **Blockchain stores commitments, never inference:** hashes, proofs, audit trail, permissions — delivering immutable, non-repudiable, multi-party-shareable logs.
- The **PearlChain** stack assembles these into a Verifiable AI Agent; the GPU-facing pipeline is `TensorRT-LLM → CC inference → execution trace → zk-STARK proof → on-chain verification`.
- Confidential ≠ safe: keep the agent-level injection/exfiltration defenses from the security lectures.

---

## Exercises

### Exercise 1 — Map the Guarantees

Take one enterprise task (e.g., confidential contract analysis). For each of the four guarantees (confidentiality, computational integrity, output integrity, auditability), name the exact layer and mechanism that delivers it, and the one residual trust assumption that remains. Produce it as a table an auditor could read.

### Exercise 2 — Attestation Gate

Write pseudocode for an attestation-gated inference service: the client sends an encrypted prompt; the server must (a) present a GPU + CPU attestation, (b) prove the bound model hash equals an approved value, and (c) only then receive the decryption key. Specify exactly what fails — and what is *not* leaked — if an attacker swaps the model.

### Exercise 3 — Draw the Hybrid Boundary

For a RAG agent answering on private documents, decide which parts you would run in a TEE and which you would prove with zk-STARK, given that a 7B-LLM proof costs hours. Justify the split by cost and by what each stakeholder most needs to verify. Then state what you would commit on-chain for each request.

### Exercise 4 — Why Not On-Chain Inference?

In two paragraphs, explain to a non-technical executive why the agent's LLM inference does **not** run on the blockchain, what *does* go on-chain, and how the enterprise still gets cryptographic auditability from that arrangement.

---

## Current as of

**June 2026.** NVIDIA Confidential Computing is production on Hopper (H100) and extends to multi-GPU on Blackwell (HGX B200, NVLink/NVSwitch, Trusted I/O); attestation via **NRAS** + a CPU TEE (Intel TDX / AMD SEV-SNP, brokered by Intel Trust Authority). ZKML tooling (EZKL, RISC Zero, Cairo/Starknet, NANOZK) is advancing fast, but **proving full LLM inference remains expensive (hours for a 7B model)**, so hybrid **TEE + zk-STARK** is the realistic enterprise architecture today — verify current proving costs before committing a design. **PearlChain** is presented here as the integrating reference platform; its layer technologies are described from their public specifications, and the platform's specific implementation details should be confirmed against its own documentation. All vendor/spec facts move quickly — treat the numbers as anchors, not constants.

---

*Next: [Lab 01](Lab-01-Research-Agent.md)*
