# Lecture 42 — Confidential Agents and Verifiable Computation

**Course:** [AI Agent Development 2026](../Guide.md) | **Previous:** [Lecture 41](Lecture-41.md) | **Next:** [Lab 01](Lab-01-Research-Agent.md)

**Outcome:** describe exactly what a signature, hardware attestation, computation proof and audit commitment establish. Each protects a different claim; stacking them does not automatically create a private, correct and safe agent.

## 1. Start with the claim

Suppose a robotics support agent analyzes private logs. The client wants to know which service received the logs, which model artifacts were authorized, which tool actions occurred and whether the returned bytes changed in transit.

| Mechanism | Useful claim | Boundary |
|---|---|---|
| Authenticated encrypted channel | Bytes reach the intended endpoint confidentially/integrally | The endpoint can still disclose plaintext |
| Confidential execution | Platform protects private state against specified host threats | Trusted guest software can leak its own data |
| Hardware attestation | Authentic evidence about measured state and supplied binding data | It does not automatically measure loaded weights or prove inference correctness |
| Computation proof | A specific encoded relation holds under the proof system's assumptions | It does not prove the relation is the intended real-world task |
| Signed/anchored audit record | A signer committed to particular record bytes/order | It does not prove all relevant events were recorded or that claims are true |

Follow the [confidential-computing course](../../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20A%20-%20GPU%20Infrastructure/Nvidia%20GPU/HPC%20Setup/Confidential-Computing-Attestation/README.md) for the full CPU/GPU protocol and actual CPU signature exercise.

## 2. Confidential GPU inference has a platform contract

Supported Hopper/Blackwell configurations protect specified paths and restrict privileged host inspection. Protection differs by mode/topology; do not describe all VRAM and every NVLink fabric as universally encrypted. Compatible CPU, firmware, drivers, GPU mode and assignment matter. There is no universal small-overhead percentage: measure your model, transfer sizes, batching, topology and latency distribution. [NVIDIA Secure AI white paper](https://docs.nvidia.com/nvidia-secure-ai-with-blackwell-and-hopper-gpus-whitepaper.pdf).

NRAS appraises signed GPU evidence; its result does not contain a magical verified `bound_model_hash` for arbitrary inference. The approved workload must validate its model manifest and bind its session key/artifact identity to evidence. TDX runtime measurements require explicit measuring software. A successful quote does not instantly detect every later weight or adapter replacement.

Use a fresh relying-party challenge, trusted verification roots, valid collateral, complete device inventory, approved measurements/state and a bound channel key. Matching CPU/GPU nonces alone does not prove the devices share the protected connection used by the workload. See [the session-binding lesson](../../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20A%20-%20GPU%20Infrastructure/Nvidia%20GPU/HPC%20Setup/Confidential-Computing-Attestation/03-Dual-Vendor-Claim-Binding-and-Verification.md).

## 3. What a computation proof actually proves

Write the relation before choosing a proof system. For a toy quantized inference relation, public inputs might be an approved program/model commitment and an output; private witness values might be the input and intermediate states. The verifier checks the proof against **that relation**, verification key, public inputs and any required assumptions.

A proof about fixed-point arithmetic is not automatically a proof that an FP16 CUDA kernel produced bit-identical values. Define rounding, overflow, quantization, nonlinear approximations, tokenization and decoding. Test the encoded computation against an independent numerical reference. A mathematically valid implementation of the wrong model remains the wrong model.

Zero knowledge hides specified private witness information beyond what the public statement reveals. It does not hide a value deliberately put into a public journal, nor prevent information revealed by outputs. Even a plain hash of a low-entropy secret can be guessed offline; commitment/privacy design is a separate requirement.

## 4. STARK and SNARK labels are not enough

RISC Zero's documented receipt types include composite/succinct STARK receipts, a Groth16 wrapper, and fake development receipts with no proof. The journal contains public outputs. A relying party must verify the intended program image and receipt/public statement; development-mode acceptance cannot be used as evidence. [RISC Zero proof system](https://dev.risczero.com/proof-system/).

Do not transfer all STARK properties to a wrapped proof. RISC Zero's Groth16 layer has a trusted-setup ceremony and elliptic-curve assumptions. A blanket “all outputs are transparent and post-quantum” claim is incorrect. State the complete proof path, hash/field/security parameters, setup assumptions and verifier version. [RISC Zero trusted setup](https://dev.risczero.com/api/trusted-setup-ceremony).

EZKL is another model-to-proof toolchain. Its supported operators, numeric conversion, visibility settings and proving configuration define the proven program. Use its actual versioned API and numerical validation; an ONNX filename alone does not establish coverage. [EZKL project](https://github.com/zkonduit/ezkl).

No proof generation was run for this lecture. There is no supported universal “seconds for small models, hours for 7B” benchmark: record architecture, context/output lengths, numeric relation, prover hardware, peak memory, proof size and verification time for a concrete experiment.

## 5. A hybrid system proves only its explicit boundary

A TEE can execute a large model while a small circuit verifies an authorization rule. That can be useful, but the small proof does **not** prove the model forward pass. Likewise, proving that a log hash was computed does not prove that the log faithfully describes GPU execution.

**Worked example:** a circuit verifies `action == STOP` whenever an authenticated sensor value is below a threshold. This proves a rule over the committed value. It does not prove sensor calibration, timestamp freshness, physical stopping distance or the LLM's reasoning. Those need separate evidence and enforcement.

If a GPU result is an external input to the proven program, explicitly name it as an assumption or verify its source in the relation. Connecting a TensorRT-LLM output to a generic execution-trace file does not generate a proof of its CUDA arithmetic.

## 6. Audit storage is an architectural choice

A signed append-only log, a transparency service or a ledger can support record integrity. A blockchain is optional and introduces consensus, finality, availability, identity/key-management and privacy assumptions. It does not guarantee event completeness, legal non-repudiation or the real-world truth of an entry. A malicious application can omit an action before logging it.

Minimize public metadata. Commitments can reveal correlations and permit guessing attacks; prompts, robotics logs and credentials should not be published simply because a record is called a proof. Reconcile authorized actions against actual tool effects and retain a recoverable event sequence.

The previous “PearlChain stack” section lacked a pinned implementation and evidence linking its claims to actual verifier code. Treat such a stack as a **proposed architecture** until a concrete repository, release, threat model and reproducible verification result are provided. This course makes no claim that a named integration supplies end-to-end confidential or verifiable LLM execution.

## 7. Complete the CPU evidence exercise

Run the [cryptographic binding lab](../../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20A%20-%20GPU%20Infrastructure/Nvidia%20GPU/HPC%20Setup/Confidential-Computing-Attestation/CPU-Binding-Lab.md). It uses real signatures over fictional claims and includes an authentic signature over an arithmetically false statement. Explain why the signature succeeds and why this is not a computation proof.

Then draw a private-log analysis protocol with an approved model manifest, client challenge, endpoint key and tool authorization. Mark which checks are cryptographic, policy-based, numerical or physical. A passing answer retains at least these limits: malicious approved software, unproven external inputs, prompt injection/tool misuse and incomplete audit capture.

**Self-check:** If every signature verifies, can the agent still send private logs to an unauthorized recipient? **Yes.** Signatures do not replace the tool's authorization/egress boundary. If a proof validates a threshold rule, is the robot safe? **No.** It proves only the stated relation, not the physical-system assumptions.
