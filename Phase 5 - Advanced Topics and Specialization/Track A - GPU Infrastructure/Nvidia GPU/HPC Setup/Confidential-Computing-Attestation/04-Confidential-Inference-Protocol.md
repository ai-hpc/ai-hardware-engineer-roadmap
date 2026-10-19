# 04 — Model Identity and Confidential Inference

**Learning target:** design the transition from verified evidence to protected model/prompt processing without claiming that a quote measures everything automatically.

## 1. Use precise names for three different mechanisms

**Confidential execution** protects data in use inside a stated TEE boundary. **Sealing** ordinarily refers to encrypting/authenticating persisted state under a key or policy associated with an environment; it is not a universal synonym for “running in a TDX VM.” A deployment needs its own persistence and key-recovery design. **TLS** protects a network session between its endpoints.

An edge proxy can terminate ordinary HTTPS outside the TEE. If it receives plaintext prompts, guest memory isolation cannot hide those prompts from it. Conversely, TLS can terminate inside an attested workload when the protocol binds the actual endpoint key to evidence. Draw exactly where decryption occurs; do not infer the boundary from an HTTPS URL.

## 2. An artifact identity covers more than weights

A deployable model manifest should identify weight shards, configuration, tokenizer, adapters, executable model code where applicable, and relevant serving configuration. Quantized conversion changes the artifact: identify the source revision, conversion recipe and resulting hashes separately.

A publisher signature is useful **when a real, independently authenticated publisher signature exists**. Do not invent a Qwen/other publisher key or imply that all model repositories publish signed manifests. Another defensible policy may pin reviewed immutable revisions and file digests through the organization's release process. That trusts the review/release authority; it is not the same as publisher endorsement. A signature says who endorsed bytes, not whether the model is safe or the inference server avoids logging.

**Example:** `weights-A`, tokenizer `T1` and template `C1` differ operationally from the same weights with tokenizer `T2` or template `C2`. A weight-only digest misses this change. Store component digests in a structured manifest, then bind the manifest digest into the session.

## 3. Measure, verify, preserve and use

TDX does not automatically extend model files into RTMR, nor does it detect every later model edit. A trusted loader must:

1. Authenticate the expected manifest through a client-approved trust path.
2. Hash/validate each artifact it will load, with strict parsing and resource bounds.
3. Prevent substitution between validation and use: use verified file descriptors/content, constrained writable state and an explicit loading pipeline.
4. Bind the actual loaded artifact identity and permitted runtime configuration to application evidence/session state.
5. Invalidate authorization when the model, adapter, executable, device assignment or protected session changes.

Measurement and access control solve different parts of this problem. An event log can explain RTMR extensions, but a quote alone cannot prove that a later unmeasured file was used. A measured malicious loader can still lie or exfiltrate; its identity must be independently reviewed. [Intel runtime measurement](https://www.intel.com/content/www/us/en/developer/articles/community/runtime-integrity-measure-and-attest-trust-domain.html).

## 4. A complete session sequence

```text
Client/relying party                         Approved confidential workload
  chooses challenge, audience, policy
  → challenge + expected artifact identity
                                            creates ephemeral endpoint key
                                            validates/loads allowed artifacts
                                            verifies protected GPU assignment
                                            collects evidence bound to transcript
  ← transcript + evidence + authenticated results + key-possession proof
  verifies roots, collateral, freshness,
  measurements, state, device inventory,
  artifact identity, key binding and proof
  atomically authorizes this pending session
  ↔ authenticated encrypted channel to the bound endpoint
  → prompt with session/request identifier
  ← response bound to that request and model session
```

For a key-release service, its release authorization must be bound to the same workload key and artifact policy. Choose an established authenticated channel/key-establishment design rather than inventing encryption by “encrypting with a public key.” Specify key confirmation, transcript authentication, unique AEAD nonces, direction-separated keys, request counters and expiry. The [CPU lab](CPU-Binding-Lab.md) demonstrates signature binding and possession only; it implements **no encrypted inference channel**.

A fresh challenge prevents stale-session reuse but does not continuously attest a long session. Define when to re-attest, how failures terminate processing, and what happens after migration, restart, firmware update, GPU reset, model replacement or key rotation. The accepted session must not silently survive a state change outside its policy.

## 5. What each trust root contributes

| Trust source | Can support | Cannot establish by itself |
|---|---|---|
| CPU endorsements and verifier | Authentic platform/TD evidence and appraised TCB | Honesty of an arbitrary guest application |
| GPU endorsements/NRAS or trusted local verifier | Authentic device evidence and accepted device state | Which prompts/weights a kernel actually processed |
| Artifact release authority | Expected bytes/configuration | Quality, safety or absence of harmful behavior |
| Approved workload and channel protocol | Artifact lifecycle and protected session behavior | Universal correctness of the model's answers |
| Client/relying-party policy | Whether these facts authorize a particular action | Truth of facts that were never authenticated |

These dependencies do not mean an attacker must compromise every signer. A bug in the approved workload, a poisoned allowlist, a replay race or a secret-logging endpoint may defeat the objective without forging any vendor signature.

## 6. Design review and worked answers

**Case A:** authentic CPU/GPU evidence, approved model, HTTPS terminates at an operator proxy, plaintext then enters the TD. **The proxy sees prompts.** Move the protected endpoint into the approved boundary or add a correctly bound application channel.

**Case B:** all weight hashes match, but the serving process logs prompts to an untrusted object store. **Attestation does not forbid this.** The approved application/configuration and egress policy must enforce the confidentiality objective.

**Case C:** attestation happened before a LoRA adapter was replaced. **The original authorization is stale for the new model state.** Revalidate and create a newly bound session, or reject the change.

**Independent assessment:** design a confidential robotics-log analysis service. Mark camera/audio data, model artifacts, logs, TLS termination and GPU transfers on the diagram. Explain how a client rejects an outdated model manifest and how authorization ends after a GPU reset. Do not claim that confidential analysis makes a robot control policy safe; behavior evaluation remains a separate requirement.
