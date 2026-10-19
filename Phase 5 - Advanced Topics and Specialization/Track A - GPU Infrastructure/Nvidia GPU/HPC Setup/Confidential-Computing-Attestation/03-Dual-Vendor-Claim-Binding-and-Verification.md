# 03 — Bind a Session, Then Apply Policy

**Learning target:** explain why two authentic reports can still authorize the wrong session. Use the [CPU lab](CPU-Binding-Lab.md) to run these cases.

## 1. Give each participant a job

The **attester** provides evidence about its environment. A **verifier** appraises evidence against endorsements and reference values. A **relying party** decides whether the result authorizes an action, such as releasing a model key. One service may implement multiple roles, but the trust relationships remain distinct. [RATS architecture](https://www.rfc-editor.org/rfc/rfc9334).

A signed output hash authenticates a commitment, not the computation that produced it. A measured application may enforce a useful execution protocol, but that assurance depends on the approved code and its TCB. There is no general conversion from “two vendor signatures” to a proof of correct inference or training.

## 2. Define an unambiguous transcript

Concatenating variable-length strings is ambiguous: `("ab", "c")` and `("a", "bc")` both produce `abc`. Choose a versioned, deterministic encoding with explicit field structure. Do not interchange raw bytes, hexadecimal text and base64 text.

Our lab's transcript contains:

| Field | Reason |
|---|---|
| Protocol version and audience | Prevent use in another protocol or service |
| Random verifier challenge | Identify a fresh pending request |
| Issue time and deadline | Bound challenge lifetime |
| Session public key | Bind the endpoint's possession-proof key; the lab establishes no encrypted channel |
| Model artifact digest | Bind the intended model/config/tokenizer manifest |
| Expected GPU identities | Prevent a weaker or incomplete device set |

The lab permits only ASCII strings, booleans, integers, null, lists and string-keyed objects, serialized with sorted keys and fixed JSON separators. This is a **local teaching encoding**, not an implementation of RFC 8785 or a vendor evidence format. Duplicate JSON keys are rejected on load. A production cross-language protocol should use a standardized encoding and cross-implementation test vectors.

Let `T` be those encoded bytes. The lab chooses:

```text
CPU binding = SHA512("course/binding/v1\0" || T)   # 64 bytes
GPU binding = SHA256("course/binding/v1\0" || T)   # 32 bytes
```

These hashes carry the same transcript through different field-size contracts. The client stores the transcript before requesting evidence and compares the complete expected value; an attester cannot select the challenge after seeing a prior report.

## 3. Freshness is more than an expiry timestamp

Suppose a token is valid from 12:00 until 13:00. It could be replayed thousands of times during that hour. A verifier challenge must be unpredictable, tied to a pending request, short-lived and consumed atomically when authorization succeeds. Concurrent requests and multiple verifier replicas need shared, transactional replay state or an equivalent protocol design.

The CPU lab consumes a challenge only after both signatures, binding, policy and session-key possession pass. It rechecks the challenge and every claim's validity window at that final authorization instant. It uses an in-memory lock to demonstrate one winner under concurrent acceptance. Restarting the process loses this state: the lab is **not** a distributed authorization service.

A nonce authenticates a challenge's occurrence in evidence. It does not prove the remote party's wall clock is trustworthy; the relying party uses its own time and explicit tolerated skew. Check collateral freshness and application session lifetime separately.

## 4. A common nonce does not prove co-location

An attacker can relay a challenge to a genuine remote CPU and another genuine remote GPU. Both reports can be authentic and contain the same challenge while the attacker's own workload runs elsewhere. Hashing both reports together does not establish a protected physical data path.

The approved guest driver/workload must authenticate the GPU endpoint it actually uses, establish the protected session, verify device/fabric membership and bind that identity to application evidence. Platform assignment and fabric attestations must match the supported deployment. Multi-GPU evidence composition is therefore partly a channel/topology problem, not just a JWT problem. [NVIDIA confidential-container architecture](https://docs.nvidia.com/datacenter/cloud-native/confidential-containers/latest/overview.html).

The lab deliberately cannot test this property. Its GPU identifier is a signed fixture claim, not proof of a local accelerator.

## 5. Separate the checks and retain one final decision

| Check | Failure example | Category |
|---|---|---|
| Trusted signature | Attacker generates its own signing key | Authenticity |
| Version/schema and exact binding | CPU from session A plus GPU from B | Protocol binding |
| Pending challenge/time/replay | Reuse an accepted bundle | Freshness |
| Allowed measurement, TCB and debug state | Authentic but disallowed application | Policy |
| Complete distinct device inventory | Empty, duplicate or missing GPU | Composition policy |
| Session-key possession | Evidence copied without the matching private key | Channel binding |
| Authorized artifact lifecycle | Validated weights replaced before use | Application enforcement |

It is useful to retain a final boolean decision **and** structured reasons, versions and statuses. The mistake is losing the evidence needed to explain or review that decision, not using a boolean anywhere.

Authenticating a report before trusting its fields is essential, but many bugs occur afterward: a wrong policy, unbound key, omitted device, stale challenge or malicious approved workload can pass every signature check.

## 6. Exercise with answers

A client accepts a model hash from the same untrusted server that supplies the evidence. Is the model independently identified? **No.** The client must pin an artifact/manifest through a trusted release process.

CPU and GPU fixtures are authentic, same challenge, same model, but the session proof is signed by another key. **Reject possession.** An authentic binding is useful only if the channel endpoint owns the bound key.

Every fixture passes and the signed result says `2 + 2 = 5`. **Signature verification can still pass.** Computation correctness is outside this fixture; a trusted application policy and separate result validation are needed.

Next: [model identity and confidential inference](04-Confidential-Inference-Protocol.md).
