# 01 — Intel TDX Attestation Chain

**Learning target:** separate what TDX measures, what a guest supplies, and what a relying party authorizes. Read [the course introduction](README.md) first.

## 1. Isolation has a defined boundary

A Trust Domain (TD) is a VM whose private memory and execution state receive TDX protections against the host/VMM under Intel's threat model. Shared pages and data deliberately sent through I/O are outside the private-memory boundary. The host can still deny service. Software inside the TD remains part of the trusted computing base: a malicious guest application can copy its own secrets out.

The TDX module is signed, loadable software executing in SEAM, with measured identity and version information. It is **not immutable code stored inside the CPU package**. CPU mechanisms, authenticated module loading and platform firmware form part of the trust chain. [Intel TDX module description](https://www.intel.com/content/www/us/en/download/738875/intel-trust-domain-extension-intel-tdx-module.html), [current specifications](https://www.intel.com/content/www/us/en/developer/tools/trust-domain-extensions/documentation.html).

## 2. Measurements are not automatic file monitoring

| Field | Size/role | Interpretation |
|---|---|---|
| MRTD | SHA-384 initial TD measurement | Initial construction events and measured initial contents; not a hash of every future process |
| RTMR[0..3] | Four extendable SHA-384 registers | Values extended by guest measurement software; meaning depends on the measured boot/runtime policy |
| REPORTDATA | 64 caller-supplied bytes | Session binding data; not an automatically checked application assertion |
| Module/TCB identity, TD attributes | Versioned quote fields | Verify module/TCB status and reject debug or other disallowed attributes |

An RTMR extension has the conceptual form `new = SHA384(old || event_digest)`, where the supplied extension value is 48 bytes. The register does not independently inspect files. Firmware, measured boot, IMA or another trusted component must choose events and extend them. An event log lets a verifier recompute the sequence, but a matching log alone cannot show that omitted events were measured. The policy and measuring software must provide that assurance. [Intel module architecture specification](https://cdrdv2.intel.com/v1/dl/getContent/733575), [Intel runtime measurement discussion](https://www.intel.com/content/www/us/en/developer/articles/community/runtime-integrity-measure-and-attest-trust-domain.html).

**Worked example:** a TD boots an approved kernel and downloads `weights.bin` afterward. If no trusted component measures or validates the file, its hash does not magically appear in MRTD or RTMR. Replacing it may leave the quote measurements unchanged. A measured loader must validate the artifact and enforce its subsequent use; lesson 04 develops that lifecycle.

## 3. Local report and remotely verifiable quote

```text
TD application supplies REPORTDATA
  → TDG.MR.REPORT produces a locally authenticated TDREPORT
  → platform quoting infrastructure validates the report
  → a certified quote-signing key signs a TD Quote
  → remote verifier checks quote, endorsements and collateral
```

The report contains a MAC and may travel through software. The secret MAC key is not exported; the statement that the *MAC bytes never leave the package* is incorrect. A remote verifier needs the quote-verification chain, not a JSON rendering of TDREPORT. SGX-based TDX quoting is a common deployment; the supported quote path depends on platform and release. The [Intel runtime guide](https://cc-enabling.trustedservices.intel.com/intel-tdx-enabling-guide/07/trust_domain_at_runtime/) documents actual quote collection and verification paths.

For the SGX-based path, the quote's attestation-key signature, QE certification data, PCK certificate chain, signed TCB information, QE identity and revocation collateral must agree. The May 2026 base specification also describes security-engine-based attestation with a Quoting TD on supported platforms; do not force its endorsement chain into an SGX-only parser. Do not replace this with an ad hoc public-key check. A third-party verification service adds that service's signing key, policy and availability to your trust assumptions.

## 4. Use real DCAP interfaces and preserve their separate outcomes

The checked Intel header exposes `tdx_qv_verify_quote(...)` and unified `tee_verify_quote(...)`. They return an API status and write separate verification/collateral results. An API return of `SGX_QL_SUCCESS` means the call completed; it is not by itself an application acceptance decision. The old imaginary `dcap_qvl.verify()` object is not a runnable SDK interface. [Pinned DCAP C header](https://github.com/intel/confidential-computing.tee.dcap/blob/7ed37274294c5dc47bda4cec601bb727be2875e0/QuoteVerification/dcap_quoteverify/inc/sgx_dcap_quoteverify.h).

For the strict DCAP acceptance example, require API success, collateral-expiration status zero and `SGX_QL_QV_RESULT_OK`, then apply the application measurement/binding policy. Non-terminal alternative results need explicit additional appraisal. [Intel DCAP library specification](https://download.01.org/intel-sgx/latest/dcap-latest/linux/docs/Intel_TDX_DCAP_Quoting_Library_API.pdf).

A verifier adapter should record:

1. API status, authenticated quote result and supplemental advisory information.
2. Collateral expiration status, reference time and applicable revocation results.
3. Parsed version/attributes/measurements from the **verified quote bytes**.
4. Exact expected REPORTDATA and pending challenge identity.
5. Policy revision and final accept/reject reason.

For example, successful API execution with expired collateral is rejected under this course's strict policy. A non-current TCB may require a separately reviewed exception in another policy; do not silently translate it to “secure.” Cached, unexpired collateral can be legitimate: an online lookup on every request is not the definition of validity. If a verifier runs on an untrusted machine, protecting its result is a further problem; QvE/result-authentication options are not interchangeable with a trusted local QVL process.

## 5. Bind exact bytes with a specified construction

Choose a protocol transcript containing a verifier-generated random challenge, audience, application/model identity, session public key and expiry. In the CPU lab:

```text
REPORTDATA = SHA512("course/binding/v1\0" || encoded_transcript)  # 64 bytes
```

This is the **course's chosen construction**, not a universal Intel convention. Some services define a different REPORTDATA composition, so an integration must use that service's documented calculation. Compare all 64 bytes. If a protocol chooses SHA-256 plus 32 zero bytes, verify the padding too; comparing only the first half leaves half the claimed binding unchecked. [Intel EAT profile](https://portal.trustauthority.intel.com/eat_profile.html).

A correct binding proves that authenticated evidence carries those bytes. It does not establish that the application was honest when choosing them. Accept only measurements of an application whose key generation, artifact validation and result handling you trust.

## Check your understanding

1. A quote has an approved MRTD but debug mode is enabled. **Reject under the strict policy; one allowed measurement does not override other attributes.**
2. A quote is authentic and REPORTDATA matches yesterday's challenge. **Reject; no current pending challenge authorizes it.**
3. A model is modified after a single measurement. **A snapshot alone does not prevent this. The workload must preserve verified artifact identity through use.**

Next: [NVIDIA evidence and verifier results](02-NVIDIA-CC-Attestation-Chain.md).
