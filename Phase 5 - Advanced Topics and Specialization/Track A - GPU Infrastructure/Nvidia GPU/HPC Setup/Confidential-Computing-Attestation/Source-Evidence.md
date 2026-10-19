# Sources and Execution Evidence

Reviewed **2026-10-18**. These links support the versioned teaching claims; they do not establish that any local hardware has confidential-computing support enabled.

| Source | Checked scope |
|---|---|
| [Intel TDX specifications hub](https://www.intel.com/content/www/us/en/developer/tools/trust-domain-extensions/documentation.html) | Current specification entry point and module/loader distinction |
| [Intel module base specification 348549-008, May 2026](https://cdrdv2.intel.com/v1/dl/getContent/733575) | RTMR extension and TDREPORT/quote architecture |
| [Intel runtime guide](https://cc-enabling.trustedservices.intel.com/intel-tdx-enabling-guide/07/trust_domain_at_runtime/) | Real report/quote collection and verification paths |
| [Pinned DCAP header](https://github.com/intel/confidential-computing.tee.dcap/blob/7ed37274294c5dc47bda4cec601bb727be2875e0/QuoteVerification/dcap_quoteverify/inc/sgx_dcap_quoteverify.h) | Real C verification functions; distinct API, quote and collateral outputs |
| [Intel runtime measurement article](https://www.intel.com/content/www/us/en/developer/articles/community/runtime-integrity-measure-and-attest-trust-domain.html) | Software must explicitly measure runtime artifacts |
| [NVIDIA Secure AI v1.3, August 2025](https://docs.nvidia.com/nvidia-secure-ai-with-blackwell-and-hopper-gpus-whitepaper.pdf) | Mode-specific CPU/GPU and NVLink protection; stated threat boundary |
| [NVIDIA SDK migration guide](https://docs.nvidia.com/attestation/attestation-client-tools-sdk/latest/migration_guide.html) | Python deprecation March 15, end of support September 15, 2026; replacement NVAT workflow |
| [NVAT 2.0.0 CLI](https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/2.0.0/sdk-cli/command-reference.html) | `verify-token`, expected nonce, relying-party policy and experimental EAR workflow |
| [NVAT verification API](https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/2.0.0/api/group__attestation__result__verification.html) | NRAS signature/submodule digest verification; EAR authentication versus semantics |
| [NVAT source release 2026.10.02](https://github.com/NVIDIA/attestation-sdk/releases/tag/2026.10.02) | Published October 7; source commit `11b2bfeb42a1158aa08744603ecb239ccb3d2265` |
| [RATS RFC 9334](https://www.rfc-editor.org/rfc/rfc9334) | Attester, verifier and relying-party roles |
| [Cryptography 46.0.5 Ed25519](https://cryptography.io/en/46.0.5/hazmat/primitives/asymmetric/ed25519/) | Actual signing/verification primitives used in the CPU lab |

The source release identifier `2026.10.02` and documentation SDK version `2.0.0` are different version labels. Do not assume they are interchangeable package names. [Downloaded-source hashes](evidence/software-sources.json) include the actual CLI verification implementation and JWT verifier examined. Intel's checked DCAP commit is dated September 9, 2026. Direct HTTP checks encountered access restrictions on three Intel HTML resources; their contents were inspected through the browser tool. The current May 2026 module specification was downloaded successfully from the specifications hub. These access restrictions are recorded rather than reported as successful downloads. Exact source URLs and available fetch hashes are in [source-check-2026-10-18.json](evidence/source-check-2026-10-18.json).

## Executed locally

- Four standard-library transcript tests.
- Nineteen optional real-cryptography tests, including atomic replay acceptance under concurrent threads, claim validity at authorization time and isolation from caller mutation of the request.
- One CLI CPU exercise with one valid fixture session and seven rejected attacks.
- Isolated CPython 3.12 environment with the three hash-pinned wheels in [the lock](labs/requirements-cpu.lock); no model download or GPU package.

The [machine-readable validation](evidence/validation.json) and [CPU report](evidence/cpu-run-2026-10-18.json) record the actual outcomes. Ed25519 key generation is random; time-based tests use an injected fixture clock. This is an educational application protocol, not a production-reviewed attestation service.

## Source-checked but not executed

No TDX quote generation, DCAP library invocation, GPU report collection, NRAS request, `nvattest` binary execution, vendor token verification, GPU readiness change or encrypted inference session was performed. The real CLI template is checked against the pinned source and documentation, with deployment policy supplied by the learner. No CPU test is counted as hardware attestation.
