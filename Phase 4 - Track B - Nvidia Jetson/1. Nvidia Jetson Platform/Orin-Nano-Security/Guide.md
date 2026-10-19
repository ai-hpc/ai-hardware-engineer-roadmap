# Jetson security: trust boundaries, updates, and model deployment

Security engineering asks what the system must protect, from whom, and under which assumptions. A robot may need to protect its boot software, model/configuration integrity, credentials, sensor data, and ability to recover after a failed update. These goals require different mechanisms.

Start with the [signed-artifact CPU lab](labs/README.md). It performs real OpenSSL signing/verification and tests tampering, target mismatch, and rollback policy. Then work through the architecture lessons below. The capstone is a threat model and an update/recovery design with reproducible negative tests.

**Version boundary, reviewed 2026-10-18:** the references below include NVIDIA's R36.4.4 Orin documentation. Use the [compatibility matrix](../../Compatibility.md) and the security documentation matching your BSP for device work. NVIDIA now marks `odmfuse.sh` deprecated in favor of Factory Secure Key Provisioning. Old examples remaining on an official page are not automatically the current provisioning workflow. No physical fuses, encrypted disks, or boot images were modified to validate this course.

## 1. Define assets and adversary capabilities

Begin with concrete failures rather than a list of security products:

| Asset or property | Example failure | Relevant boundary |
|---|---|---|
| Boot integrity | Device starts an unauthorized image | BootROM, bootloader, UEFI, kernel |
| Model/configuration integrity | Attacker substitutes weights or calibration | Update signer, verifier, application loader |
| Credential confidentiality | Device identity key copied from storage | Key provisioning and key use |
| Stored data confidentiality | Removed storage reveals recordings | Encryption and unlock policy |
| Availability | Update or malformed input prevents operation | Recovery, bounded work, resource limits |
| Command authorization | Untrusted client issues actuator requests | Application protocol and controller |

State whether the adversary can access the network, open the enclosure, replace storage, use debug ports, or execute code with normal or elevated privileges. A defense against stolen storage is not necessarily a defense against a compromised running kernel.

Map privileged components and data flows. Include the update service, model loader, camera stack, container runtime, ROS bridge, and control supervisor. Prefer a small explicit trusted computing base over assuming every process on the board is trusted.

## 2. Separate authentication, integrity, and confidentiality

A hash binds bytes to an expected digest, but an attacker who can replace both can recompute it. A digital signature authenticates signed bytes under a trusted verification key. Encryption protects confidentiality under its key/nonce/mode assumptions. Some encryption modes also authenticate; others do not.

The CPU lab signs a manifest containing target, version, size, and artifact digest. A valid signature still requires target and version checks. An authentic model may be incompatible with the application or behaviorally poor, so model evaluation remains necessary.

Authenticated transport such as TLS protects a connection under its certificate validation and endpoint assumptions. It does not replace a persistent artifact signature when content is cached, mirrored, copied offline, or installed later. Conversely, signing a model does not authenticate every command sent to the robot.

Choose key purpose and scope explicitly. A boot signing key, update signing key, device identity key, and data encryption key serve different roles. Separate them so compromise or rotation of one does not automatically require replacing every other trust relationship.

## 3. Understand the boot chain's stages

Orin's hardware-rooted boot authentication begins with immutable BootROM and configured fuse-based trust. Later stages authenticate subsequent components. NVIDIA's documentation distinguishes the SoC boot trust chain ending at the bootloader from UEFI's separate security-key scheme for its payloads. See the [Secure Boot guide](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/Security/SecureBoot.html).

These are related controls, not interchangeable status flags:

| Control | Main purpose | What it does not establish alone |
|---|---|---|
| SoC secure boot | Authenticate covered early boot components | Integrity of every runtime file |
| UEFI Secure Boot | Authenticate configured UEFI payloads | Model/configuration authorization |
| Kernel module signing | Restrict loadable kernel modules under policy | Absence of bugs in permitted modules |
| Verified read-only filesystem | Detect unauthorized block changes against a trusted root | Integrity of a separate writable data area |
| Application artifact verification | Authenticate model/configuration packages | Boot-chain or runtime isolation |

A successful boot only demonstrates that the tested configuration passed its configured checks. To evaluate enforcement, include controlled negative artifacts and verify rejection at the intended stage. Keep a development recovery method while establishing the workflow.

Do not infer secure boot from the presence of a signing package, a UEFI utility, or a file named “secure.” Record actual configuration, enrolled trust, and observed enforcement separately. Containers inherit the host boot state and kernel; a container image cannot add a hardware root of trust to an unverified host.

## 4. Treat provisioning as a board-specific lifecycle

Fuse settings are irreversible in ways ordinary configuration files are not. Key formats, hashes, byte order, dependency ordering, revocation options, and production-mode behavior are SoC-specific. Select them from the exact NVIDIA fuse specification and BSP tooling, not from a generic tutorial's hex constants.

The current referenced workflow is [Factory Secure Key and Expansion Key Provisioning](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/Security/FSKP.html). The course's disposable RSA-2048 artifact key is intentionally unrelated to a Jetson provisioning key.

A production provisioning design should answer:

1. Where are signing/decryption keys generated and stored?
2. Which keys are shared across a product family and which are device-specific?
3. How does the station identify the exact board and intended configuration?
4. Which image and key relationship is validated before irreversible settings?
5. What evidence is retained without exposing secret material?
6. How are replacement devices, key rotation, revocation, and recovery handled?

Use a written per-device manifest and controlled test units to qualify the process. A shell script that prints a hash and immediately burns it does not validate its relationship to the intended images or recovery plan. This course's CPU route ends with a reviewable provisioning design; physical provisioning requires the actual approved board configuration and equipment.

## 5. Use OP-TEE according to its actual interface

OP-TEE separates trusted applications from normal-world applications under the platform's TrustZone configuration. A client invokes a defined trusted service; this does not make every caller trustworthy or every returned value secret.

The trusted application must validate parameter types, sizes, caller permissions, and state transitions. A secure operation can still be misused if an untrusted caller may request it without authorization. Avoid placing large, complex parsing or model inference into the trusted world merely because it sounds more protected.

NVIDIA's [OP-TEE integration documentation](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/Security/OpTee.html) describes its platform-specific key and service arrangements. Do not copy historical KEK fuse names or a hardcoded trusted-application UUID and assume it represents the installed Orin service.

A hardware-derived key may remain inaccessible while a derived result or plaintext is intentionally returned to normal-world software. Read the complete data flow. “Uses a TEE” does not mean no secret-related bytes ever leave it, nor does it establish protection against all physical attacks or a malicious authorized client.

## 6. Match disk protection to the threat

Disk encryption protects stored data when the attacker lacks the unlock material. Once the running system unlocks a volume, authorized processes can read plaintext; a sufficiently privileged compromised process may also access it.

NVIDIA's documented disk-encryption path uses a trusted service to derive an unlock passphrase and a normal-world client with `cryptsetup` to unlock the volume. The key derivation can occur in the trusted world while the unlock interface still transfers a result to the client. The claim “the passphrase never leaves secure world” would therefore misdescribe that workflow. See the [disk encryption guide](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/Security/DiskEncryption.html).

AES-XTS disk encryption provides confidentiality but is not a general authenticated-integrity mechanism. If the threat includes storage modification, add an appropriate integrity/verified-boot design. A read-only verified root filesystem and writable application data have different requirements. Protect calibration and action-limit configuration if changing them can alter robot behavior.

Plan backup, recovery, and key rotation before deployment. An encrypted image that can never be recovered after a legitimate hardware replacement may fail the product's availability requirements. Conversely, a universal recovery key stored beside the device may defeat the confidentiality goal.

## 7. Design updates as authenticated transactions

An update has more states than “downloaded” and “installed”:

```text
received → signature/policy checked → written to inactive destination
         → contents verified → trial boot/use → health accepted → committed
```

Failure at each transition needs a defined recovery state. A/B slots help retain a previous image, but they do not automatically verify signatures, preserve compatible configuration, or define meaningful health.

The [signed-artifact lab](labs/README.md) checks authenticity and policy. Its minimum version is a trusted argument, not protected storage. A real rollback defense must protect both the accepted image and the state that determines the minimum permitted version.

Jetson boot-component rollback protection uses release-specific ratchet mechanisms; it is not a universal RPMB counter automatically covering kernel, root filesystem, application, and model. The [NVIDIA rollback documentation](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/Security/RollbackProtection.html) distinguishes component versions and their storage/update relationships.

Coordinate rollback protection with recovery. Advancing a version floor before proving that the new image and its recovery path work may make the old slot unusable. An application model can also require a matching tokenizer, preprocessing configuration, calibration, or runtime. Sign and validate the compatibility set.

Test interrupted writes, corrupt packages, wrong targets, old but correctly signed packages, failed health checks, and repeated reboot attempts. A health check should cover required functionality and readiness, not merely “the process exists.” The [OTA lesson](../Orin-Nano-OTA-Deep-Dive/Guide.md) develops the update workflow for the selected BSP.

## 8. Restrict runtime authority without breaking the data path

Run an application with only the file, device, network, and privilege access it needs. Separate model inference, update installation, and actuator control when their authority differs. An inference process usually does not need permission to modify boot partitions or enroll trust keys.

For systemd services, options such as `NoNewPrivileges`, filesystem protection, and resource limits can reduce authority, but test them against required camera/GPU/device access. `PrivateDevices` or an indiscriminate device-deny rule can hide required device nodes. A configuration that prevents the application from running is not a verified security improvement.

Containers share the host kernel. Device passthrough, host networking, mounted sockets, and broad privileges change the isolation boundary. Seccomp filters system calls; filesystem access controls restrict resources; neither proves that permitted native code is bug-free. Use supported defaults, narrow exceptions, and a recorded reason for each additional capability.

Treat untrusted image, telemetry, and model inputs as parsable data with bounded sizes and timeouts. An allocation failure or malformed shape should produce an explicit rejection, not an unchecked buffer access or a retry loop that starves control work.

## 9. Protect model integrity and assess its behavior

Model encryption can protect a stored file, but the deployed runtime needs usable weights or equivalent representations. Decrypted CPU/GPU buffers, plugins, and the inference process remain part of the threat model. Do not claim absolute model secrecy merely because the on-disk file is encrypted or a TEE released its key.

Verify a model package before loading executable plugins or interpreting untrusted formats. Pin source/checkpoint identity and record the evaluation evidence for the exact artifact. A signed update must still pass quality, calibration, memory, timing, and failure tests.

For a robot, keep bounded action interpretation outside unconstrained generated output. Validate units, frames, timestamps, ranges, and controller state before accepting a learned action. The [robotics foundations](../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20D%20-%20Robotics/Robotics%20Foundations/README.md) and [inference pipeline lesson](../Orin-Nano-Real-Time-Inference/Guide.md) provide executable examples.

## 10. Produce an evidence-based security design

Submit an asset/adversary table, trust-boundary diagram, key-purpose map, update state machine, recovery plan, and negative-test report. Distinguish a source-reviewed design, a CPU signature test, a development-board enforcement test, and a production provisioning qualification.

A practical review asks whether the right mechanism protects the right property:

- Can an attacker replace both an artifact and its checksum?
- Can an old correctly signed artifact be accepted?
- Can a valid package for another target be installed?
- Can the version floor or trust anchor be replaced?
- Can a failed trial update recover without violating the intended rollback policy?
- Can a correct, authentic model still produce an invalid robot action?

**Self-check:** a package passes signature verification but targets another board. Should it be installed? **Answer:** no; authenticity does not establish compatibility. A package passes every cryptographic check but fails the robot's held-out evaluation. Should it be promoted? **Answer:** no; deployment integrity and behavioral quality are separate requirements.
