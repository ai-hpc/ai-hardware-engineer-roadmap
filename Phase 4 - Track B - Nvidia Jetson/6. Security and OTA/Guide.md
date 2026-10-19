# Security and OTA: deploy an authentic, compatible, recoverable robot

**Outcome:** build an update acceptance plan that checks the exact artifact, target, version, application behavior and recovery path. **Pace:** five 60–90-minute sessions. The starting exercises run on an ordinary Linux CPU; physical secure-boot provisioning is a later, board-specific milestone.

An update can be authentic but incompatible. It can be compatible but behaviorally poor. It can work once but leave the device unable to recover after a later failure. This module connects those different acceptance questions into one deployment process.

**Version review: 18 October 2026.** Use the [Jetson compatibility matrix](../Compatibility.md) and documentation for the selected BSP. Current R39.2 security instructions use the FSKP flow, including `fskp_fuseburn.py`; an old `odmfuse.sh` recipe is not a current universal command. The R36 security deep dive remains explicitly versioned. No fuses or boot partitions were modified during this course review. [NVIDIA R39.2 secure boot](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/SD/Security/SecureBoot.html).

## 1. Write the failure you want to prevent

For a mobile robot, begin with this small example:

| Asset/property | Failure to prevent | Mechanism to investigate |
| --- | --- | --- |
| Boot software | Unauthorized early boot image starts | Configured hardware-rooted boot authentication |
| Model and calibration | Attacker replaces weights or changes units | Authenticated deployment manifest and application checks |
| Recorded sensor data | Removed storage exposes private recordings | Encryption and an appropriate unlock/recovery policy |
| Robot commands | Untrusted source issues an actuator request | Identity, authorization, freshness and bounded command interpretation |
| Availability | Interrupted update leaves neither version usable | Staging, trial selection, bounded retries and tested fallback |
| Support lifecycle | Vulnerable dependency cannot be located or updated | Artifact inventory, vulnerability handling and update ownership |

State the adversary's capabilities: network access, a user account, root access, physical storage access or debug access. A defense against a stolen powered-off disk does not imply protection against a running compromised kernel.

Draw the trust boundaries between update server, transport, local verifier, boot chain, application, model loader and controller. A container shares the host kernel; placing software in a container does not create a hardware trust root or automatically authorize its commands.

## 2. Run a real signature check

Start with the [signed-artifact CPU lab](../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Security/labs/README.md). It uses OpenSSL to generate disposable artifact-signing keys, signs a manifest, and tests tampering, wrong target, truncation and rollback policy. The test key is unrelated to Jetson fuse provisioning.

The manifest binds target, version, payload size and SHA-256 digest. Verification needs a trusted key, exact signed bytes and a strict parser. Reject a malformed manifest before interpreting it as installation instructions.

A checksum detects a difference from an expected digest. It does not authenticate an artifact when an attacker can replace both artifact and digest. A signature authenticates the covered bytes under a trusted key; it does not decide whether a version is appropriate for this board. TLS authenticates/protects a connection under its endpoint and certificate assumptions; it does not replace a persistent signature on an artifact that will be cached or installed later.

**Exercise:** modify one payload byte and keep the manifest unchanged. Then modify the manifest's target without resigning. Explain the different checks that should reject those cases. Finally sign a manifest for the wrong target using a trusted test key: the signature should pass, while target policy rejects installation.

## 3. Understand the boot-chain boundary

NVIDIA distinguishes the fuse-rooted SoC authentication chain ending at the bootloader from UEFI's security-key scheme for its payloads. Therefore a statement such as “secure boot is enabled” needs to identify the configured stage and enforcement evidence. [R39.2 secure boot architecture](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/SD/Security/SecureBoot.html).

| Layer | Acceptance evidence |
| --- | --- |
| Early boot authentication | Intended key/configuration and rejection of a controlled unauthorized boot artifact |
| UEFI payload authentication | Enrolled UEFI trust and rejection of a disallowed payload |
| Kernel/module policy | Exact kernel policy and rejection of a disallowed module |
| Root filesystem integrity | Trusted verified root and response to modified covered storage |
| Model/configuration authorization | Signed manifest, compatibility policy and loader checks |

A boot-time signature does not continuously validate every writable userspace file. A signed model can still be parsed unsafely or yield inappropriate actions. Each boundary needs its own contract and negative tests.

Use exact SoC/BSP key formats and fuse specifications. Do not carry a generic RSA-2048 example or a “128-bit SBK” statement into an Orin provisioning plan. Key purposes, supported algorithms, fuse fields and revocation mechanisms differ by platform. The [FSKP documentation](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/SD/Security/FSKP.html) describes the supported workflow.

Before irreversible provisioning, produce the board/configuration manifest, key-purpose map, signing procedure, recovery policy and test-unit results. Keep production signing material outside ordinary source/build logs. This course's CPU route produces those reviewable designs; it does not require burning fuses to learn cryptographic or update behavior.

## 4. Separate encryption from runtime isolation

Disk encryption primarily protects data at rest when the attacker lacks unlock material. Once unlocked, authorized normal-world software reads plaintext. A TEE can protect key derivation or key use while intentionally returning a result to a normal-world client.

The documented Jetson disk-unlock flow derives a passphrase through a trusted service and uses a normal-world client with `cryptsetup`. Saying that the unlock result never reaches normal-world memory misdescribes that interface. AES-XTS encryption is also not general authenticated integrity; add an appropriate integrity design when the threat includes storage modification. Work through the [security deep dive](../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Security/Guide.md) and its versioned OP-TEE/disk-encryption sources.

Separate boot signing keys, application-update keys, device identity credentials and data-encryption keys. Their compromise and rotation consequences differ. Define how a replacement board receives authorization and how legitimate data recovery works without placing a universal plaintext recovery secret beside every device.

## 5. Model update states before choosing tooling

Use this state sequence for the teaching exercise:

```text
received → signature/target/version checked → staged and verified
→ selected for trial → application checked → confirmed
                                  ↘ failed or expired → fallback
```

The [interrupted-update CPU lab](../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-OTA-Deep-Dive/labs/README.md) implements two image slots, real file writes, fsync, rename, metadata publication, bounded trials and confirmation. It terminates child processes at several installation boundaries and reopens state. Its fallback sequence is `2 → 2 → 1` after two unconfirmed trials.

These tests exercise process interruption while the OS/storage remain powered. They do not establish real power-loss durability of a Jetson, SSD cache or filesystem. Both kinds of evidence matter, and they should not share an unlabeled “power-cut passed” checkbox.

**Worked policy:** confirmed version 1 is running and candidate version 2 is staged. If the rollback floor advances to 2 before version 2 is accepted, version 1 may become forbidden just when it is needed for recovery. Coordinate the security-floor transition with acceptance and the approved recovery policy. The correct sequence depends on the product's threat model; “always allow old signed images” and “immediately forbid the old image” each have consequences.

## 6. Keep firmware, rootfs and application updates distinct

A deployment manifest should identify the compatible set of boot firmware, kernel/modules, DT, rootfs, application container, model, preprocessing and persistent-data schema. A/B rootfs alone does not atomically switch every component in that set.

The current bootloader update documentation includes UEFI capsule-based mechanisms. Debian-package updates and image-based OTA have their own supported source/target matrices and layout conditions. Do not use historical `nv_update_engine` examples as a generic R39.2 update recipe, or infer support for a new point release from an older matrix. [Bootloader update/redundancy](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/SD/Bootloader/UpdateAndRedundancy.html).

For rootfs redundancy, inspect the correct slot target and installed services. R39.2 documents a default of three failed boots and retry configuration in the range zero through three. The customer validation hook needs application-specific integration; reaching a login prompt is not a robot health check. The [rootfs lesson](../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Rootfs-and-AB-Redundancy/Guide.md) explains the documented verification flow without inventing a `mark-boot-successful` command. [NVIDIA rootfs documentation](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/SD/RootFileSystem.html).

Persistent-data migration deserves its own test. If the candidate rewrites a shared database into a format the fallback cannot read, changing slots may restore executable files while leaving the robot unusable. Use compatible readers/writers, a reversible migration, isolated data generations or an explicitly qualified recovery procedure.

## 7. Define application acceptance with real work

A health endpoint that always returns HTTP 200 only establishes that a handler answered. A stronger acceptance check exercises the required dependency and confirms the exact deployment identity.

For a perception/control service, a useful trial check may require a fresh sensor observation, the expected calibration/model identity, a valid model output, bounded processing/queue age, successful controller handshake and a working fallback path. Use a synthetic input only when the acceptance claim explicitly concerns the synthetic path; it cannot prove a physical camera is connected.

The [container CPU lab](../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Container-Fleet/labs/README.md) includes actual HTTP health behavior, malformed-input checks and a recorded container run. Connect its service health to the update trial's exact version/boot identity. An old success message from a previous boot must not confirm the new candidate.

For learned models, retain held-out quality and failure-slice results. The [CPU export/INT8 lab](../5.%20Application%20Development/5.%20ML%20and%20AI/cpu-export-quantization/Guide.md) demonstrates that a technically valid export can have much worse stop-detection recall after poor calibration. Signing that artifact does not repair its behavior.

## 8. Roll out and learn from failures

Promote a candidate through a controlled pilot before broad deployment. Define stop conditions and useful denominators: failures per trial device, accepted useful actions, stale observations, recovery events, and update completion within a stated interval. An offline unit is not automatically a successful update or a confirmed failure; track its unknown state.

Retain immutable artifact digests and versioned evaluation records, plus the minimum device evidence needed for support. Record firmware/app/model identities together. Keep credentials and private sensor content out of routine logs unless the product has an explicit authorized need and retention policy.

Use the [manufacturing module](../7.%20Compliance%20and%20Manufacturing/Guide.md) to connect deployed devices to their board revision, factory test, calibration and software identity. A new board or sensor revision can invalidate assumptions that a software-only rollout previously satisfied.

## 9. Completion task and answer key

Submit one deployment manifest, trust-boundary diagram, key-purpose map, update state machine and failure-injection report. Run the signature and interrupted-update labs; state which hardware milestones remain unexecuted.

- **Signature valid, wrong board:** reject on compatibility policy.
- **Hash changed, new hash supplied by the same untrusted download:** reject unless authenticated metadata authorizes the exact bytes.
- **Boot succeeds, camera missing:** fail the application's trial condition if the camera is required.
- **New model passes clean accuracy but fails a deployment slice:** hold promotion and investigate that slice.
- **Fallback binary starts but cannot read migrated data:** recovery is incomplete.
- **Replayed prior-boot health acknowledgement:** reject because it does not establish this trial's health.

A complete submission explains each rejection and demonstrates the corresponding executable check where the teaching environment supports it.
