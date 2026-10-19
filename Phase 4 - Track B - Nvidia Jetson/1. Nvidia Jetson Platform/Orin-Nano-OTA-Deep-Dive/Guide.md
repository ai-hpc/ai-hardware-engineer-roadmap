# Jetson updates: installation, trial, acceptance, and recovery

An update is complete when the intended software runs correctly and a defined recovery path remains. Downloading a file or rebooting successfully is only part of that process. This chapter teaches a small update protocol first, then maps its questions to the current Jetson tooling.

**Outcomes:** distinguish update layers; authenticate an artifact; explain interrupted installation; define application acceptance; and plan rollback across firmware, rootfs, containers, models, and persistent data. Budget three 60–90-minute sessions. Start with [rootfs and A/B](../Orin-Nano-Rootfs-and-AB-Redundancy/Guide.md) and [security](../Orin-Nano-Security/Guide.md).

**Reviewed 2026-10-18.** Use the [compatibility matrix](../../Compatibility.md) to choose the release. R39.2 documents Debian-package and image-based OTA, including an explicit source/target release matrix. Its Orin image-update table includes 35.5.0, 35.6.x, 36.3.0, 36.4.x, 36.5.0, and 39.2.0 as listed source releases for target 39.2.0. That is not a universal upgrade promise for arbitrary custom layouts or later point releases. Read the exact release instructions and their requirement to update both chains. [NVIDIA update mechanism](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/SD/SoftwarePackagesAndTheUpdateMechanism.html).

## 1. Identify what is changing

A robot's deployment is a compatible set of components:

| Layer | Example change | Compatibility to preserve |
| --- | --- | --- |
| Boot firmware | UEFI or platform firmware | Board identity, security state, boot media, supported next stage |
| Kernel and modules | Camera driver change | Kernel configuration, symbol versions, DT, driver/runtime expectations |
| Rootfs | OS packages and services | Runtime libraries, service configuration, users, storage layout |
| Container | Application and dependencies | Host kernel, GPU access, device permissions, interface contracts |
| Model | New weights or engine | Architecture, preprocessing, tensor shapes, runtime, calibration |
| Persistent data | Configuration or schema | Both candidate and recovery application's readers/writers |

A single deployment may change several rows. “Atomic update” must identify which components are selected together and which can remain mixed during failure. Two rootfs slots alone do not make every firmware, application, and database change one atomic operation.

Write a compatibility manifest with immutable artifact identities and explicit allowed base versions. A friendly version string is useful for humans, but use a digest to identify the exact payload.

## 2. Learn the transaction on a CPU

Complete the [interrupted-update lab](labs/README.md). It uses ordinary files in temporary directories, two image slots, version/hash metadata, bounded trial starts, and application confirmation.

The lab actually interrupts a child process after half an image, a complete image, image publication, temporary metadata, and final trial selection. On reopen, the confirmed image remains available. This validates the teaching implementation under its stated assumptions; the operating system and storage stay powered throughout.

Keep three operations separate:

- **Installation:** place and verify the complete candidate without destroying the confirmed image.
- **Trial:** start the candidate with bounded retries while retaining fallback.
- **Acceptance:** associate meaningful application health evidence with that exact boot and deployment.

The demo allows two trials, so failed acceptance produces versions `2 → 2 → 1`. Its retry counter is an ordinary JSON field. The values and representation are teaching choices, not a reverse-engineered Jetson boot record.

## 3. Authenticate the artifact and its policy

Use the [signed-artifact lab](../Orin-Nano-Security/labs/README.md) to verify actual cryptographic signatures and reject a wrong target, old version, truncated payload, or changed bytes. Carry the authenticated artifact identity into installation.

Transport security and artifact authentication solve related but different problems. A protected download connection can authenticate a server and protect transit; the artifact still needs a policy for publisher identity, target hardware, allowed versions, and offline verification. A plain SHA-256 file from the same untrusted source cannot establish who authorized the payload.

Check storage capacity, payload size limits, supported base image, security state, and the operational update window before touching the candidate. For deltas, authenticate the exact base and final reconstructed artifact; a patch that applies without errors can still be based on the wrong image.

Keep private signing keys out of the device image and build logs. The verifier's trust anchor must be protected by the actual system design. The lab's generic RSA key is not a Jetson provisioning key.

## 4. Map the layers to release-specific tools

For current boot-firmware flows, the R39.2 bootloader guide describes BUP generation followed by a UEFI capsule, with release-specific helpers and status reporting. A BUP is not a complete rootfs update. Do not assume an old `nv_update_engine` recipe describes the current platform. [Bootloader update and redundancy](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/SD/Bootloader/UpdateAndRedundancy.html).

For image-based OTA, the vendor tools include `l4t_generate_ota_package.sh` on the preparation side and `nv_ota_start.sh` on the target side. They handle supported layouts and configurations; the fleet client remains responsible for obtaining and validating the correct payload. Encrypted storage and UEFI security require their documented inputs. [Image-based OTA procedure](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/SD/SoftwarePackagesAndTheUpdateMechanism.html).

This course deliberately gives no invented universal flash command. A reproducible device recipe must record board SKU/revision, carrier, source and destination BSP, exact tool package, selected configuration, storage device/layout, security state, and recovery procedure. Attach the actual invocation only after resolving those fields against the matched documentation.

A package update modifies the installed environment according to package-manager behavior. Do not call it an A/B deployment merely because the board has two rootfs partitions. Determine which partitions and boot components the package hooks change, and test their supported rollback/recovery path.

## 5. Define health beyond process startup

An SSH connection, successful `import tensorrt`, or existing process ID says little about the robot's task. A deployment acceptance test might require:

1. The expected model and configuration hashes are loaded.
2. Required sensors produce fresh, correctly formatted observations in known frames.
3. A bounded known-input inference passes a numerical or task-specific check.
4. The control process exchanges a fresh command/acknowledgment within its deadline.
5. The watchdog path and persistent storage are operational under the defined test.

Each check needs an implementation, timeout, and failure outcome. A test fixture should avoid producing unintended physical motion. Simulated/replay health checks and device checks must be labeled separately.

Record the boot ID and deployment identity with the health result. A delayed “healthy” message from the prior process must not accept the newly started candidate. The lab rejects that stale confirmation explicitly.

On Jetson, inspect the installed validation service and its scripts before integrating application checks. The vendor flow includes customer validation before boot-status verification, but a successful Linux boot does not automatically implement your robot's task checks. [Rootfs validation services](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/SD/RootFileSystem.html).

## 6. Preserve recovery across persistent data

Suppose version 1 reads `gain` from a configuration table. Version 2 replaces it with `position_gain` and deletes the original field. If both slots use that database, falling back to version 1 can fail even though its rootfs is untouched.

One useful migration pattern is expand, migrate, then contract:

- Add a compatible representation while retaining what the old reader requires.
- Deploy readers/writers that support the transition and validate data.
- Remove the old representation only after the recovery policy no longer requires it.

This is a design pattern, not an automatic guarantee. Test both applications against data written by each relevant version. For SQLite, understand the supported schema operations and transaction behavior in the shipped version. [SQLite schema changes](https://www.sqlite.org/lang_altertable.html).

Include model caches, calibration files, identity keys, user settings, and logs in the storage plan. Decide what survives rollback, what is versioned, what can be regenerated, and what must never be replaced by a generic image.

## 7. Reconcile rollback and anti-rollback

Rollback is an availability mechanism: return to an acceptable previous deployment. Anti-rollback rejects versions below a security floor. Advancing the floor can intentionally make an old recovery image ineligible.

The CPU lab refuses to advance its floor until both image copies meet the new value. A real system might instead use a separately maintained recovery image or a different update sequence. State the invariant: after each irreversible step, which authenticated, compatible image can still boot and restore service?

Do not represent a writable text file as a hardware security counter. Jetson's boot-component rollback mechanisms have specific version and storage semantics; see the [security chapter](../Orin-Nano-Security/Guide.md). Rootfs application versions and boot firmware ratchets are not automatically one counter.

## 8. Treat container and model updates as deployments

A container image digest identifies bytes, not compatibility with the host. Test GPU/runtime compatibility, device permissions, storage paths, lifecycle, and health before switching traffic or control responsibility. Removing the previous image immediately can destroy the practical rollback path.

For a model, a symlink rename can change which pathname future opens resolve to; it does not replace an already loaded engine or finish outstanding GPU work. Define load, warm-up, input-contract validation, handover, in-flight completion, and retirement. Keep the previous model available until the new one passes the required checks.

A live controller also needs ownership rules: two application versions must not both issue commands. For a planned handover, define the authority transfer and the command freshness rule. Use the [container and fleet guide](../Orin-Nano-Container-Fleet/Guide.md) to develop the process boundary.

## 9. Choose an update framework by integration evidence

NVIDIA's platform tools, RAUC, SWUpdate, and Mender occupy different parts of an update system. A framework choice does not establish your bootloader integration, compatible partition layout, authentication policy, or health logic.

For a candidate framework, demonstrate installation into the intended inactive storage, boot selection, boot-attempt tracking, acceptance, rollback, and reporting on the exact platform. RAUC, for example, documents bootloader integration and marking slots good/bad as explicit integration responsibilities. [RAUC integration guide](https://rauc.readthedocs.io/en/latest/integration.html).

Use a comparison worksheet with the same columns for each option: supported artifacts, trust model, boot integration, storage/layout assumptions, persistent-data policy, failure evidence, operational tooling, and maintenance ownership. Link to the implementation and test results instead of labeling an untested shell wrapper “complete integration.”

## 10. Test failure and control fleet rollout

Start with deterministic local failures, then move to a recoverable hardware test setup. Cover interrupted download, rejected authentication, wrong target/base, insufficient space, failed write/read-back, trial failure, health timeout, stale confirmation, shared-data incompatibility, and exhausted recovery copies.

Physical power cuts require a separate test plan with the intended storage, firmware, power path, and recovery access. The CPU exercise's process-exit results are preparation for that test, not a substitute.

For a fleet, define the cohort, observation period, success metric, stop condition, and rollback capability before expansion. A small canary can miss a rare problem; include relevant hardware, storage, network, and workload variants. Report installations, boots, accepted deployments, rollbacks, unreachable devices, and application regressions as separate counts.

**Completion:** publish the CPU interruption evidence and a device-specific update plan with an acceptance test and recovery invariant. A hardware completion additionally needs actual logs across installation, reboot, application checks, one controlled failed trial, and recovery. No physical flashing, capsule update, or fleet rollout was executed for this chapter.
