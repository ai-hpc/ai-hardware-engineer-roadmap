# Build and maintain a Jetson image with Yocto

Yocto/OpenEmbedded describes how a system is built from recipes, configuration, and source inputs. Its value is control over those inputs and outputs. It does not automatically deliver a small image, a secure product, a fast boot, or a working update system. This chapter shows how to develop and verify those properties one at a time.

**Outcomes:** select compatible layer revisions; explain machine/distro/image roles; inspect a real recipe; distinguish metadata validation from an image build; and prepare a repeatable deployment with licenses, updates, and recovery accounted for. Budget four 60–90-minute lessons before an optional full build. Prerequisite: the [sixteen-lesson Yocto foundation course](../../../Phase%202%20-%20Embedded%20Systems/3.%20Embedded%20Linux/Yocto/Lecture/README.md).

## 1. Choose a build route from product requirements

Use Yocto when controlling the OS image and its maintenance is part of the engineering work: custom hardware, explicit package policy, reproducible inputs, distribution configuration, source traceability, or coordinated BSP releases. A standard JetPack/Ubuntu image can also be customized, versioned, tested, and deployed systematically.

There is no universal threshold at 50, 100, or 5,000 devices that makes one route correct. Compare maintenance expertise, hardware support, required packages, update model, build resources, and time to verify changes. A small specialized product may need a custom image; a large fleet may use a carefully managed standard distribution.

**Worked decision:** a team needs one supported camera and a containerized model service but does not change the kernel. First validate that application on the supported standard stack. If it later needs a maintained custom kernel, a tightly controlled package set, and a reproducible factory image, evaluate a Yocto route against those concrete requirements.

## 2. Select current, compatible releases

**Checked 2026-10-18.** NVIDIA documents an officially supported Yocto path beginning with JetPack 7.2, aligned with OE4T. That changes the old description of all Jetson Yocto work as community-only. [NVIDIA Yocto overview](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/AR/YoctoOnJetson.html).

The immutable source snapshots inspected for this chapter are:

| meta-tegra branch at review | Commit | Declared Jetson / JetPack line | Role |
| --- | --- | --- | --- |
| `wrynose` | `feab7664000f313f655d20d2f39c04432214dc25` | R39.2.1 / 7.2.1 | Current Wrynose branch; evaluate for a current image |
| `scarthgap` | `6d7cc8f27e8b3bda2abfa68b2fcfe8d6ecf10464` | R36.5.2 / 6.2.3 | Maintenance branch used for this chapter's metadata exercise |
| `master` | `95e8a869123c42f1207cc12914f9c65ce2e50fa0` | R39.2.1 / 7.2.1, `blacksail` layer compatibility | Moving development line; do not confuse with a release pin |

Sources: [Wrynose README](https://github.com/OE4T/meta-tegra/blob/feab7664000f313f655d20d2f39c04432214dc25/README.md), [Scarthgap README](https://github.com/OE4T/meta-tegra/blob/6d7cc8f27e8b3bda2abfa68b2fcfe8d6ecf10464/README.md), [master snapshot](https://github.com/OE4T/meta-tegra/blob/95e8a869123c42f1207cc12914f9c65ce2e50fa0/README.md). The [machine-readable snapshot](release-snapshot.json) retains retrieved file hashes.

The upstream Yocto release/support schedule and the BSP layer's supported combinations are separate constraints. Check both. Wrynose 6.0 and Scarthgap 5.0 are distinct LTS lines; a recipe tested on Scarthgap is not automatically tested on Wrynose. [Yocto releases](https://wiki.yoctoproject.org/wiki/Releases).

A branch name is not immutable. Pin every layer and core revision together. This chapter does not recommend the old invented `scarthgap-l4t-r36.x` branch as a universal checkout target.

## 3. Learn the layer roles

| Layer/configuration | Responsibility |
| --- | --- |
| OE-Core/Poky and BitBake | Core metadata, task engine, toolchain/base-system build |
| `meta-tegra` | Jetson BSP integration, machine definitions, platform components |
| Optional community/middleware layers | Additional packages and supported extensions |
| Product layer | Your application recipes, image composition, board adaptations and policy |
| Machine configuration | Hardware-specific choices |
| Distribution configuration | System-wide policy such as init/package/features |
| Image recipe | Runtime packages and resulting image composition |

Keep product modifications in a product layer where practical. Record when an upstream recipe is appended, patched, replaced, or given a different provider. A higher layer priority is not a universal override for every variable or version choice.

`LAYERSERIES_COMPAT` declares a layer's intended release compatibility; it is not proof of a successful build or boot. If a layer rejects the selected series, investigate the migration work. Merely adding a new name to that declaration suppresses a check without performing the migration.

## 4. Start with the supplied observable package

Use the [observation-check package lab](../../../Phase%202%20-%20Embedded%20Systems/3.%20Embedded%20Linux/Yocto/Labs/README.md). Its C program receives capture time, completion time, and an inclusive age limit. It returns distinct statuses for fresh, stale, malformed, and reversed timestamps.

The complete layer contains a recipe, licensed source, layer configuration, and image recipe. The native C suite was actually compiled and executed. The foundation course also parsed the layer with pinned Poky/BitBake and ran source fetch/unpack tasks; it explicitly did not build a complete target image.

This small application makes packaging errors visible without downloading a large model. First verify its behavior on the host. Then inspect the target compiler selection, installed path, runtime package, image inclusion, and target output. Each step produces a different piece of evidence.

The [Jetson metadata exercise](Metadata-Lab.md) adds the pinned `meta-tegra` layer and selects a real Jetson machine. It teaches source/layer compatibility before a large image build.

## 5. Understand the task graph

A typical recipe fetches source, unpacks it, applies patches, configures, compiles, installs into staging, and splits/packages the installed files. Image tasks then resolve runtime packages and construct deployable artifacts. Dependencies and classes can add tasks; the real graph is more informative than a memorized linear list.

Useful commands in an initialized, correctly configured build environment include:

```bash
bitbake-layers show-layers
bitbake-layers show-recipes observation-check
bitbake -e observation-check > observation-check.environment.txt
bitbake -c listtasks observation-check
bitbake -g observation-check
```

The environment dump includes effective values and their history; it may also contain sensitive configuration if you placed secrets in metadata. Keep credentials out of public logs and release artifacts. The graph shows build dependencies, not the application's runtime timing.

A parse pass establishes that metadata can be evaluated in that configuration. A successful compile establishes code generation for a target. Package QA, image construction, boot, hardware operation, and task-level behavior remain additional milestones.

## 6. Keep recipe staging and package installation distinct

During `do_install`, a recipe writes into its staging destination, commonly `${D}`. It should not install directly into the developer's host `/usr`. Packaging uses the staged files and package rules; the image selects runtime packages, not arbitrary recipe source directories.

A frequent mistake is compiling a binary successfully but omitting it from the installed/package file set. Another is installing it correctly but forgetting to include the runtime package in the image. The observation-check image recipe and effective `IMAGE_INSTALL` value provide an inspectable example of the latter boundary.

Use `${CC}`, `${CFLAGS}`, `${LDFLAGS}` and the appropriate build class/toolchain for the target. Running the host's native compiler inside a target recipe can produce the wrong architecture. For CUDA, the compiler frontend and target libraries have their own supported host/target combinations; use the BSP layer's recipes/classes instead of assuming an `nvcc` found in a target sysroot can execute on the build host.

**Check:** an x86-64 executable appears in an arm64 image. Which evidence catches it before flashing? Inspect the effective compiler and build log, then the executable's ELF architecture and package QA results.

## 7. Configure the board from its actual definition

Read the chosen machine configuration and included files. Developer-kit SD, external NVMe, production module, carrier, and secure-boot variants may need different machine/flash choices. Do not assume `MACHINE = "jetson-orin-nano-devkit"` selects every Orin Nano carrier/storage configuration.

For the pinned Scarthgap snapshot, inspect [the machine definition](https://github.com/OE4T/meta-tegra/blob/6d7cc8f27e8b3bda2abfa68b2fcfe8d6ecf10464/conf/machine/jetson-orin-nano-devkit.conf) rather than copying a partition number from a diagram. Preserve the resolved boot device and partition-template settings in the release record.

Board changes can require pinmux, power sequencing, device-tree bindings, driver code, firmware, and test updates. A new compatible string alone is not complete board support. Use the [kernel/DT lab](../Orin-Nano-Kernel-Internals/labs/README.md) and [camera bring-up course](../Orin-Nano-Camera-ISP-Sensor-Bringup/Guide.md) to investigate the relevant boundary.

## 8. Build in a suitable environment

Choose a supported host with adequate disk, memory, network/source access, and package prerequisites. Put the build workspace in a path without spaces. Bound build parallelism to the available resources; a large `-j` value is not free speed.

Separate downloaded sources, shared-state cache, temporary build work, deployable artifacts, and release archives. Caches can accelerate repeated builds, but a successful cache hit is not the same experiment as rebuilding everything from source. Record the distinction when investigating reproducibility.

Use source mirrors and pinned revisions so an upstream branch movement or unavailable server does not silently change a release. An offline build claim needs a tested complete local source/cache policy; setting a variable without verifying missing inputs is insufficient.

Do not respond to every failure by deleting all caches. Read the failed task's log, identify the input/configuration/tool issue, and invalidate the relevant work only when justified. Retain evidence needed to explain the failure.

## 9. Verify reproducibility at a stated level

Distinguish three claims:

- **Pinned inputs:** source and configuration identities are recorded.
- **Repeatable behavior:** independent builds pass the same functional tests.
- **Bit-for-bit reproducibility:** specified outputs are identical under a defined rebuild experiment.

Pinned layers help but do not prove the last two. Build paths, timestamps, host tools, generated data, and other inputs can affect output. Compare artifact hashes and investigate differences; the Yocto reproducibility workflow supplies tooling and test guidance. [Reproducible builds](https://docs.yoctoproject.org/test-manual/reproducible-builds.html).

Avoid embedding the current hostname or timestamp in an artifact while claiming it is byte-identical across hosts. If such metadata is required, isolate it from the reproducibility comparison or define the exact normalization policy. Preserve the original evidence as well as any normalized report.

## 10. Track components, licenses, and vulnerabilities

Use the selected release's package/license metadata and SBOM workflow. Current Yocto documentation describes SPDX output and configuration options; filenames, defaults, and class capabilities differ across releases. Inspect the actual outputs from your pinned build. [Yocto SBOM generation](https://docs.yoctoproject.org/dev-manual/sbom.html).

An SBOM is an inventory and traceability artifact. It does not automatically discharge redistribution obligations, certify the product, or prove that every vulnerability has been fixed. Retain source revisions, applied patches, license notices, and the decision record for relevant findings.

Do not delete license files or documentation indiscriminately to meet an image-size target. Decide which packages/features belong in the product, measure their cost, and keep required notices and support artifacts available through the appropriate distribution process.

Security policy, update signing, device identity, and manufacturing keys also need lifecycle ownership. The [security chapter](../Orin-Nano-Security/Guide.md) provides the trust and update boundaries.

## 11. Integrate updates and persistent data

The standard NVIDIA stack already provides update and A/B mechanisms; Yocto is not the reason those mechanisms exist. A custom image must integrate a supported boot/update flow and meaningful application acceptance. A framework dependency in a recipe does not implement that integration by itself.

Test candidate installation, trial selection, attempt exhaustion, acceptance, fallback, and persistent-data compatibility. Use the [interrupted-update lab](../Orin-Nano-OTA-Deep-Dive/labs/README.md) before mapping the design to the target. Build-time version strings, application versions, and hardware boot ratchets have different purposes.

Document the storage budget for both system slots, downloads/reconstruction, persistent data, logs, and recovery. Measure the final image and installation peak usage. No fixed “800 MB image” or “99% update success” is promised by the build system.

## 12. Validate the image in layers

A useful progression is metadata → target package → image → boot → devices → application → failure/recovery. Attach evidence at each stage:

| Stage | Evidence |
| --- | --- |
| Metadata | Exact layer revisions, parse result, effective variables/providers |
| Package | Build log, ELF architecture, installed files, QA and tests |
| Image | Package manifest, hashes, storage layout, SBOM, required services |
| Boot | Loaded kernel/DT/rootfs identities, logs, first-boot access |
| Hardware | Actual camera/GPU/network/storage operation and error paths |
| Application | Known-input quality, freshness, sustained timing and resource use |
| Recovery | Failed candidate, rollback, data compatibility and retained identity |

A boot-time report must state its endpoints. Reaching a shell is different from the first valid sensor result. Retain multiple runs and conditions; do not replace measurements with a universal sub-ten-second target.

## 13. Release a reviewable system

A release bundle should identify the source/layer lock, effective configuration, toolchain, build environment, image/package hashes, licenses/SBOM, test results, flash/update procedure, recovery method, and known limitations. Keep per-device credentials separate from generic images.

A colleague should be able to tell exactly what was built, for which hardware, with which tests, and how to recover if it fails. Automated CI helps produce that record; it does not remove the need for target hardware and application validation.

**Completion:** run the native package and metadata exercises, explain the artifact boundary they reach, and prepare a complete image/board acceptance plan. An advanced completion adds the actual target package, image, SDK if needed, physical boot, device/application tests, and recovery evidence. This chapter's retained checks do not claim a complete Jetson image build or physical deployment.
