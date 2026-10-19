# Lecture 25: Building an Embedded Linux Product with Yocto

A board image is a dependency graph, not a directory copied onto an SD card. You must select a board configuration, kernel/BSP, userspace packages, initialization policy, and deployable image format that agree. Yocto Project supplies tools and metadata to build that graph; it is not one universal Linux distribution.

This lesson joins the operating-systems material to the [sixteen-part Yocto course](../../../Phase%202%20-%20Embedded%20Systems/3.%20Embedded%20Linux/Yocto/Guide.md). Use its supplied package and already validated metadata rather than a second untested image recipe. Read [boot and Device Tree](Lecture-05.md) and [storage/update boundaries](Lecture-22.md) first.

## Four objects to distinguish

| Object | Question it answers | Example |
|---|---|---|
| Recipe | How is one component fetched, built, installed, and packaged? | `observation-check_1.0.bb` |
| Layer | Which related recipes/configurations extend the build? | `meta-robot-course` |
| Machine configuration | Which target board/architecture/BSP settings apply? | `qemux86-64` or a validated Jetson machine |
| Image recipe | Which runtime packages and image features belong in the rootfs? | `robot-course-image` |

A recipe is closer to a package-build specification than a Dockerfile. An RPM `.spec` and Debian's packaging files are different systems; Yocto recipes can produce different package formats. Target packages, native build tools, cross compilers, and SDK tools are not interchangeable.

## Follow one package through the tasks

The course's `observation-check` accepts capture time, completion time, and an allowed age. For inputs `100 125 40`, its age is 25 ms and it succeeds. For `100 141 40`, age is 41 ms and it returns the documented stale status. It rejects malformed values and a completion preceding capture before doing unsigned subtraction.

```text
source + license
    ↓ do_fetch / do_unpack
working source tree
    ↓ do_compile using the target compiler and flags
executable for the selected target
    ↓ do_install into ${D}, not into the build host's /usr
staged filesystem tree
    ↓ do_package and package QA
runtime package
    ↓ selected by IMAGE_INSTALL / dependencies
image root filesystem and deployment artifacts
```

The application's exit code is a small observable contract. Packaging it does not make Linux real-time, synchronize camera clocks, or guarantee a motor stops when data are stale. Those are larger system contracts that need separate tests.

Start with the [native package lab](../../../Phase%202%20-%20Embedded%20Systems/3.%20Embedded%20Linux/Yocto/Labs/README.md). Its seven native tests compile and exercise the program without downloading a model or building an image. Then follow the exact pinned core and layer setup there.

## A pinned baseline and its evidence

The foundation exercise uses Poky `yocto-5.0.20` at commit `64e69ed23703f6358ec431d3f5f1f8483f974cae`, BitBake 2.8.1, and a Scarthgap-compatible product layer. It is a maintained learning baseline, not a claim that Scarthgap is the latest release. Consult the Yocto [release documentation](https://docs.yoctoproject.org/) and the course's release matrix for current supported branches and host requirements.

The recorded foundation run parsed 922 recipes and completed the application's fetch/unpack boundary. It did **not** compile target packages or boot an image. This distinction matters: a recipe can parse while later failing because its target compiler, dependency, license, install path, or runtime ABI is wrong.

For a Jetson extension, use the [real BSP metadata lab](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Yocto-BSP-Production/Metadata-Lab.md). Its pinned meta-tegra revision declares R36.5.2/JetPack 6.2.3 and produced a 1,105-recipe parse with `TARGET_ARCH="aarch64"` and the observation package selected. This is an intentionally separate legacy-compatible baseline from current JetPack 7.2.x support. Current NVIDIA/OE4T integration does not retroactively change a pinned Scarthgap configuration.

## Read the evaluated configuration

The useful commands, after activating the documented build environment, are:

```bash
bitbake --version
bitbake-layers show-layers
bitbake -p
bitbake -e robot-course-image > image.environment.txt
```

These commands were exercised in the linked labs' configurations. They are not ready to run from an arbitrary directory: the required sources, host tools, and build configuration must already exist. Keep the build workspace free of spaces even though this course repository uses spaces in teaching paths.

Inspect `MACHINE`, `TARGET_ARCH`, `IMAGE_INSTALL`, `IMAGE_FSTYPES`, and variable history in the environment output. A name found in a generated shell function is not necessarily its final top-level value. Layer priority, overrides, weak defaults, and `.bbappend` matching have distinct effects; “the last layer always wins” is false.

**Worked assignment:** an included configuration sets `BOOTDEV ?= "A"`; a later machine file uses `BOOTDEV ?= "B"`. The result remains `A`, because `?=` only assigns when unset. This is the actual kind of issue exposed by the Jetson metadata exercise. Use the [BitBake assignment rules](https://docs.yoctoproject.org/bitbake/2.8/bitbake-user-manual/bitbake-user-manual-metadata.html#setting-a-default-value) rather than guessing a flash device from one line.

## Reproducibility needs evidence

Record source revisions, fetched checksums, configuration, tool versions, layer contents, licenses, and output identities. BitBake signatures and shared-state caches improve repeatability and reuse, but matching a branch name is not a complete reproducibility claim. Moving source revisions, timestamps, host dependencies, prebuilt vendor blobs, and network inputs can affect results.

Use an image manifest and package/license reports to explain what was shipped. A smaller image may reduce storage or attack surface, but deleting packages blindly can remove recovery, observability, firmware, or initialization that the product requires. Minimality is a design choice supported by acceptance tests, not a safety certification.

## Deployment is a separate engineering milestone

An ARM64 `Image` is normally an uncompressed kernel image; `Image.gz` is a compressed artifact requiring a supported decompression path. Boot firmware, DTBs/overlays, rootfs, module versions, and userspace accelerators must match the selected BSP. JetPack 5 and later Jetson workflows use UEFI; do not substitute an old CBoot flash recipe.

Flash commands and recovery pin sequences depend on the exact module, carrier, storage, BSP, and image generator. The Orin Nano developer kit's storage layout is not an arbitrary eMMC target. Use the corresponding board guide and the generated deployment workflow; there is no universal `tegraflash.py --flash all` command that makes every Yocto image bootable.

A prebuilt TensorRT engine is also an artifact with compatibility requirements. Do not put an engine built for an unrelated host/GPU/software stack into an image and call the integration complete. Record how the model is built, loaded, checked, and regenerated on supported targets.

## Completion ladder and answers

| Milestone | Evidence needed |
|---|---|
| Application behavior | Native positive, boundary, and malformed-input tests |
| Metadata | Pinned layers, successful parse, evaluated package selection |
| Target package | Compile/install/package logs, ELF architecture, package QA |
| Image | Artifact hashes, manifest, selected services/configuration |
| Board boot | Console log, rootfs identity, device and application checks |
| Updates | Authorized artifact, interruption/recovery behavior, rollback policy |

**Check:** Does `read-only-rootfs` enlarge a root partition automatically? **Answer:** No. It is an image/runtime policy feature; partition sizing and deployment layout are separate configuration.

**Check:** Does an A/B-related BSP variable implement a complete OTA product? **Answer:** No. Update selection, authentication, boot attempts, health confirmation, persistent data compatibility, and recovery must form a tested protocol.

**Check:** Does AGNOS prove all embedded AI Linux images use Yocto? **Answer:** No. AGNOS has its own image builder; use the [pinned AGNOS study](../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20E%20-%20Autonomous%20Vehicles/2.%20openpilot%20Reference%20Stack/agnos/Guide.md) for that implementation.
