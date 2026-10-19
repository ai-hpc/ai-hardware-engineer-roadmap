# Jetson lab: inspect a real BitBake configuration

**Time:** 45–75 minutes after the foundation Yocto tools are available. **Hardware:** a Linux build host; no Jetson is needed for this milestone. **Result:** an evaluated configuration and a clear account of what it proves.

Complete the [native observation-check package exercise](../../../Phase%202%20-%20Embedded%20Systems/3.%20Embedded%20Linux/Yocto/Labs/README.md) first. Its source, license, layer, application recipe, and image recipe are the inputs here. This exercise extends the pinned Scarthgap baseline with a real Jetson BSP layer. It is a maintained teaching baseline, not a claim that Scarthgap is the newest Yocto release. See the [current release matrix](Guide.md#2-select-current-compatible-releases).

## 1. Prepare a separate build directory

Use a supported Linux host with the prerequisites documented for your Yocto release. The validation host was Ubuntu 24.04.4, x86-64, Python 3.12.3. `chrpath`, `diffstat`, and `lz4c` were supplied from an isolated tools directory on that host; merely having Python and Git is not sufficient.

Use a workspace **without spaces**. The repository itself has spaces in course paths, so copy the small product layer out of the course. These commands create an example workspace under your home directory; choose a new empty directory if it already exists.

```bash
mkdir -p "$HOME/course-jetson-yocto"
cd "$HOME/course-jetson-yocto"
git clone --branch yocto-5.0.20 --depth 1 https://git.yoctoproject.org/poky
cd poky
git rev-parse HEAD
cd ..
git clone --branch scarthgap https://github.com/OE4T/meta-tegra.git
cd meta-tegra
git checkout --detach 6d7cc8f27e8b3bda2abfa68b2fcfe8d6ecf10464
cd ..
```

Verify the Poky revision is `64e69ed23703f6358ec431d3f5f1f8483f974cae`. Copy the supplied `meta-robot-course` directory next to `poky` and `meta-tegra`, using the actual location of your course checkout. Keep all three directories separate. The layer contents are linked from the foundation exercise above.

## 2. Add layers and choose the machine

From that workspace:

```bash
source poky/oe-init-build-env build-jetson-metadata
bitbake-layers add-layer ../meta-robot-course
bitbake-layers add-layer ../meta-tegra
```

At the end of `conf/local.conf`, add:

```bitbake
MACHINE = "jetson-orin-nano-devkit"
BB_NO_NETWORK = "1"
BB_NUMBER_PARSE_THREADS = "2"
```

The machine assignment overrides the template's earlier default. `BB_NO_NETWORK` makes this metadata exercise's intent explicit; it does not populate sources or prove a future offline image build. Limiting parse threads avoids needlessly consuming the whole host. Retain the normal Poky layer entries in `bblayers.conf`.

**Predict before running:** which target architecture should this machine select? What evidence would show the observation package is actually selected by the image?

## 3. Parse and inspect

```bash
bitbake --version
bitbake-layers show-layers
bitbake -p
bitbake -e robot-course-image > image.environment.txt
```

Open the environment file in an editor and search for the exact assignments below. Search for the preceding variable history as well. The file also contains generated shell functions; a simple text search can find multiple occurrences that are not effective top-level assignments.

Actual results on 2026-10-18:

| Item | Observed result |
| --- | --- |
| BitBake | 2.8.1 |
| Parse | 1,105 recipes; 2,137 targets; 64 skipped; 0 masked; 0 errors |
| `MACHINE` | `jetson-orin-nano-devkit` |
| `TARGET_ARCH` | `aarch64` |
| `IMAGE_INSTALL` | `packagegroup-core-boot observation-check` |
| `IMAGE_FSTYPES` | `tegraflash` |
| `PARTITION_LAYOUT_TEMPLATE_DEFAULT` | `flash_t234_qspi_sd.xml` |
| `TNSPEC_BOOTDEV_DEFAULT` | `mmcblk0p1` |

Whitespace in package and image-type lists is not semantically significant. These are values for this configuration, not instructions to overwrite a board's partition table. The source snapshot declares R36.5.2/JetPack 6.2.3; do not infer that a bare `L4T_VERSION` variable must appear in every image recipe's environment.

The [validation record](metadata-validation.json) retains the revisions, selected values, parse result, and hashes of the product-layer inputs and local configuration used here. It records metadata inspection only.

## 4. Investigate the surprising boot-device value

The machine file visibly contains `TNSPEC_BOOTDEV_DEFAULT ?= "mmcblk1p1"`, yet the effective value above is `mmcblk0p1`. Why?

The included `orin-nano.inc` already assigned `mmcblk0p1` using `?=`. A later `?=` assigns only if the variable is unset; it does not replace the earlier value. The environment history exposes both assignments and the final result. There is also a weak `??=` default in `tegra-common.inc`, which is a different operation.

This is a useful real example of why reading one machine-file line is insufficient. Do not “repair” the upstream configuration by guessing which device name a physical board should use. Trace the complete flash workflow and check the supported machine/storage combination first.

**Exercise:** in a disposable configuration, use `bitbake-getvar TNSPEC_BOOTDEV_DEFAULT` to inspect the history. Explain the result in three sentences: where the first ordinary default was set, why the later one did not replace it, and why a parse pass still cannot prove a bootable partition layout. The [BitBake variable assignment documentation](https://docs.yoctoproject.org/bitbake/2.8/bitbake-user-manual/bitbake-user-manual-metadata.html#setting-a-default-value) explains these operators.

## 5. Define the next milestone

Nothing in this lab executes target code, constructs a full image, flashes a board, or validates GPU/camera operation. Those milestones require additional source inputs, build resources, and board-specific acceptance checks.

For a target-package milestone, retain the compile/install/package task logs, ELF architecture, installed file manifest, and package QA output. For an image milestone, add deployable artifact identities and package contents. For a board milestone, add recovery instructions, boot logs, observation-check output, device checks, and update/recovery tests.

Do not label a metadata-only result “production image verified.” The useful achievement is that a reproducible starting configuration can be inspected before committing to a much larger build.

## Completion check

You are ready to continue when you can:

- Identify the exact core and BSP revisions and explain why both are pinned.
- Show that the product package is in the evaluated image package list.
- Explain the actual `?=` result using its variable history.
- Distinguish parse success, target compilation, image construction, and board validation.
- Name the evidence and resources required for the next milestone you choose.
