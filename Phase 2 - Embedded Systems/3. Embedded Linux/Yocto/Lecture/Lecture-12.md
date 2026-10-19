# Lecture 12 — Module 10: record what the image contains

**Course:** [Yocto guide](../Guide.md) · **Previous:** [Lecture 11](Lecture-11.md) · **Next:** [Lecture 13](Lecture-13.md)

A release should identify the software it ships and the inputs used to build it. Package manifests, license records, source archives, and an SBOM answer related but different questions.

## Follow the small package

The age-check recipe declares `LICENSE = "MIT"` and checks the supplied `LICENSE` text with `LIC_FILES_CHKSUM`. A changed license file should trigger review. That checksum does not verify all source files, establish provenance by itself, or decide obligations for a product. The [Yocto license manual](https://docs.yoctoproject.org/5.0.20/dev-manual/licenses.html) describes the metadata mechanism.

| Record | Question it answers |
|---|---|
| Package manifest | Which package names/versions were selected for this image? |
| License manifest and texts | What license information did the build record? |
| Layer commit + local diff | Which build instructions were used? |
| Recipe `SRCREV` / source checksum | Which application/library sources were selected? |
| SBOM | What component/dependency/provenance relationships were emitted? |
| Artifact checksum | Which exact image/SDK file was delivered? |

A layer commit is not the same as every recipe's Git source revision. An application may be fetched from a different repository. The supplied local source is instead versioned with the product layer; preserve that source and license together.

## Inspect effective policy first

**Documented exercise commands:**

```bash
bitbake-getvar -r observation-check LICENSE
bitbake-getvar -r observation-check LIC_FILES_CHKSUM
bitbake-getvar INHERIT_DISTRO
bitbake-getvar LICENSE_DIRECTORY
```

The Scarthgap documentation describes default `create-spdx` inheritance and a separately selectable SPDX 3 path. Inspect the actual inherited class and emitted format in your pinned checkout instead of assuming a schema from the filename. See the [release-specific SBOM guide](https://docs.yoctoproject.org/5.0.20/dev-manual/sbom.html).

After an image build, locate its package/license records and SPDX output under the effective deployment directories. Pick the age checker, one C runtime package, and one boot-related package. Trace each to its source/version, recorded license, and image inclusion. Mark fields you could not verify instead of filling them from memory.

## A review exercise

In a disposable copy, change only a comment or blank line in the local license file. Recompute its MD5 manually and predict the metadata check. Explain why the text must be reviewed even when the resulting license identifier might remain the same. Restore the original file so the course checksum remains valid.

For a release, keep source retrieval and preservation policies alongside the manifest. A manifest that names a vanished remote archive cannot recreate it. Vulnerability scanners and SBOM generators supply evidence for review; an empty report does not prove there are no vulnerabilities.

**Self-check:** does “same package names” mean “same image”? **Answer:** no. Versions, patches, build options, configuration, filesystem content, and boot artifacts may differ. Exact artifact hashes answer a different question from package identity.

**Artifact:** a three-component trace table plus source/configuration/manifest references. Target-image manifest generation is **not run** until you actually build the image.

**Previous:** [Lecture 11](Lecture-11.md) · **Next:** [Lecture 13](Lecture-13.md)
