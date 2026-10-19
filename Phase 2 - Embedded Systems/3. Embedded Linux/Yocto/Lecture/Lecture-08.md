# Lecture 8 — Module 6: select packages for an image

**Course:** [Yocto guide](../Guide.md) · **Previous:** [Lecture 07](Lecture-07.md) · **Next:** [Lecture 09](Lecture-09.md)

The application recipe answers how to build software. The image recipe answers which runtime packages belong on the target. Confusing these questions leads to “it built successfully, but the command is missing.”

## The supplied image

The [robot-course-image recipe](../Labs/meta-robot-course/recipes-core/images/robot-course-image.bb) is deliberately small:

```bitbake
require recipes-core/images/core-image-minimal.bb

SUMMARY = "Teaching image containing a timestamp contract smoke test"
IMAGE_INSTALL:append = " observation-check"
```

The leading space in the appended string matters: it separates package names. `:append` is a metadata operation, not a Python list append. A recipe name and a package name often match but are not interchangeable concepts; inspect package outputs if they differ. [Image customization manual](https://docs.yoctoproject.org/5.0.20/dev-manual/customizing-images.html).

A packagegroup is a reusable bundle of runtime dependencies, useful when several images need the same application set. One package does not need one yet. Features are larger policy choices that may affect several packages or configuration settings; adding a feature should have a reason you can test.

## Build and verify on the target

**Documented exercise, not an image build executed here:**

```bash
bitbake-getvar -r robot-course-image IMAGE_INSTALL
bitbake robot-course-image
ls tmp/deploy/images/qemux86-64
runqemu qemux86-64 robot-course-image nographic slirp
```

Inside the **guest**, run:

```bash
observation-check 100 125 40
observation-check 100 141 40
printf 'exit status: %s\n' "$?"
```

Expect `{"status":"fresh","age_ms":25}`, then `{"status":"stale","age_ms":41}`, then exit status `2`. Check `$?` immediately; another command replaces it. These exact outputs are native-tested; the guest execution is your integration gate.

Retain the image's generated package manifest in the deploy directory. Locate the age-check package and compare with the target. Record the selected rootfs artifact's exact filename and checksum; the directory may contain several formats and symlinks, not all of which are the one you booted.

## Change one package at a time

Add `strace` to the custom image, rebuild, and compare the package manifest and the same image format's size. Do not compare a compressed tarball with an uncompressed filesystem and call the difference package overhead. Image padding, filesystem free space, dependencies, and compression also affect size.

If `bitbake observation-check` succeeds but the target command is absent, inspect image selection first. If the manifest contains it but the guest does not, check which image was booted. If the file exists but execution fails, inspect permissions, architecture, dynamic loader, and libraries.

**Self-check:** does `IMAGE_INSTALL:append = "observation-check"` safely add a package? **Answer:** not reliably; without the leading space it can join the previous package name. The supplied recipe uses `" observation-check"`.

**Artifact:** before/after manifests, exact boot artifact, target outputs and exit codes, and a size comparison with matching formats.

**Previous:** [Lecture 07](Lecture-07.md) · **Next:** [Lecture 09](Lecture-09.md)
