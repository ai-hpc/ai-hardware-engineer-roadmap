# Lecture 10 — Module 8: build applications with the matching SDK

**Course:** [Yocto guide](../Guide.md) · **Previous:** [Lecture 09](Lecture-09.md) · **Next:** [Lecture 11](Lecture-11.md)

An SDK gives an application developer the target compiler, headers, libraries, and environment needed to build outside the full image tree. Its sysroot represents the intended target interface; it does not turn a host executable into target code afterward.

## Generate an image-specific SDK

**Upstream-documented exercise; SDK construction not run here:**

```bash
bitbake robot-course-image -c populate_sdk
ls tmp/deploy/sdk
```

Choose the generated installer corresponding to this build and record its checksum. Run that **exact filename** with `sh`, choose a dedicated installation directory, and read the printed setup instructions. Avoid a wildcard that accidentally executes several old installers. Source the exact `environment-setup-*` file installed for this SDK in a fresh terminal. [Standard SDK manual](https://docs.yoctoproject.org/5.0.20/sdk-manual/using.html).

```text
image metadata + package versions
          ├── target rootfs
          └── SDK toolchain + target sysroot → application binary
```

Record the image and SDK manifests together. Similar version strings alone do not prove identical ABI or package configuration.

## Rebuild the same small application

After sourcing the SDK environment, set `APP_SOURCE` to your copied `meta-robot-course/recipes-apps/observation-check/files/observation-check.c` and run:

```bash
APP_SOURCE="$HOME/yocto-course/meta-robot-course/recipes-apps/observation-check/files/observation-check.c"
${CC} ${CPPFLAGS} ${CFLAGS} ${LDFLAGS} -std=c11 "$APP_SOURCE" -o observation-check-sdk
file observation-check-sdk
```

The unquoted expansions for SDK compiler/flags are intentional: the trusted SDK defines a command plus arguments. Do not replace `${CC}` with a guessed cross-compiler name or add host include paths to make a missing target header disappear. This command is an **SDK exercise**; native tests used the host compiler and establish a different execution boundary.

Transfer the binary through your documented QEMU/board workflow and run the same fresh, boundary, stale, and future-timestamp cases from the lab README. The default minimal image does not promise an SSH server, so do not assume `scp` works without adding/configuring one or choosing another transfer method.

## A useful failure case

Compile using your host `cc` in a separate output directory, inspect it with `file`, and compare with the SDK-built artifact. On an x86 host with a QEMU x86 target, architecture alone may match; different libc/loader paths or versions can still make the host binary unsuitable. On different ISAs the mismatch is more obvious. Do not execute untrusted test files or overwrite the packaged target program during this comparison.

**Self-check:** if a library header exists on the host but is absent from the target sysroot, should you add `/usr/include` to the cross-build? **Answer:** no. Fix the target dependency/SDK content, then rebuild the matching SDK or application environment. Host headers can create a misleading successful compile with an invalid target ABI.

**Artifact:** SDK identity, source revision, compilation command, binary metadata, transfer method, and target output. Carry any mismatch into the debugging lesson.

**Previous:** [Lecture 09](Lecture-09.md) · **Next:** [Lecture 11](Lecture-11.md)
