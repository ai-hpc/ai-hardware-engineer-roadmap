# Lecture 2 — Module 0: prerequisites and host setup

**Course:** [Yocto guide](../Guide.md) · **Previous:** [Lecture 01](Lecture-01.md) · **Next:** [Lecture 03](Lecture-03.md)

A build host must support the chosen Yocto release, have enough storage and memory, and use a predictable shell environment. Verify those facts before downloading a toolchain.

## Choose a release deliberately

**Reviewed 2026-10-18:** the official release page lists **Wrynose 6.0.3** as LTS through April 2030 and **Scarthgap 5.0.20** as LTS through April 2028. Blacksail 6.1 is listed as active development. Kirkstone and Whinlatter are EOL there. Use the [official release page](https://www.yoctoproject.org/development/releases/) when revisiting this choice; an old tutorial's codename does not establish current support.

The step-by-step course uses **Scarthgap 5.0.20** with `qemux86-64`: a supported, lower-resource teaching baseline with a pinned checkout. Wrynose has a newer setup workflow; Lecture 5 explains where to follow it. Do not mix layer series just because a later release exists. A vendor board must follow its BSP compatibility matrix.

| Published quick-build baseline | Scarthgap 5.0.20 | Wrynose 6.0.3 |
|---|---|---|
| Free disk | At least 90 GB | At least 140 GB |
| RAM | At least 8 GB | At least 32 GB |
| Python | 3.8 or later | 3.9 or later |
| GCC | 8.0 or later | 10.1 or later |

These are upstream quick-build requirements, not a promise that every image fits. Multiple builds, debug artifacts, and caches need more. Check the selected release's supported host distribution and complete package list: [5.0.20 quick build](https://docs.yoctoproject.org/5.0.20/brief-yoctoprojectqs/index.html), [6.0.3 quick build](https://docs.yoctoproject.org/6.0.3/brief-yoctoprojectqs/index.html).

## Inspect the host

**Tested here on Ubuntu 24.04.4; read-only commands:**

```bash
cat /etc/os-release
uname -m
python3 --version
git --version
cc --version
df -h .
free -h
locale -a
```

Record the output, and compare it with the chosen release. Check `en_US.utf8` is available. Install missing host packages using that release's official list. Package names differ across distributions; a copied historical list may omit `zstd`, locale data, or newer tools.

Use a case-sensitive Linux filesystem and a **build path without spaces**, such as `$HOME/yocto-course`. This roadmap has spaces in directory names: copy the supplied layer into the no-space workspace before adding it. Build as your normal user. Do not run BitBake through `sudo` to work around a permissions error.

## First executable result

Run the [native package checks](../Labs/README.md#native-checks). They compile a small C program using only the host compiler and Python standard library, without downloading an image. If compilation fails, fix the native compiler first. If the command works but a BitBake host-tools check later fails, those are separate environments and requirements.

**Practice:** predict which observations fail for `capture=100`, `complete=141`, `limit=40`, then run the program. Save the exit status and output.

**Self-check:** why is enough free disk alone insufficient? **Answer:** memory, supported tools/distribution, locale, filesystem semantics, and network/source availability also affect a build. A container does not create extra RAM or disk.

**Previous:** [Lecture 01](Lecture-01.md) · **Next:** [Lecture 03](Lecture-03.md)
