# Lecture 5 — Module 3: initialize and build a pinned baseline

**Course:** [Yocto guide](../Guide.md) · **Previous:** [Lecture 04](Lecture-04.md) · **Next:** [Lecture 06](Lecture-06.md)

Start with a reference QEMU image. Add your own application only after the baseline is understandable; this separates host/setup failures from product-layer failures.

## One reproducible Scarthgap checkout

The tag `yocto-5.0.20` was checked against upstream and resolves to commit `64e69ed23703f6358ec431d3f5f1f8483f974cae`. **Tested here:** cloning this tag, environment initialization, and BitBake version reporting. Full image construction and QEMU boot are **upstream-documented exercises, not executed in this course review**.

Run from a terminal whose host prerequisites passed Lecture 2. Choose a new no-space workspace:

```bash
mkdir -p "$HOME/yocto-course"
cd "$HOME/yocto-course"
git clone --depth 1 --branch yocto-5.0.20 https://git.yoctoproject.org/poky poky
git -C poky rev-parse HEAD
source poky/oe-init-build-env build
bitbake --version
```

Compare the full commit with the value above. A detached checkout is expected for the tag. BitBake reports version `2.8.1` in this checkout. `source` changes the current directory to the build directory and configures the current shell. On a later day, return to the workspace and source the same script again; cloning is not a daily step.

Set `MACHINE = "qemux86-64"` in `conf/local.conf`, replacing any active competing assignment. Choose `BB_NUMBER_THREADS` and `PARALLEL_MAKE` for available memory/CPU; four tasks and `-j4` are a conservative starting experiment, not universal optimal values.

## Check metadata before a long build

```bash
bitbake-layers show-layers
bitbake -p
bitbake-getvar MACHINE
```

Parsing does not compile an image. If a required host tool is absent, follow the release's package list. If the shell cannot find `bitbake`, source the environment again. If a namespace/sandbox check fails, retain the diagnostic and use a supported host configuration; do not disable checks just to obtain a green terminal.

## Build, then prove what booted

**Upstream-documented workflow, not run here:**

```bash
mkdir -p logs
set -o pipefail
bitbake core-image-minimal 2>&1 | tee logs/core-image-minimal.log
ls tmp/deploy/images/qemux86-64
runqemu qemux86-64 core-image-minimal nographic slirp
```

The build may take hours and download substantial sources. Watch free disk and the task summary. On the first real failure, save the recipe/task and its log path; use Lecture 11 rather than repeatedly restarting. A normal first interrupt asks BitBake to stop cleanly after active tasks.

Inside the QEMU guest, record `uname -a`, `cat /etc/os-release`, and the login/boot log. Host and guest commands are different: your host's `/etc/os-release` does not prove the image's identity. Use the selected image's documented development login settings. Keep those settings separate from a deployment configuration.

## Current Wrynose route

For Wrynose 6.0.3, follow the [matching quick build](https://docs.yoctoproject.org/6.0.3/brief-yoctoprojectqs/index.html): it introduces `bitbake-setup`, a `poky-wrynose` configuration, build configuration fragments, and a generated `build/init-build-env`. Do not substitute that layout into the relative paths above. Record the setup-tool version, generated configuration, and resolved repository commits. The supplied teaching layer declares only Scarthgap compatibility until separately validated on Wrynose.

**Self-check:** does `bitbake -p` passing mean the target boots? **Answer:** no. It establishes metadata parsing in that host/configuration. The next distinct gates are package construction, image construction, boot, and target behavior.

**Artifact:** source commit, configuration, parse result, and—if built—image files, checksums, boot identity, and logs. Sources: [Scarthgap quick build](https://docs.yoctoproject.org/5.0.20/brief-yoctoprojectqs/index.html).

**Previous:** [Lecture 04](Lecture-04.md) · **Next:** [Lecture 06](Lecture-06.md)
