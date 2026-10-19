# Lecture 6 — Module 4: build your first application recipe

**Course:** [Yocto guide](../Guide.md) · **Previous:** [Lecture 05](Lecture-05.md) · **Next:** [Lecture 07](Lecture-07.md)

A recipe turns a known source program into files the target can install. Start with the supplied age checker: its native behavior is already tested, so you can focus on the build contract.

## Put the teaching layer in your workspace

From the **roadmap repository root**, remember its absolute path and copy the supplied layer into the no-space workspace. The guard preserves an existing working copy; later edits belong in that copy.

```bash
COURSE_REPO="$PWD"
YOCTO_WORK="$HOME/yocto-course"
if [ ! -e "$YOCTO_WORK/meta-robot-course" ]; then
    cp -a "$COURSE_REPO/Phase 2 - Embedded Systems/3. Embedded Linux/Yocto/Labs/meta-robot-course" "$YOCTO_WORK/meta-robot-course"
fi
cd "$YOCTO_WORK"
source poky/oe-init-build-env build
bitbake-layers add-layer ../meta-robot-course
bitbake-layers show-recipes observation-check
```

The copy is important: layer/build paths containing spaces are unsuitable for this beginner workflow. Lecture 7 explains the layer configuration. **Tested here:** adding a no-space copy of this layer and parsing it against the pinned Scarthgap metadata; see the [validation record](../Labs/README.md).

## Read the recipe from top to bottom

The [actual recipe](../Labs/meta-robot-course/recipes-apps/observation-check/observation-check_1.0.bb) contains:

```bitbake
SUMMARY = "Check observation age in an explicit millisecond clock domain"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://LICENSE;md5=0304269c7842bb285234556ec250ca54"
SRC_URI = "file://observation-check.c file://LICENSE"
S = "${WORKDIR}"

do_compile() {
    ${CC} ${CPPFLAGS} ${CFLAGS} ${LDFLAGS} -std=c11         "${S}/observation-check.c" -o "${B}/observation-check"
}

do_install() {
    install -d "${D}${bindir}"
    install -m 0755 "${B}/observation-check" "${D}${bindir}/observation-check"
}
```

`SRC_URI` lists local files in the recipe's `files/` directory. `S` selects where compilation reads sources in this **Scarthgap** example. `${CC}` is the configured target compiler, which can include flags; replacing it with host `gcc` can produce the wrong binary. `${D}` keeps installation inside a staging root. The `1.0` in the filename supplies the recipe version. The license checksum corresponds to the supplied license text; it is not a source-integrity or security checksum.

For a CMake or Meson project, use the appropriate inherited class rather than reimplementing its toolchain rules. `DEPENDS` names build-time recipe dependencies; `RDEPENDS:${PN}` names runtime package dependencies. The tiny checker uses the target C runtime, and normal shared-library dependency processing handles that connection. See [writing a recipe](https://docs.yoctoproject.org/5.0.20/dev-manual/new-recipe.html).

## Build and inspect the package

**Upstream-documented target-build exercise; target compilation not run in this review:**

```bash
bitbake observation-check
oe-pkgdata-util list-pkg-files observation-check
bitbake-getvar -r observation-check WORKDIR
```

Look for `/usr/bin/observation-check` in the package file list. A built package still needs image selection in Lecture 8. If a build fails, inspect the named task log before changing the recipe.

**Guided change:** add a helpful usage sentence to the C source, rebuild, and record which tasks rerun. Preserve the freshness behavior. **Self-check:** why does changing your host's `/usr/bin/gcc` not define the target compiler? **Answer:** BitBake provides a target toolchain and sysroot through recipe variables; the host compiler is used for different parts of the build.

**Artifact:** native test result, effective `PN/PV/S/CC`, package file list if built, and your explanation of every install path.

**Previous:** [Lecture 05](Lecture-05.md) · **Next:** [Lecture 07](Lecture-07.md)
