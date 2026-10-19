# Lecture 7 — Module 5: layers and small overlays

**Course:** [Yocto guide](../Guide.md) · **Previous:** [Lecture 06](Lecture-06.md) · **Next:** [Lecture 08](Lecture-08.md)

A layer gives your product changes a clear owner. Keep the course application in `meta-robot-course` so upgrading upstream metadata does not erase local edits.

## Read the layer configuration

Open [conf/layer.conf](../Labs/meta-robot-course/conf/layer.conf). `BBFILES` identifies recipes and appends, the collection name gives the layer an identity, and `LAYERDEPENDS` declares that it uses core metadata. `LAYERSERIES_COMPAT_robotcourse = "scarthgap"` states the supported series for this teaching layer. It is a claim that needs validation, not a switch that makes an incompatible layer work.

The upstream tool can create the initial skeleton for an independent layer:

```bash
bitbake-layers create-layer ../meta-my-robot
bitbake-layers add-layer ../meta-my-robot
bitbake-layers show-layers
```

These are **upstream-documented exercise commands**; use the existing supplied layer for the main course and create a second layer only for this ownership exercise. Layer priority helps choose among overlapping recipe providers. It is not a universal “every variable in the highest layer wins” rule; inspect evaluated metadata and appends. [Layer manual](https://docs.yoctoproject.org/5.0.20/dev-manual/layers.html).

## Worked overlay: package a contract note

In your separate `meta-my-robot` layer, create `recipes-apps/observation-check/observation-check_1.0.bbappend`:

```bitbake
do_install:append() {
    install -d "${D}${datadir}/observation-check"
    printf '%s\n' 'Contract: milliseconds, one clock domain, inclusive age limit.'         > "${D}${datadir}/observation-check/contract.txt"
}
FILES:${PN} += "${datadir}/observation-check"
```

This adds a packaged explanation without copying the entire base recipe. The file does not change runtime behavior; the program still receives its limit on the command line. A source or configuration file referenced by an append may also need `FILESEXTRAPATHS:prepend := "${THISDIR}/files:"`; the trailing colon separates search paths, and immediate expansion preserves the append's directory.

**Inspection exercise:**

```bash
bitbake-layers show-appends
bitbake-getvar -r observation-check do_install
bitbake observation-check
oe-pkgdata-util list-pkg-files observation-check
```

Check that the append appears, its install body is present, and the extra file reaches the package. If the base version later changes to `1.1`, a `1.0.bbappend` no longer matches. An intentional `%` version wildcard needs review against new upstream behavior; it should not hide compatibility work.

**Negative case:** rename the append to refer to a nonexistent version in your disposable exercise layer. Observe the unmatched-append diagnostic, then restore the name. Keep the failure log, not an ignored error setting.

**Self-check:** can two layers include recipes with the same name? **Answer:** yes; provider/version/priority rules then matter. Can a layer declare compatibility with Wrynose simply to suppress an error? It can change the text, but that does not validate paths, classes, API changes, or package output.

**Artifact:** a base/append diagram, the effective install function, and an extra-file package listing if built.

**Previous:** [Lecture 06](Lecture-06.md) · **Next:** [Lecture 08](Lecture-08.md)
