# Lecture 4 — Module 2: trace source to target

**Course:** [Yocto guide](../Guide.md) · **Previous:** [Lecture 03](Lecture-03.md) · **Next:** [Lecture 05](Lecture-05.md)

Follow one file through the build before editing metadata. The most common beginner confusion is treating compilation, installation into a staging directory, package creation, and image installation as the same event.

## A concrete path

```text
observation-check.c
  do_fetch / do_unpack    → source becomes available in the work directory
  do_compile              → target executable
  do_install              → staged /usr/bin/observation-check under ${D}
  do_package              → files assigned to runtime packages
  image do_rootfs         → selected packages installed into the image rootfs
  boot                    → target can execute /usr/bin/observation-check
```

`${D}` is a staging root inside the build tree. `install ... ${D}${bindir}` does not install the program into your host's `/usr/bin`. `${bindir}` expresses the target's binary-directory convention.

A recipe may create several packages: main runtime, debug symbols, development headers, or others. Building the recipe does not automatically add its package to an image. The image must select the package, directly or through runtime dependencies.

## Which file owns which decision?

| Object | Its job in this course |
|---|---|
| `conf/bblayers.conf` | Tell BitBake which metadata layers are available |
| `conf/local.conf` | Local build choices such as machine and resource settings |
| `conf/layer.conf` | Describe a layer's recipe discovery and compatibility |
| `observation-check_1.0.bb` | Build and install the age-check program |
| `robot-course-image.bb` | Select the age-check package for the image |
| `MACHINE` | Select target hardware assumptions |
| `DISTRO` | Select distribution policy |

These are the Scarthgap baseline names. Wrynose's setup tooling can manage configuration fragments and `toolcfg.conf`; the underlying ownership questions still apply.

## Inspect before running

Open the [supplied recipe](../Labs/meta-robot-course/recipes-apps/observation-check/observation-check_1.0.bb) and [image recipe](../Labs/meta-robot-course/recipes-core/images/robot-course-image.bb). Predict which task changes if you edit the C source versus if you add a package to `IMAGE_INSTALL`.

After Lecture 5 initializes the environment, these **upstream-documented inspection commands** show effective values:

```bash
bitbake-getvar -r observation-check S
bitbake-getvar -r observation-check do_install
bitbake -c listtasks observation-check
```

They require the layer to have been added in Lecture 6. The [debugging manual](https://docs.yoctoproject.org/5.0.20/dev-manual/debugging.html) explains how to inspect evaluated metadata rather than guessing from one file.

**Self-check:** the executable exists under a recipe's staging root but is absent after boot. Where do you look? **Answer:** check package file assignment, the image's selected package list, and whether you booted the matching rebuilt image. A successful compiler run only established the earliest part of that chain.

**Artifact:** annotate the source-to-target diagram with the exact recipe, package, image, and target paths from your project.

**Previous:** [Lecture 03](Lecture-03.md) · **Next:** [Lecture 05](Lecture-05.md)
