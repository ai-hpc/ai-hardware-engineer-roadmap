# Lecture 15 — Glossary and command recovery

**Course:** [Yocto guide](../Guide.md) · **Previous:** [Lecture 14](Lecture-14.md) · **Next:** [Lecture 16](Lecture-16.md)

Use this page when a command or error introduces an unfamiliar object. Each term names an actual boundary in the course project.

| Term | Plain meaning | In this course |
|---|---|---|
| Recipe | Instructions for one buildable component | `observation-check_1.0.bb` |
| Package | Installable selection of files produced by a recipe | `observation-check` containing the CLI |
| Image | Selected rootfs and associated deployment outputs | `robot-course-image` |
| Layer | Versioned group of recipes/configuration | `meta-robot-course` |
| Append | Additional metadata applied to a matching recipe | Contract-note exercise |
| Class | Reusable build behavior | Image/package/toolchain integration |
| `MACHINE` | Target hardware selection | `qemux86-64` |
| `DISTRO` | Distribution policy | `poky` in the baseline |
| `PN` / `PV` | Recipe name / version | `observation-check` / `1.0` |
| `WORKDIR` | Per-recipe build work area | Generated files, staging, task logs |
| `S` / `B` | Source directory / build directory | Recipe's compilation inputs/outputs |
| `D` | Temporary installation root | Staged target file tree |
| Sysroot | Headers/libraries/tools supplied for a build context | SDK target headers and libraries |
| BSP | Hardware-specific support metadata/software | Needed for a real board extension |
| `DEPENDS` | Build-time recipe dependencies | Ingredients needed before building |
| `RDEPENDS` | Runtime package dependencies | Software the installed package needs |
| `SRCREV` | Selected revision of a recipe's Git source | Distinct from the layer's commit |
| sstate | Reusable task outputs identified by signatures | A cache, not a release manifest |

## Recover the build shell

For the course's Scarthgap layout:

```bash
cd "$HOME/yocto-course"
source poky/oe-init-build-env build
bitbake --version
bitbake-layers show-layers
```

If you use Wrynose's setup-managed layout, source that setup's generated environment file instead. Do not combine both environments in one terminal.

## Ask a precise question with a command

| Question | Command after environment setup |
|---|---|
| Is the recipe visible? | `bitbake-layers show-recipes observation-check` |
| Which append applies? | `bitbake-layers show-appends` |
| What value was selected? | `bitbake-getvar -r observation-check S` |
| What tasks exist? | `bitbake -c listtasks observation-check` |
| Do metadata files parse? | `bitbake -p` |
| Build the package? | `bitbake observation-check` |
| What files reached its package? | `oe-pkgdata-util list-pkg-files observation-check` |
| Build the product image? | `bitbake robot-course-image` |
| Build its SDK? | `bitbake robot-course-image -c populate_sdk` |

Package-data commands require package data to have been generated. A command's existence is not evidence that it has run successfully in your workspace.

## Distinctions to retrieve from memory

Explain recipe versus package; layer commit versus recipe source commit; host versus target; image versus SDK; source download versus sstate; and parsing versus booting. For each pair, point to your actual project artifact. If you cannot, return to the lesson that produces it.

**Answer guidance:** most confusing failures cross one of these boundaries. An installed host library does not satisfy a target sysroot dependency; a visible recipe is not an installed package; cached work is not a separate source rebuild.

**Previous:** [Lecture 14](Lecture-14.md) · **Next:** [Lecture 16](Lecture-16.md)
