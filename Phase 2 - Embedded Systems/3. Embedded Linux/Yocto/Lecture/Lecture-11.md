# Lecture 11 — Module 9: Debugging builds like an engineer

**Course:** [Yocto guide](../Guide.md) | **Phase 2 — Embedded Linux, Yocto**

**Previous:** [Lecture 10](Lecture-10.md) | **Next:** [Lecture 12 — Module 10](Lecture-12.md)

---

## 1. Read failures from the top of the log

When BitBake reports failure:

1. Identify the **recipe** and **task** (`do_compile`, `do_configure`, …).
2. Open the **task log** file path printed in the error.
3. Scroll to the **first** compiler or configuration error, not the last cascading line.

## 2. High-signal commands (conceptual)

- bitbake -g TARGET (replace TARGET with your image or recipe) for dependency graph artifacts.
- `devtool` (when available in your workflow) — modify sources in a controlled workspace.

## 3. Common failure classes

| Symptom | Often means |
|---------|-------------|
| Fetch failures | Network, proxy, missing `SRC_URI` checksum, retired URLs |
| Patch rejects | Version drift; branch pin wrong |
| CMake/autotools missing deps | `DEPENDS` incomplete; wrong `PACKAGECONFIG` |
| Rootfs size explosions | accidental `-dev` packages in image |

## 4. Lab 9 — Break it on purpose

Remove a **dependency** in a *throwaway* layer copy, rebuild, and practice **tracing the log back to root cause**. Revert the change.

**Done when:** you can articulate *why* the failure belongs to fetch vs compile vs packaging.

---

**Previous:** [Lecture 10](Lecture-10.md) | **Next:** [Lecture 12 — Module 10](Lecture-12.md)
