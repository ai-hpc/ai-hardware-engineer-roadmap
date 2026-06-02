# Lecture 2 — Module 0: Prerequisites and host setup

**Course:** [Yocto guide](../Guide.md) · **Phase 2 — Embedded Linux → Yocto**

**Previous:** [Lecture 01](Lecture-01.md) · **Next:** [Lecture 03 — Module 1](Lecture-03.md)

---

## 1. Skills you should already have

- Comfortable on the **Linux command line** (paths, environment variables, `grep`, basic scripting).
- Basic embedded concepts: bootloader → kernel → root filesystem; what a **device tree** is at a high level.
- **Git workflows** (clone, branch, commit). Yocto *is* a Git-heavy ecosystem.

## 2. Host machine expectations

Yocto builds are **disk- and I/O-heavy** more than they are “mystical.”

| Resource | Practical minimum | Comfortable |
|----------|-------------------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB (small images only) | 16–32 GB |
| Disk (SSD strongly preferred) | 80 GB free | 200+ GB |
| OS | Linux (supported distro for your Yocto release) | Same, on ext4 or similar |

Use a **case-sensitive** filesystem for the build tree. On some platforms, case-insensitivity causes bizarre, hard-to-debug failures.

## 3. Lab 0 — Verify your host

- Install your distro’s build dependencies using the **official Yocto guide for your release** (package names drift by distro).
- Confirm `git`, `python3`, and a working compiler toolchain for the *host* are present.
- Create a dedicated user or directory convention so paths stay stable (teams will thank you).

**Done when:** you can clone a large Git repository and compile a small native C “hello world” on the host without errors.

---

**Previous:** [Lecture 01](Lecture-01.md) · **Next:** [Lecture 03 — Module 1](Lecture-03.md)
