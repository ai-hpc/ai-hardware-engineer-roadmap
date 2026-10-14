# Lecture 13 — Module 11: Performance, caching, and CI

**Course:** [Yocto guide](../Guide.md) | **Phase 2 — Embedded Linux, Yocto**

**Previous:** [Lecture 12](Lecture-12.md) | **Next:** [Lecture 14 — Capstone](Lecture-14.md)

---

## 1. Shared state (sstate) in one sentence

**sstate** caches task outputs so clean builds **skip work that has not changed**, when configured correctly.

## 2. CI design goals

Pin layer repositories in a manifest to exact commits, and pin recipe Git sources with `SRCREV`; these are different levels of version control. Record host/toolchain inputs and configuration. Separate download and sstate caches; a cache hit is not proof of reproducible output. See [Yocto build-history source records](https://docs.yoctoproject.org/current/dev-manual/build-quality.html).

## 3. Lab 11 — Measure a rebuild

Use separate disposable build directories to compare a warm no-op build with a build that reuses downloads and sstate. Keep the layer revisions and configuration identical; record executed tasks as well as elapsed time.

**Done when:** you can explain what you would never delete in CI vs what is safe to scrub.

---

**Previous:** [Lecture 12](Lecture-12.md) | **Next:** [Lecture 14 — Capstone](Lecture-14.md)
