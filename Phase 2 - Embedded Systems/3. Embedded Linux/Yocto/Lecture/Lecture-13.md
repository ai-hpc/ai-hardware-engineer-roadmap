# Lecture 13 — Module 11: Performance, caching, and CI

**Course:** [Yocto guide](../Guide.md) | **Phase 2 — Embedded Linux, Yocto**

**Previous:** [Lecture 12](Lecture-12.md) | **Next:** [Lecture 14 — Capstone](Lecture-14.md)

---

## 1. Shared state (sstate) in one sentence

**sstate** caches task outputs so clean builds **skip work that has not changed**, when configured correctly.

## 2. CI design goals

Use **deterministic branches** for release streams, not moving-main surprises. **Separate download and sstate directories** on shared storage with locking discipline. Pin layers with SRCREV or manifest tools appropriate to your process.

## 3. Lab 11 — Measure a rebuild

Time a **no-op rebuild** with warm cache vs after wiping tmp but keeping downloads. Record the delta.

**Done when:** you can explain what you would never delete in CI vs what is safe to scrub.

---

**Previous:** [Lecture 12](Lecture-12.md) | **Next:** [Lecture 14 — Capstone](Lecture-14.md)
