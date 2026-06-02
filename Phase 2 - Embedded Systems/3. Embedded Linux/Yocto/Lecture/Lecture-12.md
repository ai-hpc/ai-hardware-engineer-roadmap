# Lecture 12 — Module 10: Licenses, compliance, and supply-chain hygiene

**Course:** [Yocto guide](../Guide.md) | **Phase 2 — Embedded Linux, Yocto**

**Previous:** [Lecture 11](Lecture-11.md) | **Next:** [Lecture 13 — Module 11](Lecture-13.md)

---

## 1. Why this matters in hardware products

Your firmware image bundles **third-party copyrights**. Customers and acquirers ask for **software BOMs**, license texts, and source offers where required.

## 2. Practical habits

- Enable license manifest generation appropriate to your release (variable names evolve; follow current docs).
- Treat **LICENSE** and **LIC_FILES_CHKSUM** in recipes as first-class review items.
- **Pin branches** or use mirrors for anything that must be rebuildable in five years.

## 3. Lab 10 — Inspect the manifests

Generate your image and locate the **license manifest** artifacts. Pick three packages and verify: SPDX or license file is recorded, and the version matches what you thought you shipped.

---

**Previous:** [Lecture 11](Lecture-11.md) | **Next:** [Lecture 13 — Module 11](Lecture-13.md)
