# Yocto Project — Embedded Linux Distribution Engineering

<div class="course-identity auto-course" style="--course-accent: #0284c7; --course-accent-rgb: 2, 132, 199;" markdown="1">
<div class="course-identity__icon">YPEL</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Embedded Systems</p>
<p class="course-identity__title">Specialized course identity for Yocto Project — Embedded Linux Distribution Engineering.</p>
<p class="course-identity__meta">Artifact: bring-up or firmware demo · Measure: boot, latency, power, reliability</p>
</div>
</div>


A structured course for hardware and software engineers who need to **build, own, and ship** a custom embedded Linux—not just install one. The goal is clarity first: you should finish with a **mental model** that survives release renames and new boards.

**Time investment (typical):** 4–10 weeks part-time for Modules 0–6; full course including production topics 3–6 months alongside real hardware.

**What you will be able to do**

- Explain how BitBake, recipes, layers, and images relate end-to-end.
- Reproduce a build from metadata alone (team-ready, not "works on my laptop").
- Add a package, patch a component, and ship a smaller, testable image.
- Debug failures using logs and task graphs instead of guessing.
- Know when Yocto is the right tool—and when Buildroot or a vendor BSP is faster.

---

## Step-by-step lectures

Each module is a separate file under **[Lecture/](Lecture/README.md)** with labs and previous/next navigation. Work in order: [Lecture-01](Lecture/Lecture-01.md) through [Lecture-16](Lecture/Lecture-16.md).

| # | Topic | Lecture |
|---|--------|---------|
| 1 | How to use this course | [Lecture-01.md](Lecture/Lecture-01.md) |
| 2 | Module 0 — Prerequisites and host setup | [Lecture-02.md](Lecture/Lecture-02.md) |
| 3 | Module 1 — What Yocto is (and is not) | [Lecture-03.md](Lecture/Lecture-03.md) |
| 4 | Module 2 — Architecture in one picture | [Lecture-04.md](Lecture/Lecture-04.md) |
| 5 | Module 3 — First successful build | [Lecture-05.md](Lecture/Lecture-05.md) |
| 6 | Module 4 — Recipes: the atoms of the system | [Lecture-06.md](Lecture/Lecture-06.md) |
| 7 | Module 5 — Layers: how metadata stays maintainable | [Lecture-07.md](Lecture/Lecture-07.md) |
| 8 | Module 6 — Images, packages, and features | [Lecture-08.md](Lecture/Lecture-08.md) |
| 9 | Module 7 — Kernel, bootloader, device tree | [Lecture-09.md](Lecture/Lecture-09.md) |
| 10 | Module 8 — SDK and application workflow | [Lecture-10.md](Lecture/Lecture-10.md) |
| 11 | Module 9 — Debugging builds like an engineer | [Lecture-11.md](Lecture/Lecture-11.md) |
| 12 | Module 10 — Licenses, compliance, supply-chain | [Lecture-12.md](Lecture/Lecture-12.md) |
| 13 | Module 11 — Performance, caching, and CI | [Lecture-13.md](Lecture/Lecture-13.md) |
| 14 | Capstone projects | [Lecture-14.md](Lecture/Lecture-14.md) |
| 15 | Glossary and quick reference | [Lecture-15.md](Lecture/Lecture-15.md) |
| 16 | Further reading (links + maintainer tip) | [Lecture-16.md](Lecture/Lecture-16.md) |

---

All step-by-step content, labs, glossary, and external links live in the **[Lecture/](Lecture/README.md)** files above.

**Pro tip:** when documentation and the internet disagree, trust the **docs matching your exact Yocto release branch**—metadata syntax and variable names do evolve.
