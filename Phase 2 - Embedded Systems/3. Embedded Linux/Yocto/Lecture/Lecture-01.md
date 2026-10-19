# Lecture 1 — How to use this course

**Course:** [Yocto guide](../Guide.md) · **Previous:** [Course hub](../Guide.md) · **Next:** [Lecture 02](Lecture-02.md)

Build one small robot-observation package, install it into a Linux image, and prove that the image and SDK belong together. Start with a CPU reference so packaging failures do not get confused with algorithm failures.

## The project you will carry through the course

The supplied `observation-check` program accepts capture time, completion time, and an age limit in milliseconds. It emits one JSON result and returns a meaningful exit code. It teaches a small part of the [observation contract](../../../Labs/Observation-Freshness.md); it is not an inference engine or a complete robot supervisor.

```text
C source → recipe → target package → custom image → boot → target smoke test
                    ↑                       ↑
               product layer          matching SDK + manifest
```

An executable working on your laptop proves the C behavior. A package build proves a cross-build and packaging step. A booted image with a passing command proves target integration. Keep these as separate evidence levels.

## Pick the amount of work your machine supports

| Route | Work you can finish | Claim at the end |
|---|---|---|
| Native CPU route | Compile/run the supplied C program, inspect metadata, plan the image and debug fixtures | Native checks and design exercise |
| Metadata route | Also initialize pinned Poky, inspect layers, parse recipes and effective variables | Metadata validation; no target package/image claim |
| Image route | Also build, boot QEMU, generate SDK, run target checks | A demonstrated image and SDK on the named target |

See the [lab files and validation status](../Labs/README.md). First-build downloads and toolchain compilation can take hours. Budget setup separately from each 30–60 minute lesson. Stop a build when resources become insufficient or a real failure occurs; save the first failure and resume after fixing it. Do not delete caches as the first response to a problem.

## Entry check

Explain `source script` versus running `./script`, a compiler versus an executable, and a Git commit versus a moving branch. If these are unfamiliar, complete Lecture 2 before running build commands. Also run the native package checks from the lab README; they should pass before you involve BitBake.

## How to record progress

Keep source, your layer, configuration, commands, logs, target output, and a revision record in one project. Use separate folders for inputs and generated outputs. Mark each result **native tested**, **metadata tested**, **target tested**, or **not run**. A screenshot of a successful terminal is useful context, but the actual command and artifact are what another learner can reproduce.

After each session, answer its small self-check without looking. Return two days later and explain the source-to-target path to someone else. Your final assessment is the release-bundle capstone in Lecture 14, using the [course rubric](../../../../Learning-Guide.md#completion-rubric).

**Answer guidance:** `source` changes the current shell environment; executing a script changes its child process. A cross-compiler runs on the host but produces target code. A branch can move; a recorded full commit identifies one source revision.

**Previous:** [Course hub](../Guide.md) · **Next:** [Lecture 02](Lecture-02.md)
