# Lecture 13 — Module 11: understand rebuilds and caches

**Course:** [Yocto guide](../Guide.md) · **Previous:** [Lecture 12](Lecture-12.md) · **Next:** [Lecture 14](Lecture-14.md)

A faster rebuild is useful only if it produces the intended result. Learn what was reused before interpreting elapsed time.

## Three different kinds of reuse

| Item | What it preserves | What it does not prove |
|---|---|---|
| Download cache (`DL_DIR`) | Retrieved source inputs | That compilation was skipped or source is trusted |
| Shared state (`SSTATE_DIR`) | Reusable outputs of eligible tasks | That every task can be restored or output is independently reproducible |
| Existing work/stamps | Progress and task state in a build directory | That a clean worker can rebuild from the release record |

BitBake uses signatures and dependency information to decide what needs work. Changing C source, a compiler flag, or image membership can invalidate different tasks. A cached build is not a separate confirmation of bit-for-bit reproducibility. The [Yocto concepts manual](https://docs.yoctoproject.org/5.0.20/overview-manual/concepts.html) explains task signatures and shared state; [build history](https://docs.yoctoproject.org/5.0.20/dev-manual/build-quality.html) helps inspect output changes.

## A controlled rebuild experiment

Once you have built the application or image, compare these cases using the same target and configuration:

1. Repeat with unchanged sources in the same build directory.
2. Change only the application's usage text and rebuild.
3. Create a separate build directory with the same pinned inputs, pointing to the same download and sstate directories.

Record wall time, executed/restored task counts, source/configuration revisions, and output identity. Do not label all three “clean builds.” The third case has a clean work directory but warm shared caches.

**Upstream-style measurement exercise:** run from the initialized build shell:

```bash
/usr/bin/time -f 'elapsed_seconds=%e peak_rss_kib=%M' bitbake observation-check
bitbake-getvar DL_DIR
bitbake-getvar SSTATE_DIR
```

`time` reports process resource observations, not per-package energy or a whole-machine memory guarantee. Save BitBake's task summary alongside it. If the job fails, its elapsed time is a failed run, not a successful build benchmark.

## Design a small CI job

A useful job checks out exact layer/source revisions, restores explicitly named caches, records configuration, builds the requested target, runs the matching smoke tests, and archives logs/manifests/checksums. Keep credentials outside published logs. Store the tiny product layer independently of generated `tmp/` outputs.

Run an occasional rebuild in a separate controlled environment to investigate reproducibility. Compare like artifacts and explain timestamps, filesystem metadata, signing, or configuration differences. Do not delete a shared cache while other builds use it; disposable build directories are a safer place to isolate an experiment.

**Self-check:** a run restores most tasks from sstate and finishes quickly. Did it prove the sources compile from scratch? **Answer:** no. It demonstrated reuse under those signatures. A source rebuild and a cache-restoration test are different experiments.

**Artifact:** a three-row rebuild table with raw task summaries and an explanation of which work changed. Leave unexecuted rows clearly marked.

**Previous:** [Lecture 12](Lecture-12.md) · **Next:** [Lecture 14](Lecture-14.md)
