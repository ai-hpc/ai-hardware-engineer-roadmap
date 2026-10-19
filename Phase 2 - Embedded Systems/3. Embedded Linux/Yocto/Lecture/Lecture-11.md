# Lecture 11 — Module 9: debug the failing boundary

**Course:** [Yocto guide](../Guide.md) · **Previous:** [Lecture 10](Lecture-10.md) · **Next:** [Lecture 12](Lecture-12.md)

Find the first boundary where evidence disagrees with your prediction. A build error, missing package, and stale runtime result need different fixes.

## Read a failure in order

Record the target command and source/configuration revisions. In BitBake's error output, identify the **recipe and task**, then open the exact `log.do_TASK` path shown. Inspect the matching `run.do_TASK` script when you need the expanded command. Locate the earliest relevant compiler/configuration error; later errors can be consequences. [Debugging manual](https://docs.yoctoproject.org/5.0.20/dev-manual/debugging.html).

| Boundary | Example failure | Evidence to inspect |
|---|---|---|
| Parse | Invalid metadata syntax or unmatched append | Filename/line and effective layer configuration |
| Fetch/unpack | Missing local file, unreachable pinned source | `SRC_URI`, search paths, task log |
| Configure/compile | Missing target header, deliberate `#error` | Dependency, sysroot, actual compiler command |
| Install/package | File created outside staging root or not packaged | `${D}`, `FILES`, package QA output |
| Rootfs | Requested package has no provider/conflicting dependencies | Image package list and rootfs task log |
| Boot/runtime | Wrong image, wrong architecture, expired observation | Boot identity, file metadata, program status |

## Worked negative case

Use a copy of the C source from your own product layer. Add `#error COURSE_DELIBERATE_ERROR` at the top. A native compilation should fail with that marker. Restore the file and rerun the seven native tests. This is a small, bounded failure you can learn from without rebuilding an OS.

If the target toolchain has already been built, repeat through `bitbake observation-check`; identify `do_compile` in the failure and inspect its log. Do not claim this target path ran just because the native error was detected.

Now consider a different fault: rename `observation-check.c` in `SRC_URI` to a nonexistent filename in a disposable copy. This fails earlier, because there is no source to compile. Predict the boundary before running; restore the filename afterward. Save a small diff for each fault so the repair is explicit.

## Inspect values and dependencies

**Upstream-documented commands:**

```bash
bitbake-getvar -r observation-check SRC_URI
bitbake-getvar -r observation-check do_compile
bitbake -g observation-check
```

The graph command produces dependency artifacts such as `task-depends.dot`; it does not prove that a target executable was built. Use `bitbake -e observation-check` when you need full metadata history, but avoid publishing environment dumps containing private paths, mirrors, or credentials.

Avoid adding `-f`, `cleanall`, or arbitrary host include paths until you understand why the ordinary build did not do what you expected. Those actions can hide a dependency error or discard evidence.

**Self-check:** a checksum mismatch appeared after upstream source changed. Should you paste the newly printed checksum and retry? **Answer:** first inspect the actual change and intended source revision. Updating a checksum blindly removes the signal the check was designed to provide.

**Artifact:** two failure records, each with prediction, first failing task, diagnostic, minimal repair, and restored test result.

**Previous:** [Lecture 10](Lecture-10.md) · **Next:** [Lecture 12](Lecture-12.md)
