# Lecture 14 — Capstone: hand another engineer a release bundle

**Course:** [Yocto guide](../Guide.md) · **Previous:** [Lecture 13](Lecture-13.md) · **Next:** [Lecture 15](Lecture-15.md)

Complete one connected project rather than collecting unrelated screenshots. The default capstone is a QEMU observation-check appliance with a matching SDK and a repeatable acceptance procedure.

## Define acceptance before building

The packaged command must behave as follows in the target image:

| Inputs: capture, completion, limit (ms) | Expected JSON status | Exit code |
|---|---|---:|
| `100 125 40` | `fresh`, age 25 | 0 |
| `100 140 40` | `fresh`, age 40 | 0 |
| `100 141 40` | `stale`, age 41 | 2 |
| `150 140 40` | `clock_error` | 65 |
| `-1 140 40` | No JSON; usage on stderr | 64 |

This is an application acceptance fixture. It does not prove a clock is synchronized, a sensor is calibrated, or a robot is safe to actuate.

## Required handoff

Keep a pinned source/layer manifest, a reviewed configuration, the product layer, generated image and SDK identities, package/license/SBOM records, boot log, and target command results. Include how to obtain preserved sources and how to rebuild from a fresh shell. List known missing capabilities and measured resource needs.

A second learner should be able to identify the exact target, reproduce the checkout, understand which steps require a long build, boot the intended image, and run the five acceptance cases without guessing paths or commands. A successful image build with no boot evidence is an image-build milestone, not completion of the full capstone.

## Choose one independent extension

- **Runtime integration:** package the earlier Python observation replay with its exact runtime dependencies and a startup service. Preserve reset, freshness, and failure behavior; prove restart/resource cleanup on the target.
- **Hardware integration:** replace a simulated observation source with a supported peripheral. Document clocks, units, driver binding, error recovery, and capture provenance.
- **Release engineering:** rebuild the same artifact set in a second controlled environment, compare manifests/hashes, and explain every material difference.

Only choose the hardware extension when you have the required board and documentation. The QEMU route is a complete software-integration exercise, while electrical and physical timing evidence remain outside it.

## Assessment

Use the [shared completion rubric](../../../../Learning-Guide.md#completion-rubric): correctness, reasoning, measurement, reproduction, and transfer. Aim for 8/10 or better, with full correctness and reproduction. The observation CLI's native tests are the beginning of the evidence, not the entire course.

**Reduced route:** if you cannot build an image, submit native tests, parsed metadata if available, a source-to-target map, a failure/repair record, and a fully specified target acceptance procedure. Label the result **native/metadata milestone**, with image build, SDK, boot, and target execution marked **not run**. That is useful work without claiming unavailable hardware evidence.

**Self-check:** what would make the handoff fail even when your laptop can boot it? **Answer:** missing source pins, undocumented local changes, unavailable source inputs, mismatched SDK, an ambiguous boot artifact, or acceptance commands that only work in your interactive shell.

**Previous:** [Lecture 13](Lecture-13.md) · **Next:** [Lecture 15](Lecture-15.md)
