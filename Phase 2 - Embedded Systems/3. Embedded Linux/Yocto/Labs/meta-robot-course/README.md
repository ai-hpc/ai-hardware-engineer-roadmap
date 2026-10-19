# meta-robot-course

A teaching layer for the course's pinned Yocto Scarthgap 5.0.20 baseline. It supplies `observation-check` and `robot-course-image`.

Use the [lab guide](../README.md) in the roadmap for native tests, the source pin, integration commands, and the exact validation boundary. Copy this entire layer to a no-space workspace before adding it with `bitbake-layers add-layer`.

The program and metadata are covered by the roadmap's MIT license; a copy of that license accompanies the application source. Layer compatibility is declared only for Scarthgap. The Wrynose route needs its own metadata, target-build, and boot validation before adding that compatibility claim.
