# Lecture 16 — Further reading and keeping the course current

**Course:** [Yocto guide](../Guide.md) · **Previous:** [Lecture 15](Lecture-15.md) · **Next:** [Course hub](../Guide.md)

Read official documentation to answer the next concrete question in your project. Keep its version aligned with your checkout; a moving documentation homepage may describe a different setup or metadata model.

## Sources tied to the course baseline

| Need | Primary source |
|---|---|
| Choose a currently maintained series | [Yocto releases](https://www.yoctoproject.org/development/releases/) |
| Reproduce the Scarthgap setup | [5.0.20 quick build](https://docs.yoctoproject.org/5.0.20/brief-yoctoprojectqs/index.html) |
| Understand the current Wrynose setup | [6.0.3 quick build](https://docs.yoctoproject.org/6.0.3/brief-yoctoprojectqs/index.html) |
| Understand metadata operators | [BitBake 2.8 syntax](https://docs.yoctoproject.org/bitbake/2.8/bitbake-user-manual/bitbake-user-manual-metadata.html) |
| Add a layer or application | [Layers](https://docs.yoctoproject.org/5.0.20/dev-manual/layers.html), [recipes](https://docs.yoctoproject.org/5.0.20/dev-manual/new-recipe.html) |
| Select image content | [Image customization](https://docs.yoctoproject.org/5.0.20/dev-manual/customizing-images.html) |
| Trace a failure | [Debugging](https://docs.yoctoproject.org/5.0.20/dev-manual/debugging.html) |
| Use the matching toolchain | [Standard SDK](https://docs.yoctoproject.org/5.0.20/sdk-manual/using.html) |
| Preserve release evidence | [Build quality](https://docs.yoctoproject.org/5.0.20/dev-manual/build-quality.html), [licenses](https://docs.yoctoproject.org/5.0.20/dev-manual/licenses.html), [SBOM](https://docs.yoctoproject.org/5.0.20/dev-manual/sbom.html) |

## A release upgrade is an experiment

Read the target series' migration guide and the BSP's supported combinations. Create a separate branch/build directory. Resolve every layer to an exact revision; run the native application checks, parse the metadata, build the package/image/SDK, then repeat the same boot and target acceptance cases. Compare manifests and behavior before updating the release record.

Do not silently replace a frozen course pin with a moving branch. Write down why the upgrade is needed, which interfaces changed, which checks were run, and which claims remain untested. The native behavior tests can stay stable even when the build-system layout changes.

## Connect to the rest of the roadmap

Carry your package/boot evidence into [Jetson BSP production](../../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Yocto-BSP-Production/Guide.md) when you have compatible hardware. Carry the observation identity, clock, and freshness contract into [robotics](../../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20D%20-%20Robotics/Guide.md). A board image becomes useful to AI work when the complete sensor-to-runtime path is observable and recoverable.

**Independent maintenance exercise:** inspect one newer point release without changing your working baseline. Identify one relevant change, its primary-source evidence, its expected effect, and a test that would confirm it. A release name by itself is not a reason to rewrite a functioning product.

**Exit question:** can another learner tell what ran here versus what they still need to build? **Answer:** the [lab validation record](../Labs/README.md) and your own experiment record should make that distinction explicit.

**Previous:** [Lecture 15](Lecture-15.md) · **Next:** [Course hub](../Guide.md)
