# Lecture 3 - Chips, boards, flashing, and a first ESP-IDF project

**Course:** [Education guide](../Guide.md) | **Previous:** [Bytes and budgets](Lecture-02.md) | **Next:** [Choose an AI, power or gateway experiment](Lecture-04.md)

A chip is the SoC; a module adds supporting components and RF integration; a development board adds power/debug/connectors. Record all three identities and the board revision. They determine which target to compile and which physical port to use.

If you completed the [Arduino bring-up lecture](../../Lecture/Lecture-02.md), keep its boot record. This lesson offers a native ESP-IDF route. The commands are documented practice, not a claim of physical board execution in this revision.

## Use the SDK environment that belongs to the example

The [official version guide](https://docs.espressif.com/projects/esp-idf/en/stable/esp32c6/versions.html) resolved to ESP-IDF 6.1 when reviewed on 2026-10-18. Install a supported release through the [official getting-started guide](https://docs.espressif.com/projects/esp-idf/en/stable/esp32c6/get-started/index.html), then activate that release's environment. An Arduino component or Zigbee SDK example can require a different IDF line; follow its compatibility requirements rather than mixing releases.

After activation:

```bash
idf.py --version
```

Copy `examples/get-started/hello_world` from that installed SDK into a new workspace directory. This keeps generated project files separate from the SDK source. In the copied directory, for an actual C6 target:

```bash
idf.py set-target esp32c6
idf.py build
idf.py size
```

Save the target, compiler version, build log, ELF/map files, and configuration. A successful cross-build is useful even without hardware. It does not establish working pins, radio, memory options, or power.

When the correct board and port are connected, replace the placeholder in:

```bash
idf.py -p YOUR_SERIAL_PORT flash monitor
```

Observe the expected hello-world output and the example's documented restart behavior. Do not classify an intentional example restart as a crash without reading its source. Exit the monitor with Ctrl+].

## Understand the files before changing them

| File | Question it answers |
|---|---|
| Root `CMakeLists.txt` | Which project/build integration is being configured? |
| Component `CMakeLists.txt` | Which sources and dependencies belong to this component? |
| `sdkconfig` / defaults | Which concrete options and defaults selected the build? |
| `idf_component.yml` | Which managed dependencies and version constraints are requested? |
| `dependencies.lock` | Which managed dependency versions were resolved? |
| Partition table | Where do application/data/update partitions reside in flash? |

Changing target can regenerate configuration. Commit source/configuration intentionally and compare the result; do not copy an opaque `sdkconfig` from another chip and assume it is portable.

## Bring-up exercise

Change one printed line to include a project revision string. Rebuild and confirm that the new image, not a stale binary, is what boots. Reset twice, reconnect the terminal, and compare logs. Then run one GPIO or peripheral example from the same SDK release with pins verified against the board schematic.

**Answer check:** What does `idf.py size` establish? It summarizes sections of the linked image. It does not measure peak runtime heap, task stack high-water marks, dynamically allocated model activations, or total board power. Gather those separately when the application needs them.

The deliverable is a reproducible build plus clearly labeled compile-only or hardware-run evidence, followed by one small board-specific peripheral observation.
