# Lecture 5 - Arduino as an ESP-IDF component and the path to a maintainable product

**Course:** [Espressif guide](../Guide.md) | **Previous:** [Peripheral and network boundaries](Lecture-04.md)

Use Arduino as an ESP-IDF component when an existing Arduino library is useful but you need direct control of components, configuration, partitions, or native IDF features. This is a build/integration choice. Your tested data formats, ownership rules, and failure behavior should survive the move.

## Check the pair of versions

Reviewed 2026-10-18: the official component page documents Arduino ESP32 **3.3.12** with the **ESP-IDF 5.5** line; the stable Arduino release uses IDF **5.5.5**. Standalone IDF stable documentation is 6.1, which is not a reason to substitute it into this combination. Record exact revisions and the dependency lockfile. See [component instructions](https://docs.espressif.com/projects/arduino-esp32/en/latest/esp-idf_component.html) and [release 3.3.12](https://github.com/espressif/arduino-esp32/releases/tag/3.3.12).

The following is a **documented build workflow**, not a claim of a build or flash executed during this revision. Activate an installed compatible IDF environment, confirm `idf.py --version`, and work in a new project directory.

```bash
idf.py create-project-from-example 'espressif/arduino-esp32==3.3.12:hello_world'
```

Enter the generated project directory. For a C6 board, the subsequent commands are:

```bash
idf.py set-target esp32c6
idf.py menuconfig
idf.py build
idf.py size
```

Use the target matching your actual board. In menuconfig, select the intended Arduino entry-point mode and set `CONFIG_FREERTOS_HZ=1000` as required by this component. The exact dependency pin uses `==`; the vendor documentation's `^3.3.12` example permits compatible updates. The [Component Manager version rules](https://docs.espressif.com/projects/idf-component-manager/en/latest/reference/versioning.html) define these operators. Preserve `idf_component.yml`, `dependencies.lock`, project configuration, and tool versions so a later build can reproduce your choice.

After verifying the serial port and board, `idf.py -p YOUR_SERIAL_PORT flash monitor` performs deployment. Replace the placeholder; exit the monitor with Ctrl+]. A successful build establishes compile/link compatibility for that configuration. Boot, peripherals, radio coexistence, and timing still need separate hardware observations.

## Choose one entry-point model

With Arduino autostart enabled, provide `setup()` and `loop()` in C++ and let the framework manage their invocation. With autostart disabled, provide `extern "C" void app_main()`, call `initArduino()` before Arduino facilities that require initialization, and own application startup explicitly. Do not define competing startup paths accidentally. Include `Arduino.h` in `.cpp` files rather than relying on sketch preprocessing.

Separate application logic from framework adapters:

```text
components/
  sample_logic/    conversion, framing, validity rules; CPU-testable
  sensor_adapter/  Wire or native IDF peripheral access
  transport/       UART / UDP / other selected protocol
main/              configuration and ownership wiring
```

Changing adapters should not change what a timestamp means, which payloads are accepted, or how a restarted sender is identified. Move one boundary at a time and compare results against the original capture.

## Treat an update as a recovery problem

Before adding OTA, archive the partition table and identify which image authenticates the next stage. Receiving a file successfully is not equivalent to trusting it. A robust update flow checks the image, boots it in a trial state where supported, performs health checks, and confirms it only when the required functions work. Test interrupted downloads, invalid images, and failed health checks using the selected platform's documented recovery mechanism.

Similarly, “low power” is a budget: sum charge spent sampling, processing, transmitting, retrying, and sleeping. A development board's USB bridge or regulator can dominate the measured sleep current. Report measurement location, supply voltage, and workload rather than copying a chip datasheet's best-case number.

## Migration exercise and answer checks

Move the [native C framing implementation](../../labs/Framed-Telemetry.md) into a component without changing its bytes on the wire. First rerun its host tests, then replay the same fixtures through the board transport. Keep parser failures distinct from transport failures.

- **Why keep a lockfile if the manifest pins Arduino?** Its transitive dependencies also affect the build.
- **Why not immediately migrate every library?** Each changed boundary adds uncertainty; preserve known observations and test one change at a time.
- **Does using ESP-IDF make a system real-time?** No. Configuration, workload, ownership, scheduling, blocking, and measurement still determine whether the stated deadlines are met.

The course deliverable is a reproducible project with a boot record, one validated sensor interface, bounded communication behavior, and a recovery test. Continue to the [education exercises](../Education/Guide.md) for AI, power, and gateway project choices.
