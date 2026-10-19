# Compile a device tree and test an overlay

**Time:** 45–60 minutes. **Requirements:** Python 3.10+ and the `dtc`, `fdtoverlay`, and `fdtget` tools. On Ubuntu/Debian these are supplied by `device-tree-compiler`. No board, root access during the exercise, kernel modification, or firmware flashing is needed.

You will compile real device-tree sources, merge an overlay, inspect the binary result, and distinguish compiler success from a correct device contract. All addresses and compatible strings describe a **fictional teaching board**. Do not install these blobs on any hardware.

## 1. Get the files and run the lab

Place [board.dts](board.dts), [sensor-overlay.dts](sensor-overlay.dts), [device_tree_lab.py](device_tree_lab.py), and [test_device_tree.py](test_device_tree.py) in one folder. If the tools are missing, install them using your system's normal package-management workflow:

```bash
sudo apt-get install device-tree-compiler
```

Then run, without administrator privileges:

```bash
python3 device_tree_lab.py --output-dir /tmp/course-device-tree
python3 -m unittest discover -s . -p 'test_*.py' -v
```

The [retained result](example-result.json) records actual DTC 1.7.0 compilation and libfdt merging. All six native tests executed during review. If the tools are absent, the tests explicitly skip; that is not a completed native exercise.

## 2. Read the original tree

The root declares a fictional I2C controller at address `0x1000` with an address-range size `0x100`. Its child sensor uses I2C address `0x48`. These two `reg` properties have different encodings because their parent buses declare different address/size cell counts.

The labels `i2c0` and `sensor0` are references used by the source and overlay machinery. They are not Linux device numbers or globally fixed GPIO identifiers. The `compatible` strings identify which device description a driver would recognize; no such hardware driver is supplied here.

Initially both nodes have `status = "disabled"`, and the synthetic sensor has `sample-rate-hz = <100>`. The property name and acceptable values belong to this exercise's invented contract, not an upstream sensor binding.

**Check:** why does the sensor's `reg` contain only one cell? Its parent declares one address cell and zero size cells. The controller's parent uses one address and one size cell, so its `reg` contains two.

## 3. Compile and apply by hand

From the exercise directory, the underlying operations are:

```bash
dtc -@ -I dts -O dtb -o /tmp/course-device-tree/board.dtb board.dts
dtc -@ -I dts -O dtb -o /tmp/course-device-tree/sensor.dtbo sensor-overlay.dts
fdtoverlay -i /tmp/course-device-tree/board.dtb \
  -o /tmp/course-device-tree/merged.dtb /tmp/course-device-tree/sensor.dtbo
fdtget -t s /tmp/course-device-tree/merged.dtb /i2c@1000 status
fdtget -t u /tmp/course-device-tree/merged.dtb /i2c@1000/sensor@48 sample-rate-hz
```

Create the output directory first if you skipped the Python run. The final commands print **okay** and **200**. The overlay enables the controller and sensor and changes the declared rate; the controller's register range and root compatible string remain unchanged.

The `-@` option retains symbol information needed by this label-targeted overlay. The plugin overlay contains external references that are resolved when it is applied. See the [kernel overlay explanation](https://docs.kernel.org/devicetree/overlay-notes.html) for the distinction between labels and explicit target paths.

This is an offline merge into a file. It is not applying an overlay to the running kernel, changing pinmux, or programming a sensor register.

## 4. Make failures informative

The tests exercise these different failure layers:

| Change | Outcome | Lesson |
| --- | --- | --- |
| Remove a closing brace | `dtc` fails | Source syntax matters |
| Compile the base without symbols | Label overlay application fails | The needed target information is absent |
| Reference `missing_sensor` | Plugin compiles, merge fails | An external reference must eventually resolve |
| Declare 3000 Hz | Compile and merge succeed; fixture validation fails | Valid binary structure does not establish a valid device configuration |
| Change the sensor compatible string | Merge succeeds; fixture validation fails | Driver/device identity is part of the contract |
| Apply the supplied overlay | Native checks pass | The intended fixture changes are present |

The Python validation compares a small set of expected fixture properties. It is **not** a general device-tree schema validator. For a real driver, use the binding schema and the matched kernel's `dt_binding_check`/`dtbs_check` workflow, then test physical behavior. A schema-valid description still cannot prove that a wire, voltage, clock, or register table matches the board.

## 5. Optional: build an external kernel module without loading it

The separate [course_module.c](course_module.c) and [Makefile](Makefile) show a complete buildable external module with a bounded, read-only teaching parameter. It has no hardware access, IRQ, DMA, or character-device interface.

Use a temporary build directory and the headers for the kernel you intend to target:

```bash
build_dir=$(mktemp -d /tmp/course-module-build.XXXXXX)
cp course_module.c Makefile "$build_dir/"
make -C "/lib/modules/$(uname -r)/build" M="$build_dir" W=1 modules
readelf -h "$build_dir/course_module.ko"
```

Run these commands from the lab directory. A machine without matching development headers cannot complete this optional step. On a Jetson cross-build, the kernel build directory and compiler must match the target BSP; the host's `/lib/modules` path is not an arm64 cross-build recipe.

The [saved module result](module-build-result.json) records an actual **x86-64 build against Linux 6.8.0-147-generic headers**. The module was not loaded, and its init/exit/parameter behavior was not executed. An ELF file passing compilation does not prove hardware compatibility or runtime correctness. [Kernel external-module build documentation](https://docs.kernel.org/kbuild/modules.html).

## 6. Finish with a hardware mapping

Submit the original and merged properties, one compilation failure, one merge failure, and one syntactically valid but contract-invalid overlay. Explain which fields would need a real binding, schematic, pinmux configuration, and driver before deploying an analogous change.

For the optional module build, retain the target header identity, compiler, source hash, build log, and ELF architecture. Keep the evidence level explicit: **compiled**, **loaded**, and **tested with hardware** are separate milestones. Continue with the [kernel guide](../Guide.md).
