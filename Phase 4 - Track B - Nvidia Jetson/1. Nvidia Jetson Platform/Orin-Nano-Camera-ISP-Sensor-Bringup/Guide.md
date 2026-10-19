# Jetson camera bring-up: from sensor signals to trustworthy images

Camera bring-up succeeds when the system produces correctly interpreted, calibrated, timestamped images under the intended operating conditions. A visible preview is one milestone. It does not establish correct exposure controls, frame timing, robot coordinates, or long-run reliability.

This guide offers a staged path: inspect the hardware contract, validate the driver and raw stream, validate the supported ISP path, then integrate perception. Begin with the [CPU RAW10 and exposure lab](labs/README.md) and the [CPU video lab](../Orin-Nano-Video-Codec-DeepStream/labs/README.md). Both run without a sensor. Allow another two or three sessions for the board-specific investigation when hardware is available.

**Scope, reviewed 2026-10-18:** the referenced R36.4.4 camera documentation describes a JetPack 6.x teaching baseline. Use the [compatibility matrix](../../Compatibility.md) and the matching BSP documentation for your own board. This course does not claim that its device-tree or driver examples were executed on a physical camera. Do not reuse a register address, connector mapping, or mode table from a different module/sensor without verification.

## 1. Identify the actual acquisition path

Not every frame passes through the Jetson ISP. NVIDIA distinguishes these paths:

| Source/use | Typical interface | Image-processing responsibility |
|---|---|---|
| Supported CSI Bayer sensor using Jetson ISP | libargus / `nvarguscamerasrc` | Supported camera core and ISP configuration |
| CSI capture bypassing Jetson ISP | Direct V4L2 | Application or another processing component |
| USB UVC camera | V4L2 | Often the camera's internal ISP, then application processing |
| Network camera | Network transport and decoder | Often camera ISP/encoder plus receiving decoder |

The [NVIDIA camera API matrix](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/CameraDevelopment/CameraSoftwareDevelopmentSolution.html) is the release-specific reference. A V4L2 raw capture working while Argus fails can indicate a control or camera-core integration issue, not a contradiction.

A conceptual CSI path is sensor → CSI receiver → video-input capture → an optional supported ISP path → consumer. Actual buffering and engine routing depend on the API and mode. Avoid treating a simplified diagram as a register-level description of every board.

The sensor's control bus and pixel transport are separate. An I2C chip-ID read proves communication with that device under those conditions; it does not prove that CSI lanes carry valid frames. Conversely, stable capture does not prove that every exposure/gain control is programmed correctly.

## 2. Build a hardware and mode worksheet

Record the module, carrier revision, camera module, sensor revision, connector/cable, power rails, reset polarity, clock source, lane count, lane mapping, and supported mode. Obtain electrical and timing details from the actual schematics and sensor documentation.

A mode record should include:

```text
active width and height
pixel format and CSI data type
bit depth on the link and storage representation in memory
lane count and link frequency/rate
pixel clock, line length, frame length
exposure/gain ranges and units
embedded metadata lines and HDR mode, if any
virtual channel and routing, if applicable
```

Do not infer a carrier connector's available lanes from the SoC family's maximum. Verify cable orientation and voltage compatibility before power-up. Power/reset sequencing is sensor-specific; a universal “toggle this GPIO and wait 10 ms” recipe is not a bring-up procedure.

For a 1920×1080, 30 Hz, 10-bit stream, active pixel payload is `1920*1080*30*10 = 622.08 Mbit/s`. Across two lanes that is 311.04 Mbit/s per lane **before protocol/embedded-data overhead and margin**. This payload calculation is not the required sensor link-rate setting. The mode's actual transmission timing and platform constraints determine that setting.

## 3. Make driver metadata agree with the sensor

The driver controls power, mode programming, stream start/stop, and exposure/gain/frame-rate settings. Device-tree metadata describes the device, clocks, regulators, GPIOs, endpoints, and camera mode information. The media graph must connect the intended sensor endpoint to the intended capture route.

NVIDIA's [sensor software programming guide](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/CameraDevelopment/SensorSoftwareDriverProgramming.html) uses its camera framework and versioned source examples. Follow the source tree supplied with the selected BSP; older paths and bindings are not guaranteed to compile unchanged on a newer kernel.

Rather than inventing a complete sensor driver, begin from the vendor-supported driver for that exact sensor/mode and identify each adaptation:

| Driver area | Evidence required |
|---|---|
| Probe | Correct compatible string, expected device identity, errors reported |
| Power sequence | Rails, clocks, reset ordering and delays from sensor specification |
| Mode table | Register list tied to the documented resolution/rate/bit depth |
| Controls | Unit conversion, limits, rounding, and actual register update behavior |
| Stream control | Start/stop success, timeout behavior, cleanup after failure |
| Runtime resume | State restored correctly after suspend or power transition |

Check every I/O return value. A failed register write followed by a successful `stream_on` return gives userspace a false state. Grouped register updates may be needed to avoid a partially changed mode or exposure; use the sensor's documented mechanism.

The CPU timing lab uses a 74.25 MHz pixel clock, 2200 clock periods per line, and 1125 lines per frame to derive 30 Hz. It demonstrates unit conversion and quantization. Those numbers are a hypothetical mode, not a configuration to paste into an arbitrary device tree.

## 4. Validate raw capture before image aesthetics

First inspect the devices and media graph. These read-only discovery commands require the standard V4L2/media utilities:

```bash
v4l2-ctl --list-devices
media-ctl -p
```

Select the actual capture device from that output. Then inspect its formats and controls, substituting the discovered device path:

```bash
v4l2-ctl --device=/dev/video0 --list-formats-ext
v4l2-ctl --device=/dev/video0 --list-ctrls
v4l2-ctl --device=/dev/video0 --get-fmt-video
```

`/dev/video0` is an example selection, not a guaranteed camera identity. Use a stable device association in an application with multiple cameras.

Choose an advertised mode and capture a bounded number of frames with the matching V4L2 tool/sample. Save the negotiated format, `bytesperline`, `sizeimage`, sequence numbers, timestamp flags, and driver logs. The [V4L2 buffer specification](https://docs.kernel.org/userspace-api/media/v4l/buffer.html) defines buffer metadata and timestamp flags; nanosecond or microsecond representation does not itself establish timestamp accuracy.

For a raw file, store a sidecar manifest. Width and height alone are insufficient: record packing, bit alignment, endianness, stride, Bayer phase, and any embedded lines. A ten-bit CSI stream may be stored in sixteen-bit containers. The [RAW10 lab](labs/README.md) demonstrates both representations and a wrong decoder that still returns in-range values.

Prefer a sensor test pattern when supported. It separates transport/format validation from illumination, optics, and automatic exposure. Check the entire frame and several sequential frames, not just a plausible-looking thumbnail.

## 5. Interpret failures at the correct layer

| Observation | Investigate first |
|---|---|
| Sensor does not probe | Power, reset, clock, control-bus route/address, compatible string |
| Probe succeeds, no frames | Mode programming, stream start, endpoint routing, lanes/link timing |
| Frame-size mismatch | Advertised versus programmed mode, metadata lines, packing/stride |
| Regular stripes or shifted rows | Row stride, bit packing, truncation, lane/data corruption |
| Wrong colors | Bayer phase, crop/mirror changes, white balance, color interpretation |
| Raw works, Argus fails | Sensor controls, mode metadata, supported camera-core/ISP configuration |
| Periodic missing frames | Transport errors, queue pressure, control changes, timestamps/logs |
| Only multi-camera fails | Routing/virtual-channel conflicts, shared bandwidth, buffering, synchronization |

Keep the smallest failing setup: one sensor, one advertised mode, bounded capture, and a saved manifest. Changing mode tables, clock values, ISP settings, and application code simultaneously destroys useful evidence.

Kernel debug/trace interfaces depend on the BSP. Use the matching documentation and capture error counters around a reproduced failure. Do not publish invented “expected logs” with register values as if they were a measured device trace.

## 6. Treat ISP processing as a calibrated transformation

A Bayer sensor records one color-filtered value per photosite. Demosaicing estimates missing color components. Other processing can include black-level correction, lens-shading correction, denoising, white balance, color correction, tone mapping, and sharpening. Their order and implementation are platform-specific; an illustrative list is not a guarantee of a particular ISP's internal stages.

Image tuning depends on sensor, lens, mode, and illumination. A file copied from another camera can make a preview attractive while changing geometry or color statistics relevant to a model. Use supported tuning resources for the actual module. NVIDIA's camera solution guide points to partner support for additional sensor tuning.

After the raw path works, validate the supported Argus/ISP path separately. Inspect the chosen mode and capture settings. Lock or record exposure, gain, white balance, and focus when comparing two processing configurations. Otherwise an automatic control loop can change the input while you attribute the result to a software change.

A generated `videotestsrc` pipeline does not exercise CSI, the sensor driver, or the ISP. It is valuable for the later buffer/consumer stages; report that validation boundary accurately.

## 7. Relate exposure and rolling shutter to robot motion

An exposure timestamp needs a definition: start, midpoint, end, or another documented event. With rolling shutter, different rows represent different times. A single frame timestamp is an approximation unless the consumer models that readout.

Consider a hypothetical 1080-row readout with 20 µs between row starts. The last row starts `1079*20 µs = 21.58 ms` after the first. At a yaw rate of 1 rad/s, orientation changes about 0.02158 rad, or 1.24 degrees, over that interval. Treating the image as an instantaneous global-shutter observation can therefore introduce a motion-dependent geometric error.

Longer exposure can improve signal-to-noise in dim scenes while increasing motion blur and potentially constraining frame rate. Higher gain amplifies signal and noise; it does not create additional photons. Evaluate the image features or task metric under the intended robot motion and lighting.

For multi-camera systems, matching frame rates does not synchronize exposure. Hardware triggering, sensor synchronization, or a measured time relationship may be required. Quantify offset, drift, and uncertainty. A shared host timestamp assigned after dequeue can hide acquisition differences.

## 8. Add geometry and robot frames

Calibrate intrinsics for the actual resolution, crop, focus, and lens configuration. Distortion correction and resizing change the coordinate mapping used by perception. Calibrate extrinsics between cameras and the robot reference frame, with an explicit direction convention.

For a pinhole projection, `u = fx*X/Z + cx` and `v = fy*Y/Z + cy` apply to the specified camera coordinate convention and a point in front of the camera. They are not a substitute for distortion, rolling-shutter, or timestamp modeling. The [sensor-fusion course](../../../Phase%203%20-%20Artificial%20Intelligence/Track%20A%20-%20Hardware%20and%20Edge%20AI/4.%20Sensor%20Fusion/Guide.md) connects these ideas to estimation.

Save the calibration with a version and the capture mode. Test an independent target/scene rather than reporting only the fitting residual on the same calibration images. A low reprojection residual can coexist with poor coverage of depth, field of view, or motion.

Before passing an image-derived measurement to ROS, state its optical frame, timestamp, covariance/uncertainty, and transform availability. The [ROS 2 integration lab](../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20D%20-%20Robotics/Robotics%20Foundations/ROS2-Lab.md) shows why a plausible coordinate without the correct frame and time must be rejected.

## 9. Understand GMSL and virtual-channel routing

A serializer/deserializer link adds transport and control components between a remote camera and the CSI receiver. Its link rate, routing, virtual-channel identifiers, sensor control forwarding, and synchronization are separate configuration concerns.

A virtual channel distinguishes streams within the transport; it is not automatically a ROS namespace or an application camera identity. Preserve an explicit mapping from physical camera through deserializer/CSI/capture device to application source ID.

Use NVIDIA's [versioned GMSL framework documentation](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/CameraDevelopment/JetsonVirtualChannelWithGmslCameraFramework.html) and the actual serializer/deserializer vendor's supported design. Do not assume that a supported reference pairing establishes every lane count, sensor, or carrier combination.

Start with one remote camera, prove its identity and timing, then add another. Exercise disconnect/reconnect and confirm that a replacement stream cannot silently inherit another camera's calibration or source ID.

## 10. Complete a camera evidence package

The capstone contains:

1. Hardware/mode worksheet with sources and board revisions.
2. Exact BSP, driver, device-tree, and application revisions.
3. Bounded raw capture plus format/timestamp sidecar and test-pattern checks.
4. Separate ISP-path results, including controls and tuning/calibration identity.
5. Frame interval, omission, age, and exposure/synchronization observations under representative load.
6. A failure/recovery test and a clear statement of what was not exercised.

For the CPU route, submit the RAW10 byte derivation, padded-row tests, Bayer crop cases, exposure calculation, and GStreamer trace. For a hardware completion claim, add actual sensor evidence. For robot perception, add task-level evaluation under motion, lighting, and relevant failure cases.

**Self-check:** why can raw capture succeed while Argus fails? They exercise different interfaces and control/integration requirements. Why can a sharp image still be wrong for a robot? Its timestamp, camera calibration, frame transform, or preprocessing may be wrong even when the pixels look good.
