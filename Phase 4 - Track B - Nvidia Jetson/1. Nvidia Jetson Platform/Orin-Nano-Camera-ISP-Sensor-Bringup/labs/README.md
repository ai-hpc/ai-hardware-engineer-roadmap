# CPU camera lab: RAW10 layout, Bayer phase, and exposure timing

A camera can deliver the correct bytes while an application interprets them incorrectly. This lab separates three contracts: how samples are packed, which color each sample represents, and when the sensor exposes them. It uses Python only and takes about 30 minutes.

From the repository root:

```bash
python3 'Phase 4 - Track B - Nvidia Jetson/1. Nvidia Jetson Platform/Orin-Nano-Camera-ISP-Sensor-Bringup/labs/raw_camera.py'
python3 -m unittest discover -s 'Phase 4 - Track B - Nvidia Jetson/1. Nvidia Jetson Platform/Orin-Nano-Camera-ISP-Sensor-Bringup/labs' -v
```

The [implementation](raw_camera.py) is a format/timing exercise. It does not emulate a sensor, an ISP, or a Jetson driver. The eleven tests were executed with Python 3.12.3 on 2026-10-18; no hardware measurement is implied.

## 1. Decode four samples by hand

For the explicitly selected V4L2 packed 10-bit format, four samples occupy five bytes. The first four bytes contain each sample's upper eight bits. The final byte stores the two low bits of samples zero through three in bits 1:0, 3:2, 5:4, and 7:6 respectively. See the [kernel's packed RAW10 definition](https://docs.kernel.org/userspace-api/media/v4l/pixfmt-srggb10p.html).

For samples `[0, 1, 512, 1023]`, the packed bytes are:

```text
00 00 80 ff c4
```

The first sample is `(0x00 << 2) | (0xc4 & 3) = 0`. The second is `(0x00 << 2) | ((0xc4 >> 2) & 3) = 1`. Complete the remaining two before running the script.

**Answers:** `(0x80 << 2) | 0 = 512`; `(0xff << 2) | 3 = 1023`.

The tests use this independent byte fixture and exercise every possible 10-bit value in every position of a four-sample group. A round-trip test alone can miss a shared error in a packer and unpacker, so the fixed external-format example matters.

## 2. Do not confuse packed with expanded RAW10

V4L2 `RG10` uses a different memory representation: each 10-bit sample is in a little-endian 16-bit word with zero upper bits. The same values occupy eight bytes:

```text
00 00 01 00 00 02 ff 03
```

The [expanded RAW10 definition](https://docs.kernel.org/userspace-api/media/v4l/pixfmt-srggb10.html) describes that layout. A sensor transmitting ten bits per pixel on CSI does not prove that the driver's output buffer is the five-byte packed layout.

The lab has separate unpackers. Its packed-row contract requires a width divisible by four and an explicit row stride. It rejects unsupported geometry rather than guessing how a partial group was stored. Padding at the end of a row is skipped. A wrong decoder can still produce numbers between zero and 1023, so a range check alone is insufficient.

**Exercise:** how much logical payload does 1920×1080 occupy? **Answer:** packed 10-bit is 2,592,000 bytes; expanded 16-bit is 4,147,200 bytes before row padding. Always use the negotiated driver's `bytesperline` and `sizeimage` for actual allocation/indexing.

## 3. Track Bayer phase through cropping

An RGGB Bayer tile is:

```text
R G
G B
```

Cropping one column from the left changes the top-left tile to GRBG. Cropping one row changes it to GBRG. Cropping one row and one column changes it to BGGR. Cropping by two pixels in both directions preserves the phase.

The `cropped_pattern` function makes those transformations explicit. A demosaicer needs the phase of the delivered image, not merely the sensor's original phase. Mirroring, rotation, and binning modes may also alter the relevant interpretation; inspect the mode and driver contract.

This lab labels sites but does not reconstruct RGB. Demosaicing, color correction, and geometric calibration are separate operations.

## 4. Quantize exposure in a stated sensor model

For the simplified mode:

```text
pixel clock  = 74,250,000 Hz
line length  = 2,200 pixel-clock periods
frame length = 1,125 lines
line time    = 2,200 / 74,250,000 = 29.6296296 µs
frame rate   = 74,250,000 / (2,200 * 1,125) = 30 Hz
```

A requested 10 ms exposure corresponds to 337.5 lines. The lab explicitly rounds half up to 338 lines, giving **10.0148148 ms**. It reserves four lines of margin and clamps overly long requests to 1121 lines. Doubling frame length gives 15 Hz at the same pixel clock and line length.

Real sensors can use different exposure units, margins, fine integration, HDR timing, or automatic frame-length extension. The model's rounding and clamp rules are teaching choices, not values to write into an unspecified camera driver. The [sensor programming guide](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/CameraDevelopment/SensorSoftwareDriverProgramming.html) explains how mode metadata and controls must agree with the actual sensor.

## 5. Submit a format and timing contract

State the pixel format's exact name, packing, bit alignment, endianness, Bayer phase, dimensions, stride, and buffer length. Add requested and actual exposure, line/frame timing, and timestamp meaning. Explain how you would detect a wrong format that still produces plausible values.

Continue to the [camera bring-up guide](../Guide.md). For actual buffered image execution on a CPU, use the neighboring [GStreamer video lab](../../Orin-Nano-Video-Codec-DeepStream/labs/README.md).
