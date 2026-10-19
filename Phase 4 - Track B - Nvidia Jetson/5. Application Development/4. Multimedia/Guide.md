# Multimedia: preserve meaning, timing and bounded latency

<div class="course-identity auto-course" style="--course-accent: #db2777; --course-accent-rgb: 219, 39, 119;" markdown="1">
<div class="course-identity__icon">MUL</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Jetson Track</p>
<p class="course-identity__title">Specialized course identity for Multimedia.</p>
<p class="course-identity__meta">Artifact: Jetson integration demo · Measure: latency, memory, power, logs</p>
</div>
</div>

**Reviewed 18 October 2026.** Build a media path whose formats, timestamps and resource costs you can explain. Start with generated signals, then add a physical source. The practical CPU exercises run without a Jetson, microphone or camera; hardware commands below require the matching device and BSP.

**Prerequisites:** Linux processes and basic Python. Allow three 60-minute sessions: audio, video, then an integration review. Use the [platform compatibility record](../../Compatibility.md) to match driver and plugin documentation to the installed system.

## 1. Choose the learning path

| Path | What you will build and explain |
|---|---|
| [Audio foundation](Jetson-Audio-Setup-and-Development/Guide.md) | Samples versus frames, ALSA paths, physical clocks and channel mapping |
| [CPU audio lab](Jetson-Audio-Setup-and-Development/CPU-Audio-Lab.md) | Real WAV round trip, stateful resampling, aliasing and wrong-rate failure |
| [Video codecs and queueing](../../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Video-Codec-DeepStream/Guide.md) | Actual CPU encode/decode plus a frame-age and drop-policy experiment |
| [Camera geometry](../../../Phase%203%20-%20Artificial%20Intelligence/Track%20A%20-%20Hardware%20and%20Edge%20AI/4.%20Sensor%20Fusion/Camera-Time-Fusion-Lab/Guide.md) | Projection, calibration and delayed measurements |
| [LyraT I2S hardware case](ESP32-LyraT-I2S-Mic-Jetson-Orin-Nano/Guide.md) | A codec-to-Jetson clock/data contract and limits of a historical capture |

## 2. Explain each stage of the pipeline

```text
source → timestamp → format conversion → processing → encode/output
                    ↘ bounded queue ↗
```

A camera format is more than width and height. Record frame rate, pixel format, color range/matrix, stride, memory layout and timestamp clock. An audio format needs sample rate, channel order, container bits, valid bits and interleaving. Misinterpreting either can produce plausible but incorrect AI inputs.

Decode, resize, inference and encode are separate operations. A faster model does not eliminate camera buffering or an overloaded encoder. Measure capture-to-result age, queue depth, dropped frames and output quality alongside throughput.

## 3. Respect the actual module's media engines

**Orin Nano has no NVENC hardware encoder.** NVIDIA documents software H.264 encoding for this module. A family-wide plugin list that includes `nvv4l2h264enc` does not make that encoder usable on every Orin. Hardware decoder support must also be checked for the exact codec/profile/bit depth and module; do not infer it from a plugin's installed name. [Versioned Orin Nano encoding guidance](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SD/Multimedia/SoftwareEncodeInOrinNano.html).

Start with the [executed CPU codec lab](../../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Video-Codec-DeepStream/labs/README.md). For a GStreamer software-encode extension, first inspect the installed elements:

```bash
gst-inspect-1.0 --version
gst-inspect-1.0 x264enc
gst-inspect-1.0 h264parse
gst-inspect-1.0 mp4mux
```

If those plugins are available, this finite generated-stream example encodes on the CPU:

```bash
gst-launch-1.0 -e videotestsrc num-buffers=300 ! \
  'video/x-raw,width=640,height=480,framerate=30/1,format=I420' ! \
  x264enc tune=zerolatency speed-preset=ultrafast bitrate=1000 ! \
  h264parse ! mp4mux ! filesink location=encode-baseline.mp4
```

`x264enc bitrate` is in kbit/s; other encoders can use different units. End-of-stream lets the muxer finalize the container. Verify duration and decoded frame count before replacing the generated source. This command was syntax-checked, not run on Jetson. No throughput claim follows from it. [x264 element documentation](https://gstreamer.freedesktop.org/documentation/x264/index.html).

On a supported Jetson decoder, inspect `gst-inspect-1.0 nvv4l2decoder` and select the parser/demuxer for the actual stream. Use a sink appropriate to the running display session or `fakesink` for a headless decode experiment. A headless decode rate is not display latency. Family examples and their operating-system descriptions can lag a release; use installed plugin inspection and the module support matrix alongside [NVIDIA's versioned GStreamer guide](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SD/Multimedia/AcceleratedGstreamer.html).

## 4. Put a time budget on queues

At 30 frames/s, one frame interval is about 33.3 ms. Six queued frames can add roughly 200 ms before processing even starts. A queue bounded only by a large byte limit can still be too old for control.

For a live perception branch, an example policy is:

```text
queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream
```

This keeps a bounded number of waiting buffers and discards older queued buffers when full. It does not preempt a frame already being processed or prove a complete latency bound. Disabling the other two limits makes the chosen buffer limit explicit. Recording may require retaining all frames and lowering the source rate instead. [GStreamer queue semantics](https://gstreamer.freedesktop.org/documentation/coreelements/queue.html).

In the CPU queue lab, compare keep-all with drop-oldest under the same service-time sequence. Explain which policy best fits a recorder and which fits a robot reacting to the present. Count drops: silently losing evidence is not an improvement.

## 5. Add a camera after the generated path works

Discover devices and modes rather than assuming `/dev/video0`:

```bash
v4l2-ctl --list-devices
read -r -p 'Capture device path: ' VIDEO_DEVICE
v4l2-ctl --device "$VIDEO_DEVICE" --list-formats-ext
```

UVC compatibility depends on the device's exposed modes, bandwidth and host driver. Some cameras emit compressed MJPEG/H.264; treating those bytes as raw BGR will fail. CSI is a physical interface, not a guarantee that a sensor already has a compatible driver or ISP tuning. Distinguish raw Bayer capture through V4L2 from an ISP-processed path such as Argus. Use the selected release's camera support matrix and driver guide. [NVIDIA camera software architecture](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SD/CameraDevelopment/ArgusFramework/CameraSoftwareDevelopmentSolution.html).

A custom sensor needs a real binding, power sequence, clocks, endpoints and mode timings. An invented short device-tree node is not an installable driver. Keep the source-to-pixel contract with the calibration and model preprocessing artifacts.

## 6. Add audio without guessing the default route

Discover ALSA cards with `arecord -l`, named PCM routes with `arecord -L`, and PipeWire nodes with `wpctl status` when that sound server is installed. Select the actual device and supported format; `amixer set Master` is not a universal codec control.

USB capture and external I2S capture use different kernel paths. Follow the [audio lesson](Jetson-Audio-Setup-and-Development/Guide.md) for ASoC, ADMAIF routing and reference-carrier pin mappings. The CPU lab first verifies the signal-processing part without a microphone.

Bluetooth pairing alone does not select an audio route. A2DP provides a media profile, while HFP provides a telephony path with different bandwidth/latency behavior. With PipeWire/WirePlumber, discover the actual profile and node, choose the intended route, and verify capture/playback. Do not paste PulseAudio module commands into a PipeWire configuration. Classic Bluetooth audio and LE Audio are distinct capabilities; a BLE-capable radio alone does not establish either full audio path. [WirePlumber Bluetooth configuration](https://pipewire.pages.freedesktop.org/wireplumber/daemon/configuration/bluetooth.html).

## 7. Display and synchronization

The Orin Nano reference developer kit exposes DisplayPort; native HDMI and USB-C display output should not be assumed. An adapter adds its own compatibility requirements. Discover the actual display session: X11's `xrandr` commands do not configure every Wayland compositor. [Reference kit connector layout](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/hardware_layout.html).

For synchronized audio/video, identify the clock domain of each timestamp and the mapping to a common timeline. Matching callback arrival times is not enough. Keep measured capture delay and clock drift separate from a constant synchronization offset. A pipeline that starts aligned can drift over minutes.

## 8. Completion and diagnostic answers

Submit the CPU audio report, the video queue report, and a diagram with formats and timestamp meanings at each boundary. Include one deliberate failure and a correction. Hardware work additionally records the module/BSP, negotiated modes, route and observed logs.

- **High FPS but delayed reaction?** Frames can queue before inference; inspect age and occupancy.
- **A one-second WAV becomes three seconds?** The rate metadata changed without resampling.
- **`memory:NVMM` in caps proves no copies?** No. Inspect each conversion, allocation and consumer; caps describe a negotiated memory feature, not a whole-pipeline proof.
- **A codec plugin exists but encoding fails?** Check the actual module's engine, codec/profile support and caps.
- **Correctly sized capture proves no lost samples?** No. Validate a known sequence/timestamp reference and XRUN evidence.
