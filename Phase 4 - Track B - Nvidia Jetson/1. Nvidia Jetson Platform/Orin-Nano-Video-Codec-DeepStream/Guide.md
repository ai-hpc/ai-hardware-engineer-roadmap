# Jetson video pipelines: capture, decode, inference, and encoding

A useful video pipeline preserves image meaning and timing from the source to the consumer. This course starts with generated CPU video, then explains how to replace one stage at a time with Jetson acceleration. You will learn caps negotiation, buffer layout, queueing, codec timing, and inference metadata without assuming that a successful preview proves the robot sees fresh data.

Complete the [CPU video lab](labs/README.md) first. It executes real GStreamer pipelines and an FFmpeg encode/decode round trip, with tests for row padding, dropped frames, timestamps, and pixel error. Allow one session for that lab, one for the platform pipeline, and one for the capstone report.

**Hardware boundary:** Orin Nano has a hardware decoder but no NVENC hardware encoder and no DLA. Use its supported software encoding path. An NX/AGX hardware-encoding comparison requires the corresponding module and software support. NVIDIA documents [software encoding on Orin Nano](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/Multimedia/SoftwareEncodeInOrinNano.html). A plugin name or a family-level block diagram does not add an engine to a module.

## 1. Choose a supported software stack

Checked **2026-10-18**, the official DeepStream matrix lists these distinct baselines:

| DeepStream | JetPack baseline | Jetson scope in that release |
|---|---|---|
| 7.1 | 6.1 GA / L4T 36.4 | Orin family, including Nano |
| 8.0 | 7.0 GA / L4T 38.2 | AGX Thor |
| 9.0 | 7.1 GA / L4T 38.4 | AGX Thor |
| 9.1 | 7.2 GA / L4T 39.2 | Thor and Orin |

DeepStream 9.1's documented Jetson stack uses GStreamer 1.24.2 and TensorRT 10.16.1.7. This matrix names JetPack 7.2, not every later maintenance combination. Check the [official installation matrix](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html) and the course [Jetson compatibility notes](../../Compatibility.md) for the exact system you intend to run.

DeepStream 9.1 moves SDK package distribution to the NVIDIA DeepStream GitHub release and uses the `deepstream` container repository for current containers; older `deepstream-l4t` releases remain historical artifacts. The [release notes](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Release_notes.html) describe that change. Pin the selected release or image digest in your deployment record.

The traditional Python `pyds` bindings are now deprecated in current documentation, which directs new Python work toward Service Maker. Existing 7.x examples still need their matching environment. Read the [Python binding status](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Python_Sample_Apps.html) before copying an old probe callback into a new application.

## 2. Separate acquisition from compression

A CSI camera may deliver raw Bayer data, a USB camera may deliver uncompressed YUV or MJPEG, and an IP camera may deliver H.264/H.265 over a network. Those sources require different first stages:

```text
CSI raw sensor → supported camera/ISP path → raw image surface
USB raw frame → V4L2 capture → raw image buffer
USB MJPEG → V4L2 capture → JPEG decode → raw image buffer
IP compressed stream → transport/demux → parser → video decode → raw image surface
```

A parser interprets bitstream framing and codec metadata; it does not reconstruct pixels. A decoder reconstructs images. A converter changes image format, size, or memory representation. An encoder compresses images into a new bitstream. A muxer packages streams and timing into a container such as MP4.

Hardware acceleration is operation-specific. A supported hardware decoder says nothing about the availability of an encoder or a particular codec/profile/bit-depth combination. Query the installed plugins and consult the module's multimedia support table. Do not assume `/dev/video0` is a decoder: it may be a camera, and NVIDIA codec interfaces need not use the same node layout as a generic V4L2 device.

Before connecting the real camera, run the generated-source lab. Then verify camera acquisition alone using the [camera bring-up guide](../Orin-Nano-Camera-ISP-Sensor-Bringup/Guide.md). This makes a failure attributable to a specific boundary.

## 3. Read caps as a data contract

GStreamer elements exchange buffers under negotiated capabilities, or **caps**. These describe media type and properties such as format, resolution, rate, and memory features. An element cannot consume an arbitrary buffer just because the pipeline string links syntactically.

Compare:

```text
video/x-raw,format=RGB,width=640,height=480
video/x-raw,format=NV12,width=640,height=480
video/x-raw(memory:NVMM),format=NV12,width=640,height=480
video/x-h264,stream-format=byte-stream,alignment=au
```

The first three describe raw video, with different layout/memory contracts. The fourth describes a compressed stream. RGB and NV12 do not have the same number or arrangement of bytes. `memory:NVMM` identifies a negotiated NVIDIA memory feature; it is not proof that every downstream operation avoids allocations, conversions, or copies.

Use the installed tools for inspection:

```bash
gst-inspect-1.0 --version
gst-inspect-1.0 videotestsrc
gst-inspect-1.0 appsink
gst-inspect-1.0 nvv4l2decoder
gst-inspect-1.0 nvvidconv
gst-inspect-1.0 nvvideoconvert
```

The last three depend on the NVIDIA packages. Jetson's `nvvidconv` and DeepStream's `nvvideoconvert` are distinct plugins with version-specific properties; one name cannot be substituted blindly for the other. The [Jetson accelerated GStreamer documentation](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/Multimedia/AcceleratedGstreamer.html) gives a release-specific reference.

For an unresolved caps error, remove later stages until the source reaches a `fakesink`, inspect negotiated caps with `gst-launch-1.0 -v`, then add one conversion or consumer at a time. A capsfilter requests a representation; it does not itself perform conversion.

## 4. Preserve layout and ownership across engines

Raw video can be planar, semiplanar, or interleaved, and its rows can include padding. The CPU lab's width-65 RGB image has 195 logical bytes per row but a measured stride of 196. Using `width*channels` as the row stride reads the wrong bytes on later rows.

For a multi-plane surface, obtain each plane's offset, pitch, and dimensions from the API. A block-linear surface is not a row-major array suitable for arbitrary CPU indexing. Map, synchronize, transform, or copy through the supported interface before reading it.

A shared handle provides access to an allocation; it does not establish that its producer has finished. Keep the surface alive until all downstream GPU, display, encoder, or CPU consumers complete. The [memory guide](../Orin-Nano-Memory-Architecture/Guide.md) provides an allocation/lifetime worksheet.

“Zero-copy” should identify a particular avoided copy, such as importing a decoded surface directly into a supported consumer. Scaling or color conversion still reads input and writes output. Metadata extraction can be small while a pixel conversion remains expensive. An observed `cudaMemcpy` is a reason to inspect the path, not automatic proof that the entire pipeline is wrong; some transfers may be necessary for the chosen consumer.

## 5. Control queues and clocks deliberately

A GStreamer `queue` creates a separate streaming thread and buffers data until one of its enabled limits is reached. Its buffer-count, byte, and time limits interact. If you want a two-buffer limit to be the governing one, specify the other limits deliberately rather than relying on defaults:

```text
queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream
```

This drops older queued buffers when full. It does not remove the frame already being processed or other queues elsewhere in the pipeline. The [queue documentation](https://gstreamer.freedesktop.org/documentation/coreelements/queue.html) defines the behavior.

A `tee` sends data to multiple branches. Give branches appropriate queues so a slow recorder does not unintentionally block a control consumer. A queue eventually fills under sustained overload, so decide whether each branch blocks or drops. A lossless recorder and a freshness-oriented controller often need different policies.

PTS determines presentation order/time; DTS concerns decode order. With inter-frame compression, decode and display order can differ. A network jitter buffer, codec reorder buffer, muxer, sink clock, and application queue can all add latency.

GStreamer maps timestamps through segments and pipeline running time. A sensor's exposure timestamp must be related to that clock explicitly. `sync=false` on a sink disables waiting for presentation time; it does not remove upstream latency or turn PTS into wall-clock time. See the [clock and synchronization guide](https://gstreamer.freedesktop.org/documentation/application-development/advanced/clocks.html).

The CPU lab demonstrates both keep-all backpressure and dropping old frames, with real raw traces. The latter still needs an age check before a robot uses the result.

## 6. Establish a Nano-compatible encoding baseline

The CPU codec lab uses FFmpeg/libx264 and validates decoded output. For a GStreamer installation with the additional x264 and parser plugins, this is a generated-source **device-side template**:

```bash
gst-inspect-1.0 x264enc
gst-inspect-1.0 h264parse
gst-launch-1.0 -e videotestsrc num-buffers=120 ! \
  'video/x-raw,width=640,height=480,framerate=30/1,format=I420' ! \
  x264enc tune=zerolatency speed-preset=ultrafast bitrate=1000 ! \
  h264parse ! mp4mux ! filesink location=encode-baseline.mp4
```

The command requires its named plugins and is not presented as a measured Nano run. `-e` requests end-of-stream handling when the command is interrupted, which helps muxers finalize a file. Verify the resulting frame count, timestamps, duration, and decode success with ffprobe/FFmpeg before using the file as evaluation data.

Encoder settings trade bitrate, distortion, compute, and buffering. A zero-latency tune changes codec behavior but does not eliminate capture or network delay. B-frames and lookahead can improve compression while introducing buffering. Shorter keyframe intervals can improve recovery after loss while increasing bitrate. Compare task quality as well as PSNR; a small obstacle can matter more than average pixel error.

On compatible hardware with NVENC, create a separate hardware comparison using the installed encoder's documented properties. Hold source frames, resolution, rate, quality target, and measurement boundary constant. Do not copy a CPU/power/latency table from an unrelated module and label it Nano performance.

## 7. Add DeepStream one component at a time

A common analytics path is:

```text
source → decode/capture → stream mux → preprocessing → inference
       → tracking → metadata export and optional overlay/recording
```

Start from the sample configuration supplied with the selected DeepStream release and one file source. Establish correct model input normalization, labels, output parser, and coordinate transforms before adding live cameras. The [official quickstart](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html) identifies the matching sample/configuration locations.

Batching multiple streams requires preserving source identity. A batch slot is not a permanent camera ID. Record source ID, frame number, timestamp, dimensions, and any preprocessing transform with each inference result. A tracker ID is local to the tracker's lifecycle and association rules, not proof of a person's identity or a globally stable object identifier.

If inference runs only every nth frame, tracking may produce intermediate estimates. Report which results are measured detections and which are predictions. A tracker cannot recover a missing observation without assumptions about motion and association.

Custom models need more than an engine file. Check input layout, channel order, scale/mean normalization, output shapes, class mapping, thresholding, non-maximum suppression, and coordinate scaling. Validate decoded predictions against the CPU/reference framework on identical images. Use the [inference guide](../Orin-Nano-Real-Time-Inference/Guide.md) for correctness and age checks.

Metadata callbacks should keep expensive network or filesystem operations off the critical streaming path. Copy or retain metadata only through the documented ownership mechanism; a pointer into a released batch is not valid asynchronous application state. For current Python work, follow the supported Service Maker route rather than assuming historical `pyds` examples are the current API.

## 8. Diagnose failures with a small reproducible pipeline

| Symptom | First useful check |
|---|---|
| Pipeline cannot link | Inspect element pad caps and required format/memory conversion |
| Black or striped image | Verify format, plane offsets, stride, and mapping/coherency |
| Good FPS, delayed action | Trace exposure/PTS mapping and every queue |
| Correct boxes at wrong coordinates | Reverse resize, crop, padding, and display transforms |
| Recording unreadable after exit | Check EOS/mux finalization and storage errors |
| Works with one stream, fails with four | Measure batch/profile limits, peak buffers, compute, and bandwidth |
| Old output appears after an error | Check asynchronous completion and per-request buffer ownership |
| Preview works but model quality falls | Compare exact model inputs with reference preprocessing |

Capture GStreamer bus errors and warnings rather than only standard output. Bound reconnection attempts and declare stream readiness explicitly. A restarted stream may reset sequence counters or timestamps; downstream tracking and control need to recognize that boundary.

## 9. Capstone: one reproducible camera-to-result trace

Deliver a pipeline diagram, exact software/module manifest, source/caps description, buffer-lifetime table, model verification fixtures, and a paired per-frame timing trace. Include a controlled slow-consumer experiment, a source disconnect/restart, and an encoding round trip if the deployment records video.

Report frame omissions, end-to-end age, task quality, peak memory, and sustained resource use. The CPU route completes the buffer/queue/codec contracts with real execution. A Jetson claim needs a Jetson run; a physical sensor claim needs sensor evidence; robot action quality needs controlled closed-loop testing.

**Self-check:** why is a 30 FPS preview insufficient? It can hide old frames, incorrect timestamps, wrong color preprocessing, dropped measurements, or downstream action delay. A useful pipeline report checks those contracts directly.
