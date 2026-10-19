# CPU video lab: pixels, timestamps, queues, and a codec round trip

A video pipeline can display a convincing picture while quietly using old frames, indexing row padding as pixels, or changing timing during encoding. This lab checks those contracts with actual GStreamer and FFmpeg execution. It uses generated images, so no camera, display, Jetson, GPU, or network stream is required.

Allow 45–60 minutes. The layout and timestamp arithmetic tests need only Python 3.10+. The native exercises use Linux with PyGObject, GStreamer base plugins/introspection, and FFmpeg with libx264. On Ubuntu 24.04 the relevant packages are:

```bash
sudo apt-get install python3-gi gir1.2-gstreamer-1.0 \
  gir1.2-gst-plugins-base-1.0 gstreamer1.0-plugins-base ffmpeg
```

Use the distribution's `/usr/bin/python3` for its installed PyGObject bindings. A separate Python environment may not see those bindings even when the packages are installed. The recorded execution used Python 3.12.3, GStreamer 1.24.2, and FFmpeg 6.1.1 on x86-64 Linux.

## 1. Read the pixels without reading padding

The synthetic source produces RGB images with width 65 and height 48. Logical row length is `65*3 = 195` bytes. In the recorded GStreamer allocation, row stride is 196 bytes, so the allocation contains `196*48 = 9408` bytes while logical pixels occupy `65*48*3 = 9360` bytes.

Treating the whole allocation as a tightly packed 48×65×3 array would include padding and shift later rows. The [reader](video_lab.py) obtains stride and offset from `GstVideo.VideoInfo`, checks the bounds, and copies only logical RGB rows. It does not infer stride from width. This fixed CPU pipeline uses the default layout described by its caps. When adapting the reader to a camera or custom allocator, check the buffer’s [GstVideoMeta](https://gstreamer.freedesktop.org/documentation/video/gstvideometa.html) or map a [GstVideoFrame](https://gstreamer.freedesktop.org/documentation/video/video-frame.html): per-buffer plane offsets and strides can differ from that default. The supplied reader does not support arbitrary GPU memory or negative strides.

The source is black, so every logical pixel byte should be zero. The tests verify the complete logical pixel hash. Padding bytes are excluded because they are not part of the image contract. This CPU copy is deliberate for inspection; the lab makes no zero-copy claim.

## 2. Compare a blocking sink with an old-frame-dropping sink

From the repository root:

```bash
cd 'Phase 4 - Track B - Nvidia Jetson/1. Nvidia Jetson Platform/Orin-Nano-Video-Codec-DeepStream/labs'
python3 video_lab.py --keep-all > /tmp/video-keep-all.json
python3 video_lab.py > /tmp/video-drop-oldest.json
python3 -m unittest discover -s . -v
```

Both runs request 30 generated frames at 50 frames/s. The consumer sleeps 40 ms after each pulled frame, modeling a consumer slower than the 20 ms source period. The sink capacity is two frames.

- **Keep all:** a full sink blocks upstream. It preserves frames, but backpressure makes the producer/consumer fall behind the nominal timeline.
- **Drop oldest:** a full sink replaces old queued frames. The application receives fewer frames with more recent timestamps.

GStreamer 1.24 uses the appsink `drop` property for this behavior. GStreamer 1.28 introduced `leaky-type` and deprecated `drop`; the script detects the available property. It does not assume that all JetPack releases expose the latest desktop API. See the [appsink reference](https://gstreamer.freedesktop.org/documentation/app/appsink.html).

A recorded run on **2026-10-18** produced:

| Observation | Keep all | Drop oldest |
|---|---:|---:|
| Generated frames | 30 | 30 |
| Consumed frames | 30 | 17 |
| Omitted frames | 0 | 13 |
| Largest observed age at pull | 599.148 ms | 66.934 ms |

Raw [keep-all](keep-all-example.json) and [drop-oldest](drop-oldest-example.json) records are included. These figures depend on host scheduling and the final queue drain; they are not promised results. The dropping run still exceeds a hypothetical 20 ms age limit. Queue capacity alone does not establish freshness.

A synthetic live source with backpressure is not a physical camera. A real sensor may continue exposing frames while a driver drops them, blocks capture, or fills another queue. Repeat the same measurement across the real acquisition path before drawing a device conclusion.

## 3. Use the correct clock relationship

The script maps a buffer's presentation timestamp (PTS) through its segment to **pipeline running time**. It compares that with the pipeline clock minus its base time. Subtracting a stream PTS directly from Unix wall time would be meaningless.

For non-live fixtures the script reports no wall-time age: a file or generated non-live source may produce future media timestamps as fast as the CPU permits. The deterministic overflow test lets all ten non-live frames reach a dropping sink of capacity two, then verifies that frame IDs 8 and 9 remain. This proves the queue policy without relying on a lucky scheduling delay.

Study [GStreamer's clock model](https://gstreamer.freedesktop.org/documentation/application-development/advanced/clocks.html) before combining a camera's exposure timestamp with pipeline time. PTS is a presentation coordinate; it is not automatically a hardware exposure measurement.

## 4. Encode, decode, and verify a small clip

Run:

```bash
python3 codec_lab.py --output-dir /tmp/course-codec-lab
```

The [codec exercise](codec_lab.py) generates twelve 64×48 grayscale frames containing a moving bright bar. It encodes them with CPU libx264, CRF 23, no B-frames, one encoder thread, and YUV420 output, then decodes to grayscale. It preserves the command lines, raw source, encoded MP4, decoded pixels, ffprobe output, and report in the chosen directory. Use a new output directory to retain earlier runs.

The verifier checks all twelve frames, geometry, and exact presentation times using the stream's rational time base. It then calculates pixel mean squared error (MSE), maximum absolute error, and PSNR. It rejects missing pixels, missing frames, and a changed timestamp rather than treating “the decoder exited successfully” as sufficient.

The recorded fixture produced 1,830 encoded bytes, grayscale MSE **1.85072**, PSNR **45.4574 dB**, and maximum pixel error **13**. See the [actual report](codec-example.json). These are codec/fixture observations, not camera quality or robot perception metrics. Different builds or encoder settings may change them. A high PSNR can still hide a lost small obstacle; evaluate the downstream task separately.

## 5. Make and explain a failure

1. Remove one frame from the decoded bytes. The length check must fail.
2. Change one reported PTS. The rational-timestamp check must fail.
3. Treat stride as 195 instead of the reported 196. Explain why row one appears correct while later rows become wrong.
4. Increase consumer delay. Compare omitted IDs and age, then apply a separate explicit age rejection rule.
5. Change codec CRF in a copy of the script. Compare size and pixel error; do not infer a universal quality threshold from one moving bar.

The nine tests include three actual GStreamer executions and one actual FFmpeg encode/decode when dependencies are present. Missing native dependencies are reported as skips; the pure-Python checks still run. The [validation record](validation.json) identifies the executed environment and source hashes. No physical camera, DeepStream plugin, Jetson engine, or GPU was executed in this lab.

Continue to the [video pipeline guide](../Guide.md) to replace the generated source with a measured camera path and add inference metadata.
