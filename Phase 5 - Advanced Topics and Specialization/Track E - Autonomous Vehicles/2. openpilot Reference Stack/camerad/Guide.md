# Camera pipeline: preserve pixels, time and ownership

**Before:** [openpilot process map](../Guide.md), basic image formats and memory layout. **Time:** 60–90 minutes. **Artifact:** a frame contract and a source-supported latency/layout calculation. **Validation:** pinned source inspected and layout arithmetic executed; no comma camera or ISP hardware was run.

## 1. Start from an acquired frame

A sensor integrates light over an interval, emits pixels through a camera interface, and an ISP produces a usable image. The application receives a buffer and metadata. “Frame arrived” is not the same event as exposure start, exposure end or inference completion.

In the reviewed [camera metadata](https://github.com/commaai/openpilot/blob/b1e88c2357f07dfc4e0872370b64d97ad606391b/openpilot/system/camerad/cameras/camera_common.h), `FrameMetadata` includes frame ID, request ID, start/end timestamps and processing time. The Qualcomm camera path publishes acquisition metadata alongside camera state and passes image buffers through VisionIPC. Read the producer before deciding which timestamp represents the observation used by your model.

The source contains OX03C10 and OS04C10 sensor implementations. Their presence does not establish that every comma device generation uses the same sensor, resolution, ISP graph or configuration. Hardware identity belongs in the recorded experiment, not in an unversioned sensor table.

## 2. Read the actual source path

| File at the pinned snapshot | Question to answer |
|---|---|
| `openpilot/system/camerad/main.cc` | Where does the process enter the camera loop? |
| `openpilot/system/camerad/cameras/camera_qcom2.cc` | Which event signals progress, and how is metadata published? |
| `openpilot/system/camerad/cameras/spectra.cc` | How are the Qualcomm request/ISP resources configured? |
| `openpilot/system/camerad/cameras/camera_common.cc` | How are frame buffers created and sent? |
| `openpilot/system/camerad/cameras/nv12_info.py` | What are stride, aligned plane heights and allocation size? |
| `openpilot/system/camerad/sensors/sensor.h` and sensor implementations | What exposure/register/layout constraints belong to each sensor? |

The reviewed event loop uses `VIDIOC_DQEVENT` and Qualcomm request-manager events. A generic webcam `VIDIOC_DQBUF → publish` diagram is not an accurate account of this path. V4L2/media concepts still help, but the actual ioctls and ownership come from this driver integration. [Pinned camera loop](https://github.com/commaai/openpilot/blob/b1e88c2357f07dfc4e0872370b64d97ad606391b/openpilot/system/camerad/cameras/camera_qcom2.cc).

## 3. Work the NV12 allocation instead of guessing it

For an even width/height, tightly packed NV12 has an 8-bit full-resolution Y plane and interleaved UV samples at half width and half height. Its logical payload is `1.5 × width × height` bytes. That does **not** establish an ISP allocation's row stride or total size.

The pinned `get_nv12_info(1928,1208)` returns:

```text
stride = 2048 bytes
aligned Y height = 1216 rows
aligned UV height = 608 rows
buffer size = 4,804,608 bytes
logical tight payload = 3,493,536 bytes
```

The extra bytes include row/plane alignment and additional padding in this source's allocation policy. They are not extra visible pixels. The [exact helper](https://github.com/commaai/openpilot/blob/b1e88c2357f07dfc4e0872370b64d97ad606391b/openpilot/system/camerad/cameras/nv12_info.py) aligns width to 128, Y height to 32 and UV height to 16, then applies its additional buffer padding. These values were calculated by executing that pinned Python helper, not by capturing a device buffer.

**Worked indexing:** with a declared stride of 2048, Y pixel `(x=10,y=3)` is at offset `3×2048+10=6154` from the Y plane start. Using width 1928 as the stride would read offset 5794, 360 bytes too early. For UV, you also need the actual UV plane offset and subsampling convention. Never infer the offset from logical height when the plane height is aligned.

**Exercise:** make a 4×2 logical Y image inside an 8-byte row stride, fill row padding with 255 and valid pixels with 10. Compute the mean over valid pixels (10), then over all allocated Y bytes (132.5). Explain why a normalization step that averages padding can alter the model even though the image dimensions are correct.

## 4. Exposure is a delayed control loop

The pinned `CameraState::set_camera_exposure` comments and bookkeeping account for a three-frame command-to-effect delay in that camera path. At a hypothetical 20 frames/s, three frames correspond to 150 ms. That is an exposure-response model, not a measurement of camera-to-actuator latency and not a guarantee for every sensor configuration.

Trace which stored exposure value is paired with the newly measured image statistic. Pairing the new command with pixels produced under an older command introduces a feedback-model error. Aggressive gain tuning can then oscillate or saturate despite a correct arithmetic update.

For a replay study, retain frame ID, exposure/gain settings, target/measured grey fraction and acquisition timestamps. Plot delayed command versus observed brightness. Account for saturation and scene changes before attributing every brightness change to the controller. The upstream algorithm is more specific than a generic “PI controller” label.

## 5. Shared memory still needs a lifetime rule

A consumer may hold a reference to an image while the producer wants to reuse storage. The contract must say when reuse is allowed and how the consumer detects a newer generation. A pointer plus a timestamp is not sufficient if the pixels can be overwritten independently.

Draw: producer owns buffer → completes acquisition/processing → publishes buffer identity and metadata → consumer finishes required reads → storage becomes reusable. Establish the actual implementation's behavior from VisionIPC source rather than assuming this drawing supplies its protocol. Image warp, normalization, accelerator upload and history stacking can still allocate or copy data after IPC.

**Fault exercise:** copy metadata from frame `k`, but pair it with pixels from `k+1` in a replay fixture. Which checks detect it? A finite-value check will not. A frame/generation contract or controlled fixture with distinct per-frame content can.

## Completion

Submit the 6154-byte offset calculation, the padding contamination example, the exposure-delay pairing and a buffer-lifetime diagram. Carry the original acquisition timestamp into the [replay gate](../Replay-Lab.md). For kernel/process implications continue with [AGNOS](../agnos/Guide.md).
