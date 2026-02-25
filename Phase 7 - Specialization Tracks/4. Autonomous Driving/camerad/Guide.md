# camerad — Openpilot Camera Pipeline

> **Goal:** Understand how openpilot captures, processes, and delivers camera frames to the perception stack. camerad is the first process in the pipeline: raw sensor → ISP → VisionIpc → modeld.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Camera Configuration](#3-camera-configuration)
4. [Hardware Stack](#4-hardware-stack)
5. [ISP Pipeline](#5-isp-pipeline)
6. [Auto Exposure (AE)](#6-auto-exposure-ae)
7. [VisionIpc](#7-visionipc)
8. [Data Flow](#8-data-flow)
9. [Source Map](#9-source-map)
10. [Key Concepts](#10-key-concepts)

---

## 1. Overview

**camerad** is a native C++ process that:

- Captures frames from up to **three cameras** (wide road, road, driver)
- Runs them through the **Qualcomm Spectra ISP** (Image Signal Processor)
- Publishes **YUV frames** via **VisionIpc** (shared memory IPC)
- Publishes **FrameData** (metadata) via cereal messaging
- Implements **auto exposure (AE)** to adapt to lighting

**Location:** `openpilot/system/camerad/`

**Entry point:** `main.cc` → `camerad_thread()` (pinned to CPU core 6)

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              camerad                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  Sensor (I2C)     │  CSI/MIPI    │  IFE (Image Front End)  │  BPS (optional)  │
│  OX03C10/OS04C10 │  → RAW       │  Demosaic, CCM, Gamma   │  Downscale        │
│                  │              │  Vignetting correction  │  (driver cam)     │
└─────────────────────────────────────────────────────────────────────────────┘
         │                                    │
         ▼                                    ▼
   Exposure control                    YUV output
   (AE algorithm)                      VisionIpc buffers
         │                                    │
         └────────────────┬───────────────────┘
                          ▼
                   FrameData (cereal)
                   frame_id, timestamps, gain, exposure
```

**Three cameras (comma 3X):**

| Camera | Stream | Role | Focal length |
|--------|--------|------|--------------|
| **Wide road** | `VISION_STREAM_WIDE_ROAD` | Wide FoV, peripheral | 1.71 mm |
| **Road** | `VISION_STREAM_ROAD` | Main driving view | 8.0 mm |
| **Driver** | `VISION_STREAM_DRIVER` | Driver monitoring (DMS) | 1.71 mm |

---

## 3. Camera Configuration

Defined in `cameras/hw.h`:

```cpp
// Wide: fisheye, peripheral
WIDE_ROAD_CAMERA_CONFIG = {
  .camera_num = 0,
  .stream_type = VISION_STREAM_WIDE_ROAD,
  .focal_len = 1.71,
  .publish_name = "wideRoadCameraState",
  .output_type = ISP_IFE_PROCESSED,
};

// Road: main forward-facing, narrow FoV
ROAD_CAMERA_CONFIG = {
  .camera_num = 1,
  .stream_type = VISION_STREAM_ROAD,
  .focal_len = 8.0,
  .publish_name = "roadCameraState",
  .vignetting_correction = true,
  .output_type = ISP_IFE_PROCESSED,
};

// Driver: cabin-facing
DRIVER_CAMERA_CONFIG = {
  .camera_num = 2,
  .stream_type = VISION_STREAM_DRIVER,
  .focal_len = 1.71,
  .publish_name = "driverCameraState",
  .output_type = ISP_BPS_PROCESSED,  // BPS for extra processing
};
```

**Python intrinsics** (for modeld warp, calibration): `common/transformations/camera.py`

- Road: 1928×1208, focal 2648 px (OX) or 1344×760, focal ~1142 px (OS)
- Wide: 1928×1208, focal 567 px
- Driver: same as wide

---

## 4. Hardware Stack

### Sensors

| Sensor | Device | Resolution | Notes |
|--------|--------|------------|-------|
| **OX03C10** | comma 3X (tici/tizi) | 1928×1208 | OmniVision, 3 MP |
| **OS04C10** | comma 3X (mici) | 2688×1520 | OmniVision, 4 MP |
| **AR0231** | Legacy (neo) | 1164×874 | Aptina |

**Sensor interface:** I2C for register control (exposure, gain, init). MIPI CSI for image data.

**Files:** `sensors/ox03c10.cc`, `sensors/os04c10.cc`, `sensors/sensor.h`

### Qualcomm Spectra ISP

- **IFE** (Image Front End): demosaic, color correction (CCM), gamma, vignetting
- **BPS** (Bayer Processing Segment): used for driver camera (extra downscale/processing)
- **CDM** (Camera Data Mover): DMA, buffer management
- **CSIPHY**: MIPI CSI physical layer

**Kernel:** Linux V4L2 (Video4Linux2), `CAM_REQ_MGR` (Request Manager) for frame synchronization.

### V4L2: Linux Kernel Camera API

**V4L2** (Video4Linux2) is the standard Linux kernel API for video capture, output, and codecs. camerad uses V4L2 to drive the Qualcomm Spectra ISP and receive processed frames.

#### What V4L2 Provides

| Concept | Description |
|---------|-------------|
| **Device nodes** | `/dev/video0`, `/dev/video1`, … — one per capture/output device. camerad opens `/dev/video0` (sync device) for the request manager. |
| **Media Controller** | Models the pipeline: sensor → CSI → IFE → BPS → video node. Used to discover and link entities. |
| **Request API** | One *request* = one frame through the pipeline. Sensor capture, IFE, BPS all tied to the same request for synchronization. |
| **Buffer flow** | `VIDIOC_QBUF` enqueue empty buffer → hardware fills it → `VIDIOC_DQBUF` dequeue filled buffer. |

#### camerad's V4L2 Flow

```
1. Open /dev/video0 (sync/request manager device)
2. Media Controller: discover sensor, IFE, BPS entities; configure links
3. VIDIOC_REQBUFS: allocate capture buffers (YUV output)
4. VIDIOC_QUERYBUF: get buffer addresses for mmap()
5. VIDIOC_QBUF: enqueue buffers into the pipeline
6. VIDIOC_STREAMON: start streaming
7. poll(fd, POLLPRI): wait for frame-done event
8. VIDIOC_DQEVENT: dequeue event (frame completed)
9. processFrame() → YUV ready → sendFrameToVipc()
10. VIDIOC_QBUF: re-enqueue buffer for next frame
```

#### Key ioctls Used by camerad

| ioctl | Purpose |
|-------|---------|
| `VIDIOC_REQBUFS` | Allocate buffer queue (memory-mapped or DMA) |
| `VIDIOC_QUERYBUF` | Get buffer info (offset, length) for mmap |
| `VIDIOC_QBUF` | Enqueue buffer for capture |
| `VIDIOC_DQBUF` | Dequeue filled buffer (alternatively, events can signal completion) |
| `VIDIOC_DQEVENT` | Dequeue async event (e.g. frame done) — used with Request API |
| `VIDIOC_STREAMON` / `VIDIOC_STREAMOFF` | Start/stop streaming |
| `VIDIOC_S_FMT` | Set pixel format (e.g. NV12) |
| `VIDIOC_S_PARM` | Set frame rate |

#### Event-Driven Model

camerad uses **event-driven** capture, not blocking `DQBUF`:

- `poll(fd, POLLPRI)` — wait until a frame-done event is available
- `VIDIOC_DQEVENT` — dequeue the event; payload indicates which request/frame completed
- Process the frame, re-enqueue buffers, then poll again

`POLLPRI` (priority/exceptional condition) is set when an event is pending. This avoids busy-waiting and integrates cleanly with the Request API.

#### CAM_REQ_MGR (Qualcomm Request Manager)

On Qualcomm platforms, **CAM_REQ_MGR** is a kernel component that:

- **Synchronizes** sensor capture with ISP (IFE, BPS) — one request ties them together
- **Queues requests** — each request carries buffers and controls for the whole pipeline
- **Signals completion** — when the pipeline finishes a frame, an event is queued for userspace

Without the Request API, sensor and ISP could run out of sync (e.g. wrong exposure applied to a frame). The Request API guarantees: *this* sensor frame → *this* IFE config → *this* output buffer.

#### Device Layout (Qualcomm Spectra)

| Device | Role |
|--------|------|
| `/dev/video0` | Sync/request manager — camerad polls here for frame-done events |
| `/dev/media0` | Media controller — pipeline topology (sensor ↔ IFE ↔ BPS) |
| Subdevs (e.g. `/dev/v4l-subdev*`) | Sensor, IFE, BPS as separate entities; configured via media controller |

#### Debugging V4L2

```bash
# List video devices and capabilities
v4l2-ctl --list-devices

# List supported formats
v4l2-ctl -d /dev/video0 --list-formats-ext

# Media controller: show pipeline
media-ctl -d /dev/media0 -p
```

**References:** [V4L2 API (kernel.org)](https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/v4l2.html), [VIDIOC_QBUF/VIDIOC_DQBUF](https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/vidioc-qbuf.html), [V4L2 poll()](https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/func-poll.html)

---

## 5. ISP Pipeline

```
Sensor (RAW Bayer) → CSI → IFE → [BPS for driver] → YUV (NV12)
```

**Output format:** NV12 (Y plane + interleaved UV plane). Standard format for vision/ML.

**Buffer flow:**

1. `SpectraMaster` initializes `/dev/video0`, sync, ISP, ICP
2. `SpectraCamera` per camera: sensor init, IFE config, BPS config (driver only), link devices
3. `CameraBuf` allocates raw (optional) and YUV buffers
4. `VisionIpcServer` creates shared-memory buffers for consumers (modeld, etc.)
5. On V4L2 frame event: `handle_camera_event` → `processFrame` → `sendFrameToVipc` + publish FrameData

---

## 6. Auto Exposure (AE)

camerad implements **software AE** (no sensor AEC). Goal: keep a target region at ~12.5% grey (median luminance).

**Algorithm** (`camera_qcom2.cc`):

1. **Measure:** `calculate_exposure_value()` bins luminance in an AE rectangle, returns median grey fraction
2. **Target:** `target_grey_fraction` ≈ 0.125, scaled by scene brightness (darker → lower target)
3. **Control loop:** PI-like update with ~3-frame latency (sensor register buffering)
4. **Optimizer:** Brute-force over `(exposure_time, analog_gain)` to minimize `getExposureScore` (EV error)
5. **DC gain:** High conversion gain for low light; hysteresis to avoid flicker

**AE rectangles** (per camera, in pixel coords):

- Wide: `{96, 400, 1734, 524}` — lower part of frame
- Road: `{96, 160, 1734, 986}` — most of frame
- Driver: `{96, 242, 1736, 906}` — face region

**Exposure params:** `exposure_time` (µs), `analog_gain`, `dc_gain_enabled`. Sent to sensor via I2C.

---

## 7. VisionIpc

**VisionIpc** = shared-memory IPC for video frames. Avoids copies; modeld reads directly from shared buffers.

**Server (camerad):**

```cpp
VisionIpcServer v("camerad");
v.create_buffers_with_sizes(stream_type, VIPC_BUFFER_COUNT, width, height, yuv_size, stride, uv_offset);
// ...
vipc_server->send(cur_yuv_buf, &extra);  // on each frame
```

**Client (modeld):**

```python
from msgq.visionipc import VisionIpcClient, VisionStreamType
vipc = VisionIpcClient("camerad", 0, VisionStreamType.VISION_STREAM_ROAD)
# vipc.recv() → frame buffer
```

**Stream types:** `VISION_STREAM_WIDE_ROAD`, `VISION_STREAM_ROAD`, `VISION_STREAM_DRIVER`

**Buffer count:** 18 (`VIPC_BUFFER_COUNT`). Allows producer/consumer to run at different rates.

---

## 8. Data Flow

```
V4L2 poll(POLLPRI) on video0
    │
    ▼
VIDIOC_DQEVENT (frame done)
    │
    ├─► handle_camera_event()
    │       │
    │       ├─► processFrame() → YUV ready
    │       │
    │       ├─► sendFrameToVipc() → VisionIpc send
    │       │
    │       ├─► calculate_exposure_value() → grey_frac
    │       │
    │       ├─► set_camera_exposure() → I2C exposure registers
    │       │
    │       └─► pm->send("roadCameraState", FrameData)
    │
    └─► (next frame)
```

**FrameData** (cereal): `frameId`, `timestampSof`, `timestampEof`, `integLines`, `gain`, `measuredGreyFraction`, `targetGreyFraction`, `exposureValPercent`, `sensor`, `processingTime`.

---

## 9. Source Map

| Component | Path |
|-----------|------|
| Main loop | `system/camerad/main.cc` |
| Camera state, AE | `system/camerad/cameras/camera_qcom2.cc` |
| Camera buf, VIPC send | `system/camerad/cameras/camera_common.cc` |
| ISP (IFE, BPS, CDM) | `system/camerad/cameras/spectra.cc`, `cdm.cc` |
| Sensor drivers | `system/camerad/sensors/ox03c10.cc`, `os04c10.cc` |
| Camera config | `system/camerad/cameras/hw.h` |
| Intrinsics (Python) | `common/transformations/camera.py` |
| VisionIpc | `msgq/visionipc/` |

---

## 10. Key Concepts

| Concept | Description |
|---------|-------------|
| **NV12** | YUV 4:2:0: Y plane full res, UV interleaved half res. Common for ISP output and ML. |
| **VisionIpc** | Zero-copy frame sharing via shared memory. Server creates buffers; clients subscribe by stream type. |
| **FrameData** | Cereal message with frame metadata. Used for sync (frame_id, timestamps) and AE debugging. |
| **ISP** | Image Signal Processor. Converts RAW Bayer → demosaic → color correct → gamma → YUV. |
| **AE** | Auto exposure. Software loop: measure luminance → compute desired EV → set sensor exposure/gain. |
| **V4L2** | Video4Linux2 — Linux kernel API for video capture. Device nodes (`/dev/video*`), ioctls (QBUF/DQBUF), Request API. |
| **Request Manager** | V4L2 CAM_REQ_MGR: ties sensor capture to ISP pipeline. One request = one frame through the pipeline. |
| **QBUF/DQBUF** | V4L2 buffer exchange: QBUF enqueue empty buffer, DQBUF dequeue filled buffer. |

---

## Further Reading

- **V4L2 API:** [kernel.org V4L2 documentation](https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/v4l2.html) — device nodes, ioctls, buffer flow, Request API
- **Trace the pipeline:** Start at `camerad_thread()` in `camera_qcom2.cc`, follow `handle_camera_event` → `processFrame` → `sendFrameToVipc`
- **modeld consumption:** `selfdrive/modeld/modeld.py` — `VisionIpcClient` for `VISION_STREAM_ROAD` (and wide if used)
- **Calibration:** `common/transformations/camera.py`, `get_warp_matrix` for model input warp
