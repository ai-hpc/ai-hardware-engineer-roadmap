# Multimedia

**Phase 4 — Track B — Module 5.4** · Application Development

> **Focus:** Build hardware-accelerated multimedia pipelines on the **Jetson Orin Nano 8GB** — audio playback/capture, camera integration (USB and CSI), GStreamer video pipelines, display output, and video encode/decode using NVIDIA's NVENC/NVDEC engines.

**Hub:** [5. Application Development](../Guide.md)
**Dedicated audio deep dive:** [Jetson Audio Setup and Development](Jetson-Audio-Setup-and-Development/Guide.md)
**Audio application:** [ESP32-LyraT I2S Microphone Capture on Jetson Orin Nano](ESP32-LyraT-I2S-Mic-Jetson-Orin-Nano/Guide.md)

---


## 1. Audio (Linux)

Audio on Jetson uses ALSA (kernel driver) with PulseAudio or PipeWire (userspace).

If you want the full Jetson-specific model for:

- `APE` vs `HDA`
- 40-pin header `I2S2`
- device tree bring-up for custom codecs
- `amixer` routing through `ADMAIF` / `I2S`
- DAPM and ASoC debugging

use the dedicated guide: [Jetson Audio Setup and Development](Jetson-Audio-Setup-and-Development/Guide.md).

### Check audio devices

```bash
# List ALSA playback devices
aplay -l

# List ALSA capture devices
arecord -l

# List PulseAudio sinks/sources
pactl list short sinks
pactl list short sources
```

### Playback and capture

```bash
# Play a WAV file
aplay test.wav

# Record 5 seconds of audio
arecord -d 5 -f cd -t wav recording.wav

# Adjust volume
amixer set Master 80%
```

### Audio on custom carriers

The Orin Nano SoM provides I2S audio interfaces. Your carrier board needs an **audio codec** (e.g., Realtek ALC5640, TI TLV320AIC) connected via I2S + I2C control. Configure in device tree:

```dts
sound {
    compatible = "nvidia,tegra-audio-t234";
    nvidia,audio-codec = <&codec>;
    nvidia,i2s-controller = <&i2s1>;
};
```

> **Deep dive:** for the full Jetson audio stack, official NVIDIA ASoC model, 40-pin header pinmux, and custom codec bring-up flow, see [Jetson Audio Setup and Development](Jetson-Audio-Setup-and-Development/Guide.md).

---

## 2. Bluetooth audio profiles

### A2DP (stereo audio to BT speaker)

```bash
bluetoothctl
  power on
  scan on
  pair <speaker-MAC>
  connect <speaker-MAC>

# Route audio to Bluetooth
pactl set-default-sink bluez_sink.<MAC-with-underscores>.a2dp_sink
aplay test.wav   # plays through BT speaker
```

### HFP (hands-free, bidirectional)

```bash
# Enable HFP in PulseAudio
# /etc/pulse/default.pa: load-module module-bluetooth-policy
# Restart PulseAudio
pulseaudio --kill && pulseaudio --start
```

---

## 3. GStreamer — audio/video pipelines

GStreamer is the standard media framework on Jetson. NVIDIA provides hardware-accelerated plugins (`nvv4l2decoder`, `nvv4l2h264enc`, `nvarguscamerasrc`, `nv3dsink`).

### Install and verify

```bash
# GStreamer should be pre-installed in JetPack
gst-inspect-1.0 --version

# List NVIDIA plugins
gst-inspect-1.0 | grep nv
```

### Basic pipelines

```bash
# Test video (color bars)
gst-launch-1.0 videotestsrc ! autovideosink

# Test audio (sine wave)
gst-launch-1.0 audiotestsrc ! autoaudiosink

# Play a video file (HW decoded)
gst-launch-1.0 filesrc location=video.mp4 ! \
    qtdemux ! h264parse ! nvv4l2decoder ! nv3dsink
```

---

## 4. Video encoding and playback (GStreamer)

### Hardware-accelerated decode

```bash
# H.264 decode
gst-launch-1.0 filesrc location=input.mp4 ! \
    qtdemux ! h264parse ! nvv4l2decoder ! nv3dsink

# H.265/HEVC decode
gst-launch-1.0 filesrc location=input.mp4 ! \
    qtdemux ! h265parse ! nvv4l2decoder ! nv3dsink
```

### Hardware-accelerated encode

```bash
# Camera → H.264 encode → file
gst-launch-1.0 nvarguscamerasrc ! \
    'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! \
    nvv4l2h264enc bitrate=8000000 ! h264parse ! mp4mux ! \
    filesink location=output.mp4

# Camera → RTSP stream (for remote viewing)
# Use the NVIDIA DeepStream or GStreamer RTSP server
```

### Codec support on Orin Nano

| Codec | Decode | Encode | Max resolution |
|-------|--------|--------|---------------|
| **H.264** | HW | HW | 4K@60 decode, 4K@30 encode |
| **H.265 (HEVC)** | HW | HW | 4K@60 decode, 4K@30 encode |
| **VP9** | HW | — | 4K@60 decode |
| **AV1** | HW | — | 4K@60 decode |
| **JPEG** | HW | HW | — |

---

## 5. USB cameras / webcams (UVC)

Any UVC-compliant USB camera works out of the box.

```bash
# List video devices
v4l2-ctl --list-devices

# Check supported formats
v4l2-ctl -d /dev/video0 --list-formats-ext

# Capture a frame
v4l2-ctl -d /dev/video0 --set-fmt-video=width=1920,height=1080,pixelformat=MJPG \
    --stream-mmap --stream-count=1 --stream-to=frame.mjpg

# GStreamer pipeline (USB webcam → display)
gst-launch-1.0 v4l2src device=/dev/video0 ! \
    'video/x-raw,width=1280,height=720,framerate=30/1' ! \
    videoconvert ! autovideosink
```

---

## 6. CSI cameras (MIPI)

CSI cameras connect via the MIPI CSI-2 interface on the carrier board and are accessed through NVIDIA's **Argus** camera framework.

### Using nvarguscamerasrc (GStreamer)

```bash
# Preview from CSI camera 0
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! \
    'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! \
    nvvidconv ! nv3dsink

# Capture to JPEG
gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=1 ! \
    'video/x-raw(memory:NVMM),width=1920,height=1080' ! \
    nvjpegenc ! filesink location=capture.jpg
```

### Using V4L2 directly

```bash
v4l2-ctl -d /dev/video0 --set-fmt-video=width=1920,height=1080 \
    --set-ctrl bypass_mode=0 --stream-mmap --stream-count=10
```

### Camera device tree configuration

CSI cameras require device tree entries specifying the sensor driver, I2C address, MIPI lanes, and pixel format. Example for IMX219:

```dts
cam0: imx219@10 {
    compatible = "sony,imx219";
    reg = <0x10>;
    clocks = <&bpmp TEGRA234_CLK_EXTPERIPH1>;

    mode0 {
        mclk_khz = "24000";
        num_lanes = "2";
        tegra_sinterface = "serial_a";
        active_w = "3264";
        active_h = "2464";
        pixel_t = "bayer_rggb";
    };
};
```

> **Deep dive:** For full camera ISP, sensor bring-up, and multi-camera configurations see [Orin Nano Camera ISP Sensor Bring-Up](../../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Camera-ISP-Sensor-Bringup/Guide.md).

---

## 7. Display output and resolution

### HDMI/DisplayPort

```bash
# List outputs and modes
xrandr

# Set specific mode
xrandr --output HDMI-0 --mode 1920x1080 --rate 60

# Force a custom mode
cvt 1280 800 60
xrandr --newmode "1280x800_60" ...
xrandr --addmode HDMI-0 "1280x800_60"
xrandr --output HDMI-0 --mode "1280x800_60"
```

### Headless operation

For products without a display, disable display output in the device tree to speed up boot and free resources:

```bash
# Add to kernel command line
video=efifb:off
```

Or remove display-related nodes from the carrier device tree.

---

## 8. Framebuffer and DRM/KMS

### DRM/KMS (modern approach)

```bash
# List DRM devices
ls /dev/dri/

# Check connected displays
sudo cat /sys/class/drm/card*/status

# Use modetest for direct display testing
sudo modetest -M nvidia-drm -s <connector>@<crtc>:<mode>
```

### Framebuffer (legacy)

```bash
# Check framebuffer info
fbset -i

# Write test pattern
sudo apt install fbset
cat /dev/urandom > /dev/fb0   # random noise
```

---

## 9. Projects

- **Multi-camera viewer:** Display 2 CSI cameras side-by-side using GStreamer `nvcompositor` on HDMI output.
- **Video recorder:** Build a GStreamer pipeline that records H.264 video from a CSI camera to NVMe with timestamp overlay, triggered by GPIO button press.
- **Audio intercom:** Stream audio between two Jetson devices using GStreamer RTP over Ethernet.
- **ESP32-LyraT I2S mic capture:** Use a LyraT board as an I2S microphone frontend and capture raw audio through the Jetson Orin Nano 40-pin `I2S2` path.
- **RTSP camera server:** Serve a CSI camera feed as an RTSP stream accessible from VLC on any network device.

---

## 10. Resources

| Resource | Description |
|----------|-------------|
| **NVIDIA Multimedia API** | Jetson multimedia framework documentation (Argus, V4L2, GStreamer) |
| [Jetson Audio Setup and Development](Jetson-Audio-Setup-and-Development/Guide.md) | Jetson-specific ALSA, ASoC, AHUB, `APE`, device tree, pinmux, codec bring-up |
| [ESP32-LyraT I2S Microphone Capture](ESP32-LyraT-I2S-Mic-Jetson-Orin-Nano/Guide.md) | Focused Jetson 40-pin `I2S2` capture application using LyraT as an audio frontend |
| **GStreamer documentation** (gstreamer.freedesktop.org) | Pipeline syntax, plugin reference |
| **NVIDIA GStreamer plugins** | `nvarguscamerasrc`, `nvv4l2decoder`, `nvv4l2h264enc`, `nv3dsink` |
| **ALSA project** (alsa-project.org) | Advanced Linux Sound Architecture documentation |
| [Orin Nano Camera ISP](../../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Camera-ISP-Sensor-Bringup/Guide.md) | CSI sensor bring-up, ISP pipeline, multi-camera (Module 1 deep dive) |
| [Orin Nano Video Codec DeepStream](../../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Video-Codec-DeepStream/Guide.md) | Hardware codec details, DeepStream integration (Module 1 deep dive) |
