# GUI

<div class="course-identity auto-course" style="--course-accent: #16a34a; --course-accent-rgb: 22, 163, 74;" markdown="1">
<div class="course-identity__icon">GUI</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Jetson Track</p>
<p class="course-identity__title">Specialized course identity for GUI.</p>
<p class="course-identity__meta">Artifact: Jetson integration demo · Measure: latency, memory, power, logs</p>
</div>
</div>


**Phase 4 — Track B — Module 5.3** · Application Development

> **Focus:** Build graphical user interfaces for **Jetson Orin Nano 8GB** products — from lightweight embedded UIs (LVGL, framebuffer) through Qt desktop applications to web-based dashboards. Covers display setup, touch input, and GPU-accelerated rendering.

**Hub:** [5. Application Development](../Guide.md)

---


## 1. Display setup on Jetson

### Output interfaces

| Interface | Use case | Notes |
|-----------|----------|-------|
| **HDMI** | Development, desktop display | Up to 4K@60 on Orin Nano |
| **DisplayPort** | High-res monitors | Alt-mode via USB-C on some carriers |
| **eDP** | Embedded LCD panels | Requires carrier board support |
| **LVDS** (via bridge) | Industrial panels | Needs HDMI/DP-to-LVDS bridge IC |
| **DSI** | Small MIPI displays | Limited support on Orin Nano |

### Resolution and timing

```bash
# List connected displays and modes
xrandr

# Set resolution
xrandr --output HDMI-0 --mode 1920x1080 --rate 60

# For headless (virtual framebuffer)
export DISPLAY=:0
Xvfb :0 -screen 0 1920x1080x24 &
```

---

## 2. Framebuffer (Linux)

Direct framebuffer access for simple graphics without a display server.

```bash
# Check framebuffer device
ls /dev/fb*
cat /sys/class/graphics/fb0/virtual_size

# Write solid color to screen (red)
dd if=/dev/zero bs=4 count=$((1920*1080)) | \
    tr '\0' '\xff\x00\x00\xff' > /dev/fb0

# Display an image (using fbi)
sudo apt install fbi
sudo fbi -T 1 -d /dev/fb0 image.png
```

### When to use framebuffer

- **Splash screen** during boot (before X/Wayland starts)
- **Minimal status display** on small screens (no window manager needed)
- **Kiosk mode** with a single full-screen application

---

## 3. X11 and Wayland

JetPack ships with X11 (Xorg) by default. Wayland (via Weston) is available but less tested on Jetson.

```bash
# Check current display server
echo $XDG_SESSION_TYPE

# Start X11 manually (headless or remote)
startx

# For Weston (Wayland)
sudo apt install weston
weston-launch
```

### GPU acceleration

NVIDIA's L4T drivers provide GPU-accelerated OpenGL ES and EGL for both X11 and Wayland. Verify:

```bash
glxinfo | grep "OpenGL renderer"
# Should show "NVIDIA Tegra" or similar
```

---

## 4. Getting started with Qt

Qt is the most mature framework for embedded Linux GUIs with GPU acceleration.

### Install Qt on Jetson

```bash
# Qt 5 (from JetPack/Ubuntu repos)
sudo apt install qt5-default qtcreator

# Or Qt 6 (build from source or use Conan/aqt)
pip install aqtinstall
aqt install-qt linux desktop 6.6.0
```

### Minimal Qt application

```cpp
// main.cpp
#include <QApplication>
#include <QLabel>

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    QLabel label("Device Status: Running");
    label.setFont(QFont("Arial", 24));
    label.show();
    return app.exec();
}
```

```bash
# Build
qmake -project
qmake
make

# Run with EGLFS (no window manager, direct GPU)
./app -platform eglfs
```

### Qt platform plugins for embedded

| Plugin | Use case |
|--------|----------|
| **eglfs** | Full-screen, no window manager, direct GPU rendering (recommended for kiosk) |
| **linuxfb** | Framebuffer, no GPU acceleration |
| **wayland** | Wayland compositor |
| **xcb** | X11 window system |

---

## 5. Qt cross-compilation for Jetson

For faster build cycles, cross-compile Qt apps on an x86 host targeting ARM64.

### Toolchain setup

```bash
# Install cross-compiler
sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# Sysroot: copy target libraries from Jetson
rsync -avz jetson:/usr/lib/aarch64-linux-gnu/ sysroot/usr/lib/
rsync -avz jetson:/usr/include/ sysroot/usr/include/
```

### CMake toolchain file

```cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_SYSROOT /path/to/sysroot)
```

---

## 6. LVGL (Lightweight Graphics Library)

LVGL is ideal for small screens and resource-constrained displays — runs on framebuffer without a window manager.

### Setup on Jetson

```bash
git clone https://github.com/lvgl/lv_port_linux_frame_buffer.git
cd lv_port_linux_frame_buffer
mkdir build && cd build
cmake ..
make -j$(nproc)
./main   # renders to /dev/fb0
```

### When to use LVGL vs Qt

| Criteria | LVGL | Qt |
|----------|------|-----|
| **Screen size** | Small (< 7") | Any |
| **Complexity** | Simple dashboards, status screens | Complex multi-page apps |
| **Memory** | < 1 MB RAM for UI | 50+ MB |
| **GPU** | Not required (CPU rendering) | Benefits from GPU |
| **Touch** | Built-in driver support | Built-in |
| **Styling** | CSS-like themes | QSS, QML |

---

## 7. Web-based UI

Serve a web interface from the Jetson — accessible from any device with a browser.

### Architecture

```
Jetson (backend: Flask/FastAPI + frontend: React/Vue/plain HTML)
  │
  └─ Browser on phone/laptop → http://jetson-edge.local:8080 (example mDNS hostname)
```

### Advantages for embedded products

- **No display hardware needed** on the device itself
- **Cross-platform** — works from any browser
- **Easy to update** — ship new HTML/JS via OTA without reflashing
- Frameworks: Flask + HTMX (simple), FastAPI + React (full SPA)

---

## 8. Touch screen setup and calibration

### Capacitive touch (I2C)

Most modern embedded displays use capacitive touch connected via I2C. The kernel driver appears as an input device:

```bash
# List input devices
cat /proc/bus/input/devices

# Test touch events
sudo apt install evtest
sudo evtest /dev/input/eventN
```

### Calibration (if needed)

```bash
# Install xinput_calibrator (X11)
sudo apt install xinput-calibrator
xinput_calibrator

# For tslib (non-X11)
sudo apt install tslib
export TSLIB_TSDEVICE=/dev/input/eventN
ts_calibrate
ts_test
```

### Device tree for touch controller

Add the touch controller I2C device in your carrier's device tree:

```dts
&i2c1 {
    touch@38 {
        compatible = "edt,edt-ft5406";
        reg = <0x38>;
        interrupt-parent = <&gpio>;
        interrupts = <IRQ_PIN IRQ_TYPE_EDGE_FALLING>;
    };
};
```

---

## 9. Projects

- **Status kiosk:** Build a Qt EGLFS application that shows real-time GPU temperature, inference FPS, and network status on a 7" HDMI display.
- **LVGL dashboard:** Display sensor readings (I2C temperature + CAN data) on a small SPI/I2C display using LVGL on framebuffer.
- **Web config portal:** Build a Flask + HTMX web interface for your Jetson edge device that shows device status and allows Wi-Fi configuration from a phone browser.

---

## 10. Resources

| Resource | Description |
|----------|-------------|
| **Qt Documentation** (doc.qt.io) | Official Qt framework docs, EGLFS platform guide |
| **LVGL** (lvgl.io) | Lightweight graphics library, Linux framebuffer port |
| **Weston/Wayland** | Wayland compositor for embedded Linux |
| **xinput_calibrator** | Touch screen calibration tool for X11 |
| **tslib** | Touch screen library for non-X11 environments |
