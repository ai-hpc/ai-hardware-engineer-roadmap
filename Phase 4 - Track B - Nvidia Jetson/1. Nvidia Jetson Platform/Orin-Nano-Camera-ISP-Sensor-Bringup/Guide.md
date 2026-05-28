# Jetson Orin Nano 8GB -- Camera Subsystem, ISP Pipeline, and Sensor Bring-Up

<div class="course-identity auto-course" style="--course-accent: #db2777; --course-accent-rgb: 219, 39, 119;" markdown="1">
<div class="course-identity__icon">JON8</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Jetson Track</p>
<p class="course-identity__title">Specialized course identity for Jetson Orin Nano 8GB -- Camera Subsystem, ISP Pipeline, and Sensor Bring-Up.</p>
<p class="course-identity__meta">Artifact: Jetson integration demo · Measure: latency, memory, power, logs</p>
</div>
</div>


> Target: Jetson Orin Nano 8GB Developer Kit (P3767-0005, T234 SoC)
> JetPack: 6.x (L4T 36.x), Linux kernel 5.15
> Audience: Hardware engineers, BSP developers, and embedded vision engineers

---


## 1. Introduction

### 1.1 Camera Subsystem Importance in Edge AI

The camera subsystem is the primary sensor pathway for the majority of edge AI
workloads deployed on Jetson platforms. Object detection, semantic segmentation,
pose estimation, defect inspection, and autonomous navigation all begin with
pixel data captured through the camera pipeline. On the Orin Nano 8GB, every
frame traverses a purpose-built hardware path from the image sensor through
MIPI CSI-2 receivers, video input DMA engines, and a dedicated image signal
processor before reaching GPU or DLA compute units. Understanding this path at
the register and driver level is essential for anyone building production
vision systems.

Unlike commodity USB or IP cameras that present fully processed frames over a
generic transport, the Jetson camera subsystem operates on RAW sensor data and
performs image processing in dedicated silicon. This gives the system designer
control over every stage of image formation -- from black level subtraction and
lens shading correction through demosaicing, white balance, noise reduction, and
tone mapping. The tradeoff is complexity: bringing up a new sensor requires
coordinating device tree bindings, kernel-level V4L2 drivers, ISP tuning files,
and userspace capture APIs.

### 1.2 Overview of the NVIDIA Camera Pipeline

The T234 SoC in the Orin Nano implements the following camera pipeline:

```
 +----------+     +---------+     +------+     +-------+     +-------------+
 |  Image   | MIPI|  NVCSI  |     |  VI  |     |  ISP  |     |  Userspace  |
 |  Sensor  |---->| D-PHY   |---->| DMA  |---->| Proc  |---->|  libargus   |
 | (I2C cfg)|CSI-2| Rx/Deser|     |Engine|     |Engine |     |  / V4L2     |
 +----------+     +---------+     +------+     +-------+     +-------------+
       |                                                           |
       v                                                           v
   Sensor MCLK                                               GPU / DLA
   (24 MHz typ.)                                             (inference)
```

Key points:

- The sensor is configured over I2C and streams pixels over MIPI CSI-2.
- NVCSI handles the physical layer (D-PHY or C-PHY), lane synchronization, and
  packet parsing.
- VI (Video Input) performs DMA of pixel data from the CSI receiver into system
  DRAM, adding timestamps and sequence numbers.
- ISP (Image Signal Processor) processes RAW Bayer data into consumer formats
  (NV12, ARGB) through a multi-stage pipeline.
- Userspace applications access frames through either the libargus API (for
  ISP-processed output) or direct V4L2 (for RAW capture).

### 1.3 What This Guide Covers

This guide provides a complete walkthrough of every component in the path from
photon to tensor. It is organized so that each section builds on the previous:
hardware architecture first, then the software layers from kernel drivers through
userspace APIs, and finally production concerns like performance, reliability,
and debugging.

All code examples, device tree snippets, and command-line invocations are tested
against the Jetson Orin Nano 8GB with JetPack 6.x. Register addresses, clock
names, and driver paths reference the T234 BSP specifically. Other Orin-family
modules (Orin NX, AGX Orin) share most of the architecture but differ in lane
counts, ISP instances, and clock domains.

---

## 2. Camera Hardware Architecture

### 2.1 NVCSI -- CSI Receiver

The NVCSI block is the first on-chip component that touches sensor data. It
implements the MIPI D-PHY v2.1 physical layer and the CSI-2 protocol layer.

On the Orin Nano, NVCSI provides:

- **2 CSI bricks** (Brick A and Brick B)
- **Each brick contains 2 ports**, each port supporting up to 2 D-PHY data lanes
- **Ports within a brick can be combined** for x4 lane operation
- **Per-lane data rate**: up to 2.5 Gbps (D-PHY v2.1)

```
              Brick A                          Brick B
 +---------------------------+    +---------------------------+
 | Port 0      | Port 1      |    | Port 2      | Port 3      |
 | Lane 0 (D+) | Lane 2 (D+) |    | Lane 0 (D+) | Lane 2 (D+) |
 | Lane 1 (D+) | Lane 3 (D+) |    | Lane 1 (D+) | Lane 3 (D+) |
 | CLK 0       | CLK 1       |    | CLK 0       | CLK 1       |
 +---------------------------+    +---------------------------+
       |               |                |               |
       v               v                v               v
    serial_a        serial_b         serial_c        serial_d
   (tegra_sinterface names in device tree)
```

NVCSI handles:

| Function                  | Description                                         |
|---------------------------|-----------------------------------------------------|
| D-PHY calibration         | Automatic LP-HS transition calibration              |
| Lane synchronization      | Byte and word alignment across lanes                |
| Packet parsing            | Extracts data type, word count, virtual channel     |
| ECC correction            | Single-bit correction, double-bit detection         |
| CRC checking              | Per-packet payload integrity                        |
| Virtual channel demux     | Routes up to 4 VCs per port to separate VI channels |

### 2.2 VI -- Video Input

The VI block receives parsed pixel data from NVCSI and DMAs it into DRAM as
complete frames. Key responsibilities:

- **Frame assembly**: Accumulates lines from NVCSI into 2D frames
- **Timestamping**: Attaches SOF (start-of-frame) timestamps from the TSC
  (Time Stamp Counter), providing nanosecond-precision capture times
- **Sequence numbering**: Monotonically incrementing frame counter
- **Buffer management**: Works with V4L2 buffer queues (MMAP, DMABUF)
- **Crop and padding**: Hardware-level line/pixel crop

The VI exposes V4L2 video capture device nodes (`/dev/videoN`) and creates the
bridge between kernel-space camera drivers and userspace applications.

### 2.3 ISP -- Image Signal Processor

The T234 ISP is a multi-stage hardware image processing engine. On the Orin
Nano, there is a single ISP instance shared across all active camera streams.
The ISP processes RAW Bayer data through approximately 15 stages (detailed in
Section 7) and outputs processed frames in NV12 or ARGB format.

The ISP is managed exclusively by `nvargus-daemon` -- a privileged userspace
process that mediates access between applications and the ISP hardware. Direct
ISP access from kernel space or arbitrary userspace processes is not supported.

### 2.4 Block Interconnection

The complete data flow with clock domains:

```
 Sensor            NVCSI              VI                ISP            Output
 +------+  MIPI   +-------+  AXI    +-----+  AXI     +-----+  AXI   +------+
 |      |-------->|       |-------->|     |--------->|     |-------->| DRAM |
 | CMOS |  CSI-2  | D-PHY |  Bus   | DMA |  Bus    | Proc|  Bus   | (NV12)|
 |      |  Lanes  | Deser |        |     |         |     |        |      |
 +------+         +-------+        +-----+         +-----+        +------+
   |                 |                |                |
   MCLK            nvcsi_clk        vi_clk          isp_clk
   24 MHz           ~409 MHz        ~729 MHz        ~729 MHz
   (ext. osc)      (BPMP managed)  (BPMP managed)  (BPMP managed)
```

Clock verification:

```bash
# List all camera-related clocks and their rates
sudo cat /sys/kernel/debug/bpmp/debug/clk/nvcsi/rate
sudo cat /sys/kernel/debug/bpmp/debug/clk/vi/rate
sudo cat /sys/kernel/debug/bpmp/debug/clk/isp/rate

# Alternative via tegrastats (shows ISP/VI utilization)
sudo tegrastats --interval 500
```

### 2.5 Memory Architecture Implications

The Orin Nano uses unified memory (LPDDR5) shared between CPU, GPU, DLA, and
all camera/video engines. This has important implications:

- VI DMA writes compete with GPU and CPU memory traffic
- DMABUF file descriptors allow zero-copy sharing of frame buffers between VI,
  ISP, GPU, and DLA without physical memory copies
- IOMMU (SMMU) provides address translation and isolation for camera DMA engines
- The default VI buffer allocation uses IOVA addresses mapped through the SMMU

```
 +------+   +-----+   +-----+   +------+
 | VI   |   | ISP |   | GPU |   | DLA  |
 +--+---+   +--+--+   +--+--+   +--+---+
    |          |          |         |
    v          v          v         v
 +--+----------+----------+---------+--+
 |           SMMU (IOMMU)              |
 +--+----------+----------+---------+--+
    |          |          |         |
    +----------+----------+---------+
               |
    +----------+----------+
    |    LPDDR5 (shared)   |
    |    8 GB unified      |
    +-----------------------+
```

---

## 3. Supported Camera Interfaces

### 3.1 MIPI CSI-2 Protocol

The Orin Nano supports MIPI CSI-2 with D-PHY v2.1. Key protocol characteristics:

| Parameter              | Specification                                  |
|------------------------|------------------------------------------------|
| Physical layer         | D-PHY v2.1                                     |
| Max lanes per port     | 2 (combinable to 4 per brick)                  |
| Max data rate per lane | 2.5 Gbps                                       |
| Max aggregate (x4)     | 10 Gbps per brick, 20 Gbps total               |
| Clock mode             | Continuous or non-continuous                    |
| Voltage                | MIPI D-PHY levels (LP: 1.2V, HS: ~200mV diff)  |
| Lane polarity swap     | Supported in device tree                        |

### 3.2 Virtual Channels

Each CSI-2 link supports up to 4 virtual channels (VC0-VC3). Virtual channels
allow multiple logical streams over a single physical connection, commonly used
with GMSL/FPDLink deserializers:

```
 Camera 0 (VC0) --\
 Camera 1 (VC1) ---+--> Serializer --> Cable --> Deserializer --> NVCSI Port
 Camera 2 (VC2) --/                                                 |
                                                           +--------+--------+
                                                           |        |        |
                                                        VI ch0   VI ch1   VI ch2
```

Virtual channel assignment in device tree:

```dts
/* Inside the NVCSI channel node */
channel@0 {
    reg = <0>;
    /* VC is inferred from channel index, or explicitly set */
    ports {
        port@0 {
            reg = <0>;
            endpoint {
                vc-id = <0>;    /* virtual channel 0 */
                port-index = <0>;
                bus-width = <2>;
            };
        };
    };
};
```

### 3.3 Supported Data Types

| CSI-2 DT | Format     | Bits/Pixel | Packed Format | Typical Sensor          |
|----------|------------|------------|---------------|-------------------------|
| 0x2A     | RAW8       | 8          | 1 byte/px     | OV5640 (RAW mode)       |
| 0x2B     | RAW10      | 10         | 5 bytes/4px   | IMX219, IMX477          |
| 0x2C     | RAW12      | 12         | 3 bytes/2px   | IMX477, IMX708          |
| 0x2D     | RAW14      | 14         | 7 bytes/4px   | Scientific sensors      |
| 0x1E     | YUV422-8   | 16         | 2 bytes/px    | OV5640 (YUV mode)       |
| 0x1F     | YUV422-10  | 20         | 5 bytes/2px   | Rare                    |
| 0x24     | RGB888     | 24         | 3 bytes/px    | Pre-processed sensors   |
| 0x22     | RGB565     | 16         | 2 bytes/px    | Low-cost displays       |

### 3.4 Bandwidth Calculations

To determine whether a given sensor configuration fits within the available CSI
bandwidth:

```
Required bandwidth (Gbps) = Width * Height * FPS * BitsPerPixel / (1e9)
Available bandwidth       = NumLanes * LaneRate * EncodingEfficiency
```

D-PHY encoding efficiency is approximately 80% (8b/10b-like overhead plus
protocol headers).

Example calculations:

| Sensor Config              | Calculation                      | Required  | Lanes  |
|----------------------------|----------------------------------|-----------|--------|
| IMX219 3280x2464 @ 21fps  | 3280*2464*21*10 / 1e9            | 1.70 Gbps | x2 OK  |
| IMX477 4032x3040 @ 30fps  | 4032*3040*30*12 / 1e9            | 4.41 Gbps | x4 req |
| IMX708 4608x2592 @ 14fps  | 4608*2592*14*10 / 1e9            | 1.67 Gbps | x2 OK  |
| IMX477 1920x1080 @ 120fps | 1920*1080*120*12 / 1e9           | 2.99 Gbps | x4 req |

### 3.5 Physical Connector Pinout

The Orin Nano Developer Kit exposes CSI through a 22-pin FFC connector (J5)
compatible with Raspberry Pi Camera Module ribbon cables:

```
Pin  Signal          Pin  Signal
---  ------          ---  ------
 1   GND              2   CSI0_D0_N
 3   CSI0_D0_P        4   GND
 5   CSI0_D1_N        6   CSI0_D1_P
 7   GND              8   CSI0_CLK_N
 9   CSI0_CLK_P      10   GND
11   GND             12   CSI1_D0_N
13   CSI1_D0_P       14   GND
15   CSI1_D1_N       16   CSI1_D1_P
17   GND             18   CSI1_CLK_N
19   CSI1_CLK_P      20   GND
21   CAM0_PWDN       22   CAM1_PWDN
```

---

## 4. Sensor Bringup Overview

### 4.1 End-to-End Bringup Steps

Bringing up a new image sensor on the Orin Nano involves the following ordered
steps. Each step depends on the successful completion of the previous one:

```
 Step 1: Hardware verification (schematic, I2C probe, power rails)
    |
 Step 2: Device tree binding (sensor node, NVCSI port, VI channel)
    |
 Step 3: Kernel sensor driver (V4L2 subdev, I2C register sequences)
    |
 Step 4: RAW capture validation (v4l2-ctl RAW frame capture)
    |
 Step 5: ISP tuning file creation (camera_overrides.isp)
    |
 Step 6: libargus / GStreamer integration (ISP-processed output)
    |
 Step 7: Application integration (inference pipeline, recording)
```

### 4.2 Step 1 -- Hardware Verification

Before any software work, verify the electrical connection:

```bash
# Verify I2C bus is visible
sudo i2cdetect -l
# Output should include the bus your sensor is on (e.g., i2c-30)

# Scan for sensor I2C address
sudo i2cdetect -y -r 30
#      0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
# 00:          -- -- -- -- -- -- -- -- -- -- -- -- -- --
# 10: 10 -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#     ^^ IMX219 detected at 0x10

# Read sensor chip ID register (IMX219: reg 0x0000-0x0001 = 0x0219)
sudo i2ctransfer -y 30 w2@0x10 0x00 0x00 r2
# Output: 0x02 0x19  --> chip ID confirmed
```

Verify power rails with a multimeter or by reading the regulator status:

```bash
# Check regulator status (names depend on carrier board design)
cat /sys/class/regulator/regulator.*/name
cat /sys/class/regulator/regulator.*/microvolts
```

Key power rails for typical camera modules:

| Rail    | Typical Voltage | Purpose                        |
|---------|-----------------|--------------------------------|
| AVDD    | 2.8V            | Analog supply (pixel array)    |
| DVDD    | 1.05-1.2V       | Digital core                   |
| IOVDD   | 1.8V            | I/O and MIPI PHY               |

### 4.3 Step 2 -- Device Tree Binding

Create a device tree overlay that defines three interconnected nodes:

1. **Sensor I2C node** -- under the I2C bus controller
2. **NVCSI channel** -- under `host1x/nvcsi`, linking sensor to VI
3. **tegra-camera-platform** -- module registration and bandwidth hints

A minimal device tree structure (detailed in Section 5):

```dts
/* Sensor on I2C bus 30, address 0x10, CSI port 0, 2 lanes */
&cam_i2c {
    sensor@10 {
        compatible = "sony,imx219";
        reg = <0x10>;
        /* ... clock, regulator, mode properties ... */
        port {
            sensor_out: endpoint {
                port-index = <0>;
                bus-width = <2>;
                remote-endpoint = <&csi_in0>;
            };
        };
    };
};
```

### 4.4 Step 3 -- Sensor Driver

If a driver already exists in the NVIDIA kernel tree
(`kernel/nvidia/drivers/media/i2c/`), enable it in the kernel config. Otherwise,
write a new V4L2 subdev driver (detailed in Section 6). The driver must:

- Probe via I2C and verify the chip ID
- Program mode-specific register tables on `s_stream(1)`
- Implement format enumeration and negotiation
- Expose V4L2 controls for gain, exposure, and frame rate

### 4.5 Step 4 -- RAW Capture Validation

Once the driver loads and `/dev/videoN` appears:

```bash
# Verify the device node exists
v4l2-ctl --list-devices

# List supported formats
v4l2-ctl -d /dev/video0 --list-formats-ext

# Capture 10 RAW frames
v4l2-ctl -d /dev/video0 \
    --set-fmt-video=width=3280,height=2464,pixelformat=RG10 \
    --stream-mmap --stream-count=10 --stream-to=raw_capture.raw

# Quick check: file size should be width * height * 2 * 10 frames
ls -la raw_capture.raw
# Expected: 3280 * 2464 * 2 * 10 = 161,587,200 bytes
```

View the RAW file with a Bayer viewer or convert with `ffmpeg`:

```bash
ffmpeg -f rawvideo -pix_fmt bayer_rggb16le \
    -s 3280x2464 -i raw_capture.raw \
    -vframes 1 -pix_fmt rgb24 frame0.png
```

### 4.6 Step 5 -- ISP Tuning

For ISP-processed output, a tuning file matching the sensor must be present at
`/var/nvidia/nvcam/settings/camera_overrides.isp`. NVIDIA provides default
tuning files for supported sensors (IMX219, IMX477, IMX708). Custom sensors
require the NVIDIA ISP Tuning Tool (detailed in Section 8).

### 4.7 Step 6 -- Libargus Validation

With the ISP tuning file in place, validate full-pipeline capture:

```bash
# Restart nvargus-daemon to pick up new tuning
sudo systemctl restart nvargus-daemon

# Test with argus_camera sample app
cd /usr/src/jetson_multimedia_api/argus/build
./samples/oneShot/argus_oneshot --device 0

# Or via GStreamer
gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=30 ! \
    'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! \
    nvjpegenc ! multifilesink location="frame_%03d.jpg"
```

### 4.8 Bringup Checklist

| Step | Verification Command                          | Expected Result               |
|------|-----------------------------------------------|-------------------------------|
| 1    | `sudo i2cdetect -y -r <bus>`                  | Sensor address visible        |
| 2    | `dtc -I fs /proc/device-tree \| grep <compat>`| Sensor node in live DT        |
| 3    | `lsmod \| grep <driver>`                      | Driver module loaded          |
| 3    | `dmesg \| grep <sensor_name>`                 | "probed successfully"         |
| 4    | `v4l2-ctl --list-devices`                     | /dev/videoN present           |
| 4    | `v4l2-ctl --stream-mmap --stream-count=1`     | Frame captured, no errors     |
| 5    | `ls /var/nvidia/nvcam/settings/camera_*`      | ISP file present              |
| 6    | `gst-launch-1.0 nvarguscamerasrc ...`         | Live preview visible          |

---

## 5. Device Tree Configuration

### 5.1 Overview of Camera Device Tree Nodes

The Jetson camera subsystem requires three coordinated device tree node groups.
These nodes establish the hardware topology that the kernel camera stack uses to
discover sensors, configure CSI receivers, and route pixel data:

1. **Sensor I2C node** -- Declares the sensor on its I2C bus with clock,
   regulator, GPIO, and mode properties.
2. **NVCSI/VI graph bindings** -- Defines the media graph using OF (Open
   Firmware) graph port/endpoint pairs that connect the sensor output to the
   NVCSI input and the NVCSI output to the VI input.
3. **tegra-camera-platform node** -- Registers modules with the camera platform
   driver for bandwidth management and ISP assignment.

### 5.2 Sensor Node Properties

The sensor node lives under the I2C controller that the sensor is physically
wired to. On the Orin Nano Developer Kit, the camera connector I2C bus is
typically `cam_i2c` (I2C bus 30, controller at `0x3180000`).

```dts
&cam_i2c {
    status = "okay";

    imx219_cam0: imx219@10 {
        compatible = "sony,imx219";
        reg = <0x10>;

        /* External clock (EXTPERIPH1 routed to CAM_MCLK) */
        clocks = <&bpmp_clks TEGRA234_CLK_EXTPERIPH1>;
        clock-names = "extperiph1";
        mclk = "extperiph1";
        clock-frequency = <24000000>;

        /* Regulator supplies */
        avdd-supply = <&cam0_avdd_2v8>;    /* 2.8V analog */
        iovdd-supply = <&cam0_iovdd_1v8>;  /* 1.8V I/O    */
        dvdd-supply = <&cam0_dvdd_1v2>;    /* 1.2V core   */

        /* Power/reset GPIOs */
        reset-gpios = <&gpio CAM0_RST_L GPIO_ACTIVE_LOW>;
        pwdn-gpios  = <&gpio CAM0_PWDN GPIO_ACTIVE_HIGH>;

        /* Physical dimensions for ISP lens shading */
        physical_w = "3.680";   /* mm, active area width  */
        physical_h = "2.760";   /* mm, active area height */

        /* Sensor mode definitions */
        mode0 {
            mclk_khz = "24000";
            num_lanes = "2";
            tegra_sinterface = "serial_a";
            phy_mode = "DPHY";
            discontinuous_clk = "yes";
            dpcm_enable = "false";
            cil_settletime = "0";       /* 0 = auto-calculate */

            active_w = "3280";
            active_h = "2464";
            mode_type = "bayer";
            pixel_phase = "rggb";
            csi_pixel_bit_depth = "10";
            readout_orientation = "0";

            line_length = "3448";
            inherent_gain = "1";
            mclk_multiplier = "25";     /* pixel_clk / mclk */
            pix_clk_hz = "182400000";

            gain_factor = "16";
            framerate_factor = "1000000";
            exposure_factor = "1000000";
            min_gain_val = "16";        /* 1x in Q4 */
            max_gain_val = "170";       /* 10.66x   */
            step_gain_val = "1";
            default_gain = "16";
            min_exp_time = "13";        /* us */
            max_exp_time = "683709";    /* us */
            step_exp_time = "1";
            default_exp_time = "2495";
            min_framerate = "2000000";  /* 2 fps * 1e6 */
            max_framerate = "21000000"; /* 21 fps * 1e6 */
            step_framerate = "1";
            default_framerate = "21000000";

            embedded_metadata_height = "2";
        };

        mode1 {
            /* 1920x1080 @ 30fps cropped mode */
            mclk_khz = "24000";
            num_lanes = "2";
            tegra_sinterface = "serial_a";
            phy_mode = "DPHY";
            discontinuous_clk = "yes";
            dpcm_enable = "false";
            cil_settletime = "0";

            active_w = "1920";
            active_h = "1080";
            mode_type = "bayer";
            pixel_phase = "rggb";
            csi_pixel_bit_depth = "10";
            readout_orientation = "0";

            line_length = "3448";
            inherent_gain = "1";
            mclk_multiplier = "25";
            pix_clk_hz = "182400000";

            gain_factor = "16";
            framerate_factor = "1000000";
            exposure_factor = "1000000";
            min_gain_val = "16";
            max_gain_val = "170";
            step_gain_val = "1";
            default_gain = "16";
            min_exp_time = "13";
            max_exp_time = "683709";
            step_exp_time = "1";
            default_exp_time = "2495";
            min_framerate = "2000000";
            max_framerate = "30000000";
            step_framerate = "1";
            default_framerate = "30000000";

            embedded_metadata_height = "2";
        };

        /* OF graph endpoint connecting sensor to NVCSI */
        ports {
            #address-cells = <1>;
            #size-cells = <0>;
            port@0 {
                reg = <0>;
                imx219_out0: endpoint {
                    port-index = <0>;
                    bus-width = <2>;
                    remote-endpoint = <&csi_in0>;
                };
            };
        };
    };
};
```

### 5.3 CSI Endpoint Configuration

The NVCSI channel node defines how the CSI receiver connects to both the sensor
(input side) and the VI (output side):

```dts
&host1x {
    nvcsi@15a00000 {
        status = "okay";
        num-channels = <1>;

        channel@0 {
            reg = <0>;
            ports {
                #address-cells = <1>;
                #size-cells = <0>;

                /* Input port: from sensor */
                port@0 {
                    reg = <0>;
                    csi_in0: endpoint@0 {
                        port-index = <0>;    /* CSI port A */
                        bus-width = <2>;     /* 2 data lanes */
                        remote-endpoint = <&imx219_out0>;
                    };
                };

                /* Output port: to VI */
                port@1 {
                    reg = <1>;
                    csi_out0: endpoint@1 {
                        remote-endpoint = <&vi_in0>;
                    };
                };
            };
        };
    };

    vi@15c10000 {
        status = "okay";
        num-channels = <1>;

        ports {
            #address-cells = <1>;
            #size-cells = <0>;

            port@0 {
                reg = <0>;
                vi_in0: endpoint {
                    port-index = <0>;
                    bus-width = <2>;
                    remote-endpoint = <&csi_out0>;
                };
            };
        };
    };
};
```

### 5.4 Lane Mapping and Polarity

If the PCB layout requires lane swapping or polarity inversion:

```dts
endpoint {
    port-index = <0>;
    bus-width = <4>;
    /* Lane remapping: physical lane N carries logical lane M */
    lane-swizzle = <2 3 0 1>;   /* swap lane pairs */
    /* Polarity inversion for specific lanes */
    lane-polarity = <0 1 0 0>;  /* invert lane 1 polarity */
};
```

### 5.5 Pixel Clock and Timing

The `pix_clk_hz` property must match the sensor's actual pixel clock for the
given mode. This value is used by the VI to validate incoming frame timing and
by the camera platform driver for bandwidth allocation.

Calculate from the sensor datasheet:

```
pixel_clock = line_length * (frame_height + VBlanking) * frame_rate

Example (IMX219 mode0):
  line_length   = 3448 pixels
  frame_height  = 2464 + 112 (VBlank) = 2576 lines
  frame_rate    = 21 fps
  pixel_clock   = 3448 * 2576 * 21 = 186,474,048 ~ 182,400,000 (datasheet)
```

### 5.6 tegra-camera-platform Node

```dts
tegra-camera-platform {
    compatible = "nvidia,tegra-camera-platform";
    num_csi_lanes = <4>;           /* total available lanes */
    max_lane_speed = <2500000>;    /* kHz, per lane */
    min_bits_per_pixel = <10>;
    vi_peak_byte_per_pixel = <2>;
    vi_bw_margin_pct = <25>;       /* 25% bandwidth headroom */
    isp_peak_byte_per_pixel = <5>;
    isp_bw_margin_pct = <25>;

    modules {
        module0 {
            badge = "imx219_rear";
            position = "rear";
            orientation = "1";

            drivernode0 {
                pcl_id = "v4l2_sensor";
                devname = "imx219 30-0010";
                proc-device-tree = "/proc/device-tree/cam_i2c/imx219@10";
            };
        };
    };
};
```

### 5.7 Applying Device Tree Overlays

```bash
# Compile the overlay
dtc -I dts -O dtb -@ -o imx219-overlay.dtbo imx219-overlay.dts

# Deploy to boot partition
sudo cp imx219-overlay.dtbo /boot/

# Edit extlinux.conf to apply overlay
sudo nano /boot/extlinux/extlinux.conf
# Add under the LABEL entry:
#     FDTOVERLAYS /boot/imx219-overlay.dtbo

# Reboot and verify
sudo reboot
# After reboot:
dtc -I fs /proc/device-tree 2>/dev/null | grep imx219
```

---

## 6. Sensor Driver Development

### 6.1 Driver Architecture

A Jetson camera sensor driver is a standard Linux I2C client driver that
registers a V4L2 subdevice. The driver must implement:

- **I2C probe**: Verify chip ID, initialize power supplies and clocks
- **V4L2 subdev video ops**: `s_stream` to start/stop the sensor
- **V4L2 subdev pad ops**: Format enumeration and negotiation
- **V4L2 controls**: Exposure, gain, frame rate, test pattern

```
 +---------------------------+
 | V4L2 Subdev Interface     |
 |   .video_ops (s_stream)   |
 |   .pad_ops (get/set_fmt)  |
 |   .ctrl_handler (AE/gain) |
 +---------------------------+
 | I2C Register Interface    |
 |   sensor_write_reg()      |
 |   sensor_read_reg()       |
 +---------------------------+
 | I2C Client Driver         |
 |   .probe / .remove        |
 +---------------------------+
```

### 6.2 I2C Register Programming

Most image sensors use 16-bit register addresses with 8-bit data values.
Some (like the IMX477) use 16-bit addresses with 8-bit or 16-bit data depending
on the register.

```c
/* 16-bit address, 8-bit data write */
static int sensor_write_reg(struct i2c_client *client, u16 addr, u8 val)
{
    u8 buf[3] = { addr >> 8, addr & 0xFF, val };
    struct i2c_msg msg = {
        .addr  = client->addr,
        .flags = 0,
        .len   = 3,
        .buf   = buf,
    };
    int ret = i2c_transfer(client->adapter, &msg, 1);
    if (ret != 1) {
        dev_err(&client->dev, "I2C write failed: addr=0x%04x val=0x%02x ret=%d\n",
                addr, val, ret);
        return ret < 0 ? ret : -EIO;
    }
    return 0;
}

/* 16-bit address, 8-bit data read */
static int sensor_read_reg(struct i2c_client *client, u16 addr, u8 *val)
{
    u8 addr_buf[2] = { addr >> 8, addr & 0xFF };
    struct i2c_msg msgs[2] = {
        { .addr = client->addr, .flags = 0,            .len = 2, .buf = addr_buf },
        { .addr = client->addr, .flags = I2C_M_RD,     .len = 1, .buf = val      },
    };
    int ret = i2c_transfer(client->adapter, msgs, 2);
    if (ret != 2)
        return ret < 0 ? ret : -EIO;
    return 0;
}

/* Write a table of register values (terminated by {0xFFFF, 0xFF}) */
static int sensor_write_table(struct i2c_client *client,
                              const struct reg_pair *table)
{
    int ret;
    for (; table->addr != 0xFFFF; table++) {
        if (table->addr == 0xFFFE) {
            usleep_range(table->val * 1000, table->val * 1000 + 500);
            continue;
        }
        ret = sensor_write_reg(client, table->addr, table->val);
        if (ret)
            return ret;
    }
    return 0;
}
```

### 6.3 V4L2 Subdev Operations

```c
static int sensor_s_stream(struct v4l2_subdev *sd, int enable)
{
    struct sensor_priv *priv = to_sensor_priv(sd);
    int ret;

    if (enable) {
        /* Power on sequence: AVDD -> DVDD -> IOVDD -> MCLK -> reset */
        ret = sensor_power_on(priv);
        if (ret)
            return ret;

        /* Write mode register table */
        ret = sensor_write_table(priv->client,
                                 priv->modes[priv->current_mode].reg_list);
        if (ret) {
            sensor_power_off(priv);
            return ret;
        }

        /* Apply current control values */
        ret = __v4l2_ctrl_handler_setup(&priv->ctrl_handler);
        if (ret) {
            sensor_power_off(priv);
            return ret;
        }

        /* Start streaming: set MIPI output enable */
        ret = sensor_write_reg(priv->client, REG_MODE_SELECT, 0x01);
    } else {
        /* Stop streaming */
        ret = sensor_write_reg(priv->client, REG_MODE_SELECT, 0x00);
        sensor_power_off(priv);
    }
    return ret;
}

static int sensor_enum_mbus_code(struct v4l2_subdev *sd,
                                  struct v4l2_subdev_state *state,
                                  struct v4l2_subdev_mbus_code_enum *code)
{
    if (code->index > 0)
        return -EINVAL;
    code->code = MEDIA_BUS_FMT_SRGGB10_1X10;
    return 0;
}

static int sensor_get_fmt(struct v4l2_subdev *sd,
                           struct v4l2_subdev_state *state,
                           struct v4l2_subdev_format *fmt)
{
    struct sensor_priv *priv = to_sensor_priv(sd);
    const struct sensor_mode *mode = &priv->modes[priv->current_mode];

    fmt->format.width      = mode->width;
    fmt->format.height     = mode->height;
    fmt->format.code       = mode->mbus_code;
    fmt->format.field      = V4L2_FIELD_NONE;
    fmt->format.colorspace = V4L2_COLORSPACE_RAW;
    return 0;
}

static int sensor_set_fmt(struct v4l2_subdev *sd,
                           struct v4l2_subdev_state *state,
                           struct v4l2_subdev_format *fmt)
{
    struct sensor_priv *priv = to_sensor_priv(sd);
    int i;

    /* Find closest matching mode */
    for (i = 0; i < priv->num_modes; i++) {
        if (priv->modes[i].width == fmt->format.width &&
            priv->modes[i].height == fmt->format.height) {
            priv->current_mode = i;
            break;
        }
    }

    return sensor_get_fmt(sd, state, fmt);
}

static const struct v4l2_subdev_video_ops sensor_video_ops = {
    .s_stream = sensor_s_stream,
};

static const struct v4l2_subdev_pad_ops sensor_pad_ops = {
    .enum_mbus_code  = sensor_enum_mbus_code,
    .get_fmt         = sensor_get_fmt,
    .set_fmt         = sensor_set_fmt,
    .enum_frame_size = sensor_enum_frame_size,
};

static const struct v4l2_subdev_ops sensor_subdev_ops = {
    .video = &sensor_video_ops,
    .pad   = &sensor_pad_ops,
};
```

### 6.4 Mode Tables

Mode tables define the register sequences for each supported resolution/fps
combination:

```c
struct reg_pair {
    u16 addr;
    u8  val;
};

/* IMX219: 3280x2464 @ 21fps, 2-lane, RAW10 */
static const struct reg_pair mode_3280x2464_regs[] = {
    /* PLL settings for 24 MHz input, 182.4 MHz pixel clock */
    { 0x0301, 0x05 },  /* VT_PIX_CLK_DIV */
    { 0x0303, 0x01 },  /* VT_SYS_CLK_DIV */
    { 0x0304, 0x03 },  /* PRE_PLL_CLK_VT_DIV */
    { 0x0305, 0x03 },  /* PRE_PLL_CLK_OP_DIV */
    { 0x0306, 0x00 },  /* PLL_VT_MPY [10:8] */
    { 0x0307, 0x39 },  /* PLL_VT_MPY [7:0] = 57 */

    /* Frame geometry */
    { 0x0340, 0x09 },  /* FRM_LENGTH_A [15:8] = 2576 */
    { 0x0341, 0xD0 },  /* FRM_LENGTH_A [7:0]  */
    { 0x0342, 0x0D },  /* LINE_LENGTH_A [15:8] = 3448 */
    { 0x0343, 0x78 },  /* LINE_LENGTH_A [7:0]  */

    /* Active area */
    { 0x0344, 0x00 },  /* X_ADDR_START [11:8] */
    { 0x0345, 0x00 },  /* X_ADDR_START [7:0]  */
    { 0x0346, 0x00 },  /* Y_ADDR_START [11:8] */
    { 0x0347, 0x00 },  /* Y_ADDR_START [7:0]  */
    { 0x0348, 0x0C },  /* X_ADDR_END [11:8] = 3279 */
    { 0x0349, 0xCF },
    { 0x034A, 0x09 },  /* Y_ADDR_END [11:8] = 2463 */
    { 0x034B, 0x9F },

    /* Output size */
    { 0x034C, 0x0C },  /* X_OUTPUT_SIZE = 3280 */
    { 0x034D, 0xD0 },
    { 0x034E, 0x09 },  /* Y_OUTPUT_SIZE = 2464 */
    { 0x034F, 0xA0 },

    /* MIPI output: 2-lane, RAW10 */
    { 0x0114, 0x01 },  /* CSI_LANE_MODE = 2 lanes */
    { 0x0128, 0x00 },  /* DPHY_CTRL = auto */
    { 0x012A, 0x18 },  /* EXCK_FREQ [15:8] = 24 MHz */
    { 0x012B, 0x00 },

    { 0xFFFF, 0xFF },  /* End of table sentinel */
};

struct sensor_mode {
    u32 width;
    u32 height;
    u32 max_fps;
    u64 pixel_clk;
    u32 line_length;
    u32 mbus_code;
    const struct reg_pair *reg_list;
};

static const struct sensor_mode imx219_modes[] = {
    {
        .width      = 3280,
        .height     = 2464,
        .max_fps    = 21,
        .pixel_clk  = 182400000,
        .line_length = 3448,
        .mbus_code  = MEDIA_BUS_FMT_SRGGB10_1X10,
        .reg_list   = mode_3280x2464_regs,
    },
    {
        .width      = 1920,
        .height     = 1080,
        .max_fps    = 30,
        .pixel_clk  = 182400000,
        .line_length = 3448,
        .mbus_code  = MEDIA_BUS_FMT_SRGGB10_1X10,
        .reg_list   = mode_1920x1080_regs,
    },
};
```

### 6.5 Gain and Exposure Controls

```c
static int sensor_set_gain(struct sensor_priv *priv, u32 gain_val)
{
    /*
     * IMX219 analog gain register (0x0157):
     *   gain = 256 / (256 - reg_val)
     *   reg_val = 256 - (256 / gain)
     *
     * gain_val is in Q4 fixed point (16 = 1.0x, 170 = 10.625x)
     */
    u32 gain_linear = gain_val;  /* already in sensor units */
    u8 reg_val = (u8)(256 - (256 * 16 / gain_linear));

    return sensor_write_reg(priv->client, 0x0157, reg_val);
}

static int sensor_set_exposure(struct sensor_priv *priv, u32 exp_us)
{
    /*
     * Exposure = coarse_time * line_length / pixel_clock
     * coarse_time = exp_us * pixel_clock / (line_length * 1e6)
     */
    u32 coarse = (u32)((u64)exp_us * priv->pixel_clk /
                       ((u64)priv->line_length * 1000000ULL));

    /* Clamp to valid range: 1 to frame_length - 4 */
    coarse = clamp_t(u32, coarse, 1, priv->frame_length - 4);

    sensor_write_reg(priv->client, 0x015A, (coarse >> 8) & 0xFF);
    return sensor_write_reg(priv->client, 0x015B, coarse & 0xFF);
}

/* V4L2 control handler */
static int sensor_s_ctrl(struct v4l2_ctrl *ctrl)
{
    struct sensor_priv *priv = container_of(ctrl->handler,
                                            struct sensor_priv, ctrl_handler);
    switch (ctrl->id) {
    case V4L2_CID_GAIN:
        return sensor_set_gain(priv, ctrl->val);
    case V4L2_CID_EXPOSURE:
        return sensor_set_exposure(priv, ctrl->val);
    case V4L2_CID_VFLIP:
        return sensor_write_reg(priv->client, 0x0172, ctrl->val ? 0x02 : 0x00);
    case V4L2_CID_HFLIP:
        return sensor_write_reg(priv->client, 0x0172, ctrl->val ? 0x01 : 0x00);
    default:
        return -EINVAL;
    }
}

static const struct v4l2_ctrl_ops sensor_ctrl_ops = {
    .s_ctrl = sensor_s_ctrl,
};

static void sensor_init_controls(struct sensor_priv *priv)
{
    v4l2_ctrl_handler_init(&priv->ctrl_handler, 4);

    v4l2_ctrl_new_std(&priv->ctrl_handler, &sensor_ctrl_ops,
                      V4L2_CID_GAIN, 16, 170, 1, 16);
    v4l2_ctrl_new_std(&priv->ctrl_handler, &sensor_ctrl_ops,
                      V4L2_CID_EXPOSURE, 13, 683709, 1, 2495);
    v4l2_ctrl_new_std(&priv->ctrl_handler, &sensor_ctrl_ops,
                      V4L2_CID_HFLIP, 0, 1, 1, 0);
    v4l2_ctrl_new_std(&priv->ctrl_handler, &sensor_ctrl_ops,
                      V4L2_CID_VFLIP, 0, 1, 1, 0);

    priv->subdev.ctrl_handler = &priv->ctrl_handler;
}
```

### 6.6 Driver Probe and Registration

```c
static int sensor_probe(struct i2c_client *client)
{
    struct sensor_priv *priv;
    u8 chip_id_h, chip_id_l;
    int ret;

    priv = devm_kzalloc(&client->dev, sizeof(*priv), GFP_KERNEL);
    if (!priv)
        return -ENOMEM;

    priv->client = client;

    /* Get clock */
    priv->mclk = devm_clk_get(&client->dev, "extperiph1");
    if (IS_ERR(priv->mclk))
        return dev_err_probe(&client->dev, PTR_ERR(priv->mclk),
                             "Failed to get MCLK\n");

    /* Enable clock for chip ID read */
    clk_set_rate(priv->mclk, 24000000);
    clk_prepare_enable(priv->mclk);
    usleep_range(5000, 10000);

    /* Read and verify chip ID */
    ret = sensor_read_reg(client, 0x0000, &chip_id_h);
    ret |= sensor_read_reg(client, 0x0001, &chip_id_l);
    if (ret || chip_id_h != 0x02 || chip_id_l != 0x19) {
        dev_err(&client->dev, "Chip ID mismatch: 0x%02x%02x\n",
                chip_id_h, chip_id_l);
        clk_disable_unprepare(priv->mclk);
        return -ENODEV;
    }
    dev_info(&client->dev, "IMX219 detected (chip ID: 0x%02x%02x)\n",
             chip_id_h, chip_id_l);

    clk_disable_unprepare(priv->mclk);

    /* Initialize V4L2 subdev */
    v4l2_i2c_subdev_init(&priv->subdev, client, &sensor_subdev_ops);
    priv->subdev.flags |= V4L2_SUBDEV_FL_HAS_DEVNODE;

    /* Initialize controls */
    sensor_init_controls(priv);

    /* Initialize media entity pads */
    priv->pad.flags = MEDIA_PAD_FL_SOURCE;
    priv->subdev.entity.function = MEDIA_ENT_F_CAM_SENSOR;
    ret = media_entity_pads_init(&priv->subdev.entity, 1, &priv->pad);
    if (ret)
        return ret;

    /* Register subdev */
    ret = v4l2_async_register_subdev(&priv->subdev);
    if (ret) {
        media_entity_cleanup(&priv->subdev.entity);
        return ret;
    }

    i2c_set_clientdata(client, priv);
    return 0;
}

static const struct of_device_id sensor_of_match[] = {
    { .compatible = "sony,imx219" },
    { },
};
MODULE_DEVICE_TABLE(of, sensor_of_match);

static struct i2c_driver sensor_driver = {
    .driver = {
        .name           = "imx219",
        .of_match_table = sensor_of_match,
    },
    .probe = sensor_probe,
    .remove = sensor_remove,
};
module_i2c_driver(sensor_driver);
```

---

## 7. ISP Pipeline Deep Dive

### 7.1 ISP Processing Stages

The T234 ISP processes RAW Bayer data through a fixed-function pipeline with
approximately 15 stages. Each stage is individually configurable through the
ISP tuning file.

```
 RAW Bayer Input (from VI DMA buffer)
    |
    v
 [1. Linearization]       -- Correct sensor non-linearity (pedestal removal)
    |
    v
 [2. Black Level Sub.]    -- Subtract per-channel optical black reference
    |
    v
 [3. Bad Pixel Corr.]     -- Detect and interpolate stuck/hot/dead pixels
    |                         (static table + dynamic detection)
    |
    v
 [4. Lens Shading Corr.]  -- Compensate for radial brightness falloff
    |                         (per-channel 2D gain mesh)
    |
    v
 [5. Green Imbalance]     -- Correct Gr/Gb channel mismatch
    |
    v
 [6. Demosaicing]         -- Interpolate missing color channels
    |                         (directional edge-aware algorithm)
    |
    v
 [7. Color Correction]    -- 3x3 CCM (Color Correction Matrix) per illuminant
    |                         Transforms sensor color space to sRGB
    |
    v
 [8. Auto White Balance]  -- Per-frame WB gain computation (R, Gr, Gb, B gains)
    |                         using scene statistics
    |
    v
 [9. Noise Reduction]     -- Spatial NR: bilateral/non-local means
    |                         Temporal NR: motion-compensated accumulation
    |                         Chroma NR: separate UV denoising
    |
    v
 [10. Auto Exposure]      -- Histogram-based exposure computation
    |                          (feeds back to sensor gain/integration time)
    |
    v
 [11. Tone Mapping]       -- Global gamma curve + local tone mapping
    |                          (HDR compression / shadow lift)
    |
    v
 [12. Color Space Conv.]  -- Convert from linear RGB to target colorspace
    |                          (BT.601 / BT.709 for YUV output)
    |
    v
 [13. Edge Enhancement]   -- Unsharp mask or detail-preserving sharpening
    |                          (coring threshold to avoid noise amplification)
    |
    v
 [14. Chroma Suppression] -- Reduce color artifacts at high-contrast edges
    |
    v
 [15. Format Conversion]  -- Pack into output format (NV12, ARGB, etc.)
    |
    v
 Processed Output (to DRAM, then to GPU/DLA/display)
```

### 7.2 Demosaicing

The ISP's demosaicing algorithm reconstructs full RGB at each pixel from the
Bayer color filter array pattern. The T234 ISP uses a directional
interpolation method:

```
 Bayer pattern (RGGB):      After demosaic:
 R G R G R G               (R,G,B) (R,G,B) (R,G,B) ...
 G B G B G B               (R,G,B) (R,G,B) (R,G,B) ...
 R G R G R G               (R,G,B) (R,G,B) (R,G,B) ...
 G B G B G B               (R,G,B) (R,G,B) (R,G,B) ...
```

Quality depends on:
- Correct Bayer phase (`pixel_phase` in device tree: rggb, bggr, grbg, gbrg)
- Sensor optical quality (aliasing degrades demosaic output)
- Appropriate anti-aliasing filter on the sensor

### 7.3 White Balance

Auto White Balance (AWB) uses per-frame statistics collected by the ISP:

- **Grey World assumption**: The average scene color should be neutral grey
- **Illuminant estimation**: Matches statistics to known illuminant chromaticities
- **Gain application**: Multiplies R, Gr, Gb, B channels by computed gains

```
 WB Gains:  [R_gain]   [1.0]     [1.0]     [B_gain]
            Daylight:   ~1.5      1.0       1.0       ~1.2
            Tungsten:   ~2.0      1.0       1.0       ~0.8
            Fluorescent:~1.6      1.0       1.0       ~1.1
```

### 7.4 Noise Reduction

The ISP provides three NR stages:

| NR Type    | Domain   | Method                        | Controls                   |
|------------|----------|-------------------------------|----------------------------|
| Spatial    | Luma     | Edge-aware bilateral filter   | Strength, radius           |
| Spatial    | Chroma   | Guided chroma filter          | Strength (higher OK)       |
| Temporal   | Luma+C   | Motion-compensated averaging  | Weight, motion threshold   |

Temporal NR is particularly effective on the Orin Nano because it uses the
ISP's internal motion estimation without consuming GPU cycles. However, it
introduces one frame of latency and can cause ghosting on fast-moving objects.

### 7.5 Tone Mapping

The tone mapping stage applies a transfer function to convert linear-light
pixel values to a display-referred encoding:

- **Global tone curve**: A 1D LUT (typically sRGB gamma or a custom curve)
- **Local tone mapping**: Adapts the curve spatially to lift shadows and
  compress highlights, useful for high dynamic range scenes

```
 Input (linear)     Global Gamma        Local TM
 +-------+         +-------+           +-------+
 |       |  --->   |   /   |  --->     | ///   |
 |       |         |  /    |           | //    |
 |       |         | /     |           |//     |
 +-------+         +-------+           +-------+
  0   1.0           0   1.0             0   1.0
```

### 7.6 Edge Enhancement

Edge enhancement (sharpening) uses an unsharp mask approach:

```
 sharpened = original + strength * (original - blurred)
```

Key parameters:
- **Strength**: Amplification factor (too high = halo artifacts)
- **Radius**: Blur kernel size for the unsharp mask
- **Coring threshold**: Minimum edge magnitude to sharpen (rejects noise)

### 7.7 ISP Tuning File Format

The ISP tuning file (`camera_overrides.isp`) is a binary blob with a
defined header structure. It is not human-editable -- it must be generated by
NVIDIA's ISP Tuning Tool or exported from a tuning session.

The file encodes per-illuminant parameter sets:

```
 camera_overrides.isp
 +--------------------+
 | Header             |
 |   version, sensor  |
 +--------------------+
 | Illuminant D65     |
 |   CCM, WB gains    |
 |   NR params        |
 +--------------------+
 | Illuminant TL84    |
 |   CCM, WB gains    |
 |   NR params        |
 +--------------------+
 | Illuminant A       |
 |   CCM, WB gains    |
 |   NR params        |
 +--------------------+
 | Shared params      |
 |   Gamma, sharpen   |
 |   Bad pixel table  |
 |   Lens shading     |
 +--------------------+
```

---

## 8. ISP Tuning and Calibration

### 8.1 ISP Tuning Tools

NVIDIA provides the ISP Tuning Tool (NvISPTuner) as part of the Camera
Development Kit, available to registered NVIDIA Developer Program members.
The tool is a Windows/Linux GUI application that connects to a running Jetson
target over the network.

Workflow:

```
 [Developer Workstation]              [Jetson Orin Nano]
 +--------------------+   network    +-------------------+
 | NvISPTuner GUI     |<------------>| nvisp-tuner-agent |
 | - Load RAW frames  |              | - Capture RAW     |
 | - Adjust params    |              | - Apply settings  |
 | - Preview results  |              | - Export .isp file|
 +--------------------+              +-------------------+
```

### 8.2 The camera_overrides.isp File

The ISP tuning file lives at:

```
/var/nvidia/nvcam/settings/camera_overrides.isp
```

When `nvargus-daemon` starts and opens a camera, it searches for this file. If
found, it overrides the built-in default tuning. Multiple sensor tunings can
coexist in the same file, keyed by sensor name and mode.

```bash
# Check if the tuning file exists and its size
ls -la /var/nvidia/nvcam/settings/camera_overrides.isp

# Back up before modifying
sudo cp /var/nvidia/nvcam/settings/camera_overrides.isp \
        /var/nvidia/nvcam/settings/camera_overrides.isp.bak

# After deploying a new tuning file, restart the daemon
sudo systemctl restart nvargus-daemon
```

### 8.3 Tuning for IMX219

The IMX219 (Raspberry Pi Camera Module v2) is the most commonly used sensor on
Jetson development kits. NVIDIA ships a default tuning file for it.

Key tuning considerations for IMX219:

| Parameter            | Recommendation                                     |
|----------------------|----------------------------------------------------|
| Black level          | ~64 DN for RAW10 (sensor's OB level)               |
| Color matrix (D65)   | Derive from Macbeth chart capture under daylight    |
| Color matrix (TL84)  | Derive from Macbeth chart under fluorescent         |
| Noise reduction      | Moderate luma NR; aggressive chroma NR acceptable   |
| Sharpening           | Light sharpening; sensor has no AA filter           |
| Lens shading         | Capture flat field for each corner/edge calibration |

### 8.4 Tuning for IMX477

The IMX477 (Raspberry Pi HQ Camera) produces 12-bit RAW at higher quality:

```bash
# Verify IMX477 is detected and producing 12-bit output
v4l2-ctl -d /dev/video0 --list-formats-ext
# Should show: 'RG12' (SRGGB12)
```

Key differences from IMX219 tuning:

| Parameter            | IMX219           | IMX477                          |
|----------------------|------------------|---------------------------------|
| Bit depth            | 10-bit           | 12-bit                          |
| Black level          | ~64 DN           | ~256 DN                         |
| Dynamic range        | ~60 dB           | ~72 dB                          |
| Noise floor          | Higher           | Lower (larger pixels)           |
| Lens shading         | Moderate falloff | Interchangeable lens dependent  |

### 8.5 Tuning for IMX708

The IMX708 (Raspberry Pi Camera Module v3) introduces autofocus and HDR:

```bash
# IMX708 may advertise multiple formats including HDR modes
v4l2-ctl -d /dev/video0 --list-formats-ext
# Look for: RG10 (standard), various HDR DT codes
```

The IMX708 supports in-sensor HDR (DOL-HDR) which requires additional ISP
tuning for tone mapping the extended dynamic range.

### 8.6 Calibration Procedure

Standard ISP calibration sequence:

1. **Black level calibration**: Cap the lens, capture 100 frames, compute mean
   per-channel.

2. **Lens shading calibration**: Illuminate an integrating sphere or flat
   white target, capture at each supported mode, generate per-channel gain mesh.

3. **Color calibration**: Capture an X-Rite ColorChecker (Macbeth chart) under
   D65 (daylight), TL84 (fluorescent), and Illuminant A (tungsten). Compute
   CCM for each illuminant by least-squares fitting.

4. **Noise profiling**: Capture a grey target at multiple gain/exposure
   settings. Measure noise variance vs. signal to parameterize NR curves.

5. **Gamma/tone curve**: Select sRGB gamma (2.2) or a custom curve optimized
   for the application (e.g., flatter curve for machine vision).

6. **Sharpening**: Capture a resolution target (ISO 12233), adjust sharpening
   strength until MTF50 meets requirements without ringing artifacts.

```bash
# Capture RAW for calibration at specific gain
v4l2-ctl -d /dev/video0 \
    --set-ctrl=gain=16,exposure=50000 \
    --set-fmt-video=width=3280,height=2464,pixelformat=RG10 \
    --stream-mmap --stream-count=100 \
    --stream-to=calibration_gain1x.raw
```

---

## 9. V4L2 and Media Controller Framework

### 9.1 Media Graph Topology

The Jetson camera stack uses the Linux Media Controller framework to represent
the hardware pipeline as a directed graph of entities and links:

```bash
# Display the full media graph
media-ctl -p

# Typical output for a single IMX219:
Media controller API version 6.1.0

Media device information
------------------------
driver          tegra-camrtc-capture
model           NVIDIA Tegra Video Input Device
serial
bus info
hw revision     0x3
driver version  6.1.0

Device topology
- entity 1: imx219 30-0010 (1 pad, 1 link)
             type V4L2 subdev subtype Sensor
             device node name /dev/v4l-subdev0
        pad0: Source
                [fmt:SRGGB10_1X10/3280x2464]
                -> "nvcsi-0":0 [ENABLED]

- entity 2: nvcsi-0 (2 pads, 2 links)
             type V4L2 subdev subtype Unknown
             device node name /dev/v4l-subdev1
        pad0: Sink
                <- "imx219 30-0010":0 [ENABLED]
        pad1: Source
                -> "vi-0":0 [ENABLED]

- entity 3: vi-0 (1 pad, 1 link)
             type Node subtype V4L
             device node name /dev/video0
        pad0: Sink
                <- "nvcsi-0":1 [ENABLED]
```

### 9.2 v4l2-ctl Usage

```bash
# List all video devices
v4l2-ctl --list-devices

# Query device capabilities
v4l2-ctl -d /dev/video0 --all

# List supported pixel formats
v4l2-ctl -d /dev/video0 --list-formats-ext

# Set format and capture
v4l2-ctl -d /dev/video0 \
    --set-fmt-video=width=1920,height=1080,pixelformat=RG10 \
    --stream-mmap=4 --stream-count=100 --stream-to=/dev/null

# Query and set controls
v4l2-ctl -d /dev/video0 --list-ctrls
v4l2-ctl -d /dev/video0 --set-ctrl=gain=32
v4l2-ctl -d /dev/video0 --set-ctrl=exposure=30000

# Get current control values
v4l2-ctl -d /dev/video0 --get-ctrl=gain
v4l2-ctl -d /dev/video0 --get-ctrl=exposure

# Capture with verbose timing (shows per-frame timestamps)
v4l2-ctl -d /dev/video0 \
    --set-fmt-video=width=1920,height=1080,pixelformat=RG10 \
    --stream-mmap --stream-count=60 --verbose
```

### 9.3 media-ctl Pipeline Configuration

```bash
# Set format on the sensor subdev pad
media-ctl -V '"imx219 30-0010":0 [fmt:SRGGB10_1X10/3280x2464]'

# Set format on the NVCSI subdev (must match sensor output)
media-ctl -V '"nvcsi-0":0 [fmt:SRGGB10_1X10/3280x2464]'

# Verify link status
media-ctl -l '"imx219 30-0010":0 -> "nvcsi-0":0 [1]'

# Print current graph with format info
media-ctl -p --print-dot > camera_graph.dot
dot -Tpng camera_graph.dot -o camera_graph.png
```

### 9.4 Subdev Routing

For multi-stream configurations (e.g., GMSL deserializer with multiple VCs),
subdev routing configures how streams map to pads:

```bash
# Example: MAX9296A deserializer with 2 virtual channels
media-ctl -R '"max9296 30-0048":0 -> "max9296 30-0048":4 [1], \
              "max9296 30-0048":1 -> "max9296 30-0048":5 [1]'
```

### 9.5 V4L2 Buffer Management

The preferred buffer mode for zero-copy operation is DMABUF:

```c
/* Request DMABUF buffers */
struct v4l2_requestbuffers req = {
    .count  = 4,
    .type   = V4L2_BUF_TYPE_VIDEO_CAPTURE,
    .memory = V4L2_MEMORY_DMABUF,
};
ioctl(fd, VIDIOC_REQBUFS, &req);

/* Export buffer as DMABUF fd */
struct v4l2_exportbuffer expbuf = {
    .type  = V4L2_BUF_TYPE_VIDEO_CAPTURE,
    .index = 0,
};
ioctl(fd, VIDIOC_EXPBUF, &expbuf);
int dmabuf_fd = expbuf.fd;
/* This fd can be imported by CUDA, display, or encoder */

/* Queue buffer */
struct v4l2_buffer buf = {
    .type   = V4L2_BUF_TYPE_VIDEO_CAPTURE,
    .memory = V4L2_MEMORY_MMAP,
    .index  = 0,
};
ioctl(fd, VIDIOC_QBUF, &buf);

/* Start streaming */
int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
ioctl(fd, VIDIOC_STREAMON, &type);

/* Dequeue captured frame */
ioctl(fd, VIDIOC_DQBUF, &buf);
/* buf.timestamp contains the capture timestamp */
/* buf.sequence contains the frame sequence number */
```

---

## 10. Libargus and Argus Camera API

### 10.1 Architecture

Libargus is NVIDIA's proprietary camera API that provides full ISP-processed
output through an EGLStream-based interface:

```
 +---------------------+       +---------------------+
 | Application         |       | nvargus-daemon      |
 | (links libargus.so) |       | (system service)    |
 |                     | IPC   |                     |
 | CameraProvider  ----------> | Sensor Driver (V4L2)|
 | CaptureSession  ----------> | ISP Control         |
 | OutputStream    ----------> | Buffer Management   |
 | Request         <---------- | Metadata/Stats      |
 +---------------------+       +---------------------+
           |                              |
           v                              v
     EGLStream output              Hardware (NVCSI/VI/ISP)
     (NV12 in NVMM memory)
```

### 10.2 Key Interfaces

| Interface                   | Purpose                                          |
|-----------------------------|--------------------------------------------------|
| `ICameraProvider`           | Enumerate cameras, create sessions               |
| `ICaptureSession`           | Manage capture lifecycle                         |
| `IRequest`                  | Configure per-frame capture parameters           |
| `IEGLOutputStreamSettings`  | Set output resolution, format, buffer count      |
| `ISourceSettings`           | Control exposure time, gain, frame duration       |
| `IAutoControlSettings`      | Enable/disable AE, AWB, configure ROIs           |
| `ICaptureMetadata`          | Read back actual exposure, gain, AWB values       |
| `IDenoiseSettings`          | Control NR mode and strength                      |
| `IEdgeEnhanceSettings`      | Control sharpening mode and strength              |

### 10.3 Complete Capture Session Example

```cpp
#include <Argus/Argus.h>
#include <EGLStream/EGLStream.h>
#include <EGLStream/NV/ImageNativeBuffer.h>

using namespace Argus;

int main()
{
    /* Create camera provider */
    UniqueObj<CameraProvider> provider(CameraProvider::create());
    ICameraProvider *iProvider = interface_cast<ICameraProvider>(provider);
    if (!iProvider) {
        fprintf(stderr, "Failed to create CameraProvider\n");
        return 1;
    }

    /* Get camera device list */
    std::vector<CameraDevice*> devices;
    iProvider->getCameraDevices(&devices);
    if (devices.empty()) {
        fprintf(stderr, "No cameras found\n");
        return 1;
    }
    printf("Found %zu camera(s)\n", devices.size());

    /* Create capture session for first camera */
    UniqueObj<CaptureSession> session(
        iProvider->createCaptureSession(devices[0]));
    ICaptureSession *iSession = interface_cast<ICaptureSession>(session);

    /* Query sensor modes */
    ISensorMode *sensorMode;
    {
        ICameraProperties *iCamProps =
            interface_cast<ICameraProperties>(devices[0]);
        std::vector<SensorMode*> modes;
        iCamProps->getAllSensorModes(&modes);
        printf("Available sensor modes:\n");
        for (size_t i = 0; i < modes.size(); i++) {
            ISensorMode *m = interface_cast<ISensorMode>(modes[i]);
            Size2D<uint32_t> res = m->getResolution();
            printf("  Mode %zu: %ux%u\n", i, res.width(), res.height());
        }
        sensorMode = interface_cast<ISensorMode>(modes[0]);
    }

    /* Configure output stream */
    UniqueObj<OutputStreamSettings> streamSettings(
        iSession->createOutputStreamSettings(STREAM_TYPE_EGL));
    IEGLOutputStreamSettings *iStreamSettings =
        interface_cast<IEGLOutputStreamSettings>(streamSettings);
    iStreamSettings->setPixelFormat(PIXEL_FMT_YCbCr_420_888);
    iStreamSettings->setResolution(Size2D<uint32_t>(1920, 1080));
    iStreamSettings->setMetadataEnable(true);

    UniqueObj<OutputStream> stream(
        iSession->createOutputStream(streamSettings.get()));

    /* Create frame consumer */
    UniqueObj<EGLStream::FrameConsumer> consumer(
        EGLStream::FrameConsumer::create(stream.get()));
    IFrameConsumer *iConsumer =
        interface_cast<IFrameConsumer>(consumer);

    /* Create and configure capture request */
    UniqueObj<Request> request(iSession->createRequest());
    IRequest *iRequest = interface_cast<IRequest>(request);
    iRequest->enableOutputStream(stream.get());

    /* Set per-frame controls */
    ISourceSettings *iSourceSettings =
        interface_cast<ISourceSettings>(iRequest->getSourceSettings());
    iSourceSettings->setSensorMode(sensorMode);
    iSourceSettings->setExposureTimeRange(
        Range<uint64_t>(33000000ULL, 33000000ULL));  /* 33ms = 30fps */
    iSourceSettings->setGainRange(Range<float>(1.0f, 1.0f));

    /* Set auto-control parameters */
    IAutoControlSettings *iAutoControl =
        interface_cast<IAutoControlSettings>(
            iRequest->getAutoControlSettings());
    iAutoControl->setAeAntibandingMode(AE_ANTIBANDING_MODE_AUTO);
    iAutoControl->setAwbMode(AWB_MODE_AUTO);

    /* Submit repeating request */
    iSession->repeat(request.get());

    /* Capture loop */
    for (int i = 0; i < 300; i++) {
        UniqueObj<EGLStream::Frame> frame(
            iConsumer->acquireFrame(1000000000ULL));  /* 1 sec timeout */
        if (!frame) {
            fprintf(stderr, "Frame acquire timeout\n");
            continue;
        }

        IFrame *iFrame = interface_cast<IFrame>(frame);
        printf("Frame %d: number=%u timestamp=%lu\n",
               i, iFrame->getNumber(),
               (unsigned long)iFrame->getTime());

        /* Access capture metadata */
        const ICaptureMetadata *meta =
            interface_cast<const ICaptureMetadata>(iFrame->getMetadata());
        if (meta) {
            printf("  Exposure: %lu ns, Gain: %.2f, AWB: (%.2f, %.2f, %.2f, %.2f)\n",
                   (unsigned long)meta->getSensorExposureTime(),
                   meta->getSensorAnalogGain(),
                   meta->getAwbGains().r(), meta->getAwbGains().gEven(),
                   meta->getAwbGains().gOdd(), meta->getAwbGains().b());
        }

        /* Get NV12 image for further processing */
        EGLStream::Image *image = iFrame->getImage();
        EGLStream::NV::IImageNativeBuffer *iNativeBuf =
            interface_cast<EGLStream::NV::IImageNativeBuffer>(image);
        if (iNativeBuf) {
            int fd = iNativeBuf->createNvBuffer(
                Size2D<uint32_t>(1920, 1080),
                NvBufferColorFormat_NV12,
                NvBufferLayout_Pitch);
            /* fd is a DMABUF file descriptor -- pass to CUDA, encoder, etc. */
            /* ... process frame ... */
            NvBufferDestroy(fd);
        }
    }

    /* Stop capture */
    iSession->stopRepeat();
    iSession->waitForIdle();

    return 0;
}
```

### 10.4 Building Argus Applications

```bash
# Argus samples are in the Multimedia API
cd /usr/src/jetson_multimedia_api/argus

# Build all samples
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# Run the one-shot sample
./samples/oneShot/argus_oneshot

# Run with specific camera and mode
./samples/oneShot/argus_oneshot --device 0 --mode 1
```

### 10.5 Per-Frame Control

Libargus supports per-frame control changes without stopping the capture
session:

```cpp
/* Change exposure for the next frame */
ISourceSettings *src = interface_cast<ISourceSettings>(
    iRequest->getSourceSettings());
src->setExposureTimeRange(Range<uint64_t>(16000000ULL, 16000000ULL));
src->setGainRange(Range<float>(2.0f, 2.0f));

/* Submit as a single-shot request (overrides repeat for one frame) */
iSession->capture(request.get());
```

---

## 11. GStreamer Camera Pipeline

### 11.1 NVIDIA GStreamer Elements

NVIDIA provides several GStreamer elements for camera capture on Jetson:

| Element             | Source     | ISP | NVMM  | Use Case                      |
|---------------------|------------|-----|-------|-------------------------------|
| `nvarguscamerasrc`  | libargus   | Yes | Yes   | Production ISP-processed      |
| `nvv4l2camerasrc`   | V4L2       | No  | Yes   | RAW/YUV sensors, ISP bypass   |
| `v4l2src`           | V4L2       | No  | No    | Debug only, CPU memory        |

### 11.2 nvarguscamerasrc Properties

```bash
# Inspect all properties
gst-inspect-1.0 nvarguscamerasrc

# Key properties:
#   sensor-id        : Camera index (0, 1, ...)
#   sensor-mode      : Sensor mode index (-1 = auto)
#   num-buffers      : Number of frames to capture (-1 = infinite)
#   wbmode           : White balance mode (0=off, 1=auto, 2-9=presets)
#   aelock           : Lock auto exposure (true/false)
#   awblock          : Lock auto white balance (true/false)
#   exposuretimerange: "min max" in nanoseconds
#   gainrange        : "min max" analog gain
#   ispdigitalgainrange: "min max" ISP digital gain
#   tnr-mode         : Temporal noise reduction (0=off, 1=fast, 2=HQ)
#   tnr-strength     : TNR strength (0.0 - 1.0)
#   ee-mode          : Edge enhancement (0=off, 1=fast, 2=HQ)
#   ee-strength      : Edge enhancement strength (0.0 - 1.0)
#   saturation       : Color saturation (0.0 - 2.0)
```

### 11.3 Pipeline Construction Examples

```bash
# Basic preview (display on screen)
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! \
    'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! \
    nvvidconv ! nv3dsink

# H.264 recording at 8 Mbps
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! \
    'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! \
    nvv4l2h264enc bitrate=8000000 insert-sps-pps=true ! \
    h264parse ! mp4mux ! filesink location=recording.mp4

# H.265 recording (better compression)
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! \
    'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! \
    nvv4l2h265enc bitrate=6000000 ! h265parse ! \
    matroskamux ! filesink location=recording.mkv

# JPEG snapshot (30 frames, save as individual JPEGs)
gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=30 ! \
    'video/x-raw(memory:NVMM),width=3280,height=2464,framerate=21/1' ! \
    nvjpegenc quality=95 ! multifilesink location="snap_%04d.jpg"

# RTSP streaming (requires gst-rtsp-server)
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! \
    'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! \
    nvv4l2h264enc bitrate=4000000 insert-sps-pps=true ! \
    h264parse ! rtph264pay ! udpsink host=224.1.1.1 port=5000

# RAW capture bypassing ISP (via nvv4l2camerasrc)
gst-launch-1.0 nvv4l2camerasrc device=/dev/video0 ! \
    'video/x-raw(memory:NVMM),format=GRAY16_LE,width=3280,height=2464' ! \
    nvvidconv ! 'video/x-raw,format=GRAY8' ! \
    filesink location=raw_capture.gray
```

### 11.4 Caps Negotiation

The `memory:NVMM` annotation is critical. It indicates that buffers reside in
NVIDIA-managed memory (NVMM) accessible to all hardware accelerators. Breaking
the NVMM chain forces an expensive device-to-host copy:

```bash
# GOOD: entire pipeline stays in NVMM
nvarguscamerasrc ! 'video/x-raw(memory:NVMM),...' ! nvv4l2h264enc ! ...

# BAD: nvvidconv without NVMM output forces copy to CPU memory
nvarguscamerasrc ! 'video/x-raw(memory:NVMM),...' ! nvvidconv ! \
    'video/x-raw,format=BGRx' ! videoconvert ! ...
#    ^--- CPU memory, full copy from GPU memory

# ACCEPTABLE: when CPU access is required (e.g., OpenCV)
nvarguscamerasrc ! 'video/x-raw(memory:NVMM),...' ! nvvidconv ! \
    'video/x-raw,format=BGRx' ! videoconvert ! \
    'video/x-raw,format=BGR' ! appsink
```

### 11.5 Multi-Camera GStreamer

```bash
# Dual camera capture to separate files
gst-launch-1.0 \
    nvarguscamerasrc sensor-id=0 ! \
        'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! \
        nvv4l2h264enc bitrate=4000000 ! h264parse ! \
        splitmuxsink location=cam0_%05d.mp4 max-size-time=60000000000 \
    nvarguscamerasrc sensor-id=1 ! \
        'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! \
        nvv4l2h264enc bitrate=4000000 ! h264parse ! \
        splitmuxsink location=cam1_%05d.mp4 max-size-time=60000000000
```

---

## 12. Multi-Camera Synchronization

### 12.1 Hardware Sync Triggers

For applications requiring frame-level synchronization across multiple cameras
(stereo vision, surround view), hardware triggering is essential. The typical
approach uses the sensor's FSIN (frame sync input) or XVS pin:

```
                    GPIO (from Orin Nano)
                     |
          +----------+----------+
          |                     |
   +------v------+       +-----v-------+
   | Sensor A    |       | Sensor B    |
   | FSIN pin    |       | FSIN pin    |
   | (ext. trig) |       | (ext. trig) |
   +-------------+       +-------------+
          |                     |
     CSI Port A            CSI Port B
```

Configure the sensor for external trigger mode (sensor-specific registers):

```c
/* IMX477 external trigger mode example */
/* Set trigger mode register */
sensor_write_reg(client, 0x0106, 0x01);  /* EXT_TRIG_MODE = enabled */

/* Configure Orin Nano GPIO as trigger output */
/* In device tree: */
// trigger-gpio = <&gpio TEGRA234_MAIN_GPIO(H, 6) GPIO_ACTIVE_HIGH>;
```

```bash
# Generate hardware trigger pulse from userspace (for testing)
echo 427 > /sys/class/gpio/export
echo out > /sys/class/gpio/gpio427/direction

# Toggle at desired frame rate (30 Hz = 33.3ms period)
while true; do
    echo 1 > /sys/class/gpio/gpio427/value
    sleep 0.001
    echo 0 > /sys/class/gpio/gpio427/value
    sleep 0.0323
done
```

### 12.2 Frame-Start Signaling

The VI block timestamps each frame with a TSC (Time Stamp Counter) value at the
SOF (Start of Frame) event. These timestamps enable software-level
synchronization even without hardware triggers:

```bash
# Capture with timestamp output
v4l2-ctl -d /dev/video0 --stream-mmap --stream-count=10 --verbose 2>&1 | \
    grep -E "seq|timestamp"
# Output:
#   seq: 0, timestamp: 1234567.890123
#   seq: 1, timestamp: 1234567.923456
```

### 12.3 Software Sync Strategies

When hardware sync is not available, software synchronization matches frames
from multiple cameras by timestamp proximity:

```python
import threading
import queue
import cv2

class SyncedCapture:
    def __init__(self, cam_ids, max_drift_ms=5.0):
        self.max_drift = max_drift_ms / 1000.0
        self.queues = {}
        self.caps = {}

        for cam_id in cam_ids:
            pipeline = (
                f"nvarguscamerasrc sensor-id={cam_id} ! "
                f"video/x-raw(memory:NVMM),width=1920,height=1080,"
                f"framerate=30/1 ! nvvidconv ! "
                f"video/x-raw,format=BGRx ! videoconvert ! "
                f"video/x-raw,format=BGR ! appsink drop=true"
            )
            self.caps[cam_id] = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            self.queues[cam_id] = queue.Queue(maxsize=2)

    def _capture_thread(self, cam_id):
        while True:
            ret, frame = self.caps[cam_id].read()
            if not ret:
                break
            ts = self.caps[cam_id].get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            try:
                self.queues[cam_id].put_nowait((ts, frame))
            except queue.Full:
                self.queues[cam_id].get()
                self.queues[cam_id].put((ts, frame))

    def get_synced_frames(self):
        """Return dict of {cam_id: frame} with closest timestamps."""
        frames = {}
        for cam_id in self.queues:
            ts, frame = self.queues[cam_id].get(timeout=1.0)
            frames[cam_id] = (ts, frame)

        # Find reference timestamp (first camera)
        ref_ts = list(frames.values())[0][0]

        # Verify all frames within drift tolerance
        for cam_id, (ts, frame) in frames.items():
            if abs(ts - ref_ts) > self.max_drift:
                return None  # Frames too far apart, retry
        return {cid: f for cid, (_, f) in frames.items()}
```

### 12.4 VI Channel Allocation

Each active camera stream consumes one VI channel. The Orin Nano supports up to
the number of channels corresponding to its CSI port configuration:

| Configuration          | VI Channels Used | Max Simultaneous Streams |
|------------------------|------------------|--------------------------|
| 4x sensors at x2 lanes | 4                | 4                        |
| 2x sensors at x4 lanes | 2                | 2                        |
| 1x sensor + 1x deser (2 VC) | 3         | 3                        |

---

## 13. Camera to CUDA Zero-Copy

### 13.1 Zero-Copy Data Path

The zero-copy path avoids any CPU-side memory copies between camera capture and
GPU processing:

```
 VI DMA --> DRAM (physical) --> ISP (same DRAM) --> DRAM (NV12)
                                                       |
                                                  DMABUF fd
                                                       |
                                              EGLImage mapping
                                                       |
                                              CUDA device pointer
                                                       |
                                              CUDA kernel launch
```

All stages reference the same physical memory through different virtual
address mappings. The DMABUF file descriptor acts as the portable handle.

### 13.2 DMABUF to EGLImage to CUDA

```cpp
#include <cuda_runtime.h>
#include <cuda_egl_interop.h>
#include "nvbuf_utils.h"
#include "NvBufSurface.h"

void process_frame_zero_copy(int dmabuf_fd, int width, int height)
{
    EGLDisplay egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);

    /* Step 1: Create EGLImage from DMABUF fd */
    EGLImageKHR egl_image = NvEGLImageFromFd(egl_display, dmabuf_fd);
    if (egl_image == EGL_NO_IMAGE_KHR) {
        fprintf(stderr, "Failed to create EGLImage from DMABUF\n");
        return;
    }

    /* Step 2: Register EGLImage with CUDA */
    cudaGraphicsResource_t cuda_resource;
    cudaError_t err = cudaGraphicsEGLRegisterImage(
        &cuda_resource, egl_image,
        cudaGraphicsRegisterFlagsReadOnly);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA EGL register failed: %s\n",
                cudaGetErrorString(err));
        NvDestroyEGLImage(egl_display, egl_image);
        return;
    }

    /* Step 3: Get mapped CUDA pointer */
    cudaEglFrame egl_frame;
    err = cudaGraphicsResourceGetMappedEglFrame(&egl_frame, cuda_resource, 0, 0);

    /* Step 4: Launch CUDA kernel directly on the mapped memory */
    /* For NV12: Y plane at pPitch[0], UV plane at pPitch[1] */
    unsigned char *y_plane  = (unsigned char *)egl_frame.frame.pPitch[0];
    unsigned char *uv_plane = (unsigned char *)egl_frame.frame.pPitch[1];

    dim3 block(32, 32);
    dim3 grid((width + 31) / 32, (height + 31) / 32);
    my_nv12_processing_kernel<<<grid, block>>>(
        y_plane, uv_plane, width, height,
        egl_frame.pitch);  /* stride in bytes */

    cudaDeviceSynchronize();

    /* Step 5: Cleanup */
    cudaGraphicsUnregisterResource(cuda_resource);
    NvDestroyEGLImage(egl_display, egl_image);
}
```

### 13.3 NvBufSurface Access

For cases where you need both CPU and GPU access to the same buffer:

```cpp
#include "NvBufSurface.h"

void access_frame_nvbufsurface(int dmabuf_fd)
{
    NvBufSurface *surf = NULL;

    /* Get NvBufSurface from DMABUF fd */
    NvBufSurfaceFromFd(dmabuf_fd, (void **)&surf);

    /* Map for CPU access (required before CPU read/write) */
    NvBufSurfaceMap(surf, 0, -1, NVBUF_MAP_READ);
    NvBufSurfaceSyncForCpu(surf, 0, -1);

    /* Access pixel data on CPU */
    unsigned char *y_data =
        (unsigned char *)surf->surfaceList[0].mappedAddr.addr[0];
    unsigned char *uv_data =
        (unsigned char *)surf->surfaceList[0].mappedAddr.addr[1];

    int y_stride = surf->surfaceList[0].pitch;
    int width    = surf->surfaceList[0].width;
    int height   = surf->surfaceList[0].height;

    printf("Frame: %dx%d, stride=%d, format=%d\n",
           width, height, y_stride, surf->surfaceList[0].colorFormat);

    /* Example: compute average luminance */
    uint64_t sum = 0;
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            sum += y_data[row * y_stride + col];
        }
    }
    printf("Average luminance: %.1f\n", (double)sum / (width * height));

    /* Unmap after CPU access */
    NvBufSurfaceUnMap(surf, 0, -1);
}
```

### 13.4 Avoiding CPU-Side Copies

Common pitfalls that break zero-copy:

| Mistake                                        | Fix                                        |
|------------------------------------------------|--------------------------------------------|
| Using `v4l2src` instead of `nvarguscamerasrc`  | Switch to `nvarguscamerasrc`               |
| GStreamer caps without `memory:NVMM`           | Add `(memory:NVMM)` to caps               |
| `cv2.VideoCapture` default backend             | Use `CAP_GSTREAMER` with NVMM pipeline    |
| Calling `NvBufSurfaceMap` unnecessarily        | Use CUDA mapping for GPU-only access       |
| Allocating new buffers per frame               | Pre-allocate buffer pool, recycle FDs      |

---

## 14. Camera to DLA/TensorRT Pipeline

### 14.1 End-to-End Architecture

A production inference pipeline from camera to model output:

```
 Camera Sensor
    |
 NVCSI/VI (RAW capture)
    |
 ISP (Bayer -> NV12)
    |
 nvarguscamerasrc (GStreamer / libargus)
    |
 nvvideoconvert (NV12 -> RGBA, resize to model input)
    |
 nvinfer / nvinferserver (TensorRT inference on GPU or DLA)
    |
 Application logic (post-processing, tracking, alerts)
```

### 14.2 DeepStream Pipeline

NVIDIA DeepStream provides the highest-performance camera-to-inference pipeline:

```bash
# Install DeepStream (if not already installed)
sudo apt-get install deepstream-7.0

# Minimal DeepStream pipeline with camera input
gst-launch-1.0 \
    nvarguscamerasrc sensor-id=0 ! \
    'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! \
    m.sink_0 nvstreammux name=m batch-size=1 width=1920 height=1080 ! \
    nvinfer config-file-path=/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt ! \
    nvvideoconvert ! nvdsosd ! nv3dsink
```

### 14.3 TensorRT Engine on DLA

Build a TensorRT engine targeting the DLA:

```python
import tensorrt as trt

def build_dla_engine(onnx_path, engine_path, dla_core=0):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"ONNX parse error: {parser.get_error(i)}")
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)

    # Enable DLA
    config.default_device_type = trt.DeviceType.DLA
    config.DLA_core = dla_core
    config.set_flag(trt.BuilderFlag.FP16)      # DLA requires FP16 or INT8
    config.set_flag(trt.BuilderFlag.GPU_FALLBACK)  # fallback for unsupported layers

    serialized = builder.build_serialized_network(network, config)
    with open(engine_path, 'wb') as f:
        f.write(serialized)
    print(f"DLA engine saved to {engine_path}")

build_dla_engine("yolov8n.onnx", "yolov8n_dla.engine", dla_core=0)
```

### 14.4 Camera-to-DLA Python Pipeline

```python
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class CameraDLAPipeline:
    def __init__(self, engine_path, sensor_id=0,
                 input_size=(640, 640)):
        self.input_size = input_size

        # Open camera via GStreamer (NVMM path)
        pipeline = (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM),width=1920,height=1080,"
            f"framerate=30/1 ! nvvidconv ! "
            f"video/x-raw,format=BGRx ! videoconvert ! "
            f"video/x-raw,format=BGR ! appsink drop=true max-buffers=2"
        )
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        # Load TensorRT engine
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.d_input = cuda.mem_alloc(
            1 * 3 * input_size[0] * input_size[1] * 4)  # FP32
        self.d_output = cuda.mem_alloc(
            1 * 84 * 8400 * 4)  # YOLOv8 output shape
        self.h_output = np.empty((1, 84, 8400), dtype=np.float32)
        self.stream = cuda.Stream()

    def preprocess(self, frame):
        """Resize, normalize, transpose to NCHW."""
        img = cv2.resize(frame, self.input_size)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.expand_dims(img, 0)   # Add batch dim
        return np.ascontiguousarray(img)

    def infer(self, frame):
        """Run inference on a single frame."""
        input_data = self.preprocess(frame)

        # Copy input to device
        cuda.memcpy_htod_async(self.d_input, input_data, self.stream)

        # Execute inference
        self.context.execute_async_v2(
            bindings=[int(self.d_input), int(self.d_output)],
            stream_handle=self.stream.handle)

        # Copy output to host
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()

        return self.h_output

    def run(self):
        """Main capture-infer loop."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            detections = self.infer(frame)
            # Post-process detections ...
            # (NMS, coordinate scaling, class filtering)

pipeline = CameraDLAPipeline("yolov8n_dla.engine", sensor_id=0)
pipeline.run()
```

### 14.5 nvinfer Configuration

For DeepStream's `nvinfer` element, the configuration file specifies the model
and preprocessing parameters:

```ini
# config_infer_primary.txt
[property]
gpu-id=0
net-scale-factor=0.00392157   # 1/255
model-engine-file=yolov8n_dla.engine
labelfile-path=labels.txt
batch-size=1
process-mode=1                # 1=primary detector
model-color-format=0          # 0=RGB
network-mode=1                # 0=FP32, 1=FP16, 2=INT8
num-detected-classes=80
interval=0                    # infer every frame
gie-unique-id=1
output-blob-names=output0

[class-attrs-all]
pre-cluster-threshold=0.25
nms-iou-threshold=0.45
```

---

## 15. Performance Optimization

### 15.1 Frame Rate Tuning

Achieving maximum frame rate requires coordination across the full pipeline:

| Bottleneck          | Diagnostic                                    | Optimization                                |
|---------------------|-----------------------------------------------|---------------------------------------------|
| Sensor output rate  | Check sensor datasheet for max FPS at mode     | Use binned/cropped modes                    |
| CSI bandwidth       | Calculate: W*H*FPS*BPP vs lane capacity        | Increase lanes or reduce resolution         |
| VI DMA              | `cat /sys/kernel/debug/camera/vi/*/status`     | Ensure sufficient DRAM bandwidth            |
| ISP throughput      | `tegrastats` ISP utilization                   | Reduce resolution or use ISP bypass         |
| Encoder             | GStreamer pipeline latency measurement          | Use hardware encoder, tune bitrate          |
| Application         | Profile with `nsys` or `nvprof`                | Async processing, pipelined buffers         |

### 15.2 Latency Reduction

End-to-end latency from photon to processed result:

```
 Sensor exposure    ~33 ms  (at 30 fps)
 Sensor readout     ~15 ms  (rolling shutter, resolution dependent)
 CSI transfer       < 1 ms
 VI DMA             < 1 ms
 ISP processing     ~5-10 ms
 CUDA/DLA inference ~5-20 ms (model dependent)
 -----------------------------------------
 Total              ~60-80 ms typical (30 fps)
```

Latency reduction strategies:

```bash
# 1. Reduce exposure time (brighter scene or wider aperture)
gst-launch-1.0 nvarguscamerasrc exposuretimerange="5000000 10000000" ! ...

# 2. Use higher frame rate mode (reduces per-frame delay)
gst-launch-1.0 nvarguscamerasrc sensor-mode=1 ! \
    'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=60/1' ! ...

# 3. Minimize buffer count (2 buffers instead of default 4)
gst-launch-1.0 nvarguscamerasrc ! \
    'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! \
    queue max-size-buffers=1 leaky=downstream ! ...

# 4. Use appsink with drop=true to always get the latest frame
... ! appsink drop=true max-buffers=1 sync=false
```

### 15.3 Buffer Management

Proper buffer management is critical for sustained throughput:

```c
/* Pre-allocate a fixed pool of DMABUF buffers */
#define NUM_BUFFERS 4

NvBufSurf::NvCommonAllocateParams params;
params.memType     = NVBUF_MEM_SURFACE_ARRAY;
params.width       = 1920;
params.height      = 1080;
params.layout      = NVBUF_LAYOUT_PITCH;
params.colorFormat = NVBUF_COLOR_FORMAT_NV12;
params.memtag      = NvBufSurfaceTag_CAMERA;

int dmabuf_fds[NUM_BUFFERS];
for (int i = 0; i < NUM_BUFFERS; i++) {
    NvBufSurf::NvAllocate(&params, 1, &dmabuf_fds[i]);
}

/* Use buffer pool in capture loop -- never allocate per frame */
int buf_idx = 0;
while (capturing) {
    int fd = dmabuf_fds[buf_idx];
    /* ... capture into fd, process fd, recycle ... */
    buf_idx = (buf_idx + 1) % NUM_BUFFERS;
}

/* Free on shutdown */
for (int i = 0; i < NUM_BUFFERS; i++) {
    NvBufSurf::NvDestroy(dmabuf_fds[i]);
}
```

### 15.4 ISP Bypass for RAW Capture

When ISP processing is not needed (e.g., for offline processing or custom
GPU-based ISP), bypass the ISP to reduce latency and power:

```bash
# Direct RAW capture via V4L2 (no ISP)
v4l2-ctl -d /dev/video0 \
    --set-fmt-video=width=3280,height=2464,pixelformat=RG10 \
    --stream-mmap=4 --stream-count=300 --stream-to=/dev/null

# Measure RAW capture frame rate
v4l2-ctl -d /dev/video0 \
    --set-fmt-video=width=1920,height=1080,pixelformat=RG10 \
    --stream-mmap=4 --stream-count=300 2>&1 | tail -5
# Look for: "fps: XX.XX"

# GStreamer RAW capture
gst-launch-1.0 nvv4l2camerasrc device=/dev/video0 ! \
    'video/x-raw(memory:NVMM),width=1920,height=1080' ! \
    fakesink sync=false
```

### 15.5 Clock Frequency Optimization

Ensure camera clocks are running at maximum rates for peak throughput:

```bash
# Check current clock rates
sudo cat /sys/kernel/debug/bpmp/debug/clk/vi/rate
sudo cat /sys/kernel/debug/bpmp/debug/clk/isp/rate
sudo cat /sys/kernel/debug/bpmp/debug/clk/nvcsi/rate

# Set maximum performance mode (disables DVFS throttling)
sudo nvpmodel -m 0           # MAXN power mode
sudo jetson_clocks            # Lock clocks to max frequency

# Verify clocks are at maximum
sudo jetson_clocks --show
```

---

## 16. Production Deployment

### 16.1 Reliability Considerations

Production camera systems must handle continuous 24/7 operation. Key concerns:

- **Memory leaks**: Monitor NVMM memory usage over extended runs
- **Buffer exhaustion**: Pre-allocate fixed pools, never dynamic allocation
- **Daemon crashes**: Configure automatic restart for `nvargus-daemon`
- **Sensor hangs**: Implement watchdog for I2C communication failures

### 16.2 Watchdog for Camera Hangs

```bash
# Configure nvargus-daemon for automatic restart
sudo mkdir -p /etc/systemd/system/nvargus-daemon.service.d/
sudo tee /etc/systemd/system/nvargus-daemon.service.d/override.conf << 'EOF'
[Service]
WatchdogSec=15
Restart=always
RestartSec=3
StartLimitIntervalSec=60
StartLimitBurst=5
EOF

sudo systemctl daemon-reload
sudo systemctl restart nvargus-daemon
```

Application-level watchdog:

```python
import threading
import time
import subprocess

class CameraWatchdog:
    def __init__(self, timeout_sec=10):
        self.timeout = timeout_sec
        self.last_frame_time = time.time()
        self.lock = threading.Lock()
        self._running = True

    def feed(self):
        """Call this every time a frame is successfully captured."""
        with self.lock:
            self.last_frame_time = time.time()

    def _watchdog_thread(self):
        while self._running:
            with self.lock:
                elapsed = time.time() - self.last_frame_time
            if elapsed > self.timeout:
                print(f"WATCHDOG: No frame for {elapsed:.1f}s, resetting camera")
                self._reset_camera()
            time.sleep(1.0)

    def _reset_camera(self):
        """Reset camera subsystem."""
        subprocess.run(["sudo", "systemctl", "restart", "nvargus-daemon"],
                       timeout=10)
        time.sleep(3)
        with self.lock:
            self.last_frame_time = time.time()

    def start(self):
        t = threading.Thread(target=self._watchdog_thread, daemon=True)
        t.start()

    def stop(self):
        self._running = False
```

### 16.3 Error Recovery

Common failure modes and recovery strategies:

| Failure Mode                   | Detection                        | Recovery                           |
|--------------------------------|----------------------------------|------------------------------------|
| I2C bus hang                   | I2C timeout in dmesg             | Reset I2C controller, re-probe     |
| CSI lane sync loss             | NVCSI CRC/ECC errors             | Toggle sensor reset GPIO           |
| ISP processing stall           | Frame timeout in nvargus-daemon  | Restart nvargus-daemon             |
| Sensor firmware crash          | Chip ID read returns 0xFF        | Power cycle sensor (GPIO toggle)   |
| Buffer pool exhaustion         | ENOMEM from VIDIOC_QBUF          | Flush pipeline, re-allocate        |

Sensor power cycle via GPIO:

```bash
# Reset sensor via GPIO (CAM0_PWDN)
echo 0 > /sys/class/gpio/gpio<pwdn_pin>/value   # power down
sleep 0.1
echo 1 > /sys/class/gpio/gpio<pwdn_pin>/value   # power up
sleep 0.5

# Or via device tree reset GPIO (handled by driver)
# reset-gpios = <&gpio CAM0_RST_L GPIO_ACTIVE_LOW>;
```

### 16.4 Thermal Impact of ISP

The ISP is a significant heat contributor during continuous processing:

```bash
# Monitor thermal zones during camera operation
watch -n 1 'paste <(cat /sys/class/thermal/thermal_zone*/type) \
                   <(cat /sys/class/thermal/thermal_zone*/temp) | \
            awk "{printf \"%-20s %5.1f C\n\", \$1, \$2/1000}"'
```

| Temperature Zone | Normal Range | Warning Threshold | Action Required            |
|------------------|------------- |-------------------|----------------------------|
| CPU-therm        | 40-65 C      | 85 C              | Improve airflow            |
| GPU-therm        | 40-65 C      | 85 C              | Reduce GPU load            |
| CV-therm (ISP)   | 40-70 C      | 90 C              | Reduce ISP resolution/FPS  |
| SOC-therm        | 40-70 C      | 97 C              | Thermal shutdown imminent  |

Thermal mitigation strategies for camera workloads:

- Use a heatsink and fan (active cooling) for continuous ISP processing
- Reduce ISP output resolution when thermal headroom is low
- Disable temporal noise reduction (TNR) to reduce ISP power draw
- Consider ISP bypass with GPU-based processing if GPU thermals have more
  headroom

### 16.5 Long-Running Stability Checklist

1. Enable log rotation to prevent disk fill from `nvargus-daemon` logs.
2. Monitor NVMM memory usage: `cat /sys/kernel/debug/nvmap/iovmm/clients`
3. Track V4L2 sequence numbers to detect dropped frames.
4. Implement frame content health checks:

```python
import hashlib
import numpy as np

class FrameHealthChecker:
    def __init__(self):
        self.prev_hash = None
        self.frozen_count = 0
        self.black_threshold = 5      # mean pixel value
        self.frozen_limit = 30        # frames

    def check(self, frame_data):
        """Returns (is_healthy, reason) tuple."""
        arr = np.frombuffer(frame_data, dtype=np.uint8)

        # Check for black frame
        mean_val = arr.mean()
        if mean_val < self.black_threshold:
            return False, f"Black frame (mean={mean_val:.1f})"

        # Check for saturated frame
        if mean_val > 250:
            return False, f"Saturated frame (mean={mean_val:.1f})"

        # Check for frozen frame (identical to previous)
        curr_hash = hashlib.md5(frame_data).hexdigest()
        if curr_hash == self.prev_hash:
            self.frozen_count += 1
            if self.frozen_count > self.frozen_limit:
                return False, f"Frozen frame ({self.frozen_count} identical)"
        else:
            self.frozen_count = 0
        self.prev_hash = curr_hash

        return True, "OK"
```

---

## 17. Common Issues and Debugging

### 17.1 Blank / No Frames

**Symptom**: `/dev/video0` exists but `v4l2-ctl --stream-mmap` produces empty
or zero-byte frames.

**Diagnostic steps**:

```bash
# Step 1: Verify sensor is responding on I2C
sudo i2cdetect -y -r 30
# If sensor address missing: check power rails, I2C pull-ups, connector seating

# Step 2: Check kernel logs for camera errors
dmesg | grep -iE "nvcsi|vi\b|imx|cam|csi|i2c" | tail -30

# Step 3: Verify NVCSI status
cat /sys/kernel/debug/camera/nvcsi/nvcsi0/status 2>/dev/null

# Step 4: Verify VI status
cat /sys/kernel/debug/camera/vi/status 2>/dev/null

# Step 5: Check if sensor is actually streaming
# (sensor should be writing to MIPI lines after s_stream(1))
dmesg | grep "s_stream"
```

### 17.2 Color Artifacts (Green/Purple Tint)

**Symptom**: Image has a strong green, purple, or unnatural color cast.

**Root cause**: Bayer pattern phase mismatch between the device tree and the
actual sensor output.

```
 Correct RGGB:              Wrong (e.g., BGGR applied to RGGB sensor):
 R G R G                    Interpreted as B G B G
 G B G B                    Interpreted as G R G R
 (Natural colors)           (Purple/magenta cast)
```

**Fix**: Verify the `pixel_phase` property in the device tree mode definition
matches the sensor datasheet:

```dts
/* Must match the actual sensor Bayer pattern */
pixel_phase = "rggb";    /* Most Sony IMX sensors */
/* Other options: "bggr", "grbg", "gbrg" */
```

Also verify the `mbus_code` in the sensor driver:

```c
/* For RGGB RAW10: */
.mbus_code = MEDIA_BUS_FMT_SRGGB10_1X10,  /* RGGB */
/* For BGGR RAW10: */
.mbus_code = MEDIA_BUS_FMT_SBGGR10_1X10,  /* BGGR */
/* For GRBG RAW10: */
.mbus_code = MEDIA_BUS_FMT_SGRBG10_1X10,  /* GRBG */
/* For GBRG RAW10: */
.mbus_code = MEDIA_BUS_FMT_SGBRG10_1X10,  /* GBRG */
```

### 17.3 I2C Timeouts

**Symptom**: `dmesg` shows I2C transfer failures, sensor probe fails.

```bash
# Common dmesg output:
# [  12.345] i2c i2c-30: sendbytes: NAK bailout
# [  12.346] imx219 30-0010: failed to read chip ID

# Diagnostic:
sudo i2cdetect -y -r 30
# If address shows as UU: driver already bound (unbind first for raw access)
# If address missing: hardware issue

# Check I2C bus speed (should be 100kHz or 400kHz for most sensors)
cat /sys/class/i2c-adapter/i2c-30/device/speed_mode 2>/dev/null

# Verify sensor address matches device tree reg property
# IMX219 = 0x10, IMX477 = 0x1A, IMX708 = 0x10, OV5647 = 0x36
```

**Common causes and fixes**:

| Cause                        | Fix                                          |
|------------------------------|----------------------------------------------|
| Wrong I2C address in DT      | Check sensor datasheet for SADDR pin config  |
| Missing pull-ups (1.8V)      | Add 2.2k pull-ups on SDA/SCL                |
| Sensor not powered           | Verify AVDD, DVDD, IOVDD rails with DMM     |
| Reset GPIO held active       | Check GPIO polarity in device tree            |
| I2C bus contention            | Ensure no address conflicts on the bus       |

### 17.4 CSI Errors

**Symptom**: Frames are corrupted, have line tears, or VI reports errors.

```bash
# Check NVCSI error counters
cat /sys/kernel/debug/camera/nvcsi/nvcsi0/status
# Look for: CRC errors, ECC errors, header errors

# Check VI error counters
cat /sys/kernel/debug/camera/vi/vi*/status
# Look for: overflow, short frame, spurious data

# Common CSI error messages in dmesg:
# "NVCSI: cil_intr_status: 0x00000004"     --> CRC error
# "VI: capture status error: 0x00000001"    --> frame timeout
# "NVCSI: phy clock settle time mismatch"   --> timing issue
```

**CSI timing issues** (settle time):

```dts
/* If auto-calculation fails, manually set CIL settle time */
mode0 {
    cil_settletime = "0";    /* 0 = auto (recommended first) */
    /* If auto fails, calculate from D-PHY spec:
       T_HS-SETTLE = 85ns + 6*UI  (where UI = 1/DataRate)
       For 2.5 Gbps: UI = 0.4ns
       T_HS-SETTLE = 85 + 6*0.4 = 87.4 ns
       Register value = T_HS-SETTLE / T_CLK (NVCSI clock period)
    */
};
```

### 17.5 dmesg Camera Debug Flags

Enable verbose camera subsystem debugging:

```bash
# Enable NVCSI debug output
echo 1 > /sys/module/nvcsi/parameters/dbg_mask  2>/dev/null

# Enable VI debug output
echo 0xff > /sys/module/tegra_video/parameters/debug  2>/dev/null

# Enable sensor driver debug (if supported)
echo 7 > /proc/sys/kernel/printk  # raise printk level

# Monitor all camera-related kernel messages in real time
dmesg -wH | grep -iE "nvcsi|vi\b|isp|imx|cam|csi|i2c" &

# Enable nvargus-daemon verbose logging
sudo systemctl stop nvargus-daemon
sudo nvargus-daemon --verbose=7  # max verbosity
```

### 17.6 v4l2-compliance Testing

Run the V4L2 compliance test suite to verify driver correctness:

```bash
# Install v4l2-compliance (if not present)
sudo apt-get install v4l-utils

# Run compliance test on the capture device
v4l2-compliance -d /dev/video0 2>&1 | tee v4l2_compliance.log

# Run compliance test on the sensor subdevice
v4l2-compliance -d /dev/v4l-subdev0 2>&1 | tee subdev_compliance.log

# Key tests to pass:
#   VIDIOC_QUERYCAP       -- device capability reporting
#   VIDIOC_ENUM_FMT       -- format enumeration
#   VIDIOC_S_FMT          -- format setting
#   VIDIOC_REQBUFS        -- buffer allocation
#   VIDIOC_STREAMON/OFF   -- streaming start/stop
#   Buffer exchange       -- queue/dequeue cycle
```

### 17.7 Complete Debugging Checklist

```bash
#!/bin/bash
# camera_debug_dump.sh -- Collect all camera diagnostic information

echo "=== System Info ==="
cat /etc/nv_tegra_release
uname -a
dpkg -l | grep -i jetpack

echo "=== Camera Devices ==="
v4l2-ctl --list-devices
ls -la /dev/video* /dev/v4l-subdev* 2>/dev/null

echo "=== Media Controller Topology ==="
media-ctl -p 2>/dev/null

echo "=== Device Tree Camera Nodes ==="
dtc -I fs /proc/device-tree 2>/dev/null | grep -A5 "tegra-camera-platform"
dtc -I fs /proc/device-tree 2>/dev/null | grep -B2 -A10 "imx\|ov5\|cam_sensor"

echo "=== I2C Buses ==="
i2cdetect -l
for bus in 30 31 32 33; do
    echo "--- I2C bus $bus ---"
    sudo i2cdetect -y -r $bus 2>/dev/null
done

echo "=== Kernel Modules ==="
lsmod | grep -iE "tegra_video|nvcsi|imx|sensor|cam"

echo "=== Camera Kernel Logs ==="
dmesg | grep -iE "nvcsi|vi\b|isp|imx|ov5|cam|csi|mclk" | tail -50

echo "=== Clock Rates ==="
for clk in nvcsi vi isp extperiph1; do
    rate=$(sudo cat /sys/kernel/debug/bpmp/debug/clk/$clk/rate 2>/dev/null)
    echo "$clk: $rate Hz"
done

echo "=== NVCSI Status ==="
cat /sys/kernel/debug/camera/nvcsi/nvcsi0/status 2>/dev/null

echo "=== VI Status ==="
cat /sys/kernel/debug/camera/vi/status 2>/dev/null

echo "=== Thermal ==="
paste <(cat /sys/class/thermal/thermal_zone*/type) \
      <(cat /sys/class/thermal/thermal_zone*/temp) 2>/dev/null | \
    awk '{printf "%-20s %5.1f C\n", $1, $2/1000}'

echo "=== nvargus-daemon Status ==="
systemctl status nvargus-daemon --no-pager

echo "=== ISP Tuning File ==="
ls -la /var/nvidia/nvcam/settings/camera_overrides.isp 2>/dev/null || \
    echo "WARNING: No ISP tuning file found"

echo "=== NvMap Memory Usage ==="
cat /sys/kernel/debug/nvmap/iovmm/clients 2>/dev/null | head -20
```

### 17.8 Diagnostic Quick Reference

| Symptom                    | First Check                           | Likely Cause                    |
|----------------------------|---------------------------------------|---------------------------------|
| No `/dev/video*`           | `dmesg \| grep imx`                  | Driver not loaded or probe fail |
| Black frames               | `i2cdetect -y -r 30`                 | Sensor not streaming            |
| Green/purple tint          | `pixel_phase` in device tree          | Wrong Bayer phase               |
| Horizontal line tears      | CSI lane count in DT                  | Lane mismatch or signal issue   |
| Frame drops at high FPS    | `tegrastats` ISP utilization          | ISP or bandwidth bottleneck     |
| nvargus-daemon crash       | `journalctl -u nvargus-daemon`        | ISP tuning file mismatch       |
| I2C NAK errors             | Multimeter on SDA/SCL                 | Missing pull-ups or power       |
| Exposure not changing      | V4L2 control values                   | Driver not applying controls    |
| CUDA mapping fails         | EGLDisplay initialization             | EGL not initialized             |
| GStreamer pipeline fails    | `gst-inspect-1.0 nvarguscamerasrc`    | Missing NVIDIA GStreamer plugins|

---

*This guide targets JetPack 6.x on the Jetson Orin Nano 8GB (Tegra234,
P3767-0005). Register addresses, device tree paths, clock names, and driver
interfaces may differ on other Jetson platforms or JetPack versions. Always
cross-reference with the official NVIDIA L4T documentation and the sensor
manufacturer's datasheet for your specific hardware configuration.*

