# Jetson Orin Nano 8GB -- Power Management, Thermal Design, and Energy Optimization

<div class="course-identity auto-course" style="--course-accent: #2563eb; --course-accent-rgb: 37, 99, 235;" markdown="1">
<div class="course-identity__icon">JON8</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Jetson Track</p>
<p class="course-identity__title">Specialized course identity for Jetson Orin Nano 8GB -- Power Management, Thermal Design, and Energy Optimization.</p>
<p class="course-identity__meta">Artifact: Jetson integration demo · Measure: latency, memory, power, logs</p>
</div>
</div>


A deep-dive engineering guide for the NVIDIA Jetson Orin Nano 8GB development kit and
production module, covering the T234 SoC power architecture, thermal subsystem, DVFS
framework, and real-world optimization techniques for edge AI deployments.

Target audience: embedded systems engineers, hardware integration teams, and anyone
shipping Jetson-based products that must operate reliably within strict power and
thermal envelopes.

---

## 1. Introduction

### 1.1 Power Efficiency as an Edge AI Differentiator

Edge AI devices live under constraints that data-center GPUs never face. A Jetson Orin
Nano deployed inside a traffic camera housing, a drone payload bay, or an autonomous
mobile robot must deliver useful inference throughput while staying within a fixed power
budget -- often drawn from a battery, a PoE link, or a shared DC rail that also feeds
sensors, radios, and actuators.

The Orin Nano 8GB ships with two official power modes:

| Mode   | TDP   | CPU Cores | CPU Max Freq | GPU Max Freq | DLA Max Freq | EMC Max Freq |
|--------|-------|-----------|--------------|--------------|--------------|--------------|
| 15W    | 15 W  | 6         | 1.51 GHz     | 625 MHz      | 614 MHz      | 2133 MHz     |
| 7W     | 7 W   | 4         | 1.13 GHz     | 420 MHz      | 414 MHz      | 2133 MHz     |

These numbers define the upper bound. In practice, actual power draw depends on
workload, DVFS state, peripheral activity, and ambient temperature. Understanding how
to control and monitor every variable in that chain is the subject of this guide.

### 1.2 Thermal Constraints in Embedded Deployments

The T234 SoC die temperature must stay below 97 degrees C (the hard thermal shutdown
threshold). NVIDIA defines a recommended operating range where the junction temperature
stays at or below 87 degrees C for sustained workloads. Beyond that point the firmware
begins throttling clocks, and beyond 97 degrees C the device powers off to prevent
damage.

In a sealed IP67 enclosure at 50 degrees C ambient -- a common scenario for outdoor
industrial deployments -- the thermal headroom shrinks dramatically. The difference
between a device that runs at full throughput and one that throttles to half speed often
comes down to:

- Heatsink design (thermal resistance from junction to ambient)
- Fan strategy (active vs passive, duty cycle, control algorithm)
- Software power management (clock gating, core parking, DLA offload)
- Workload scheduling (duty-cycling inference, batching, idle sleep)

This guide treats power and thermal as two sides of the same coin, because every watt
the SoC consumes becomes heat that must be removed.

### 1.3 Guide Scope and Conventions

All commands are shown as run on the Orin Nano target under JetPack 6.x (L4T 36.x)
with a standard Ubuntu 22.04 rootfs. Sysfs paths are given as they appear on this
platform; some paths differ between JetPack versions. Unless noted, commands require
root or sudo.

Throughout the guide:

- `$` prefix indicates a regular user shell prompt.
- `#` prefix indicates a root shell prompt.
- File paths under `/sys` and `/proc` are kernel-exported interfaces.
- Register addresses are from the T234 TRM (Technical Reference Manual).

---

## 2. Power Architecture

### 2.1 T234 SoC Power Domains

The T234 SoC inside the Orin Nano organizes its logic into several independently
controllable power domains. Each domain can be power-gated (completely turned off) or
clock-gated (logic stopped but power retained for fast resume).

Key power domains:

| Domain          | Contains                                     | Power-Gatable |
|-----------------|----------------------------------------------|---------------|
| CPU cluster 0   | ARM Cortex-A78AE cores 0-3                  | Per-core       |
| CPU cluster 1   | ARM Cortex-A78AE cores 4-5                  | Per-core       |
| GPU             | Ampere GPU (1024 CUDA cores on full Orin)    | Yes            |
| DLA0            | Deep Learning Accelerator engine 0           | Yes            |
| DLA1            | Deep Learning Accelerator engine 1           | Yes            |
| PVA             | Programmable Vision Accelerator              | Yes            |
| VIC             | Video Image Compositor                       | Yes            |
| NVDEC           | Video decoder                                | Yes            |
| NVENC           | Video encoder                                | Yes            |
| NVJPG           | JPEG encoder/decoder                         | Yes            |
| SE              | Security Engine (AES, SHA, RNG)              | No             |
| APE             | Audio Processing Engine                      | Yes            |

On the Orin Nano specifically, some engines present on the full Orin SoC are fused off
or limited. The Orin Nano 8GB has access to 6 CPU cores, a reduced GPU configuration,
and one DLA instance (DLA0). DLA1 and PVA are not available on the Orin Nano SKU.

### 2.2 PMIC and Voltage Rails

The Orin Nano module uses a multi-output PMIC (Power Management IC) to generate the
voltage rails the SoC requires. The key rails are:

| Rail Name       | Nominal Voltage | Supplies                    |
|-----------------|----------------|-----------------------------|
| VDD_CPU         | 0.6-1.1 V      | CPU core logic              |
| VDD_GPU         | 0.6-1.1 V      | GPU core logic              |
| VDD_SOC         | 0.8 V           | SoC peripherals, fabric     |
| VDD_DDR         | 1.1 V (LPDDR5) | Memory I/O and controller   |
| VDD_1V8         | 1.8 V           | I/O, PLL, analog            |
| VDD_3V3         | 3.3 V           | Carrier board I/O, SD card  |
| VDD_IN          | 5-20 V          | Module input supply          |

VDD_CPU and VDD_GPU are variable-voltage rails controlled by DVFS. The PMIC adjusts
these voltages in real time based on the operating frequency requested by the kernel's
clock framework.

### 2.3 Power Sequencing

The T234 follows a strict power-up sequence controlled by the PMIC and the boot ROM:

1. VDD_IN applied (5 V minimum from carrier board).
2. PMIC enables core rails in order: VDD_SOC, VDD_DDR, VDD_1V8, VDD_CPU, VDD_GPU.
3. POR (Power-On Reset) de-asserts once all rails are stable.
4. Boot ROM begins executing from on-chip SRAM.
5. MB1 (Microboot 1) configures DRAM, loads MB2.
6. MB2 loads UEFI/CBoot, which loads the Linux kernel.

Power-down reverses this sequence. The PMIC handles sequencing automatically; incorrect
sequencing (e.g., from a poorly designed carrier board) can cause latch-up or permanent
damage.

### 2.4 Module vs Carrier Board Power Budget

The module VDD_IN rail draws power for the SoC, DRAM, and on-module regulators. The
carrier board adds its own consumption for USB hubs, Ethernet PHYs, display bridges,
and other peripherals. A typical development kit breakdown at 15W mode under load:

```
Module (SoC + DRAM + PMIC losses):    ~12 W
Carrier board peripherals:             ~3-5 W
Total at the DC jack:                  ~15-17 W
```

In a custom carrier board design, stripping unnecessary peripherals can reduce system
power by 2-3 W, which matters significantly in battery-powered applications.

---

## 3. nvpmodel Deep Dive

### 3.1 What nvpmodel Does

`nvpmodel` is NVIDIA's power mode management tool. It reads a configuration file that
defines named power modes, each specifying:

- How many CPU cores are online
- Maximum CPU frequency
- Maximum GPU frequency
- Maximum DLA frequency
- Maximum EMC (memory controller) frequency
- Power budget ceiling

The tool then applies these constraints by writing to sysfs nodes and communicating
with the NVPVA and thermal frameworks.

### 3.2 Built-in Power Modes

The Orin Nano 8GB ships with these modes in `/etc/nvpmodel.conf`:

```
# Query current mode
$ sudo nvpmodel -q

# List available modes
$ sudo nvpmodel -q --verbose
```

Mode 0 -- 15W (MAXN):

| Parameter            | Value        |
|----------------------|-------------|
| Online CPU cores     | 6 (0-5)     |
| CPU max freq         | 1510 MHz    |
| GPU max freq         | 625 MHz     |
| DLA max freq         | 614 MHz     |
| EMC max freq         | 2133 MHz    |
| Power budget         | 15000 mW    |

Mode 1 -- 7W:

| Parameter            | Value        |
|----------------------|-------------|
| Online CPU cores     | 4 (0-3)     |
| CPU max freq         | 1126 MHz    |
| GPU max freq         | 420 MHz     |
| DLA max freq         | 414 MHz     |
| EMC max freq         | 2133 MHz    |
| Power budget         | 7000 mW     |

### 3.3 Switching Power Modes

```bash
# Switch to 15W mode
$ sudo nvpmodel -m 0

# Switch to 7W mode
$ sudo nvpmodel -m 1

# Query current mode
$ sudo nvpmodel -q
# Output: NV Power Mode: MODE_15W

# The mode persists across reboots (stored in /var/lib/nvpmodel/)
```

After switching modes, the kernel adjusts CPU online/offline status and sets frequency
ceilings. DVFS continues to operate within these new bounds.

### 3.4 The nvpmodel.conf Configuration File

The configuration file lives at `/etc/nvpmodel.conf`. Here is the annotated structure:

```ini
# /etc/nvpmodel.conf -- Orin Nano 8GB
# NVIDIA Power Model Configuration

# ------------------------------------------------------------------
# PARAM TYPE: CLOCK
#   Specifies clock constraints per power mode.
#   Fields:
#     CPU_ONLINE   -- bitmask or count of online cores
#     TPC_POWER_GATING -- GPU TPC power gating
#     GPU_POWER_GATING -- GPU power gating
#     ...
#
# PARAM TYPE: POWER_BUDGET
#   Specifies the OC (over-current) or total power cap in mW.
# ------------------------------------------------------------------

###########################
# MODE 0: 15W (MAXN)
###########################
< POWER_MODEL ID=0 NAME=MODE_15W >
  CPU_ONLINE CORE_0 1
  CPU_ONLINE CORE_1 1
  CPU_ONLINE CORE_2 1
  CPU_ONLINE CORE_3 1
  CPU_ONLINE CORE_4 1
  CPU_ONLINE CORE_5 1
  TPC_POWER_GATING TPC_PG_MASK 0
  GPU_POWER_GATING GPU_PG_MASK 0
  CPU_A78_0 MIN_FREQ 729600
  CPU_A78_0 MAX_FREQ 1510400
  GPU MIN_FREQ 0
  GPU MAX_FREQ 625000
  GPU_POWER_GATING GPU_PG_MASK 0
  DLA0_CORE MAX_FREQ 614400
  DLA0_FALCON MAX_FREQ 294400
  EMC MAX_FREQ 2133000
  POWER_BUDGET CPU 15000
< /POWER_MODEL >

###########################
# MODE 1: 7W
###########################
< POWER_MODEL ID=1 NAME=MODE_7W >
  CPU_ONLINE CORE_0 1
  CPU_ONLINE CORE_1 1
  CPU_ONLINE CORE_2 1
  CPU_ONLINE CORE_3 1
  CPU_ONLINE CORE_4 0
  CPU_ONLINE CORE_5 0
  TPC_POWER_GATING TPC_PG_MASK 1
  GPU_POWER_GATING GPU_PG_MASK 1
  CPU_A78_0 MIN_FREQ 729600
  CPU_A78_0 MAX_FREQ 1126400
  GPU MIN_FREQ 0
  GPU MAX_FREQ 420000
  DLA0_CORE MAX_FREQ 414700
  DLA0_FALCON MAX_FREQ 294400
  EMC MAX_FREQ 2133000
  POWER_BUDGET CPU 7000
< /POWER_MODEL >
```

### 3.5 Creating a Custom Power Mode

You can add custom modes for workload-specific tuning. Example: a 10W mode that keeps
all 6 CPU cores online but limits GPU frequency for a CPU-heavy pre-processing
pipeline:

```ini
###########################
# MODE 2: CUSTOM 10W (CPU-heavy)
###########################
< POWER_MODEL ID=2 NAME=MODE_10W_CPU >
  CPU_ONLINE CORE_0 1
  CPU_ONLINE CORE_1 1
  CPU_ONLINE CORE_2 1
  CPU_ONLINE CORE_3 1
  CPU_ONLINE CORE_4 1
  CPU_ONLINE CORE_5 1
  TPC_POWER_GATING TPC_PG_MASK 0
  GPU_POWER_GATING GPU_PG_MASK 0
  CPU_A78_0 MIN_FREQ 729600
  CPU_A78_0 MAX_FREQ 1510400
  GPU MIN_FREQ 0
  GPU MAX_FREQ 306000
  DLA0_CORE MAX_FREQ 414700
  DLA0_FALCON MAX_FREQ 294400
  EMC MAX_FREQ 2133000
  POWER_BUDGET CPU 10000
< /POWER_MODEL >
```

After editing `/etc/nvpmodel.conf`, apply the new mode:

```bash
$ sudo nvpmodel -m 2
$ sudo nvpmodel -q
# Output: NV Power Mode: MODE_10W_CPU
```

### 3.6 nvpmodel and DVFS Interaction

nvpmodel sets frequency ceilings (MAX_FREQ), not fixed frequencies. The DVFS governors
still scale clocks up and down within the allowed range. This means:

- In 15W mode with an idle system, the GPU may clock down to near zero.
- In 7W mode under full load, clocks hit the 7W-mode ceiling and stay there.
- `jetson_clocks` (Section 5) overrides DVFS by pinning clocks at the nvpmodel ceiling.

The hierarchy is: `nvpmodel` ceiling > `DVFS governor` dynamic scaling > `jetson_clocks` lock.

---

## 4. DVFS (Dynamic Voltage and Frequency Scaling)

### 4.1 How DVFS Works on Jetson

DVFS is the primary runtime power optimization mechanism. The kernel continuously
adjusts the operating frequency (and corresponding voltage) of the CPU, GPU, and EMC
based on workload demand. Lower frequency means lower voltage, and since dynamic power
scales as P = C * V^2 * f, even a small frequency reduction yields a significant power
saving.

The T234 exposes DVFS control through the standard Linux cpufreq (for CPUs) and
devfreq (for GPU and EMC) frameworks.

### 4.2 CPU Frequency Scaling

The Orin Nano CPU cores belong to a single cpufreq policy (all cores in a cluster share
the same frequency):

```bash
# List available CPU frequencies
$ cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies
729600 1036800 1267200 1420800 1510400

# Current frequency
$ cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
1510400

# Current governor
$ cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
schedutil

# Set governor
$ echo "schedutil" | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# Force a specific frequency (requires governor = userspace)
$ echo "userspace" | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
$ echo 1036800 | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
```

### 4.3 CPU Governor Types

| Governor     | Behavior                                                        |
|-------------|------------------------------------------------------------------|
| `schedutil`  | Default. Scales frequency based on CPU scheduler utilization.   |
| `ondemand`   | Scales frequency based on CPU load sampling.                    |
| `conservative`| Like ondemand but ramps up/down more gradually.               |
| `performance`| Locks at maximum allowed frequency. High power.                |
| `powersave`  | Locks at minimum frequency. Lowest power, worst performance.   |
| `userspace`  | Allows manual frequency setting via scaling_setspeed.           |

For most edge AI workloads, `schedutil` provides the best balance. It reacts quickly to
inference bursts and drops frequency during idle gaps.

```bash
# List available governors
$ cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors
schedutil ondemand conservative performance powersave userspace

# View schedutil parameters (kernel 5.15+)
$ ls /sys/devices/system/cpu/cpufreq/schedutil/
rate_limit_us
```

### 4.4 GPU Frequency Scaling

The GPU uses the `devfreq` framework:

```bash
# GPU devfreq path
GPU_PATH="/sys/devices/platform/gpu.0/devfreq/17000000.gpu"

# Available GPU frequencies
$ cat ${GPU_PATH}/available_frequencies
306000000 420000000 522000000 625000000

# Current GPU frequency
$ cat ${GPU_PATH}/cur_freq
306000000

# GPU governor
$ cat ${GPU_PATH}/governor
nvhost_podgov

# Available governors
$ cat ${GPU_PATH}/available_governors
nvhost_podgov userspace performance simple_ondemand

# Set GPU to a fixed frequency
$ echo "userspace" | sudo tee ${GPU_PATH}/governor
$ echo 522000000 | sudo tee ${GPU_PATH}/userspace/set_freq
```

The default GPU governor `nvhost_podgov` is NVIDIA's custom governor that monitors GPU
activity and scales frequency to maintain a target utilization. It ramps up quickly when
CUDA kernels launch and drops when the GPU goes idle.

### 4.5 EMC (Memory) Clock Scaling

The External Memory Controller frequency determines DRAM bandwidth:

```bash
# EMC devfreq path
EMC_PATH="/sys/kernel/actmon_avg_activity/mc_all"

# Or through devfreq
EMC_DEVFREQ="/sys/devices/platform/bus@0/31b0000.emc/devfreq/31b0000.emc"

# Available EMC frequencies
$ cat ${EMC_DEVFREQ}/available_frequencies
204000000 408000000 665000000 800000000 1065000000 1331000000 1600000000 2133000000

# Current EMC frequency
$ cat ${EMC_DEVFREQ}/cur_freq

# EMC governor
$ cat ${EMC_DEVFREQ}/governor
```

EMC scaling is often overlooked. For workloads with low memory bandwidth requirements
(small models, low batch sizes), capping the EMC frequency can save 0.5-1.5 W:

```bash
# Cap EMC at 1600 MHz instead of 2133 MHz
$ echo 1600000000 | sudo tee ${EMC_DEVFREQ}/max_freq
```

### 4.6 DVFS Latency and Transition Behavior

Frequency transitions are not instantaneous:

| Domain | Transition Time    | Notes                              |
|--------|-------------------|------------------------------------|
| CPU    | ~10-50 us         | Voltage ramp dominates             |
| GPU    | ~20-100 us        | PLL relock may be needed           |
| EMC    | ~1-5 us           | DLL recalibration on some steps    |

For latency-sensitive inference pipelines, the transition overhead may cause jitter.
Locking frequencies with `jetson_clocks` eliminates this at the cost of higher average
power.

---

## 5. jetson_clocks

### 5.1 What jetson_clocks Does

`jetson_clocks` is a shell script (located at `/usr/bin/jetson_clocks`) that:

1. Sets all CPU governors to `performance` (max frequency).
2. Sets the GPU governor to fixed max frequency.
3. Sets the EMC to max frequency.
4. Optionally maximizes fan speed.

In other words, it disables DVFS and locks every clock domain at its nvpmodel ceiling.

```bash
# Apply max clocks (within current nvpmodel constraints)
$ sudo jetson_clocks

# Show current clock status
$ sudo jetson_clocks --show
```

Example output of `jetson_clocks --show`:

```
SOC family:tegra234  Machine:NVIDIA Orin Nano Developer Kit
Online CPUs: 0-5
cpu0: Online=1 Governor=performance MinFreq=1510400 MaxFreq=1510400 CurrentFreq=1510400
cpu1: Online=1 Governor=performance MinFreq=1510400 MaxFreq=1510400 CurrentFreq=1510400
cpu2: Online=1 Governor=performance MinFreq=1510400 MaxFreq=1510400 CurrentFreq=1510400
cpu3: Online=1 Governor=performance MinFreq=1510400 MaxFreq=1510400 CurrentFreq=1510400
cpu4: Online=1 Governor=performance MinFreq=1510400 MaxFreq=1510400 CurrentFreq=1510400
cpu5: Online=1 Governor=performance MinFreq=1510400 MaxFreq=1510400 CurrentFreq=1510400
GPU MinFreq=625000000 MaxFreq=625000000 CurrentFreq=625000000
EMC CurrentFreq=2133000000
Fan: speed=255
```

### 5.2 Store and Restore

Before running `jetson_clocks`, save the current DVFS state so you can revert later:

```bash
# Save current state
$ sudo jetson_clocks --store /tmp/clocks_backup.conf

# Lock all clocks at max
$ sudo jetson_clocks

# ... run benchmarks, profiling, etc. ...

# Restore original DVFS state
$ sudo jetson_clocks --restore /tmp/clocks_backup.conf
```

The store file is a plain text file capturing governor names and frequency limits for
each domain.

### 5.3 When to Use jetson_clocks

Use `jetson_clocks` for:

- **Benchmarking**: Eliminates frequency-scaling variability. Results are reproducible.
- **Latency-critical inference**: Removes DVFS transition jitter. Every frame processed
  at the same clock speed.
- **Thermal testing**: Forces maximum sustained power draw to stress-test cooling.

Do NOT use `jetson_clocks` for:

- **Battery-powered devices**: Constant max clocks drain batteries rapidly.
- **Thermally constrained enclosures**: May trigger throttling or shutdown.
- **Production idle periods**: Wastes power when there is no work to do.

### 5.4 Impact on Power and Thermal

Typical measurements on Orin Nano 8GB with `jetson_clocks` in 15W mode:

| Scenario                          | Power (VDD_IN) | SoC Temp  |
|-----------------------------------|----------------|-----------|
| Idle, DVFS active (no jetson_clocks) | ~3.5 W       | ~38 C     |
| Idle, jetson_clocks active        | ~6.5 W         | ~45 C     |
| Full inference, DVFS active       | ~12 W          | ~65 C     |
| Full inference, jetson_clocks     | ~14 W          | ~72 C     |

The idle penalty of `jetson_clocks` is substantial (~3 W) because the CPU and GPU run
at max frequency even with no work. This is pure waste in production deployments with
variable workloads.

---

## 6. Power Measurement

### 6.1 INA3221 Power Monitor

The Orin Nano developer kit includes an INA3221 three-channel power monitor IC on the
carrier board. It measures voltage and current on key power rails via I2C.

The INA3221 provides:

- Shunt voltage measurement (proportional to current through a sense resistor)
- Bus voltage measurement
- Calculated power = V_bus * I_shunt

### 6.2 Sysfs Power Interface

Power measurements are exposed through sysfs:

```bash
# Find INA3221 devices
$ ls /sys/bus/i2c/drivers/ina3221/

# Typical paths (may vary by board revision)
INA_PATH="/sys/bus/i2c/drivers/ina3221/"

# List available channels
$ find /sys/bus/i2c/drivers/ina3221/ -name "in_power*_input" 2>/dev/null

# Alternative: use the hwmon interface
$ find /sys/class/hwmon/ -name "in_power*_input" 2>/dev/null
```

On the Orin Nano devkit, the key power rails monitored are:

| Channel | Rail         | What it measures                        |
|---------|-------------|------------------------------------------|
| 0       | VDD_IN      | Total module input power                 |
| 1       | VDD_CPU_GPU_CV | CPU + GPU + Computer Vision power     |
| 2       | VDD_SOC     | SoC domain power                         |

```bash
# Read total input power (in milliwatts)
$ cat /sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon*/in_power0_input
12500

# Read voltage (in millivolts)
$ cat /sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon*/in_voltage0_input
5025

# Read current (in milliamps)
$ cat /sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon*/in_current0_input
2488
```

### 6.3 tegrastats Power Readings

`tegrastats` is NVIDIA's real-time system monitor. It reports power alongside CPU/GPU
utilization, temperature, and memory usage:

```bash
# Run tegrastats with 1-second interval
$ sudo tegrastats --interval 1000

# Example output line:
# RAM 2048/7620MB (lfb 1234x4MB) SWAP 0/3810MB (cached 0MB)
# CPU [34%@1510,22%@1510,18%@1510,45%@1510,12%@1510,8%@1510]
# GR3D_FREQ 62%@625 VIC_FREQ 0%@0
# APE 174 MTS fg 0% bg 0%
# POWER CPU/GPU/SOC 3521/1205/2150 VDD_IN 8234 VDD_CPU_GPU_CV 4726 VDD_SOC 2150

# Fields explained:
#   CPU/GPU/SOC      -- individual domain power in mW
#   VDD_IN           -- total module input power in mW
#   VDD_CPU_GPU_CV   -- combined CPU+GPU+CV power in mW
#   VDD_SOC          -- SoC domain power in mW
```

### 6.4 Scripted Power Logging

For profiling AI workloads, log power data to a CSV for post-analysis:

```bash
#!/bin/bash
# power_log.sh -- Log power data from INA3221 to CSV
# Usage: sudo ./power_log.sh <duration_seconds> <output_file>

DURATION=${1:-60}
OUTPUT=${2:-power_log.csv}
INTERVAL=0.1  # 100ms sampling

INA_BASE="/sys/bus/i2c/drivers/ina3221"
# Find the correct INA device -- adjust address as needed
INA_DEV=$(find ${INA_BASE} -maxdepth 1 -name "1-004*" | head -1)
HWMON=$(find ${INA_DEV}/hwmon -maxdepth 1 -name "hwmon*" | head -1)

echo "timestamp_ms,vdd_in_mw,vdd_cpu_gpu_cv_mw,vdd_soc_mw" > "${OUTPUT}"

START=$(date +%s%N)
END=$(( $(date +%s) + DURATION ))

while [ $(date +%s) -lt ${END} ]; do
    NOW=$(( ($(date +%s%N) - START) / 1000000 ))
    P0=$(cat ${HWMON}/in_power0_input 2>/dev/null || echo 0)
    P1=$(cat ${HWMON}/in_power1_input 2>/dev/null || echo 0)
    P2=$(cat ${HWMON}/in_power2_input 2>/dev/null || echo 0)
    echo "${NOW},${P0},${P1},${P2}" >> "${OUTPUT}"
    sleep ${INTERVAL}
done

echo "Logged ${DURATION}s of power data to ${OUTPUT}"
```

### 6.5 Per-Rail Power Measurement for Custom Carrier Boards

On custom carrier boards, you may add additional INA3221 or INA226 sensors on specific
rails (e.g., VDD_GPU only, or a dedicated 3.3V peripheral rail). The kernel INA driver
supports multiple devices:

```
# Device tree snippet for adding an INA226 at I2C address 0x41
ina226@41 {
    compatible = "ti,ina226";
    reg = <0x41>;
    shunt-resistor = <5000>;  /* 5 milliohm sense resistor */
};
```

After adding the device tree node and rebuilding, the new sensor appears under
`/sys/class/hwmon/` with standard voltage, current, and power attributes.

---

## 7. Thermal Architecture

### 7.1 Thermal Zones

The Linux thermal framework organizes temperature sensors into thermal zones. Each zone
represents a physical sensor location:

```bash
# List all thermal zones
$ ls /sys/class/thermal/thermal_zone*/type

# Typical zones on Orin Nano
$ for z in /sys/class/thermal/thermal_zone*/; do
    echo "$(cat ${z}/type): $(cat ${z}/temp) m-degC"
done

# Example output:
# CPU-therm: 42500 m-degC      (42.5 C)
# GPU-therm: 41000 m-degC      (41.0 C)
# SOC0-therm: 42000 m-degC     (42.0 C)
# SOC1-therm: 41500 m-degC     (41.5 C)
# SOC2-therm: 42000 m-degC     (42.0 C)
# tj-therm: 42500 m-degC       (junction temperature)
# PMIC-Die: 50000 m-degC       (PMIC internal)
```

Temperature values are reported in millidegrees Celsius. Divide by 1000 for degrees C.

### 7.2 Trip Points

Each thermal zone has trip points that trigger actions when crossed:

```bash
# List trip points for CPU thermal zone
ZONE="/sys/class/thermal/thermal_zone0"
$ cat ${ZONE}/type
CPU-therm

$ for i in $(seq 0 10); do
    TYPE=$(cat ${ZONE}/trip_point_${i}_type 2>/dev/null) || break
    TEMP=$(cat ${ZONE}/trip_point_${i}_temp 2>/dev/null)
    HYST=$(cat ${ZONE}/trip_point_${i}_hyst 2>/dev/null)
    echo "Trip ${i}: type=${TYPE} temp=${TEMP} hyst=${HYST}"
done

# Example output:
# Trip 0: type=active   temp=50000  hyst=0       (fan starts)
# Trip 1: type=active   temp=60000  hyst=0       (fan increases)
# Trip 2: type=active   temp=70000  hyst=0       (fan at high)
# Trip 3: type=passive  temp=82000  hyst=0       (throttling begins)
# Trip 4: type=passive  temp=87000  hyst=0       (aggressive throttling)
# Trip 5: type=critical temp=97000  hyst=0       (emergency shutdown)
```

Trip point types:

| Type       | Action                                                      |
|-----------|--------------------------------------------------------------|
| `active`   | Activates cooling devices (fans) at increasing duty cycles  |
| `passive`  | Reduces clock frequencies to lower heat generation          |
| `hot`      | Advisory; may trigger additional software throttling        |
| `critical` | Emergency thermal shutdown to prevent hardware damage       |

### 7.3 Cooling Devices

Cooling devices are bound to thermal zones and respond to trip points:

```bash
# List cooling devices
$ ls /sys/class/thermal/cooling_device*/type

# Example:
# cooling_device0: cpu-cluster0 (CPU frequency throttling)
# cooling_device1: cpu-cluster1
# cooling_device2: gpu (GPU frequency throttling)
# cooling_device3: pwm-fan (physical fan)

# Check current state and max state
$ cat /sys/class/thermal/cooling_device3/cur_state
5
$ cat /sys/class/thermal/cooling_device3/max_state
10
```

### 7.4 Thermal Governors

The thermal governor decides how aggressively to activate cooling devices:

```bash
# Current thermal governor for a zone
$ cat /sys/class/thermal/thermal_zone0/policy
step_wise

# Available governors
$ cat /sys/class/thermal/thermal_zone0/available_policies
step_wise bang_bang user_space power_allocator
```

| Governor           | Behavior                                                    |
|-------------------|--------------------------------------------------------------|
| `step_wise`       | Default. Steps cooling state up/down by 1 per polling cycle. Smooth, gradual response. |
| `bang_bang`        | Binary on/off control. Above trip = max cooling, below = off. Used for fans with no PWM. |
| `user_space`      | No automatic control. Userspace daemon manages cooling.      |
| `power_allocator` | IPA (Intelligent Power Allocation). Distributes a thermal budget across multiple heat sources using PID control. |

For most Orin Nano deployments, `step_wise` works well. For advanced thermal management
in multi-zone enclosures, `power_allocator` (IPA) can optimize the balance between CPU
and GPU thermal budgets.

### 7.5 Thermal Zone to Cooling Device Binding

The kernel binds cooling devices to thermal zones through the device tree. You can
inspect current bindings:

```bash
# Show which cooling device is bound to which zone trip point
$ for zone in /sys/class/thermal/thermal_zone*/; do
    ZTYPE=$(cat ${zone}/type)
    for trip_dir in ${zone}/cdev*; do
        [ -d "${trip_dir}" ] || continue
        CDEV=$(cat ${trip_dir}/type 2>/dev/null || echo "unknown")
        TRIP=$(cat ${trip_dir}/trip_point 2>/dev/null || echo "?")
        echo "${ZTYPE} -> ${CDEV} at trip ${TRIP}"
    done
done
```

---

## 8. Thermal Management in Software

### 8.1 Reading Thermal Zones Programmatically

Python example for continuous thermal monitoring:

```python
#!/usr/bin/env python3
"""thermal_monitor.py -- Monitor all thermal zones on Jetson Orin Nano."""

import os
import time
import glob

THERMAL_BASE = "/sys/class/thermal"

def read_thermal_zones():
    """Read all thermal zone temperatures."""
    zones = {}
    for zone_path in sorted(glob.glob(os.path.join(THERMAL_BASE, "thermal_zone*"))):
        try:
            with open(os.path.join(zone_path, "type")) as f:
                name = f.read().strip()
            with open(os.path.join(zone_path, "temp")) as f:
                temp_mc = int(f.read().strip())
            zones[name] = temp_mc / 1000.0
        except (IOError, ValueError):
            continue
    return zones

def main():
    print(f"{'Time':>8s}", end="")
    # First pass to get zone names
    zones = read_thermal_zones()
    for name in zones:
        print(f"  {name:>12s}", end="")
    print()

    start = time.monotonic()
    while True:
        elapsed = time.monotonic() - start
        zones = read_thermal_zones()
        print(f"{elapsed:8.1f}", end="")
        for temp in zones.values():
            print(f"  {temp:12.1f}", end="")
        print(flush=True)
        time.sleep(1.0)

if __name__ == "__main__":
    main()
```

### 8.2 Thermal Throttling Behavior

When the SoC temperature exceeds a passive trip point, the thermal framework reduces
clock frequencies. The throttling is progressive:

```
Temperature Range         Action
---------------------------------------------------------
< 82 C                   No throttling. Full performance.
82 C - 87 C              Mild throttling. CPU/GPU freq reduced by 10-30%.
87 C - 92 C              Aggressive throttling. Freq reduced by 30-60%.
92 C - 97 C              Severe throttling. Minimum frequencies.
>= 97 C                  Emergency shutdown (critical trip point).
```

You can observe throttling in real time:

```bash
# Watch CPU frequency drop under thermal load
$ watch -n 0.5 "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq && \
  cat /sys/class/thermal/thermal_zone0/temp"
```

### 8.3 Modifying Trip Points

On JetPack 6.x, trip points can be adjusted at runtime (if the driver supports
writable trip points):

```bash
# Lower the passive throttling trip point from 82C to 75C
# for more conservative thermal behavior
ZONE="/sys/class/thermal/thermal_zone0"
$ echo 75000 | sudo tee ${ZONE}/trip_point_3_temp

# Raise the active fan trip point to delay fan activation
$ echo 55000 | sudo tee ${ZONE}/trip_point_0_temp
```

**Warning**: Never raise the critical trip point (97 C). Lowering it is acceptable for
added safety margin.

For persistent changes across reboots, modify the device tree thermal zone definitions
and rebuild the DTB:

```dts
/* Excerpt from tegra234-thermal.dtsi */
cpu_thermal: cpu-thermal {
    polling-delay = <1000>;       /* ms between polls */
    polling-delay-passive = <500>; /* ms during throttling */

    trips {
        cpu_sw_throttle: cpu-sw-throttle {
            temperature = <82000>;
            hysteresis = <0>;
            type = "passive";
        };
        cpu_critical: cpu-critical {
            temperature = <97000>;
            hysteresis = <0>;
            type = "critical";
        };
    };
};
```

### 8.4 PWM Fan Control via Sysfs

The Orin Nano devkit fan is controlled through the PWM subsystem:

```bash
# Find the fan PWM device
$ find /sys/class/hwmon/ -name "pwm1" 2>/dev/null

# Typical path
FAN_PWM="/sys/class/hwmon/hwmon2/pwm1"

# Read current PWM value (0-255)
$ cat ${FAN_PWM}
128

# Set fan to full speed
$ echo 255 | sudo tee ${FAN_PWM}

# Set fan to 50% duty cycle
$ echo 128 | sudo tee ${FAN_PWM}

# Set fan to off (use with caution)
$ echo 0 | sudo tee ${FAN_PWM}

# Enable manual control (disable thermal governor control)
$ echo 1 | sudo tee ${FAN_PWM}_enable

# Return control to the thermal governor
$ echo 2 | sudo tee ${FAN_PWM}_enable
```

### 8.5 Fan Speed / Temperature Curve

The default thermal governor uses a stepped curve. You can inspect and modify it:

```bash
# The thermal framework's fan curve is defined by the active trip points
# and the cooling device states. Each state maps to a PWM value.

# Example default curve (approximate):
#   < 50C  -> fan off (state 0, PWM 0)
#   50-55C -> state 2, PWM ~80
#   55-60C -> state 4, PWM ~130
#   60-65C -> state 6, PWM ~180
#   65-70C -> state 8, PWM ~210
#   > 70C  -> state 10, PWM 255

# To see current state
$ cat /sys/class/thermal/cooling_device3/cur_state
```

---

## 9. Power Optimization Strategies

### 9.1 CPU Core Parking

Offlining unused CPU cores saves both static and dynamic power. On a 6-core Orin Nano,
if your inference pipeline uses 2 cores, park the others:

```bash
# Offline cores 2-5 (keep cores 0 and 1 online)
$ for core in 2 3 4 5; do
    echo 0 | sudo tee /sys/devices/system/cpu/cpu${core}/online
done

# Verify
$ cat /sys/devices/system/cpu/online
0-1

# Bring cores back online
$ for core in 2 3 4 5; do
    echo 1 | sudo tee /sys/devices/system/cpu/cpu${core}/online
done
```

Power savings from core parking on Orin Nano (measured at VDD_IN):

| Online Cores | Idle Power | Savings vs 6 cores |
|-------------|------------|---------------------|
| 6 (all)     | ~3.5 W     | baseline            |
| 4           | ~3.0 W     | ~0.5 W              |
| 2           | ~2.7 W     | ~0.8 W              |
| 1           | ~2.5 W     | ~1.0 W              |

### 9.2 GPU Clock Management

When no GPU work is pending, the GPU can be power-gated entirely:

```bash
# Check GPU power state
$ cat /sys/devices/platform/gpu.0/devfreq/17000000.gpu/cur_freq
0   # 0 means GPU is idle/power-gated

# Force GPU to lowest frequency when active
GPU_DEV="/sys/devices/platform/gpu.0/devfreq/17000000.gpu"
$ echo 306000000 | sudo tee ${GPU_DEV}/max_freq

# Allow full range again
$ echo 625000000 | sudo tee ${GPU_DEV}/max_freq
```

### 9.3 DLA vs GPU Power Comparison

The DLA (Deep Learning Accelerator) is far more power-efficient than the GPU for
supported operations. On the Orin Nano (DLA0 only):

| Workload (ResNet-50)         | Engine | Throughput  | Power   | Perf/Watt      |
|-----------------------------|--------|-------------|---------|----------------|
| FP16, batch=1               | GPU    | ~180 fps    | ~8 W    | ~22.5 fps/W    |
| INT8, batch=1               | GPU    | ~320 fps    | ~9 W    | ~35.5 fps/W    |
| INT8, batch=1               | DLA    | ~120 fps    | ~3 W    | ~40.0 fps/W    |
| INT8, batch=1               | DLA+GPU| ~380 fps    | ~10 W   | ~38.0 fps/W    |

The DLA delivers better perf-per-watt but lower absolute throughput. For power-
constrained applications, offloading entire models or specific layers to DLA is a
key optimization strategy. Use TensorRT's `--useDLACore=0` flag to target DLA0.

```bash
# Run TensorRT with DLA
$ /usr/src/tensorrt/bin/trtexec \
    --onnx=resnet50.onnx \
    --int8 \
    --useDLACore=0 \
    --allowGPUFallback \
    --workspace=1024 \
    --verbose
```

### 9.4 Memory Bandwidth Optimization

LPDDR5 memory consumes significant power at high bandwidth. Strategies to reduce
memory power:

1. **Reduce model size**: INT8 quantization halves memory traffic vs FP16.
2. **Use smaller input resolutions**: 640x480 vs 1920x1080 reduces feature map sizes.
3. **Batch processing**: Amortizes memory overhead across multiple inputs.
4. **Cap EMC frequency**: If bandwidth is not the bottleneck, lower EMC saves power.

```bash
# Profile memory bandwidth usage
$ sudo tegrastats --interval 500 | while read line; do
    echo "$line" | grep -oP 'EMC_FREQ \K[0-9]+%'
done

# If EMC utilization is consistently below 50%, cap the frequency
EMC_DEV="/sys/devices/platform/bus@0/31b0000.emc/devfreq/31b0000.emc"
$ echo 1600000000 | sudo tee ${EMC_DEV}/max_freq
```

### 9.5 Peripheral Power Management

Disable peripherals you are not using:

```bash
# Disable USB if not needed (saves ~0.3 W)
$ echo "suspend" | sudo tee /sys/bus/usb/devices/usb1/power/level
$ echo "suspend" | sudo tee /sys/bus/usb/devices/usb2/power/level

# Disable Wi-Fi if using Ethernet only
$ sudo nmcli radio wifi off
$ sudo rfkill block wifi

# Disable Bluetooth
$ sudo rfkill block bluetooth

# Disable display output if headless
$ echo "off" | sudo tee /sys/class/drm/card0-HDMI-A-1/status 2>/dev/null
```

---

## 10. Dynamic Power Management

### 10.1 Runtime PM Framework

Linux Runtime PM allows individual devices to enter low-power states when idle:

```bash
# Check runtime PM status of a device
$ cat /sys/devices/platform/gpu.0/power/runtime_status
suspended   # or "active"

# Check runtime PM configuration
$ cat /sys/devices/platform/gpu.0/power/control
auto   # "auto" means runtime PM is enabled

# Set autosuspend delay (ms before device suspends after going idle)
$ echo 500 | sudo tee /sys/devices/platform/gpu.0/power/autosuspend_delay_ms

# Enable runtime PM for a device
$ echo "auto" | sudo tee /sys/devices/platform/gpu.0/power/control
```

### 10.2 Autosuspend for Peripherals

Configure aggressive autosuspend for peripherals to save power between accesses:

```bash
# USB autosuspend (default is often 2000ms)
$ echo 100 | sudo tee /sys/bus/usb/devices/*/power/autosuspend_delay_ms 2>/dev/null

# I2C controller autosuspend
$ for dev in /sys/bus/platform/devices/*.i2c/power/; do
    echo "auto" | sudo tee ${dev}/control 2>/dev/null
    echo 100 | sudo tee ${dev}/autosuspend_delay_ms 2>/dev/null
done

# SPI controller autosuspend
$ for dev in /sys/bus/platform/devices/*.spi/power/; do
    echo "auto" | sudo tee ${dev}/control 2>/dev/null
    echo 100 | sudo tee ${dev}/autosuspend_delay_ms 2>/dev/null
done
```

### 10.3 Suspend to RAM (SC7/S2R)

The Orin Nano supports Suspend to RAM (called SC7 in NVIDIA terminology), which powers
down almost everything except DRAM self-refresh and a small always-on domain:

```bash
# Initiate suspend
$ sudo systemctl suspend

# Or directly through sysfs
$ echo mem | sudo tee /sys/power/state

# Check supported sleep states
$ cat /sys/power/state
freeze mem

# Power consumption in SC7: approximately 100-200 mW (module only)
```

### 10.4 Wake Sources

Configure what can wake the system from suspend:

```bash
# List wake-capable devices
$ cat /proc/acpi/wakeup  # May be empty on ARM; use sysfs instead

# Check if a device is a wake source
$ cat /sys/devices/platform/3610000.xhci/power/wakeup
enabled

# Enable wake from USB
$ echo "enabled" | sudo tee /sys/devices/platform/3610000.xhci/power/wakeup

# Enable wake from GPIO (e.g., button press)
# This is configured in the device tree pinctrl node:
#   nvidia,wake-gpio = <&gpio TEGRA234_MAIN_GPIO(A, 5) GPIO_ACTIVE_LOW>;

# Enable wake from RTC (timed wakeup)
$ sudo rtcwake -m mem -s 60  # Suspend and wake after 60 seconds
```

### 10.5 Suspend/Resume Latency

| Transition     | Typical Latency  | Notes                              |
|---------------|------------------|------------------------------------|
| Active -> SC7  | ~200-500 ms      | Includes device driver suspend     |
| SC7 -> Active  | ~300-800 ms      | DRAM recalibration, driver resume  |

For applications requiring faster wake, consider using CPU idle states (C-states)
instead of full suspend. The deepest C-state (C7) provides significant power savings
with microsecond-level wake latency.

---

## 11. Battery-Powered Operation

### 11.1 Power Budget Planning

For battery-powered Jetson Orin Nano deployments, power budgeting is critical:

```
Battery capacity planning example:
  Battery:       50 Wh (e.g., 4S LiPo, ~14.8V nominal)
  DC-DC loss:    ~10% (buck converter to 5V)
  Available:     45 Wh at module input

  Scenario A: Continuous inference at 15W mode
    Runtime = 45 Wh / 15 W = 3.0 hours

  Scenario B: Continuous inference at 7W mode
    Runtime = 45 Wh / 7 W = 6.4 hours

  Scenario C: Duty-cycled (30% active at 15W, 70% SC7 at 0.2W)
    Average power = 0.30 * 15 + 0.70 * 0.2 = 4.64 W
    Runtime = 45 Wh / 4.64 W = 9.7 hours
```

### 11.2 Duty-Cycling Inference

For applications that do not require continuous inference (e.g., periodic environmental
monitoring), duty-cycling dramatically extends battery life:

```bash
#!/bin/bash
# duty_cycle_inference.sh -- Run inference periodically, sleep between cycles.

INFERENCE_CMD="/opt/myapp/run_inference.sh"
ACTIVE_SECONDS=10     # Run inference for 10 seconds
SLEEP_SECONDS=50      # Sleep for 50 seconds (1 inference per minute)

while true; do
    # Wake up: lock clocks for consistent inference performance
    sudo jetson_clocks --store /tmp/clk_backup.conf
    sudo jetson_clocks

    # Run inference
    timeout ${ACTIVE_SECONDS} ${INFERENCE_CMD}

    # Restore DVFS and prepare for sleep
    sudo jetson_clocks --restore /tmp/clk_backup.conf

    # Enter low-power state
    sudo rtcwake -m mem -s ${SLEEP_SECONDS}
done
```

### 11.3 Aggressive Idle Power Reduction

To minimize power between inference cycles without full suspend:

```bash
#!/bin/bash
# low_power_idle.sh -- Enter lowest possible idle power state.

# Park all but one CPU core
for core in 1 2 3 4 5; do
    echo 0 > /sys/devices/system/cpu/cpu${core}/online
done

# Drop CPU to minimum frequency
echo powersave > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# Drop EMC to minimum
EMC_DEV="/sys/devices/platform/bus@0/31b0000.emc/devfreq/31b0000.emc"
echo userspace > ${EMC_DEV}/governor
echo 204000000 > ${EMC_DEV}/min_freq

# Disable USB
for usb in /sys/bus/usb/devices/usb*/power/level; do
    echo "suspend" > ${usb} 2>/dev/null
done

# Result: idle power drops to approximately 1.5-2.0 W
```

### 11.4 Power Supply Considerations

When designing a battery power supply for the Orin Nano:

| Parameter           | Requirement                                 |
|--------------------|----------------------------------------------|
| Input voltage       | 5V-20V (module VDD_IN via carrier board)    |
| Peak current (5V)   | 3A minimum for 15W mode + carrier overhead  |
| Inrush current      | ~5A for <10ms during power-on               |
| Voltage ripple      | <100 mV peak-to-peak recommended            |
| Brown-out           | Module may hang if VDD_IN drops below 4.5V  |

Use a buck converter with fast transient response. The Orin Nano's power consumption
can change by several watts within milliseconds when GPU inference starts, causing
voltage dips on weak supplies.

---

## 12. Power Profiling for AI Workloads

### 12.1 Methodology

A rigorous power profiling methodology for AI workloads:

1. Set a known power mode (nvpmodel).
2. Optionally lock clocks (jetson_clocks) for reproducibility.
3. Let the system reach thermal steady state (5-10 minutes warm-up).
4. Begin power logging.
5. Run the inference workload for a measured duration.
6. Compute average power, peak power, and energy per inference.

```bash
#!/bin/bash
# profile_inference_power.sh -- Profile power consumption of an inference workload.

MODEL=$1          # e.g., resnet50.engine
DURATION=120      # seconds
LOG_FILE="power_profile_$(date +%Y%m%d_%H%M%S).csv"

if [ -z "$MODEL" ]; then
    echo "Usage: $0 <tensorrt_engine_file>"
    exit 1
fi

# Step 1: Set power mode
sudo nvpmodel -m 0
sleep 2

# Step 2: Lock clocks for reproducible results
sudo jetson_clocks --store /tmp/clk.bak
sudo jetson_clocks
sleep 2

# Step 3: Warm-up (let thermals stabilize)
echo "Warming up for 60 seconds..."
/usr/src/tensorrt/bin/trtexec --loadEngine=${MODEL} --duration=60 --streams=1 \
    > /dev/null 2>&1

# Step 4: Start power logging in background
echo "Starting power log..."
sudo bash -c '
INA_HW=$(find /sys/bus/i2c/drivers/ina3221/*/hwmon -maxdepth 1 -name "hwmon*" | head -1)
echo "timestamp_ms,vdd_in_mw,temp_cpu_mc" > '"${LOG_FILE}"'
START=$(date +%s%N)
while true; do
    NOW=$(( ($(date +%s%N) - START) / 1000000 ))
    P=$(cat ${INA_HW}/in_power0_input 2>/dev/null || echo 0)
    T=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo 0)
    echo "${NOW},${P},${T}" >> '"${LOG_FILE}"'
    sleep 0.1
done
' &
LOG_PID=$!

# Step 5: Run workload
echo "Running inference for ${DURATION} seconds..."
/usr/src/tensorrt/bin/trtexec --loadEngine=${MODEL} --duration=${DURATION} --streams=1 \
    2>&1 | tee /tmp/trtexec_output.log

# Step 6: Stop logging
sudo kill ${LOG_PID} 2>/dev/null
wait ${LOG_PID} 2>/dev/null

# Step 7: Compute statistics
echo ""
echo "=== Power Profile Results ==="
python3 -c "
import csv
powers = []
with open('${LOG_FILE}') as f:
    reader = csv.DictReader(f)
    for row in reader:
        p = int(row['vdd_in_mw'])
        if p > 0:
            powers.append(p)

if powers:
    avg = sum(powers) / len(powers)
    peak = max(powers)
    print(f'Average power:  {avg:.0f} mW ({avg/1000:.2f} W)')
    print(f'Peak power:     {peak:.0f} mW ({peak/1000:.2f} W)')
    print(f'Samples:        {len(powers)}')
    duration_s = len(powers) * 0.1
    energy_j = avg / 1000.0 * duration_s
    print(f'Total energy:   {energy_j:.1f} J over {duration_s:.1f} s')
"

# Restore clocks
sudo jetson_clocks --restore /tmp/clk.bak
echo "Profile saved to ${LOG_FILE}"
```

### 12.2 Perf-Per-Watt Metrics

The key metric for edge AI is throughput per watt:

```
Perf/Watt = Throughput (fps or inferences/sec) / Average Power (W)
```

Example comparison table for common models on Orin Nano 8GB, 15W mode:

| Model              | Precision | FPS    | Avg Power | Perf/Watt  |
|-------------------|-----------|--------|-----------|------------|
| ResNet-50          | FP16      | ~180   | 8.5 W     | 21.2       |
| ResNet-50          | INT8      | ~320   | 9.0 W     | 35.6       |
| YOLOv8-S (det)     | FP16      | ~90    | 9.2 W     | 9.8        |
| YOLOv8-S (det)     | INT8      | ~150   | 9.5 W     | 15.8       |
| MobileNetV2        | INT8      | ~700   | 7.0 W     | 100.0      |
| EfficientNet-B0    | INT8      | ~450   | 7.5 W     | 60.0       |
| SSD-MobileNet-V2   | INT8      | ~250   | 8.0 W     | 31.3       |

### 12.3 Comparing Models by Energy Efficiency

For applications where you need N inferences per hour, energy-per-inference is more
meaningful than throughput:

```
Energy/Inference (mJ) = Average Power (mW) / Throughput (inferences/sec)
```

```python
#!/usr/bin/env python3
"""energy_compare.py -- Compare models by energy per inference."""

models = [
    # (name,        precision, fps,  avg_power_w)
    ("ResNet-50",   "FP16",    180,  8.5),
    ("ResNet-50",   "INT8",    320,  9.0),
    ("YOLOv8-S",    "FP16",    90,   9.2),
    ("YOLOv8-S",    "INT8",    150,  9.5),
    ("MobileNetV2", "INT8",    700,  7.0),
]

print(f"{'Model':<20s} {'Prec':<6s} {'FPS':>6s} {'Power':>7s} {'mJ/inf':>8s}")
print("-" * 52)
for name, prec, fps, power in models:
    energy_mj = (power * 1000) / fps
    print(f"{name:<20s} {prec:<6s} {fps:>6.0f} {power:>6.1f}W {energy_mj:>7.1f}")
```

Output:

```
Model                Prec     FPS   Power   mJ/inf
----------------------------------------------------
ResNet-50            FP16      180    8.5W    47.2
ResNet-50            INT8      320    9.0W    28.1
YOLOv8-S             FP16       90    9.2W   102.2
YOLOv8-S             INT8      150    9.5W    63.3
MobileNetV2          INT8      700    7.0W    10.0
```

---

## 13. Thermal Design for Enclosures

### 13.1 Thermal Resistance Model

The thermal path from the SoC die to ambient air can be modeled as a series of
thermal resistances:

```
T_junction = T_ambient + P_dissipated * (R_jc + R_cs + R_sa)

Where:
  T_junction   = SoC die temperature (C)
  T_ambient    = Surrounding air temperature (C)
  P_dissipated = Power dissipated by the SoC (W)
  R_jc         = Junction-to-case thermal resistance (C/W)  -- SoC package
  R_cs         = Case-to-sink thermal resistance (C/W)      -- thermal pad/paste
  R_sa         = Sink-to-ambient thermal resistance (C/W)   -- heatsink + airflow
```

Typical values for the Orin Nano module:

| Parameter | Value            | Notes                              |
|-----------|------------------|------------------------------------|
| R_jc      | ~1.5 C/W         | Fixed by NVIDIA module design      |
| R_cs      | ~0.2-0.5 C/W     | Depends on thermal interface used  |
| R_sa      | ~2-10 C/W        | Depends on heatsink design         |

### 13.2 Heatsink Selection

Example calculation for a sealed enclosure at 50 C ambient:

```
Target:  T_junction <= 87 C (NVIDIA recommended max for sustained operation)
Budget:  T_junction - T_ambient = 87 - 50 = 37 C thermal headroom
Power:   15 W at full 15W mode

Required total R_theta:
  R_total = 37 / 15 = 2.47 C/W

  R_jc = 1.5 C/W (fixed)
  R_cs = 0.3 C/W (good thermal pad)
  R_sa = 2.47 - 1.5 - 0.3 = 0.67 C/W

A heatsink with R_sa <= 0.67 C/W at natural convection is challenging.
Options:
  a) Add a fan: forced convection reduces R_sa to 0.3-0.5 C/W
  b) Reduce power mode to 7W: R_sa = (37 - 7*1.8) / 7 = 3.49 C/W (easy)
  c) Use the enclosure as heatsink: bond module to metal enclosure wall
```

### 13.3 Passive vs Active Cooling

| Approach        | R_sa Range | Pros                           | Cons                          |
|----------------|------------|--------------------------------|-------------------------------|
| Passive (fins)  | 3-10 C/W  | No moving parts, silent, no fan failure | Large, heavy, limited power budget |
| Active (fan)    | 0.3-2 C/W | Compact, high cooling capacity | Noise, fan wear, dust ingress |
| Enclosure-bonded| 1-3 C/W   | No heatsink needed, sealed     | Enclosure surface gets hot    |
| Heat pipe       | 0.5-1 C/W | Remote heat dissipation        | Cost, design complexity       |

### 13.4 Thermal Interface Materials

The interface between the Orin Nano module and the heatsink is critical:

| Material           | Thermal Conductivity | Thickness | R_cs      | Notes              |
|-------------------|---------------------|-----------|-----------|---------------------|
| Thermal paste      | 4-12 W/mK           | ~50 um    | 0.05-0.15 | Best performance, messy |
| Thermal pad (soft) | 3-6 W/mK            | 0.5-1 mm  | 0.2-0.5   | Easy assembly       |
| Thermal pad (high) | 8-17 W/mK           | 0.5-1 mm  | 0.1-0.2   | Expensive           |
| Gap filler         | 1-5 W/mK            | 1-5 mm    | 0.5-2.0   | Fills large gaps    |

For the Orin Nano module, a 1 mm soft thermal pad (e.g., Bergquist Gap Pad 5000S35 or
Fujipoly SARCON) provides good performance with easy assembly. The module's thermal
interface surface has defined keep-out zones -- consult the Orin Nano module datasheet
for the exact TIM placement area.

### 13.5 Ambient Temperature Derating

NVIDIA specifies the Orin Nano for operation from -25 C to +80 C (module surface
temperature). In practice, the useful operating range depends on your cooling solution:

```
Maximum Sustainable Power vs. Ambient Temperature
(Assuming R_total = 4 C/W, passive cooling)

Ambient (C)  | Max Sustained Power (W) | Notes
-------------|-------------------------|---------------------------
25           | (87-25)/4 = 15.5 W      | Full 15W mode possible
35           | (87-35)/4 = 13.0 W      | 15W mode, some headroom
45           | (87-45)/4 = 10.5 W      | 15W mode may throttle
50           | (87-50)/4 =  9.3 W      | 7W mode recommended
60           | (87-60)/4 =  6.8 W      | 7W mode, tight
70           | (87-70)/4 =  4.3 W      | Custom low-power mode needed
```

---

## 14. Custom Fan Control

### 14.1 Simple Threshold-Based Fan Script

```bash
#!/bin/bash
# simple_fan_control.sh -- Threshold-based fan control for Orin Nano.
# Run as root. Adjust FAN_PWM path for your board.

FAN_PWM="/sys/class/hwmon/hwmon2/pwm1"
TEMP_ZONE="/sys/class/thermal/thermal_zone0/temp"

# Thresholds (millidegrees C)
THRESH_OFF=40000       # Below 40C: fan off
THRESH_LOW=50000       # 40-50C: low speed
THRESH_MED=60000       # 50-60C: medium speed
THRESH_HIGH=70000      # 60-70C: high speed
                       # Above 70C: full speed

# PWM values (0-255)
PWM_OFF=0
PWM_LOW=80
PWM_MED=140
PWM_HIGH=200
PWM_FULL=255

# Take manual control
echo 1 > ${FAN_PWM}_enable

cleanup() {
    echo "Restoring automatic fan control..."
    echo 2 > ${FAN_PWM}_enable
    exit 0
}
trap cleanup SIGINT SIGTERM

echo "Custom fan control active. Ctrl+C to stop."

while true; do
    TEMP=$(cat ${TEMP_ZONE})

    if [ ${TEMP} -lt ${THRESH_OFF} ]; then
        PWM=${PWM_OFF}
    elif [ ${TEMP} -lt ${THRESH_LOW} ]; then
        PWM=${PWM_LOW}
    elif [ ${TEMP} -lt ${THRESH_MED} ]; then
        PWM=${PWM_MED}
    elif [ ${TEMP} -lt ${THRESH_HIGH} ]; then
        PWM=${PWM_HIGH}
    else
        PWM=${PWM_FULL}
    fi

    echo ${PWM} > ${FAN_PWM}
    sleep 2
done
```

### 14.2 PID-Based Fan Controller

A PID controller provides smoother fan operation with less audible speed changes:

```python
#!/usr/bin/env python3
"""pid_fan_control.py -- PID-based fan controller for Jetson Orin Nano.
Run as root: sudo python3 pid_fan_control.py
"""

import time
import signal
import sys

# Configuration
FAN_PWM_PATH = "/sys/class/hwmon/hwmon2/pwm1"
TEMP_PATH = "/sys/class/thermal/thermal_zone0/temp"  # CPU thermal zone
TARGET_TEMP = 65.0    # Target temperature in Celsius
POLL_INTERVAL = 2.0   # Seconds between updates

# PID gains (tune these for your enclosure and heatsink)
KP = 8.0    # Proportional gain
KI = 0.5    # Integral gain
KD = 2.0    # Derivative gain

# PWM limits
PWM_MIN = 0       # Minimum PWM (fan off)
PWM_MAX = 255     # Maximum PWM (full speed)
PWM_KICKSTART = 100  # Minimum PWM to start the fan spinning

# Anti-windup limit for integral term
INTEGRAL_MAX = 200.0

class PIDFanController:
    def __init__(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_pwm = 0
        self.fan_running = False

    def read_temp(self):
        """Read SoC temperature in Celsius."""
        with open(TEMP_PATH) as f:
            return int(f.read().strip()) / 1000.0

    def write_pwm(self, value):
        """Write PWM value (0-255) to fan."""
        value = max(PWM_MIN, min(PWM_MAX, int(value)))
        with open(FAN_PWM_PATH, "w") as f:
            f.write(str(value))
        self.prev_pwm = value
        return value

    def enable_manual_control(self):
        """Switch fan to manual PWM control."""
        with open(FAN_PWM_PATH + "_enable", "w") as f:
            f.write("1")

    def restore_auto_control(self):
        """Restore fan to automatic thermal governor control."""
        with open(FAN_PWM_PATH + "_enable", "w") as f:
            f.write("2")

    def update(self):
        """Run one PID update cycle."""
        temp = self.read_temp()
        error = temp - TARGET_TEMP

        # PID terms
        p_term = KP * error
        self.integral += error * POLL_INTERVAL
        self.integral = max(-INTEGRAL_MAX, min(INTEGRAL_MAX, self.integral))
        i_term = KI * self.integral
        d_term = KD * (error - self.prev_error) / POLL_INTERVAL
        self.prev_error = error

        # Calculate PWM output
        output = p_term + i_term + d_term

        # Below target with integral wound down: allow fan off
        if temp < (TARGET_TEMP - 5.0) and output < PWM_KICKSTART:
            pwm = PWM_MIN
            self.fan_running = False
        elif output > 0:
            # Ensure minimum PWM to overcome fan stiction
            pwm = max(PWM_KICKSTART, output) if not self.fan_running else output
            self.fan_running = (pwm >= PWM_KICKSTART)
        else:
            pwm = PWM_MIN
            self.fan_running = False

        actual_pwm = self.write_pwm(pwm)
        return temp, actual_pwm, p_term, i_term, d_term

    def run(self):
        """Main control loop."""
        self.enable_manual_control()
        print(f"PID fan controller started. Target: {TARGET_TEMP} C")
        print(f"{'Time':>6s} {'Temp':>6s} {'PWM':>4s} {'P':>7s} {'I':>7s} {'D':>7s}")

        start = time.monotonic()
        try:
            while True:
                temp, pwm, p, i, d = self.update()
                elapsed = time.monotonic() - start
                print(f"{elapsed:6.0f} {temp:6.1f} {pwm:4d} {p:7.1f} {i:7.1f} {d:7.1f}")
                time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            pass
        finally:
            self.restore_auto_control()
            print("\nRestored automatic fan control.")

def main():
    controller = PIDFanController()

    def signal_handler(sig, frame):
        controller.restore_auto_control()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    controller.run()

if __name__ == "__main__":
    main()
```

### 14.3 Quiet Operation Strategies

For noise-sensitive environments (medical devices, home automation, conference rooms):

1. **Use a large passive heatsink**: Eliminate the fan entirely if the power budget
   allows (typically 7W mode only at reasonable ambient temperatures).

2. **Slow fan curve with hysteresis**: Prevent the fan from cycling on and off:

```bash
# Add hysteresis: fan turns ON at 60C, OFF at 50C
# (10 degree hysteresis prevents rapid cycling)
```

3. **Cap fan PWM below audible threshold**: Many 5V fans become inaudible below
   40% duty cycle (~PWM 100). Cap the maximum PWM and accept higher temperatures:

```bash
# Cap fan at 40% duty cycle
$ echo 100 | sudo tee /sys/class/hwmon/hwmon2/pwm1
```

4. **Use a larger, slower fan**: A 60mm or 80mm fan at 30% duty moves more air with
   less noise than the stock 40mm fan at 80%.

5. **Reduce power mode**: Running at 7W instead of 15W reduces heat output by more
   than half, often eliminating the need for active cooling entirely.

### 14.4 Fan Failure Detection

Monitor fan tachometer feedback to detect failures:

```bash
# Read fan RPM (if tachometer is available)
$ cat /sys/class/hwmon/hwmon2/fan1_input
3500

# A reading of 0 RPM when PWM > 0 indicates a stuck or dead fan
```

```python
#!/usr/bin/env python3
"""fan_watchdog.py -- Alert on fan failure."""

import os
import time
import subprocess

FAN_RPM_PATH = "/sys/class/hwmon/hwmon2/fan1_input"
FAN_PWM_PATH = "/sys/class/hwmon/hwmon2/pwm1"
CHECK_INTERVAL = 10  # seconds
FAILURE_THRESHOLD = 3  # consecutive failures before alerting

consecutive_failures = 0

while True:
    try:
        with open(FAN_PWM_PATH) as f:
            pwm = int(f.read().strip())
        with open(FAN_RPM_PATH) as f:
            rpm = int(f.read().strip())

        if pwm > 50 and rpm == 0:
            consecutive_failures += 1
            if consecutive_failures >= FAILURE_THRESHOLD:
                # Fan commanded on but not spinning -- failure
                msg = f"FAN FAILURE DETECTED: PWM={pwm}, RPM={rpm}"
                print(msg)
                # Reduce power to prevent thermal runaway
                os.system("sudo nvpmodel -m 1")  # Switch to 7W mode
                # Send alert (customize for your alerting system)
                subprocess.run(["logger", "-p", "daemon.crit", msg])
        else:
            consecutive_failures = 0
    except (IOError, ValueError) as e:
        print(f"Error reading fan status: {e}")

    time.sleep(CHECK_INTERVAL)
```

---

## 15. Production Power Optimization

### 15.1 Disabling Unused Peripherals in Device Tree

For production devices, disable unused peripherals at the device tree level for
maximum power savings. This prevents the kernel from even initializing the hardware:

```dts
/* In your custom device tree overlay */

/* Disable USB 3.0 if not used */
&xusb {
    status = "disabled";
};

/* Disable DisplayPort if headless */
&display {
    status = "disabled";
};

/* Disable PCIe if not used */
&pcie0 {
    status = "disabled";
};

/* Disable HDMI */
&hdmi {
    status = "disabled";
};

/* Disable camera ISP if not using cameras */
&vi {
    status = "disabled";
};

/* Disable audio */
&tegra_sound {
    status = "disabled";
};
```

Apply the overlay during the build process:

```bash
# Compile the overlay
$ dtc -I dts -O dtb -o custom-overlay.dtbo custom-overlay.dts

# Apply to the base DTB
$ fdtoverlay -i tegra234-p3767-0003-p3768-0000-a0.dtb \
    -o merged.dtb custom-overlay.dtbo

# Flash with the modified DTB
$ sudo cp merged.dtb /boot/dtb/
```

### 15.2 Custom nvpmodel for Specific Workloads

Production devices rarely need the generic 15W or 7W modes. Create a mode tuned for
your exact workload profile:

```ini
# Example: Security camera running YOLOv8 at 15 fps target
# Profile shows: GPU is the bottleneck, CPU usage is low (pre/post processing)
# DLA handles half the layers, GPU handles the rest

< POWER_MODEL ID=3 NAME=MODE_SECURITY_CAM >
  # Only 2 CPU cores needed for pre/post processing
  CPU_ONLINE CORE_0 1
  CPU_ONLINE CORE_1 1
  CPU_ONLINE CORE_2 0
  CPU_ONLINE CORE_3 0
  CPU_ONLINE CORE_4 0
  CPU_ONLINE CORE_5 0

  # CPU doesn't need max frequency
  CPU_A78_0 MIN_FREQ 729600
  CPU_A78_0 MAX_FREQ 1036800

  # GPU at moderate clock (enough for 15 fps)
  GPU MIN_FREQ 0
  GPU MAX_FREQ 420000
  GPU_POWER_GATING GPU_PG_MASK 0

  # DLA at moderate clock
  DLA0_CORE MAX_FREQ 414700
  DLA0_FALCON MAX_FREQ 294400

  # Memory -- measured that 1600 MHz is sufficient
  EMC MAX_FREQ 1600000

  # Measured total: ~6W average
  POWER_BUDGET CPU 8000
< /POWER_MODEL >
```

### 15.3 Power-Gating Unused Engines

Runtime power-gating of engines not needed by your application:

```bash
# Power-gate the video encoder (not needed for inference-only)
$ echo "auto" | sudo tee /sys/devices/platform/*/nvenc/power/control

# Power-gate the video decoder
$ echo "auto" | sudo tee /sys/devices/platform/*/nvdec/power/control

# Power-gate JPEG engine
$ echo "auto" | sudo tee /sys/devices/platform/*/nvjpg/power/control

# Verify engines are suspended
$ cat /sys/devices/platform/*/nvenc/power/runtime_status
suspended
```

### 15.4 Kernel Configuration for Power

Strip the kernel of unnecessary drivers and features for production:

```bash
# Disable kernel features that consume power
# (in .config or menuconfig)

# Disable debug features
CONFIG_DEBUG_INFO=n
CONFIG_DYNAMIC_DEBUG=n
CONFIG_FTRACE=n

# Disable unnecessary filesystems
CONFIG_EXT4_FS=n        # If using only rootfs on eMMC with a minimal FS
CONFIG_NFS_FS=n         # No network filesystem

# Reduce timer frequency (saves CPU wakeups)
CONFIG_HZ=100           # Instead of 250 or 1000

# Enable aggressive idle
CONFIG_CPU_IDLE=y
CONFIG_ARM_CPUIDLE=y
CONFIG_CPU_FREQ_DEFAULT_GOV_SCHEDUTIL=y
```

### 15.5 Systemd Service Optimization

Disable unnecessary services in production:

```bash
# Audit running services
$ systemctl list-units --type=service --state=running

# Disable services not needed for inference
$ sudo systemctl disable --now snapd.service
$ sudo systemctl disable --now ModemManager.service
$ sudo systemctl disable --now NetworkManager-wait-online.service
$ sudo systemctl disable --now avahi-daemon.service
$ sudo systemctl disable --now bluetooth.service
$ sudo systemctl disable --now cups.service
$ sudo systemctl disable --now wpa_supplicant.service  # If using Ethernet only

# Disable graphical desktop if headless
$ sudo systemctl set-default multi-user.target
$ sudo systemctl disable --now gdm3.service

# Power saved: 0.5-1.5 W from removing desktop + services
```

---

## 16. Monitoring and Alerting

### 16.1 jtop (jetson-stats)

`jtop` is the go-to monitoring tool for Jetson platforms. It provides a real-time
terminal UI showing CPU, GPU, memory, power, thermal, and fan status:

```bash
# Install jetson-stats
$ sudo pip3 install jetson-stats

# Run jtop
$ sudo jtop

# jtop provides multiple pages:
#   Page 1 (GPU):    GPU utilization, frequency, temperature
#   Page 2 (CPU):    Per-core CPU utilization and frequency
#   Page 3 (ENGINE): DLA, PVA, NVENC, NVDEC utilization
#   Page 4 (FAN):    Fan speed, PWM, temperature
#   Page 5 (POWER):  Per-rail power consumption
#   Page 6 (INFO):   JetPack version, board info, nvpmodel mode
```

Using jtop programmatically for logging:

```python
#!/usr/bin/env python3
"""jtop_logger.py -- Log Jetson stats to CSV using jetson-stats API."""

from jtop import jtop
import csv
import time
import sys

OUTPUT = sys.argv[1] if len(sys.argv) > 1 else "jtop_log.csv"
DURATION = int(sys.argv[2]) if len(sys.argv) > 2 else 300  # 5 minutes

with jtop() as jetson:
    with open(OUTPUT, "w", newline="") as csvfile:
        writer = None
        start = time.monotonic()

        while jetson.ok() and (time.monotonic() - start) < DURATION:
            row = {
                "timestamp": f"{time.monotonic() - start:.1f}",
                "cpu_temp": jetson.temperature.get("CPU", 0),
                "gpu_temp": jetson.temperature.get("GPU", 0),
                "gpu_util": jetson.gpu.get("val", 0) if jetson.gpu else 0,
                "power_total": jetson.power[0].get("tot", {}).get("power", 0)
                    if jetson.power else 0,
                "fan_speed": jetson.fan.get("speed", 0) if jetson.fan else 0,
                "ram_used_mb": jetson.memory.get("RAM", {}).get("used", 0) // (1024*1024)
                    if jetson.memory else 0,
            }

            # Add per-CPU utilization
            if jetson.cpu:
                for i, cpu in enumerate(jetson.cpu.values()):
                    if isinstance(cpu, dict):
                        row[f"cpu{i}_util"] = cpu.get("val", 0)
                        row[f"cpu{i}_freq"] = cpu.get("freq", {}).get("cur", 0)

            if writer is None:
                writer = csv.DictWriter(csvfile, fieldnames=row.keys())
                writer.writeheader()

            writer.writerow(row)

print(f"Logged {DURATION}s of data to {OUTPUT}")
```

### 16.2 Building a Power/Thermal Dashboard

For production monitoring, expose metrics via Prometheus and visualize with Grafana:

```python
#!/usr/bin/env python3
"""jetson_prometheus_exporter.py -- Expose Jetson power and thermal metrics."""

from prometheus_client import start_http_server, Gauge
import glob
import time
import os

# Define Prometheus metrics
TEMP_GAUGE = Gauge("jetson_temperature_celsius", "Temperature", ["zone"])
POWER_GAUGE = Gauge("jetson_power_milliwatts", "Power consumption", ["rail"])
FAN_GAUGE = Gauge("jetson_fan_pwm", "Fan PWM value (0-255)")
CPU_FREQ_GAUGE = Gauge("jetson_cpu_freq_khz", "CPU frequency", ["core"])
GPU_FREQ_GAUGE = Gauge("jetson_gpu_freq_hz", "GPU frequency")

def read_file(path):
    """Read and return contents of a sysfs file."""
    try:
        with open(path) as f:
            return f.read().strip()
    except (IOError, FileNotFoundError):
        return None

def collect_metrics():
    """Collect all metrics from sysfs."""
    # Temperatures
    for zone in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        name = read_file(os.path.join(zone, "type"))
        temp = read_file(os.path.join(zone, "temp"))
        if name and temp:
            TEMP_GAUGE.labels(zone=name).set(int(temp) / 1000.0)

    # Power (INA3221)
    for hwmon in glob.glob("/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*"):
        for power_file in sorted(glob.glob(os.path.join(hwmon, "in_power*_input"))):
            idx = power_file.split("in_power")[1].split("_")[0]
            rail_names = {
                "0": "VDD_IN",
                "1": "VDD_CPU_GPU_CV",
                "2": "VDD_SOC"
            }
            rail = rail_names.get(idx, f"rail_{idx}")
            val = read_file(power_file)
            if val:
                POWER_GAUGE.labels(rail=rail).set(int(val))

    # Fan
    for pwm in glob.glob("/sys/class/hwmon/*/pwm1"):
        val = read_file(pwm)
        if val:
            FAN_GAUGE.set(int(val))

    # CPU frequencies
    for i in range(6):
        freq = read_file(f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq")
        if freq:
            CPU_FREQ_GAUGE.labels(core=str(i)).set(int(freq))

    # GPU frequency
    for devfreq in glob.glob("/sys/devices/platform/gpu.0/devfreq/*/cur_freq"):
        val = read_file(devfreq)
        if val:
            GPU_FREQ_GAUGE.set(int(val))

def main():
    start_http_server(9100)
    print("Prometheus exporter running on :9100")
    while True:
        collect_metrics()
        time.sleep(1.0)

if __name__ == "__main__":
    main()
```

### 16.3 Automated Thermal Protection Script

A watchdog script that takes protective action before thermal shutdown:

```bash
#!/bin/bash
# thermal_watchdog.sh -- Automated thermal protection for production Jetson.
# Install as a systemd service for production use.

CRITICAL_TEMP=90000     # 90C: emergency power reduction
WARNING_TEMP=85000      # 85C: switch to low power mode
NORMAL_TEMP=75000       # 75C: safe to return to normal mode
CHECK_INTERVAL=5        # seconds

CURRENT_MODE=$(sudo nvpmodel -q 2>/dev/null | grep -oP 'MODE_\S+' || echo "unknown")
REDUCED=false
LOG_TAG="thermal-watchdog"

log() {
    logger -t ${LOG_TAG} "$1"
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1"
}

get_max_temp() {
    local max=0
    for zone in /sys/class/thermal/thermal_zone*/temp; do
        local t=$(cat ${zone} 2>/dev/null || echo 0)
        [ ${t} -gt ${max} ] && max=${t}
    done
    echo ${max}
}

log "Thermal watchdog started. Warning=${WARNING_TEMP}m, Critical=${CRITICAL_TEMP}m"

while true; do
    TEMP=$(get_max_temp)
    TEMP_C=$((TEMP / 1000))

    if [ ${TEMP} -ge ${CRITICAL_TEMP} ]; then
        log "CRITICAL: ${TEMP_C}C -- Forcing minimum power mode"
        sudo nvpmodel -m 1  # 7W mode
        # Additionally cap CPU to minimum
        for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq; do
            echo 729600 > ${cpu} 2>/dev/null
        done
        REDUCED=true

    elif [ ${TEMP} -ge ${WARNING_TEMP} ]; then
        if [ "${REDUCED}" = false ]; then
            log "WARNING: ${TEMP_C}C -- Switching to 7W mode"
            sudo nvpmodel -m 1
            REDUCED=true
        fi

    elif [ ${TEMP} -lt ${NORMAL_TEMP} ] && [ "${REDUCED}" = true ]; then
        log "NORMAL: ${TEMP_C}C -- Restoring original power mode"
        sudo nvpmodel -m 0
        REDUCED=false
    fi

    sleep ${CHECK_INTERVAL}
done
```

Systemd service file for the watchdog:

```ini
# /etc/systemd/system/thermal-watchdog.service
[Unit]
Description=Thermal Watchdog for Jetson Orin Nano
After=nvpmodel.service

[Service]
Type=simple
ExecStart=/usr/local/bin/thermal_watchdog.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
# Install the watchdog service
$ sudo cp thermal_watchdog.sh /usr/local/bin/
$ sudo chmod +x /usr/local/bin/thermal_watchdog.sh
$ sudo cp thermal-watchdog.service /etc/systemd/system/
$ sudo systemctl daemon-reload
$ sudo systemctl enable --now thermal-watchdog.service
```

### 16.4 Power Budget Alerting

Monitor total power consumption and alert when it exceeds expected bounds:

```python
#!/usr/bin/env python3
"""power_budget_alert.py -- Alert when power exceeds budget."""

import time
import subprocess

INA_POWER_PATH = None  # Will be auto-detected

# Find INA3221 power node
import glob
for path in glob.glob("/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/in_power0_input"):
    INA_POWER_PATH = path
    break

if not INA_POWER_PATH:
    print("ERROR: Cannot find INA3221 power monitor")
    exit(1)

BUDGET_MW = 15000    # 15W budget
ALERT_THRESHOLD = 0.95  # Alert at 95% of budget
SUSTAINED_SECONDS = 10   # Must exceed for this long

over_budget_start = None

while True:
    try:
        with open(INA_POWER_PATH) as f:
            power_mw = int(f.read().strip())

        if power_mw > BUDGET_MW * ALERT_THRESHOLD:
            if over_budget_start is None:
                over_budget_start = time.monotonic()
            elif (time.monotonic() - over_budget_start) > SUSTAINED_SECONDS:
                msg = (f"POWER ALERT: {power_mw}mW sustained > "
                       f"{BUDGET_MW * ALERT_THRESHOLD:.0f}mW for "
                       f"{SUSTAINED_SECONDS}s")
                print(msg)
                subprocess.run(["logger", "-p", "daemon.warning", msg])
                over_budget_start = None  # Reset after alerting
        else:
            over_budget_start = None

    except (IOError, ValueError):
        pass

    time.sleep(1.0)
```

---

## 17. Common Issues and Debugging

### 17.1 Thermal Throttling Symptoms

**Symptom**: Inference throughput drops after running for several minutes.

**Diagnosis**:

```bash
# Check if thermal throttling is active
$ cat /sys/class/thermal/thermal_zone*/temp
# If any zone is above 82000 (82C), throttling is likely active.

# Check if CPU frequency is being capped
$ cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
# Compare with scaling_max_freq. If cur < max, throttling is active.

# Check for throttling messages in dmesg
$ dmesg | grep -i throt
# Look for: "soctherm: OC ALARM" or "tegra_soctherm: throttle"

# Monitor in real time
$ sudo tegrastats --interval 1000
# Watch for declining clock frequencies and rising temperatures.
```

**Resolution**:

1. Improve cooling (larger heatsink, better thermal interface, add fan).
2. Reduce power mode (`nvpmodel -m 1`).
3. Reduce workload (lower inference rate, smaller model, lower resolution).
4. Improve airflow in enclosure.

### 17.2 Power Budget Exceeded Warnings

**Symptom**: System logs show power budget warnings; performance is inconsistent.

```bash
# Check for power capping messages
$ dmesg | grep -i "power budget\|power cap\|overcurrent"

# Check current power mode and budget
$ sudo nvpmodel -q --verbose
```

**Diagnosis**: The SoC's internal power estimator has detected that actual power
consumption exceeds the configured budget. The firmware reduces clocks to bring power
back within budget.

**Resolution**:

```bash
# If you have adequate cooling and power supply, increase the budget
# by using a higher power mode or creating a custom mode with a
# higher POWER_BUDGET value.

# Verify your power supply can handle the load
# Check VDD_IN voltage under load -- should not dip below 4.5V
$ cat /sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/in_voltage0_input
# Should read close to 5000 (5.0V) under load
```

### 17.3 Fan Failures

**Symptom**: Temperature rises rapidly despite fan being configured.

```bash
# Check if fan is receiving PWM signal
$ cat /sys/class/hwmon/hwmon2/pwm1
# Non-zero value means the controller is driving the fan.

# Check fan tachometer
$ cat /sys/class/hwmon/hwmon2/fan1_input
# 0 RPM with non-zero PWM = fan is stuck or disconnected.

# Check fan connector
# The Orin Nano devkit uses a 4-pin PWM fan header.
# Pin 1: GND (black)
# Pin 2: +5V (red)
# Pin 3: Tachometer (yellow)
# Pin 4: PWM control (blue)

# Test with manual full speed
$ echo 1 | sudo tee /sys/class/hwmon/hwmon2/pwm1_enable
$ echo 255 | sudo tee /sys/class/hwmon/hwmon2/pwm1
# If fan still does not spin, it is likely a hardware failure.
```

### 17.4 Unexpectedly High Idle Power

**Symptom**: System draws 5-7 W at idle when it should draw 3-4 W.

**Diagnosis checklist**:

```bash
# 1. Check if jetson_clocks is active (locks all clocks at max)
$ sudo jetson_clocks --show
# If all governors are "performance", jetson_clocks is active.
# Fix: sudo jetson_clocks --restore

# 2. Check for rogue processes keeping CPU busy
$ top -bn1 | head -20
# Look for processes consuming CPU. Common culprits:
#   - Xorg/gdm3 (desktop environment)
#   - thermald or power-daemon spinning
#   - Python scripts polling sensors too aggressively

# 3. Check GPU activity
$ cat /sys/devices/platform/gpu.0/devfreq/17000000.gpu/cur_freq
# Should be near 0 or minimum at idle. If at max, something is using the GPU.

# 4. Check if USB devices are preventing suspend
$ lsusb
# Disconnect unnecessary USB devices.

# 5. Check EMC frequency
$ cat /sys/devices/platform/bus@0/31b0000.emc/devfreq/31b0000.emc/cur_freq
# Should be at minimum at idle. High EMC freq at idle wastes ~0.5-1W.

# 6. Check nvpmodel mode
$ sudo nvpmodel -q
# Ensure you are in the intended power mode.
```

**Quick fix for high idle power**:

```bash
# Disable desktop environment
$ sudo systemctl set-default multi-user.target
$ sudo systemctl stop gdm3

# Ensure DVFS is active
$ echo schedutil | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Enable GPU runtime PM
$ echo auto | sudo tee /sys/devices/platform/gpu.0/power/control

# Kill unnecessary background services
$ sudo systemctl stop snapd cups avahi-daemon bluetooth
```

### 17.5 System Hangs Under Heavy Thermal Load

**Symptom**: System becomes unresponsive during sustained high-power workloads.

```bash
# Check if this is a thermal shutdown
# After reboot, examine the boot logs:
$ journalctl -b -1 | grep -i "thermal\|shutdown\|critical\|emergency"

# Check PMIC thermal status (if accessible)
$ cat /sys/class/thermal/thermal_zone*/type
# Look for PMIC-Die zone. PMIC overtemperature can also cause shutdown.

# Monitor all thermal zones before reproducing the issue
$ watch -n 0.5 'for z in /sys/class/thermal/thermal_zone*/; do
    echo "$(cat ${z}/type): $(cat ${z}/temp)"
done'
```

**Prevention**:

1. Lower the critical trip point by 5-10 degrees for earlier warning.
2. Implement the thermal watchdog script from Section 16.3.
3. Add serial console logging to capture thermal events before shutdown.
4. Ensure the PMIC also has adequate thermal relief (often overlooked).

### 17.6 Inconsistent Power Readings

**Symptom**: Power readings from tegrastats and INA3221 sysfs do not match, or readings
seem unreasonably high or low.

```bash
# tegrastats reads from the same INA3221 but may apply averaging or offsets.
# For raw readings, use sysfs directly:

# Read all three channels
$ for i in 0 1 2; do
    HWMON=$(find /sys/bus/i2c/drivers/ina3221/*/hwmon -maxdepth 1 -name "hwmon*" | head -1)
    V=$(cat ${HWMON}/in_voltage${i}_input 2>/dev/null || echo "N/A")
    I=$(cat ${HWMON}/in_current${i}_input 2>/dev/null || echo "N/A")
    P=$(cat ${HWMON}/in_power${i}_input 2>/dev/null || echo "N/A")
    echo "Channel ${i}: ${V} mV, ${I} mA, ${P} mW"
done

# Check if the INA3221 sense resistor values are correct
# (misconfigured shunt resistance gives wrong current/power readings)
$ cat /sys/bus/i2c/drivers/ina3221/*/shunt_resistor* 2>/dev/null
```

**Notes on measurement accuracy**:

- INA3221 has approximately 1% accuracy on voltage and 1.5% on current.
- Software power readings include PMIC conversion losses (the SoC receives less
  power than VDD_IN shows).
- For precise measurements, use an external power analyzer (e.g., Keithley or
  Monsoon) on the VDD_IN rail.

### 17.7 nvpmodel Fails to Switch Modes

**Symptom**: `nvpmodel -m <mode>` returns an error or the mode does not take effect.

```bash
# Check the nvpmodel service status
$ sudo systemctl status nvpmodel.service

# Examine the configuration file for syntax errors
$ sudo nvpmodel -p /etc/nvpmodel.conf --verbose

# Check if another process holds the nvpmodel lock
$ ls -la /var/lock/nvpmodel

# Force remove stale lock (use with caution)
$ sudo rm /var/lock/nvpmodel/nvpmodel

# Retry
$ sudo nvpmodel -m 0
```

Common causes:

1. **Syntax error in nvpmodel.conf**: Validate the file after any edits. The parser is
   strict about whitespace and keywords.
2. **Stale lock file**: If the system crashed during a mode switch, the lock file may
   persist. Remove it manually.
3. **Kernel driver not loaded**: Some nvpmodel operations require specific kernel modules
   (e.g., `tegra_cpufreq`). Check `lsmod | grep tegra`.

### 17.8 Debugging Power States with Kernel Tracing

For deep debugging of power management issues, use ftrace:

```bash
# Enable cpufreq tracing
$ echo 1 | sudo tee /sys/kernel/debug/tracing/events/power/cpu_frequency/enable
$ echo 1 | sudo tee /sys/kernel/debug/tracing/events/power/cpu_idle/enable

# Enable GPU devfreq tracing
$ echo 1 | sudo tee /sys/kernel/debug/tracing/events/devfreq/devfreq_frequency/enable

# Enable thermal tracing
$ echo 1 | sudo tee /sys/kernel/debug/tracing/events/thermal/thermal_temperature/enable

# Start tracing
$ echo 1 | sudo tee /sys/kernel/debug/tracing/tracing_on

# Run your workload...
sleep 10

# Stop tracing
$ echo 0 | sudo tee /sys/kernel/debug/tracing/tracing_on

# Read the trace
$ sudo cat /sys/kernel/debug/tracing/trace | head -100

# Example trace output:
#  trtexec-1234  [003] .... 45.123456: cpu_frequency: state=1510400 cpu_id=3
#  kworker/0:2   [000] .... 45.234567: thermal_temperature: thermal_zone=CPU-therm
#                                        id=0 temp_prev=65000 temp=66500
#  kworker/1:0   [001] .... 45.345678: devfreq_frequency: parent=17000000.gpu
#                                        freq=625000000 prev_freq=420000000 load=85
```

### 17.9 Quick Reference: Key Sysfs Paths

```
Power Management:
  /etc/nvpmodel.conf                                    nvpmodel configuration
  /var/lib/nvpmodel/                                    nvpmodel state persistence
  /sys/devices/system/cpu/cpu*/online                   CPU core online/offline
  /sys/devices/system/cpu/cpu*/cpufreq/                 CPU frequency control
  /sys/devices/platform/gpu.0/devfreq/17000000.gpu/     GPU frequency control
  /sys/devices/platform/bus@0/31b0000.emc/devfreq/      EMC frequency control

Thermal:
  /sys/class/thermal/thermal_zone*/temp                 Temperature readings
  /sys/class/thermal/thermal_zone*/type                 Zone names
  /sys/class/thermal/thermal_zone*/trip_point_*_temp    Trip point temperatures
  /sys/class/thermal/thermal_zone*/trip_point_*_type    Trip point types
  /sys/class/thermal/thermal_zone*/policy               Thermal governor
  /sys/class/thermal/cooling_device*/cur_state          Cooling device state

Fan:
  /sys/class/hwmon/hwmon*/pwm1                          Fan PWM value (0-255)
  /sys/class/hwmon/hwmon*/pwm1_enable                   Fan control mode
  /sys/class/hwmon/hwmon*/fan1_input                    Fan RPM tachometer

Power Monitoring:
  /sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/           INA3221 power monitor
    in_power0_input                                      Power in mW
    in_voltage0_input                                    Voltage in mV
    in_current0_input                                    Current in mA

Runtime PM:
  /sys/devices/*/power/runtime_status                   Device power state
  /sys/devices/*/power/control                          auto/on
  /sys/devices/*/power/autosuspend_delay_ms             Autosuspend delay

Suspend:
  /sys/power/state                                      Available sleep states
```

### 17.10 Quick Reference: Essential Commands

```bash
# Power mode management
sudo nvpmodel -q                        # Query current mode
sudo nvpmodel -m 0                      # Set 15W mode
sudo nvpmodel -m 1                      # Set 7W mode
sudo nvpmodel -q --verbose              # Detailed mode info

# Clock management
sudo jetson_clocks                      # Lock all clocks at max
sudo jetson_clocks --show               # Show current clock state
sudo jetson_clocks --store FILE         # Save current state
sudo jetson_clocks --restore FILE       # Restore saved state

# Monitoring
sudo tegrastats --interval 1000         # Real-time system stats
sudo jtop                               # Interactive Jetson monitor

# Thermal
watch -n 1 cat /sys/class/thermal/thermal_zone*/temp  # Watch temps

# Power measurement
watch -n 1 cat /sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/in_power0_input

# Suspend
sudo rtcwake -m mem -s 60               # Suspend, wake after 60s
sudo systemctl suspend                  # Suspend immediately
```

---

## Summary

Power and thermal management on the Jetson Orin Nano is not a single configuration
step but a continuous engineering discipline. The key principles:

1. **Measure first**: Use INA3221 and tegrastats to understand your actual power
   profile before optimizing. Assumptions about where power goes are often wrong.

2. **Match mode to workload**: The 15W and 7W built-in modes are starting points.
   Custom nvpmodel modes tuned to your specific workload yield the best efficiency.

3. **Let DVFS work**: Avoid `jetson_clocks` in production unless latency jitter
   matters more than power savings. The DVFS governors are well-tuned and save
   significant power during idle and light-load periods.

4. **Offload to DLA**: For supported network architectures, DLA delivers 30-50%
   better perf-per-watt than the GPU. Use TensorRT's DLA targeting for power-
   constrained deployments.

5. **Design the thermal solution early**: The enclosure, heatsink, and fan strategy
   must be designed together with the power budget. A thermal bottleneck reduces
   effective compute capacity regardless of how well the software is optimized.

6. **Monitor in production**: Deploy power and thermal monitoring with alerting.
   Thermal throttling is silent performance regression -- you will not notice it
   without instrumentation.

7. **Duty-cycle when possible**: For applications that do not require continuous
   inference, duty-cycling with suspend-to-RAM can extend battery life by 3-5x
   compared to continuous operation.

---

## References

- NVIDIA Jetson Orin Nano Module Data Sheet (DA-11060-001)
- NVIDIA Jetson Orin Nano Developer Kit Carrier Board Specification
- NVIDIA Jetson Linux Developer Guide -- Power Management
- NVIDIA Jetson Linux Developer Guide -- Thermal Management
- T234 SoC Technical Reference Manual (TRM)
- Linux Kernel Documentation: cpu-freq, devfreq, thermal
- INA3221 Triple-Channel Power Monitor Datasheet (Texas Instruments)
