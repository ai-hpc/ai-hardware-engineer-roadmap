# Project: A Measured Wireless Video Link with FPGA Components

**Parent:** [Track A — FPGA development](../1.%20Xilinx%20FPGA%20Development/Guide.md). **Prerequisites:** checked RTL streams, Zynq memory/DMA ownership, Linux networking, and a working CPU video pipeline.

Build a small correct link first, then justify whether higher resolution, FPGA preprocessing, or a different MAC is useful. A 4K60 wireless system and an ASIC transition are advanced extensions, not an eight-week outcome guaranteed by a bill of materials. No camera, RF link, VCU, or FPGA board run has been performed for this course review.

## Choose a first completion target

Use one camera or deterministic test pattern, one encoder, one network path, one decoder, and one display or checked output file. Start with a mode that the actual camera, encoder, receiver, and link support together, such as a modest resolution/frame rate chosen after capability inspection. Make resolution, frame rate, codec/profile, bitrate, maximum queue age, and loss/recovery policy explicit.

**Example teaching requirement:** 640×360 at 30 frames/s, H.264 at a requested 2 Mbit/s, with a bounded two-frame application queue. These are initial assumptions to test, not hardware performance claims. Define whether overload drops the oldest frame, drops a whole encoded access unit, requests recovery, or stops the stream. Dropping arbitrary bytes of a compressed stream is not equivalent to dropping a raw frame.

Run the course's [CPU video/codec lab](../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Video-Codec-DeepStream/labs/README.md) before buying hardware. It provides executable queue and codec exercises; its host results do not validate an RF path or a board encoder.

## Milestone 1: Establish versions and actual capabilities

Create a manifest with board revision/FPGA part, camera/sensor mode, radio frontend, host OS, kernel, driver, FPGA image, codec implementation, and tools. Save commits/hashes and the selected upstream board profile. Use the documented imaging workflow for that exact board; do not copy an unverified download filename or a destructive `/dev/sdb` command from another person's machine.

The [openwifi software repository](https://github.com/open-sdr/openwifi) and [hardware repository](https://github.com/open-sdr/openwifi-hw) describe supported platforms, matching build flow, and known issues. The reviewed public README describes a 20 MHz baseline and lists distinct `antsdr`, `e310v2`, `antsdr_e200`, and `zcu102_fmcs2` targets. Similar product names do not make their images interchangeable. A ZCU106 port requires an actual porting assessment; it is not merely changing an MMCM to a PLL. Upstream best-case throughput figures are not guaranteed goodput for your channel, radio, modulation, aggregation, or workload.

On Raspberry Pi, use the current [camera software documentation](https://www.raspberrypi.com/documentation/computers/camera_software.html): inspect `rpicam-hello --list-cameras` and `rpicam-vid --help` on the target. Older `vcgencmd get_camera`, legacy camera-enable instructions, and OMX plugin recipes are not a universal modern bring-up path. Pi 5 uses software video encoding; do not assume a Pi 4 V4L2 hardware encoder exists on it.

Begin radio work with the upstream-supported configuration and an appropriate controlled test setup. Follow the board's RF power/connection requirements and the project's spectrum-use requirements. A bare cable between transmit/receive ports is not automatically a suitable attenuated test link.

**Pass condition:** the manifest identifies a documented image/board combination and captures actual camera/encoder/decoder capabilities. A shopping estimate is not completion evidence.

## Milestone 2: Make a local video path reproducible

First capture a bounded file, decode it, and count/inspect frames. Then run the same encoder/decoder over local networking. Record requested and negotiated caps, actual dimensions/frame rate, encoded bitrate, CPU load, dropped frames, and timestamps. A pipeline that runs forever is not a ten-second recording just because a comment says so; use the tool's duration/frame-count option and confirm clean end-of-stream.

A requested bitrate is not the exact output rate for every short window. A ten-second, 20 Mbit/s payload would be approximately `20,000,000 × 10 / 8 = 25,000,000 bytes` before container/network overhead, but real rate control can vary. Inspect the produced file instead of asserting an exact size.

For GStreamer, use `gst-inspect-1.0` to confirm each selected element. Do not prescribe `v4l2h264enc`/`v4l2h264dec` across boards without checking availability. A raw RTP/H.264 sender needs a receiver with matching RTP caps, payload type, clock rate, and depayload/parser/decoder path. Add an explicit jitter/reorder policy; a UDP socket buffer is not itself an RTP jitter buffer. `sync=false` changes sink scheduling, not every upstream queue or the sensor's latency.

**Pass condition:** a bounded, correctly decoded stream with a saved capability report and reproducible commands. Keep a known input so image corruption can be distinguished from network failure.

## Milestone 3: Route separate wired LANs correctly

For a routed design with a TX host connected to one SDR and an RX host connected to another, use separate subnets. Example addresses below are a topology exercise; configure only dedicated test devices using their supported network management.

| Device/interface | Example address | Network |
|---|---|---|
| TX host Ethernet | 10.10.1.2/24 | TX LAN |
| TX SDR Ethernet | 10.10.1.1/24 | TX LAN |
| TX SDR wireless | 10.10.3.1/30 | Radio transit |
| RX SDR wireless | 10.10.3.2/30 | Radio transit |
| RX SDR Ethernet | 10.10.2.1/24 | RX LAN |
| RX host Ethernet | 10.10.2.2/24 | RX LAN |

Forwarding and firewall rules must permit the intended traffic on both SDR routers. The route relationships are:

```text
TX host: 10.10.2.0/24 via 10.10.1.1
TX SDR:  10.10.2.0/24 via 10.10.3.2
RX SDR:  10.10.1.0/24 via 10.10.3.1
RX host: 10.10.1.0/24 via 10.10.2.1
```

NAT is not necessary for this complete private routed topology. If both separated Ethernet LANs use the same subnet, a host may ARP locally for the remote address instead of using the intended router. Bridging is a different design with different radio-mode requirements; do not silently mix the two.

Inspect `ip address`, `ip route`, and `ip route get <destination>` on each device. Test the immediate neighbor, then the opposite radio, then the opposite host. Pinging the TX radio's own wireless address from TX does not test the link. Verify the return route as carefully as the forward route.

Measure sustainable application goodput and packet loss at the offered rate with a bounded tool run before adding video. TCP throughput, UDP offered load, UDP received payload, and PHY rate are different quantities. Save actual values and conditions; do not claim 100 Mbit/s just from a modulation label.

## Milestone 4: Budget bytes, bursts, and age

Uncompressed packed 1920×1080 RGB8 at 60 frames/s needs:

```text
1920 × 1080 × 3 × 60 = 373,248,000 bytes/s
                         = 2.986 Gbit/s payload
```

NV12 at the same dimensions/rate uses 1.5 bytes/pixel in the ideal packed layout, about 1.493 Gbit/s. Stride padding and extra memory passes increase traffic. Compression changes network payload but introduces rate-control bursts, reference-frame dependencies, and recovery behavior.

If encoded video averages 20 Mbit/s while the measured sustainable received payload is 25 Mbit/s, the mean utilization is 80%; that alone does not prove a bounded queue. An I-frame burst, retransmissions, or a brief fade can create backlog. If the link stops for 100 ms while 20 Mbit/s continues arriving, backlog grows by 2 Mbit = 250,000 bytes. At 25 Mbit/s service and continuing 20 Mbit/s arrivals, draining that extra backlog needs 0.4 seconds under this simple constant-rate model.

At 60 frames/s, one frame interval is 16.67 ms. Capture exposure/readout, encoder buffering, transmission, decoder reordering, and display scanout can overlap or add delay depending on the implementation. Do not promise a 5 ms whole-frame capture step without specifying the sensor mode and timing reference.

**Pass condition:** report queue bounds, actual losses, recovery time, and source-to-display age under normal load and at least one impairment. A smooth average FPS can conceal old frames.

## Milestone 5: Measure latency with a meaningful clock boundary

A timestamp drawn onto a frame measures only the point at which that timestamp was generated. Comparing timestamps from different machines requires a documented synchronization method and error bound. One GPIO high on TX and another low on RX are different electrical signals, not automatically one pulse.

Choose a measurement design: a shared instrument observing stimulus and displayed response, synchronized hardware timestamps with quantified uncertainty, or an optical event plus display observation. Define exposure/readout/scanout position and measurement resolution. Report distribution, sample count, outliers, and dropped frames. Separate sensor-to-display latency from encoder-only time and network round-trip time.

## Milestone 6: Add one FPGA function

Insert a checked transform or packet adapter into the working pipeline. Use the [ready/valid RTL lab](../1.%20Xilinx%20FPGA%20Development/labs/README.md) and [HLS arithmetic lab](../4.%20High-Level%20Synthesis%20%28HLS%29/labs/README.md) as starting points. Preserve frame IDs, timestamps, valid-byte masks, and packet boundaries under backpressure.

For a width converter from 256 bits to 32 bits, a full input beat contains eight full output beats. A partial final input beat needs an explicit byte-valid rule; `TLAST` must move to the correct final accepted output beat. Holding `TVALID` without acceptance must not advance the fragment counter. Add tests for stalls on every fragment, reset mid-packet, maximum packet length, and malformed length before connecting DMA.

A VCU design needs the exact supported device, platform reference design, codec software, licensing, clocks, and memory integration. MIPI D-PHY is not a generic LVDS pin assignment: use the board schematic, supported IP/PHY design, lane mapping, voltage/bank constraints, and sensor-specific driver. Do not invent register APIs or sensor initialization from a few function names.

**Pass condition:** numerical/byte parity, backpressure/reset simulation, implemented timing/resource reports, then measured board behavior. The CPU baseline remains available for comparison.

## Optional extension: local time gates and a real MAC

Run the [local slot-gate RTL lab](labs/README.md). Its ten actual simulator variants check cycles, guard windows, wrap, reset, and invalid configurations. It does not align two radios' clocks or implement packet admission, carrier sensing, ACKs, retries, or a wireless timing protocol.

At an assumed 100 MHz, 100 µs is 10,000 clock cycles. A counter incrementing every clock does not count microseconds unless a proper conversion/enable exists. A real scheduled link also needs synchronization acquisition, drift bounds, propagation and radio delays, missed-beacon recovery, and an airtime admission rule. Keep the upstream working MAC until these requirements and tests exist.

## Optional extension: 4K and ASIC feasibility

Increase resolution only after measuring each stage's headroom. Select one supported sensor/codec/radio mode and revisit the traffic, memory, power, thermal, and latency budgets. A cooperative scheduler is not automatically faster or more deterministic than an RTOS/Linux implementation; long nonpreemptive work can block urgent events. A task comment assigning “core 1” does not create an SMP or AMP configuration.

For an ASIC study, inventory ownership, source availability, license/redistribution rights, synthesizability, memory macros, analog/PHY requirements, verification, test/DFT, timing corners, packaging, and software dependencies. A vendor hardened VCU or FPGA IP block cannot be assumed portable to an ASIC process. Node, die area, power, cost, and schedule require an actual architecture and implementation study. A defensible feasibility report is a valid outcome; invented tapeout dates and area targets are not.

## Submission

Deliver the compatibility manifest, topology, bounded local-video test, actual network and end-to-end measurements, failure/recovery trace, and FPGA evidence for stages you ran. Mark each stage host-only, simulated, implemented, or hardware-tested. Explain the next limiting resource with data. The project is complete at the explicitly chosen milestone, even if the evidence says a lower resolution or an unchanged MAC is the right design.
