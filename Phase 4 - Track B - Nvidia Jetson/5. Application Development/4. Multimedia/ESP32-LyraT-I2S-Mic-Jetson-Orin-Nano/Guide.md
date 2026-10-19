# LyraT to Jetson I2S: a clock contract and a historical capture

<div class="course-identity auto-course" style="--course-accent: #16a34a; --course-accent-rgb: 22, 163, 74;" markdown="1">
<div class="course-identity__icon">ELIM</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Jetson Track</p>
<p class="course-identity__title">Specialized course identity for ESP32-LyraT I2S Microphone Capture on Jetson Orin Nano - Application Guide.</p>
<p class="course-identity__meta">Artifact: Jetson integration demo · Measure: latency, memory, power, logs</p>
</div>
</div>

**Reviewed 18 October 2026.** This hardware case explains how an ESP32-LyraT V4.3 can expose an ES8388 microphone stream to a Jetson Orin Nano reference carrier. It preserves a reported May 2026 experiment without claiming that the firmware is bundled, currently reproduced, or bit-exact.

Begin with the [audio foundation](../Jetson-Audio-Setup-and-Development/Guide.md) and complete the [actual CPU audio lab](../Jetson-Audio-Setup-and-Development/CPU-Audio-Lab.md). That path has runnable code and tests. The physical integration below is an advanced extension requiring the exact board, complete firmware source and a matching Jetson BSP.

## 1. Understand the two buses

```text
ESP32 ── I2C control ──> ES8388 registers
microphones ──> ES8388 ADC ── I2S samples ──> Jetson I2S2 DIN
ESP32 ── BCLK/LRCK and codec MCLK ──> configured receivers
```

I2C configures the codec; it does not carry the microphone samples. In this architecture the ESP32 owns codec initialization and generates the sample clocks. Jetson consumes those clocks and captures the codec's digital output. Reading samples into ESP32 DMA is a useful observation, but Jetson can listen to the same data line independently.

Keep one owner for each driven signal and codec-control path. Connecting two clock outputs together is not a synchronization method. If Jetson is to own the codec instead, design the control and pin-isolation changes explicitly rather than leaving both processors active.

## 2. Verify the precise board and source

The checked ESP-ADF `release/v2.x` revision is `d6e1ef5ccf29ca52dcb8332a3232505366755012` (31 August 2026). Its LyraT V4.3 board pin function specifies MCLK GPIO0, BCLK GPIO5, WS GPIO25, data out GPIO26 and data in GPIO35. The directions are from the **ESP32** perspective. GPIO35 receives the codec's ADC output. [Pinned board source](https://github.com/espressif/esp-adf/blob/d6e1ef5ccf29ca52dcb8332a3232505366755012/components/audio_board/lyrat_v4_3/board_pins_config.c).

The official hardware guide has moved to the [multimedia boards documentation](https://docs.espressif.com/projects/esp-adf/en/latest/multimedia-boards/dev-boards/board-esp32-lyrat-v4.3.html). Match the printed board revision and schematic before using JP4 names. Other LyraT variants are separate designs.

This source inspection confirms pin definitions; it does not establish the exact firmware that produced the historical recording. A branch name is also insufficient for reproduction: record the ADF commit, IDF commit, submodules, component lock and configuration.

## 3. Wire the capture direction only

Power down before wiring, use the specified logic levels and share ground. For the stated V4.3/reference-carrier configuration:

| LyraT JP4 signal | ESP32 connection | Jetson 40-pin connection | Driver |
|---|---|---|---|
| SCLK/BCLK | GPIO5 | pin 12, I2S2 SCLK | ESP32 |
| LRCK/WS | GPIO25 | pin 35, I2S2 FS | ESP32 |
| ASDOUT | GPIO35 input | pin 38, I2S2 DIN | ES8388 ADC |
| GND | ground | pin 6 or another verified ground | shared reference |

MCLK GPIO0 may be essential to the **codec** even when no MCLK wire goes to Jetson. Keep Jetson's header 7 MCLK disconnected in this capture arrangement. The reference carrier's I2S2 pins are documented in the [versioned NVIDIA audio guide](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SD/Communications/AudioSetupAndDevelopment.html#board-interfaces).

Leave Jetson pin 40 DOUT disconnected. LyraT's DSDIN is already attached to ESP32 GPIO26, normally an output: adding Jetson playback to that net can create two drivers. A playback extension requires isolating or explicitly disabling the existing driver and validating the new route.

## 4. Make the clock and framing contract explicit

For a candidate 48 kHz stereo stream, specify all of these independently:

- 48,000 LRCK frames/s, and which device drives LRCK/BCLK.
- Two slots; actual wire slot width, which may exceed the 16 valid bits.
- Philips I2S versus left-justified framing, data delay and clock edge.
- Left/right slot assignment and sign/bit alignment.
- Codec MCLK and ADC configuration, including input selection and gain.
- Jetson's memory format, for example stereo S16_LE after any driver conversion.

With two 16-bit wire slots, BCLK is `48,000 × 2 × 16 = 1.536 MHz`. With two 32-bit slots it is `3.072 MHz`. A stereo S16_LE application stream is 192,000 bytes/s in either case. A DMA byte count therefore does not identify wire slot width or prove oscillator accuracy.

ESP-IDF's current channel API and legacy I2S driver are distinct interfaces. An ADF revision may wrap one of them. Use the API and configuration corresponding to the pinned project; combining old `i2s_config_t` fragments with new channel handles is not a working port. [ESP-IDF 5.5.4 I2S reference](https://docs.espressif.com/projects/esp-idf/en/v5.5.4/esp32/api-reference/peripherals/i2s.html).

## 5. Know which firmware is actually available

The old course referenced a local project named `lyrat_jp4_passthrough` under `esp-adf/examples/recorder/`. That project and its binary are **not included in this repository**. The upstream [minimal capture example request, issue 1607](https://github.com/espressif/esp-adf/issues/1607), was still open at this review. Do not run a flash command with invented artifact paths or infer that a successful upstream build contains this custom application.

A reproducible implementation must configure the V4.3 board and ES8388 microphone ADC, start the selected I2S clock/receive path, drain buffers if that driver requires it, and expose the intended JP4 signals. Retain the source/configuration and inspect the resulting waveforms. Build and flash using that project's generated instructions and partition layout, not hardcoded offsets copied from another build.

**Learner design exercise:** list the initialization order, source revision, signal ownership, format and observations needed before connecting Jetson. A correct answer includes codec register configuration, I2S clock start, a known audio stimulus, measured BCLK/LRCK, and a sample-format check. Writing a new driver is outside the beginner completion path.

## 6. Bring up Jetson in observable stages

Use the installed BSP's pinmux/device-tree workflow, not a generic “JetPack 6 or newer” promise. Configure I2S2 as the consumer of external clocks and validate the required DAI format. The old `cbm-cfm` terminology describes the codec-side clock provider in that binding; confirm the exact current binding rather than treating the text as a universal mixer value.

1. Discover the actual ALSA APE card and capture PCM with `arecord -l` and `arecord -L`.
2. Inspect available controls and establish the configured I2S2→ADMAIF route described in the [audio lesson](../Jetson-Audio-Setup-and-Development/Guide.md).
3. Verify the source clocks and waveform at the Jetson pins. An ALSA card or mixer control can exist before this physical path works.
4. Negotiate the actual rate/channel/container format, capture a known stimulus, and retain kernel/ALSA XRUN logs.
5. Decode the recording and check each channel separately. A clip that is audible can still have swapped channels or repeated samples.

A bounded capture command, after identifying a compatible configured PCM:

```bash
read -r -p 'Configured ALSA capture PCM: ' CAPTURE_PCM
arecord -D "$CAPTURE_PCM" --dump-hw-params -d 1 -r 48000 -c 2 -f S16_LE /tmp/lyrat-probe.wav
arecord -D "$CAPTURE_PCM" -d 5 -r 48000 -c 2 -f S16_LE lyrat.wav
```

These commands were syntax-checked only. Neither current Jetson capture nor firmware flashing was performed during this course review.

## 7. Interpret the historical evidence accurately

The earlier course recorded a **reported 11 May 2026** LyraT V4.3 experiment with custom `lyrat_jp4_passthrough` firmware, ESP-ADF `release/v2.x` and ESP-IDF 5.5.3. It reported audible speech on Jetson and the following file statistics. The original binary/source and WAV are not bundled, so these observations cannot be independently remeasured here.

| Reported observation | What it supports |
|---|---|
| 5.000 s, 48 kHz, stereo S16_LE | A declared format and duration |
| 480,000 scalar sample values | 240,000 stereo frames, if reported as interleaved samples |
| 960,044 bytes total | Consistent with 960,000 PCM bytes plus a minimal 44-byte header |
| Peak amplitude 0.233582; RMS 0.021088 | Digital amplitude statistics of that clip |
| Mean amplitude −0.000075 | That clip's mean; not a calibrated analog offset measurement |
| Speech reportedly audible | A useful functional listening observation |

WAV headers are not always 44 bytes; metadata and format chunks can differ. A valid file of the expected length does not prove zero lost samples: dropped samples can be replaced, repeated or concealed. An XRUN-free log is stronger evidence for that driver path, but bit-exact transport still needs a known sequence/reference.

The recorded RMS corresponds to about `20 log10(0.021088) = −33.5 dBFS` using a full-scale amplitude of 1. It is not microphone SPL or SNR. SNR requires an appropriate signal/noise measurement; SoX's “rough frequency” is not a reliable speech fundamental-frequency estimator.

Reported ESP32 per-second byte counters fluctuated around the expected 192,000 bytes/s. Buffer granularity, scheduling intervals and measurement windows can explain differences, but only clock/timestamp measurements can distinguish them from actual rate error. The counts do not prove a stable physical clock.

## 8. Strengthen the experiment

| Claim | Evidence needed |
|---|---|
| Correct sample format | Known signed values or calibrated tone; byte/order inspection |
| Correct channel identity | Stimulate one input at a time and compare both channels |
| Correct physical clock | Measure BCLK/LRCK and count samples over an independent time reference |
| No lost/repeated samples | Known sequence or continuity marker plus driver counters |
| Speech quality | Labelled audio, noise conditions and downstream error metrics |
| Long-run synchronization | Capture timestamps and drift across the intended duration |

Complete the CPU lab first, including the chunk-boundary and wrong-header failures. For hardware completion, attach a source/configuration manifest, wiring diagram, negotiated parameters, waveform evidence and a known-stimulus result. If the hardware is unavailable, submit the clock/frame calculations and explain exactly which claims remain unmeasured.

**Diagnostic answer:** a known-good ESP32 receive path narrows the search, but it does not prove the external wire, Jetson clock polarity, DAI framing or ALSA channel interpretation. Debug each boundary with a measurement.
