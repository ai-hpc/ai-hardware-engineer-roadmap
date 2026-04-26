# ESP32-LyraT I2S Microphone Capture on Jetson Orin Nano - Application Guide

> **Goal:** Use an **ESP32-LyraT audio dev kit** as a practical I2S microphone/audio frontend for the **Jetson Orin Nano 8GB / Super Developer Kit** 40-pin header, so you can test Jetson I2S capture, AHUB routing, and JetPack audio debugging before designing a custom microphone board.

**Hub:** [Multimedia](../Guide.md)  
**Audio foundation:** [Jetson Audio Setup and Development](../Jetson-Audio-Setup-and-Development/Guide.md)

---

## 1. What this application is testing

This is not a normal "plug in a USB mic" test.

This application tests the real embedded-audio path:

```text
microphone frontend
  -> codec / ADC
  -> I2S serial audio
  -> Jetson 40-pin header I2S2
  -> AHUB / ADMAIF
  -> ALSA capture
  -> recording, ASR, wake-word, or voice pipeline
```

The ESP32-LyraT is useful here because it already has:

- onboard microphones
- an `ES8388` audio codec
- an exposed I2S header
- ESP-ADF examples for configuring the board-side audio path

The important correction:

> Audio is not carried over I2C. On the LyraT, I2C controls the codec registers. The audio samples move over I2S.

So if you say "I2C mic" in this project, usually you mean one of these:

- **I2S microphone / I2S audio stream:** digital audio data path
- **I2C-controlled codec:** control path for a codec such as `ES8388`

---

## 2. Recommended first architecture

Use the LyraT as the audio frontend and let the ESP32 side configure the codec.

```text
ESP32-LyraT
  onboard mics
      |
      v
  ES8388 codec / ADC
      |
      | I2S clocks + ADC data on JP4
      v
Jetson Orin Nano 40-pin header
  I2S2 DIN / SCLK / FS
      |
      v
Jetson AHUB -> ADMAIF1 -> ALSA arecord
```

This is the lowest-risk bring-up model because:

- ESP32-LyraT already knows how to initialize the `ES8388`
- Jetson only has to prove that it can receive I2S
- you avoid fighting two hosts trying to control the same codec over I2C

Later, for a real product, replace the LyraT with a dedicated codec or ADC board that Jetson controls directly.

---

## 3. What not to do first

Do not start by trying to make Jetson directly control the LyraT `ES8388` over I2C.

That path is possible in theory, but it is a poor first test because:

- the `ES8388` is already wired to the ESP32 on the LyraT
- the ESP32 firmware normally owns codec initialization
- Jetson would need an ASoC codec driver and machine-driver binding
- two hosts on the same codec control path can conflict
- you may need board rework or firmware changes to fully tri-state the ESP32 side

For first bring-up, treat the LyraT like a configured I2S audio source.

---

## 4. Hardware prerequisites

- Jetson Orin Nano 8GB / Super Developer Kit
- JetPack 6.x / L4T 36.x or newer
- ESP32-LyraT V4.3 or similar LyraT variant
- USB cable for powering and flashing the LyraT
- jumper wires
- logic analyzer or oscilloscope, strongly recommended
- optional powered speakers or headphones for LyraT-side sanity tests

Voltage rule:

> Jetson 40-pin header signals are 3.3 V logic. Do not connect 5 V UART/I2S/control signals.

The LyraT I2S header signals are ESP32-side 3.3 V logic, so direct signal wiring is reasonable. Still share ground.

---

## 5. Jetson 40-pin I2S2 pins

For the Orin Nano developer kit 40-pin header, the audio guide uses the header-exposed `I2S2` path:

| Jetson function | 40-pin header pin | Direction in this test |
|---|---:|---|
| `I2S2_SCLK` / bit clock | `12` | LyraT -> Jetson if LyraT is master |
| `I2S2_FS` / LRCK / frame sync | `35` | LyraT -> Jetson if LyraT is master |
| `I2S2_DIN` | `38` | LyraT audio data -> Jetson |
| `I2S2_DOUT` | `40` | Jetson playback data -> LyraT, optional |
| `AUD_MCLK` | `7` | optional; do not connect for first capture test |
| `GND` | `6`, `9`, `14`, etc. | common ground |

For microphone capture, the minimum useful wiring is:

```text
LyraT bit clock  -> Jetson pin 12
LyraT LRCK       -> Jetson pin 35
LyraT ADC data   -> Jetson pin 38
LyraT GND        -> Jetson GND
```

Do not connect both sides as clock masters. If LyraT drives `SCLK` and `LRCK`, Jetson must be configured as the I2S clock slave for that DAI link.

---

## 6. ESP32-LyraT JP4 I2S header

Espressif documents the LyraT V4.3 I2S header `JP4` as exposing the board I2S signals:

| LyraT JP4 signal | ESP32 pin | Meaning for this test |
|---|---|---|
| `MCLK` | `GPIO0` | master clock; usually leave unconnected for first Jetson capture test |
| `SCLK` | `GPIO5` | I2S bit clock |
| `LRCK` | `GPIO25` | I2S left/right frame clock |
| `DSDIN` | `GPIO26` | data into ES8388 DAC, useful for playback into LyraT |
| `ASDOUT` | `GPIO35` | ADC data out from ES8388, useful for mic capture into Jetson |
| `GND` | `GND` | common reference |

For capture from LyraT microphones into Jetson:

| LyraT signal | Jetson signal | Jetson pin |
|---|---|---:|
| `SCLK` | `I2S2_SCLK` | `12` |
| `LRCK` | `I2S2_FS` | `35` |
| `ASDOUT` | `I2S2_DIN` | `38` |
| `GND` | `GND` | `6` / `9` / `14` |

Optional playback direction:

| Jetson signal | LyraT signal | Purpose |
|---|---|---|
| `I2S2_DOUT` pin `40` | `DSDIN` | send Jetson audio into LyraT DAC |

Keep playback disconnected until capture works.

---

## 7. Bring-up plan

Use this order. It prevents chasing software routing while the wire-level clocks are still wrong.

1. Prove Jetson internal audio routing.
2. Prove LyraT microphones and codec work on the LyraT side.
3. Make LyraT output a stable I2S mic stream.
4. Verify `SCLK`, `LRCK`, and `ASDOUT` on a scope or logic analyzer.
5. Enable Jetson 40-pin `I2S2` pinmux.
6. Configure Jetson audio route from `I2S2` to `ADMAIF1`.
7. Record raw audio with `arecord`.
8. Only then add GStreamer, wake-word, ASR, or beamforming.

---

## 8. Jetson-side setup

### Confirm JetPack and ALSA devices

```bash
cat /etc/nv_tegra_release
uname -a

aplay -l
arecord -l
amixer -c APE controls | grep -E 'I2S2|ADMAIF'
```

If `APE` does not exist, stop and fix the Jetson audio stack before wiring external hardware.

### Enable 40-pin I2S2 pinmux

Use Jetson-IO if your image supports it:

```bash
sudo /opt/nvidia/jetson-io/jetson-io.py
```

Select the 40-pin header configuration that enables `I2S2`, save the overlay, and reboot.

After reboot:

```bash
amixer -c APE controls | grep I2S2
```

### Internal loopback first

Before using the LyraT, prove that Jetson-side routing is alive:

```bash
amixer -c APE cset name="I2S2 Mux" "ADMAIF1"
amixer -c APE cset name="ADMAIF1 Mux" "I2S2"
amixer -c APE cset name="I2S2 Loopback" "on"

aplay -D hw:APE,0 test.wav &
arecord -D hw:APE,0 -r 48000 -c 2 -f S16_LE jetson-i2s2-loopback.wav
```

This does not prove the external pins work. It proves the AHUB route is sane.

---

## 9. LyraT-side setup

On the LyraT, use ESP-ADF or ESP-IDF firmware that:

- initializes the `ES8388`
- selects the onboard microphone input
- configures sample rate, usually `48000`
- configures stereo or mono capture
- drives I2S clocks consistently
- leaves the ADC data visible on `ASDOUT`

For the first Jetson test, choose a boring format:

| Parameter | Recommended first value |
|---|---|
| Sample rate | `48000` Hz |
| Channels | `2` |
| Sample format | `S16_LE` or 16-bit samples inside 32-bit I2S slots |
| Clock role | LyraT master, Jetson slave |
| Data source | LyraT onboard microphone path through `ES8388` ADC |

Expected wire-level clocks for a common stereo 48 kHz setup:

| Signal | Expected behavior |
|---|---|
| `LRCK` | 48 kHz |
| `SCLK` | commonly 1.536 MHz or 3.072 MHz depending on slot width |
| `ASDOUT` | toggles when microphone signal is active |

If `LRCK` or `SCLK` is not present, Jetson cannot capture anything.

---

## 10. External capture on Jetson

Once the LyraT is generating clocks and ADC data, route Jetson capture:

```bash
amixer -c APE cset name="ADMAIF1 Mux" "I2S2"
```

Try a conservative capture:

```bash
arecord -D hw:APE,0 -r 48000 -c 2 -f S16_LE lyrat-i2s-capture.wav
```

Inspect the file:

```bash
file lyrat-i2s-capture.wav
aplay lyrat-i2s-capture.wav
```

If playback is silent, inspect signal level:

```bash
sox lyrat-i2s-capture.wav -n stat
```

Useful quick checks:

- silence with no clocks: wiring or LyraT firmware problem
- silence with clocks: I2S data line, codec gain, or route problem
- distorted audio: bit depth, slot width, or master/slave mismatch
- one channel dead: codec input route or channel ordering problem

---

## 11. Important limitation: this may need a real ASoC binding

Jetson audio is not just "GPIO plus arecord."

For a robust external I2S capture card, Jetson normally needs:

- `I2S2` pinmux as SFIO
- an enabled `i2s2` controller
- a sound-card DAI link
- codec or dummy-codec binding
- correct master/slave clock configuration
- correct sample format and TDM/I2S settings

If you only use mixer commands and `arecord`, you may reach the limit of what the default Jetson sound card exposes. That is normal.

For a production board, use the full path from the audio deep dive:

```text
device tree overlay
  -> codec or dummy-codec node
  -> DAI link for I2S2
  -> AHUB route
  -> ALSA PCM
```

---

## 12. Direct Jetson control of LyraT `ES8388`

This is the advanced path, not the first bring-up path.

It would look like this:

```text
Jetson I2C -> ES8388 control registers
Jetson I2S2 <-> ES8388 I2S audio
ESP32 side disabled or kept from driving the same bus
```

Why it is hard:

- the LyraT was designed for ESP32 to own the codec
- the ESP32 and `ES8388` are already connected
- Jetson and ESP32 must not both drive I2C/I2S
- Linux must have a working `ES8388` codec driver and DAI binding
- board-level pull-ups and strap behavior may matter

Use this path only if you intentionally turn the LyraT into a codec breakout board.

For a real Jetson audio product, it is usually cleaner to design or buy a small `ES7210`, `ES8388`, `TLV320`, or similar codec board intended to be controlled by the Jetson.

---

## 13. Debug checklist

### Jetson checks

```bash
arecord -l
aplay -l
amixer -c APE controls | grep I2S2
amixer -c APE controls | grep ADMAIF
dmesg | grep -i -E 'asoc|tegra|i2s|audio'
```

### Wire checks

Use a scope or logic analyzer:

- `LRCK` toggles at the expected sample rate
- `SCLK` toggles continuously during capture
- `ASDOUT` changes when sound reaches the microphone
- no two devices are driving the same clock line
- ground is shared

### Format checks

Try common capture variants:

```bash
arecord -D hw:APE,0 -r 48000 -c 1 -f S16_LE test-mono.wav
arecord -D hw:APE,0 -r 48000 -c 2 -f S16_LE test-stereo.wav
arecord -D hw:APE,0 -r 48000 -c 2 -f S32_LE test-stereo-32.wav
```

Use the one that matches the LyraT firmware's I2S slot format.

---

## 14. What success looks like

Minimum success:

- LyraT outputs stable I2S clocks
- Jetson `I2S2` is enabled on the 40-pin header
- `ADMAIF1` routes from `I2S2`
- `arecord` creates a non-silent WAV file
- channel count and sample rate are correct

Better success:

- waveforms show clean clocks and data
- spoken audio is intelligible
- gain is not clipped
- left/right channels are understood
- recording works repeatedly after reboot

Product-level success:

- proper codec or ADC driver
- stable device-tree overlay
- reproducible ALSA route setup
- GStreamer pipeline
- application-level audio health checks
- wake-word or ASR pipeline using the captured audio

---

## 15. Next application steps

Once the raw capture path works:

- convert the `arecord` path into a GStreamer capture pipeline
- add WebRTC VAD or wake-word detection
- test local ASR such as Whisper or a smaller streaming ASR model
- replace LyraT with a dedicated Jetson-controlled codec board
- design a real microphone frontend for the smart-speaker product path

Example GStreamer capture starting point:

```bash
gst-launch-1.0 alsasrc device=hw:APE,0 ! \
  audio/x-raw,rate=48000,channels=2,format=S16LE ! \
  wavenc ! filesink location=lyrat-i2s-gst.wav
```

---

## 16. References

- [NVIDIA Jetson Linux Developer Guide - Audio Setup and Development](https://docs.nvidia.com/jetson/archives/r38.4/DeveloperGuide/SD/Communications/AudioSetupAndDevelopment.html)
- [NVIDIA Jetson Linux Developer Guide - Configuring the Jetson Expansion Headers](https://docs.nvidia.com/jetson/archives/r38.4/DeveloperGuide/HR/ConfiguringTheJetsonExpansionHeaders.html)
- [ESP32-LyraT V4.3 Hardware Reference](https://espressif-docs.readthedocs-hosted.com/projects/esp-adf/en/latest/design-guide/dev-boards/board-esp32-lyrat-v4.3.html)
- [ESP32-LyraT V4.3 schematic](https://dl.espressif.com/dl/schematics/esp32-lyrat-v4.3-schematic.pdf)
- [Jetson Audio Setup and Development](../Jetson-Audio-Setup-and-Development/Guide.md)
