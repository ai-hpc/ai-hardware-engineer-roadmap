# Jetson Audio Setup and Development - Project Guide

> **Goal:** Learn how audio really works on the **Jetson Orin Nano 8GB Developer Kit** so you can move from "a USB headset plays sound" to "a custom codec or smart-speaker audio path works on the 40-pin header" using NVIDIA's **ALSA + ASoC + AHUB** model.

**Hub:** [Multimedia](../Guide.md)  
**Primary official source:** [NVIDIA Jetson Linux Developer Guide - Audio Setup and Development](https://docs.nvidia.com/jetson/archives/r38.4/DeveloperGuide/SD/Communications/AudioSetupAndDevelopment.html)  
**Related local guides:** [Application Development](../../Guide.md) · [Orin Nano GPIO / SPI / I2C / CAN](../../../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-GPIO-SPI-I2C-CAN/Guide.md)

---

## 1. Why this project matters

If you want to build an **AI smart speaker**, **voice appliance**, **robot intercom**, **kiosk**, or **multimedia edge device**, audio bring-up on Jetson is not just:

```bash
aplay sound.wav
```

Jetson audio usually crosses several layers at once:

- Linux user space tools like `aplay`, `arecord`, and `amixer`
- ALSA and ASoC inside the kernel
- NVIDIA's **Audio Processing Engine (APE)** and **Audio Hub (AHUB)**
- physical interfaces like **USB audio**, **DisplayPort audio**, or **I2S on the 40-pin header**
- device tree, pinmux, clocks, and codec routing

This guide is the Jetson-side software course that pairs naturally with product ideas like:

- far-field microphone systems
- local voice assistants
- smart speakers
- audio capture and playback appliances

---

## 2. The mental model first

Think of Jetson audio as a layered pipeline:

```text
Your app
  |
  |-- aplay / arecord / amixer / GStreamer / PipeWire / PulseAudio
  |
ALSA user-space libraries
  |
ALSA / ASoC kernel stack
  |
  |-- machine driver
  |-- platform driver
  |-- codec driver
  |
APE / AHUB inside the Jetson SoC
  |
  |-- ADMAIF
  |-- I2S / DMIC / DSPK / Mixer / AMX / ADX / SFC / ASRC
  |
Physical interface
  |
  |-- USB headset / speaker
  |-- DP monitor audio
  |-- custom I2S codec on the 40-pin header
```

If audio fails, the bug is usually in one of these places:

- the sound card never registered
- the codec was not probed
- the pins are still GPIO instead of audio SFIO
- the device tree does not describe the route correctly
- the DAPM path is incomplete

---

## 3. What the Orin Nano dev kit exposes for audio

From NVIDIA's Jetson Linux guide, the **Jetson Orin Nano / Orin NX dev kit** exposes these main audio paths:

| Interface | What it gives you | Pinmux needed | Sound card |
|---|---|---:|---|
| 40-pin expansion header | `I2S2` on the header | Yes | `APE` |
| M.2 Key E slot | `I2S4` path | No | `APE` |
| DisplayPort | monitor audio through HDA | No | `HDA` |
| USB host | USB speakers, headsets, microphones | No special pinmux | USB card created when device is plugged in |

For the **40-pin header** on Orin Nano dev kits, NVIDIA documents these audio pins:

| Signal | Header pin |
|---|---:|
| `I2S2 FS` | `35` |
| `I2S2 SCLK` | `12` |
| `I2S2 DIN` | `38` |
| `I2S2 DOUT` | `40` |
| `AUD_MCLK` | `7` |

That is the main path you use when you want:

- an external DAC / ADC
- an audio codec board
- a custom amplifier board
- a product-specific speaker / microphone design

---

## 4. Start with the simplest audio path first

Do not start with a custom codec unless you really need to.

Use this order:

1. **USB audio**
   - easiest path
   - class-compliant devices often work immediately
   - best for first microphone-array experiments
2. **DisplayPort monitor audio**
   - good for "does Jetson audio basically work?" testing
3. **40-pin I2S + external codec**
   - correct path for real embedded product design
   - highest bring-up effort

For smart-speaker-style work, this usually means:

- use a **USB mic array** or USB headset first to validate software
- then move to **custom I2S playback/capture hardware**
- only then start full product tuning

---

## 5. First commands to run on a Jetson

Before changing anything, inspect the sound cards already present.

```bash
cat /proc/asound/cards
aplay -l
arecord -l
ls /dev/snd/pcmC?D*
```

What to look for:

- `HDA` usually means display audio such as **DP**
- `APE` usually means the Jetson **AHUB / ASoC** path
- a USB headset or microphone will usually appear as an extra USB sound card after plugging it in

This matters because the device names you use depend on the card:

- **DisplayPort / HDMI-style audio**

```bash
aplay -Dhw:HDA,<devID> sound.wav
```

- **USB audio**

```bash
aplay -Dhw:<cardID>,<devID> sound.wav
arecord -Dhw:<cardID>,<devID> -r 48000 -c 2 -f S16_LE test.wav
```

- **APE / AHUB path**

```bash
aplay -D hw:APE,0 sound.wav
arecord -D hw:APE,0 -r 48000 -c 2 -f S16_LE cap.wav
```

One important detail from NVIDIA's guide: **card indexes are not guaranteed to stay stable across boots**. Use `/proc/asound/cards` to check what your current board actually registered.

---

## 6. The key Jetson audio terms, made simple

### ALSA

This is the standard Linux audio framework.

You touch it every day with:

- `aplay`
- `arecord`
- `amixer`

### ASoC

This is ALSA's **System-on-Chip audio layer** for embedded processors.

On Jetson, ASoC is what connects:

- Jetson internal audio hardware
- external codecs
- sound-card registration
- user-space mixer controls

### APE

The **Audio Processing Engine** is Jetson's dedicated audio hardware block.

It lets Jetson handle audio with less CPU involvement than doing everything in general-purpose software.

### AHUB

The **Audio Hub** is the internal audio routing fabric inside the SoC.

It connects modules like:

- `ADMAIF`
- `I2S`
- `DMIC`
- `DSPK`
- `Mixer`
- `AMX`
- `ADX`
- `SFC`
- `ASRC`

### DAPM

**Dynamic Audio Power Management** is ALSA's routing and power graph.

In practice, this means:

- the kernel tracks which audio path is really active
- only the needed parts should power on
- if the route is incomplete, playback or capture may silently fail

### Device Tree

This is where Jetson learns:

- which codec exists
- which I2C or SPI bus talks to it
- which I2S controller is used
- what DAI links exist
- which widgets and routes make up the sound card

---

## 7. The three ASoC driver roles you must understand

NVIDIA's guide breaks Jetson ASoC into three important pieces.

### Platform driver

This is the Jetson-side block that handles PCM registration and memory/audio transfer.

On Jetson, **ADMAIF** is the big one to remember here.

### Codec driver

This configures the actual codec or codec-like endpoint.

Examples:

- an external I2S audio codec
- some internal AHUB modules
- a codec on a custom audio board

### Machine driver

This is the piece that binds everything into a registered sound card.

It tells Linux:

- which platform driver and codec driver belong together
- which DAI links exist
- how widgets and routes are described

If you remember only one sentence, remember this:

> The **machine driver** is the part that turns "Jetson plus codec" into a usable Linux sound card.

---

## 8. The AHUB blocks that matter most in practice

You do not need to master every AHUB block on day one. Start with these:

| Block | What it means in practice |
|---|---|
| `ADMAIF` | bridge between memory and the AHUB |
| `I2S` | main serial audio link to external codecs |
| `DMIC` | digital microphone controller |
| `DSPK` | digital speaker output path |
| `Mixer` | combine streams |
| `AMX` | pack multiple streams together, useful for TDM-style layouts |
| `ADX` | unpack a TDM stream into separate streams |
| `SFC` | sample-rate conversion |
| `ASRC` | asynchronous sample-rate conversion |

For most first products:

- **speaker / headphone output**: think `ADMAIF -> I2S -> codec`
- **microphone capture**: think `codec or DMIC -> I2S/DMIC -> ADMAIF`
- **advanced mic-array / TDM work**: think `AMX` / `ADX`

---

## 9. Jetson Orin Nano 40-pin header audio bring-up

NVIDIA's official guide is very clear on one important rule:

**Audio pins must be configured as SFIO, not plain GPIO.**

On Orin Nano dev kits, the 40-pin header audio path needs pinmux configuration for `I2S2`.

### Practical meaning

If you wire a codec board to pins `12`, `35`, `38`, and `40`, but those pins are still configured as GPIOs:

- clocks may never appear
- frame sync may never toggle
- playback and capture will fail even though your driver looks correct

### First pinmux step

Use Jetson-IO to switch the header into the audio function group before debugging the codec:

```bash
sudo /opt/nvidia/jetson-io/jetson-io.py
```

Then confirm your pinmux and device tree match the use case you want.

---

## 10. Bring-up flow for a custom codec board

This is the shortest correct mental checklist for custom audio hardware on Jetson.

### Step 1: make sure the hardware choice is sane

Before software:

- the codec must support the I2S mode you want
- required clocks and supplies must exist
- Linux kernel support must exist for the codec
- the board must expose the needed control bus, usually I2C

### Step 2: add or enable the codec node

NVIDIA's guide shows the normal pattern:

- put the codec under its control bus node, usually I2C
- set the codec status to `"okay"`
- add any required supply, GPIO, and clock properties

### Step 3: enable the Jetson I2S node

For the Orin-series 40-pin header, NVIDIA documents the exposed I2S controller address as:

```text
0x02901100
```

The I2S node must be enabled in device tree with status `"okay"`.

### Step 4: configure the sound node

This is where most people get stuck.

The sound node must describe:

- the DAI link between Jetson and the codec
- widgets such as microphone, line-in, headphone, speaker
- routing between Jetson-side DAPM widgets and codec-side widgets
- clock and format assumptions

### Step 5: confirm the codec actually probed

If the codec did not probe, the sound card will never work.

Useful checks:

```bash
dmesg | grep -i asoc
cat /sys/kernel/debug/asoc/components
i2cdetect -y -r <bus-number>
```

If the codec is missing from `asoc/components`, fix probing before touching mixer controls.

---

## 11. The first routing examples you should know

NVIDIA's guide uses the `APE` card with `ADMAIF` channels and AHUB mux controls.

The key idea is:

- `ADMAIF<i>` names an internal AHUB channel
- `hw:APE,<i-1>` is the PCM device you use from user space

That off-by-one detail confuses almost everyone once.

### I2S playback through AHUB

```bash
amixer -c APE cset name="I2S2 Mux" ADMAIF1
aplay -D hw:APE,0 sound.wav
```

### I2S capture through AHUB

```bash
amixer -c APE cset name="ADMAIF1 Mux" I2S2
arecord -D hw:APE,0 -r 48000 -c 2 -f S16_LE cap.wav
```

### Internal loopback for debug

```bash
amixer -c APE cset name="I2S2 Mux" ADMAIF1
amixer -c APE cset name="ADMAIF1 Mux" I2S2
amixer -c APE cset name="I2S2 Loopback" on
```

This is a good sanity test when you are trying to answer:

- is the AHUB route alive?
- is the I2S block active?
- is the failure in routing or in the external analog side?

---

## 12. USB and monitor audio are still important

Do not ignore the simpler paths.

### USB audio

USB speakers, microphones, and headsets are often the fastest way to unblock application work:

```bash
aplay -Dhw:<cardID>,<devID> sound.wav
arecord -Dhw:<cardID>,<devID> -r 48000 -c 2 -f S16_LE cap.wav
```

This is especially useful for:

- speech pipeline validation
- AI assistant prototypes
- first-pass mic-array experiments

### DisplayPort audio

If your monitor has speakers:

```bash
aplay -Dhw:HDA,<devID> sound.wav
```

This is a good basic confidence test that the Linux audio stack is alive even before custom hardware works.

---

## 13. Troubleshooting in the right order

NVIDIA's troubleshooting flow is worth following almost exactly.

### Problem 1: no sound cards found

Start with:

```bash
cat /proc/asound/cards
dmesg | grep -i asoc
```

Common causes:

- codec driver not enabled in kernel
- codec did not probe
- bad I2C wiring
- wrong device tree node or wrong bus

### Problem 2: sound card exists, but codec path is incomplete

Check:

```bash
cat /sys/kernel/debug/asoc/components
```

If the codec is missing, stop there and fix the codec probe path first.

### Problem 3: sound exists in software but nothing is audible

This is often a DAPM or routing issue.

Enable tracing:

```bash
for i in `find /sys/kernel/debug/tracing/events -name "enable" | grep snd_soc_`; do
  echo 1 | sudo tee "$i" >/dev/null
done

sudo cat /sys/kernel/debug/tracing/trace_pipe | grep '\*'
```

What you want to see:

- a real DAPM path from source to sink
- the expected widgets turning on during playback or capture

### Problem 4: the route looks right, but the pins are dead

Check:

- header pins are configured as SFIO, not GPIO
- the I2S node has `status = "okay"`
- clocks and frame sync exist on the wires

At this point, use a scope or logic analyzer. Software alone will not answer a clocking bug.

### Problem 5: the interface works sometimes, but not reliably

Look for:

- wrong codec clock source
- mismatched master/slave assumptions
- unsupported frame format
- bad sample-rate assumptions

---

## 14. Recommended learning path for real products

If your real target is an audio product, follow this order:

1. **USB headset or USB mic array**
   - validate app stack, recording, playback, streaming
2. **DisplayPort monitor audio**
   - quick sanity check for output
3. **40-pin I2S playback with a known codec board**
   - validate pinmux, clocks, DTS, and routing
4. **40-pin capture path**
   - validate microphones, gains, and DAPM routes
5. **multichannel / TDM / smart-speaker work**
   - only after the simpler mono or stereo path is stable

That order is slower emotionally, but faster technically.

---

## 15. Product ideas

- **AI smart speaker bring-up:** use a USB mic array first, then move playback to an external I2S codec + amplifier.
- **Voice appliance:** local wake word, capture, playback, and network streaming on one Jetson.
- **Robot intercom:** two-way audio over Ethernet or Wi-Fi using Jetson user-space apps and ALSA endpoints.
- **Kiosk audio subsystem:** DP monitor audio first, then migrate to a dedicated amplifier board for product hardware.

---

## 16. What success looks like

You are in good shape when you can explain:

- why a Jetson audio device shows up as `APE`, `HDA`, or USB
- why the 40-pin audio path needs pinmux on Orin Nano
- what the machine driver, codec driver, and platform driver each do
- how `ADMAIF` relates to `hw:APE,<n>`
- why a DAPM path matters
- which part of the pipeline is broken when sound fails

That is the point where you stop "trying random mixer commands" and start doing real Jetson audio engineering.

---

## 17. References

- [NVIDIA Jetson Linux Developer Guide - Audio Setup and Development](https://docs.nvidia.com/jetson/archives/r38.4/DeveloperGuide/SD/Communications/AudioSetupAndDevelopment.html)
- [ALSA Project](https://www.alsa-project.org/)
- [ASoC DAPM documentation](https://www.kernel.org/doc/html/latest/sound/soc/dapm.html)
- [Jetson Expansion Header configuration](https://docs.nvidia.com/jetson/archives/r38.4/DeveloperGuide/HR/ConfiguringTheJetsonExpansionHeaders.html)
