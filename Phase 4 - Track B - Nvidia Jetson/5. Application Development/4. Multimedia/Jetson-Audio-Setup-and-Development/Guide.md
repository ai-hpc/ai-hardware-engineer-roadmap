# Jetson audio: samples, clocks and a working capture path

<div class="course-identity auto-course" style="--course-accent: #7e22ce; --course-accent-rgb: 126, 34, 206;" markdown="1">
<div class="course-identity__icon">JASA</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Jetson Track</p>
<p class="course-identity__title">Specialized course identity for Jetson Audio Setup and Development - Project Guide.</p>
<p class="course-identity__meta">Artifact: Jetson integration demo · Measure: latency, memory, power, logs</p>
</div>
</div>

**Reviewed 18 October 2026.** Learn to describe an audio stream precisely, identify the actual device path, and validate samples before feeding a speech model. The [CPU audio lab](CPU-Audio-Lab.md) runs real NumPy/WAV processing without a microphone. Jetson/codec capture commands below are source-reviewed hardware extensions, not tests executed here.

**Prerequisites:** binary integers, basic Linux and [peripheral access](../../1.%20Peripheral%20Access/Guide.md). Allow two 60-minute sessions. **Completion artifact:** a known-format WAV, a channel map, a rate/latency calculation and an observed failure from the CPU lab or actual capture.

## 1. Audio has several different paths

```text
USB microphone → USB audio driver → ALSA PCM → application
I2S codec → I2S / AHUB routing / ADMAIF → ALSA PCM → application
application → ALSA HDA device → DisplayPort sink
```

USB audio does not pass through Jetson's I2S2/AHUB route. A desktop sound server such as PipeWire may sit above ALSA and add mixing, format conversion and buffering. A working USB microphone therefore does not validate a custom I2S codec, while a broken I2S route does not imply that every audio device is broken.

ASoC connects the CPU-side interface, codec components and board-specific machine description. A codec's control bus sets registers; the sample bus carries audio. I2C register access is not an audio stream. [Linux ASoC architecture](https://docs.kernel.org/sound/soc/overview.html).

## 2. Describe samples, frames and buffers

A **sample** is one channel's value at one time. A **frame** contains one sample from every channel. Four-channel S16_LE has `4 × 2 = 8` bytes per frame. At 48,000 frames/s, its memory data rate is 384,000 bytes/s.

A **period** of 480 frames lasts 10 ms at 48 kHz. A four-period buffer has 40 ms capacity. Capacity is not automatically the current delay: occupancy, application batching, hardware FIFO and sound-server buffering matter. Record negotiated parameters and actual timestamps. [ALSA PCM model](https://www.alsa-project.org/alsa-doc/alsa-lib/pcm.html).

| Field | Example | Common error |
|---|---|---|
| Rate | 48,000 frames/s | Relabel 48 kHz samples as 16 kHz |
| Channels/order | mic-left, mic-right, reference, unused | Average unrelated or opposite-polarity channels |
| Container format | signed 16-bit little-endian | Parse as unsigned or wrong byte order |
| Valid sample bits | 16 or 24 | Confuse valid bits with container or slot width |
| Interleaving | L0,R0,L1,R1… | Treat stereo bytes as a mono time series |
| Time | sample index + capture clock mapping | Timestamp callback completion as sample acquisition |

S24_3LE packs 24 bits in three bytes; S32_LE stores four bytes per sample and can carry a signal with fewer valid bits. Interpret alignment/sign extension from the device/driver contract. A filename ending `.wav` does not prove any particular sample format. [Python WAV support](https://docs.python.org/3/library/wave.html).

## 3. Inspect the registered devices

```bash
cat /proc/asound/cards
aplay -l
arecord -l
aplay -L
arecord -L
```

Lowercase `-l` lists hardware devices; uppercase `-L` includes named PCM configurations/plugins. Card indexes can change. Choose a stable card identifier and the actual device/direction. `hw:` requests the hardware format directly; `plughw:` may convert. A convenient converted stream can hide the actual device rate, so record which path was used.

For an actual supported capture device, inspect its hardware parameters and negotiate the rate, channels and format before recording. One source-reviewed example, **only if the device supports stereo 48 kHz PCM16**, is:

```bash
read -r -p 'ALSA capture PCM (from arecord -L): ' CAPTURE_PCM
arecord -D "$CAPTURE_PCM" --dump-hw-params -d 1 -r 48000 -c 2 -f S16_LE /tmp/probe.wav
arecord -D "$CAPTURE_PCM" -d 5 -r 48000 -c 2 -f S16_LE capture.wav
```

An unsupported-format error is useful information; do not respond by randomly changing all three fields until a file appears. Inspect permissions and current owners if the device is busy. If using PipeWire, inspect `wpctl status` and the negotiated graph rate/quantum; ALSA defaults and sound-server defaults are not necessarily identical. [PipeWire configuration](https://docs.pipewire.org/page_man_pipewire_conf_5.html).

## 4. Match the reference carrier and the installed BSP

Use the [current Jetson compatibility record](../../../Compatibility.md). The checked Jetson Linux 39.2.1 audio reference separates carrier tables. For the Orin NX/Nano reference carrier it lists:

| Path | Reference connector signals |
|---|---|
| I2S2 | FS35, SCLK12, DIN38, DOUT40; configure pinmux |
| Audio master-clock option | Header7; verify actual direction/configuration |
| I2S4 | M.2 Key E; not a promise that the installed radio exposes audio |
| HDA | DisplayPort; discover its PCM device |

DMIC3 on header32/16 is listed under **AGX Orin**, not the NX/Nano carrier table. Do not copy that mapping to Orin Nano based on the SoC's internal DMIC instances. [NVIDIA board interfaces](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SD/Communications/AudioSetupAndDevelopment.html#board-interfaces).

A custom carrier needs its own schematic and pinmux/device-tree configuration. Orin Nano's reference kit has DisplayPort, not a native HDMI connector or USB-C display output; an adapter is a separate component.

## 5. Configure routing and clocks as separate contracts

An ASoC card can register while its route is incomplete. DAPM tracks connected audio components and powers the active path. Mixer settings, stream state and board routes all affect whether samples flow. [DAPM model](https://docs.kernel.org/sound/soc/dapm.html).

For the NVIDIA APE path, ADMAIF1 corresponds to PCM device0 within the actual APE card. Inspect available controls before using these route examples:

```bash
amixer -c APE controls
amixer -c APE cget name='ADMAIF1 Mux'
# After selecting a configured external I2S2 capture path:
amixer -c APE cset name='ADMAIF1 Mux' I2S2
# For the reverse playback route:
amixer -c APE cset name='I2S2 Mux' ADMAIF1
```

These commands select internal routes; they do not instantiate a missing codec, provide external clocks or define a complete TDM layout. Newer NVIDIA guidance moves PCM override configuration toward DAI device-tree nodes; old JetPack 6 mixer-control scripts are not universal current recipes. [Versioned NVIDIA audio reference](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SD/Communications/AudioSetupAndDevelopment.html).

Designate one driver for each clock. The bit clock shifts data; the frame clock marks audio frames. For four 32-bit TDM slots at 48 kHz:

`BCLK = 48,000 × 4 × 32 = 6.144 MHz`.

The application may still store four 16-bit values per frame. Physical wire width and memory representation are different. Codec MCLK requirements depend on that codec and its PLL; never assume MCLK=BCLK. [Linux audio clocking](https://docs.kernel.org/sound/soc/clocking.html).

A three-microphone ADC may transmit four slots. Configure the transport's actual slots/masks, then map the active channels into the application. Asking `arecord -c 3` does not reprogram every codec and link into three-slot TDM. Validate channel identities with a known per-channel stimulus.

## 6. Debug the smallest claim first

| Failure | Next evidence |
|---|---|
| No card | Driver probe, codec bus, device tree and kernel log |
| Card exists, all zeros | Route, mute/gain, clock presence, codec ADC configuration |
| I2S reset/timeout | Clock source and actual BCLK/FS waveform, then driver state |
| Repeated/pitched samples | Sample rate, word alignment, frame format and channel parsing |
| XRUN | Buffer occupancy, missed deadlines, competing work and process scheduling |
| Clicks at chunk boundaries | Filter/resampler state, dropped frames, gain discontinuities |
| Gradual queue growth | Capture/playback clock mismatch and rate negotiation |

Use a logic analyzer/oscilloscope to confirm physical clock/frame timing when available. A DMA byte counter alone cannot measure oscillator accuracy. An internal digital loopback can validate parts of the software route without proving external microphone quality or wiring.

An XRUN is an underrun/overrun, not a request to maximize all clocks automatically. Measure the cause, choose a period/buffer budget, and retest under intended load. Increasing capacity can reduce overruns while increasing sample age.

## 7. Prepare audio for AI without changing its meaning

A model expecting mono 16 kHz needs explicit channel selection/downmix and proper sample-rate conversion. Keeping every third 48 kHz sample aliases frequencies above 8kHz. Merely changing a WAV header makes one second of audio last three seconds and turns a 1kHz tone into approximately 333Hz.

The [CPU lab](CPU-Audio-Lab.md) implements a stateful, causal low-pass FIR followed by 3:1 decimation. It preserves filter history and decimation phase across irregular chunks. Resetting either on each callback produces discontinuities or timing drift.

A capture timestamp needs a defined relation to the sample index and hardware/host clock. Linux exposes several audio/system timestamp concepts; a callback's receipt time also includes buffering. At 50 ppm mismatch, two nominally 48 kHz clocks diverge by 1,440 samples, or 30 ms, in 10 minutes. Clock synchronization or adaptive rate correction must be designed rather than inferred from an initially aligned recording. [ALSA timestamping](https://docs.kernel.org/sound/designs/timestamping.html).

For ASR, retain a transcript-labelled evaluation set and report word error rate by microphone, noise, language and distance. Separate capture/endpointing delay from model computation and finalization. A synthetic tone exercise verifies signal plumbing, not speech accuracy. Continue with [Voice AI](../../../../Phase%203%20-%20Artificial%20Intelligence/Track%20A%20-%20Hardware%20and%20Edge%20AI/5.%20Voice%20AI/Guide.md).

## 8. Completion

Run the CPU lab, explain the 6.144 MHz wire clock and30 ms drift calculation, and submit one deliberate wrong-format/rate/channel result. For hardware work, add negotiated PCM parameters, a measured clock/route record, a known stimulus and XRUN logs. Keep the [LyraT integration case](../ESP32-LyraT-I2S-Mic-Jetson-Orin-Nano/Guide.md) scoped to its exact firmware and BSP.

**Diagnostic answer:** a correctly sized audible WAV is encouraging, but it does not prove bit-exact transport, correct channel order, zero lost samples, calibrated amplitude or clock accuracy. Each needs an independent reference or measurement.

The [source review record](labs/source-evidence-2026-10-18.json) retains checked URLs, download hashes and the pinned LyraT board source. These checks validate source provenance; hardware stages remain unexecuted.
