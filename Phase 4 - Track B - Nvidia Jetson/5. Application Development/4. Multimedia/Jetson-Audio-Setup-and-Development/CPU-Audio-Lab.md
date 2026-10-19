# CPU audio lab: preserve rate, channels and streaming state

**Time:** 45–75 minutes. **Tools:** Python 3.12 and NumPy 2.3.5 on the tested Linux x86_64 host. No microphone, model, GPU or audio playback is required. The four-channel signal is generated from equations and uses no external recordings.

## 1. Predict three failures

A one-second 48 kHz fixture contains:

- Channel 0: amplitude 0.4 at 1 kHz plus amplitude 0.2 at 12 kHz.
- Channel 1: amplitude 0.3 at 2 kHz.
- Channel 2: the negative of channel 0.
- Channel 3: zero, representing an unused slot.

Before running, predict:

1. Keeping every third sample maps the 12 kHz component to a 4 kHz alias at 16 kHz sampling.
2. Averaging channels 0 and 2 gives zero even though each has a strong signal.
3. Writing the original 48,000 samples into a 16 kHz WAV changes duration to 3 seconds; it does not resample.

## 2. Set up and run

From the repository root:

```bash
python3.12 -m venv /tmp/course-audio-cpu
/tmp/course-audio-cpu/bin/python -m pip install --require-hashes -r \
  'Phase 4 - Track B - Nvidia Jetson/5. Application Development/4. Multimedia/Jetson-Audio-Setup-and-Development/labs/requirements-linux-cpu-py312.txt'
/tmp/course-audio-cpu/bin/python \
  'Phase 4 - Track B - Nvidia Jetson/5. Application Development/4. Multimedia/Jetson-Audio-Setup-and-Development/labs/audio_lab.py' \
  --output /tmp/course-audio-fixture
```

The lock pins the one required wheel and hash. The review reused the already-installed NumPy 2.3.5 environment. Outputs are a four-channel 48 kHz WAV, correctly filtered 16 kHz WAV, naive decimation WAV, wrong-header WAV, and a JSON report. They total well under 1 MB.

## 3. Trace the actual math

[The implementation](labs/audio_lab.py) converts normalized float samples into signed little-endian PCM16 using rounding and clipping, then reloads the WAV. Quantization step is `1/32768`; an in-range rounded value has error at most half a step. Values beyond the representable endpoints clip, so that error bound no longer applies.

The 511-tap low-pass is a normalized sinc with 7 kHz cutoff multiplied by a Hann window. It is deliberately a readable fixed-ratio teaching filter, not a general-purpose production resampler. `numpy.sinc(x)` means `sin(pi*x)/(pi*x)`. [NumPy definition](https://numpy.org/doc/stable/reference/generated/numpy.sinc.html).

The causal convolution is:

`y[n] = sum(h[k] * x[n-k], k=0..510)`.

Then retain every third output. A symmetric 511-tap FIR has group delay 255 input samples, or 5.3125 ms at 48 kHz. Filtering cannot remove this causal delay for free. The lab retains it and excludes the startup/end margins from its tone-amplitude comparison.

For each chunk, the implementation preserves the last 510 inputs and the global decimation phase. The optional test compares 37 irregular chunks of seeded random input with an independent full convolution. Matching only chunks whose lengths happen to be divisible by 3 would miss a phase bug.

## 4. Read the measured result

Actual CPU run on 18 October 2026, retained in the [report](labs/measured-run-2026-10-18.json):

| Check | Observed |
|---|---:|
| Input frames/channels |48,000 /4|
| Output frames at 16 kHz |16,000|
| PCM round-trip max error |1.50e-5|
| Streamed versus one-block filter max error |0|
| Wanted 1 kHz amplitude after filtering |0.400001|
| Naive 4 kHz alias amplitude |0.200005|
| Filtered 4 kHz alias amplitude |1.34e-6|
| Opposite-channel average RMS |0|
| Wrong-header duration |3 seconds|

The 103.5 dB reduction is a comparison at this specific synthetic alias tone after trimming; it is **not** a broadband microphone SNR or a guarantee over every stopband frequency. The generated PCM already has quantization effects. No ALSA, physical codec, Jetson or speech model was exercised.

## 5. Tests

Portable accounting tests:

```bash
python3 -m unittest discover -s \
  'Phase 4 - Track B - Nvidia Jetson/5. Application Development/4. Multimedia/Jetson-Audio-Setup-and-Development/labs' -p 'test_*.py' -v
```

Actual NumPy/WAV tests:

```bash
/tmp/course-audio-cpu/bin/python -m unittest discover -s \
  'Phase 4 - Track B - Nvidia Jetson/5. Application Development/4. Multimedia/Jetson-Audio-Setup-and-Development/labs/optional' -v
```

Three portable and six optional tests cover memory/wire accounting, drift, independent PCM byte fixtures, clipping, WAV metadata, stateful convolution, alias rejection, channel cancellation and malformed input.

## 6. Exercises with diagnostic answers

1. Run again with `--taps 63` and a different `--output` directory. Predict delay and measure the alias result. **Answer:** delay becomes 31/48000≈0.646 ms; shorter filters generally have a wider transition region, so evaluate frequencies rather than assuming the same rejection. The [executed 63-tap variation](labs/measured-short-filter-2026-10-18.json) measured 85.1 dB rejection of this alias, compared with 103.5 dB for 511 taps.
2. Reset `history` for each chunk in a temporary copy. **Answer:** the independent random-signal test fails at boundaries; a clean single-block result did not validate streaming.
3. Why not always average all channels? **Answer:** channels can be microphones with different phase, loopback references or empty slots. Use a documented map and deliberate mixing policy.
4. What changes for 44.1 kHz→16 kHz? **Answer:** the ratio is 160/441, not 1/3. Use a correctly designed rational/polyphase converter and preserve its state; this fixed-factor class is not that implementation.
5. Why can a nominal 48 kHz capture gradually overrun a nominal 48 kHz playback queue? **Answer:** physical clocks differ; buffers absorb only a finite mismatch. Measure timestamp/sample-count drift and use a bounded correction policy.

**Submit:** your original report, a changed-filter result and one explanation of a failure. Return to [audio bring-up](Guide.md) before trying these contracts on hardware.
