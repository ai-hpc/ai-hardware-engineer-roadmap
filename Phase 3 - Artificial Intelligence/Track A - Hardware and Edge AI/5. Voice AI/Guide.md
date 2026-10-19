# Module 6A — Voice AI

<div class="course-identity auto-course" style="--course-accent: #0891b2; --course-accent-rgb: 8, 145, 178;" markdown="1">
<div class="course-identity__icon">M6VA</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · AI Workloads</p>
<p class="course-identity__title">Specialized course identity for Module 6A — Voice AI.</p>
<p class="course-identity__meta">Artifact: model or workload study · Measure: accuracy, latency, memory, throughput</p>
</div>
</div>


**Parent:** [Phase 3 — Artificial Intelligence](../../Guide.md) · Track A

> *Speech-to-text, text-to-speech, and real-time voice pipelines — the audio workloads that drive edge AI hardware.*

**Prerequisites:** Module 1 (Neural Networks — transformers, attention), Module 2 (Frameworks — PyTorch).

**Role targets:** Voice AI Engineer · Speech/Audio ML Engineer · Edge Audio Engineer · Conversational AI Engineer

---

**Learning sequence:** validate the audio contract → measure recognition → separate streaming/endpoint latency → synthesize audio → connect a small, expiring intent to the robot adapter. Model/API references below were checked on 2026-10-18; pretrained speech inference and target-device timing are optional exercises, not measured results from this course revision.

## Why Voice AI Matters for Hardware Engineers

Voice systems need explicit budgets for audio buffering, endpoint detection, inference, and response playback. Choose these from the interaction design and measure the complete path:

- **STT (Speech-to-Text):** Measure first partial result, final transcript, and real-time factor separately
- **TTS (Text-to-Speech):** Implementations include autoregressive, non-autoregressive, flow, and diffusion approaches
- **On-device voice:** Some deployments require local processing; verify model acquisition, logs, telemetry, and fallback behavior as well as inference location
- **VAD (Voice Activity Detection):** Always-on, ultra-low-power — runs on MCU or dedicated DSP (L4/L5 hardware)

| Voice task | Compute pattern | Hardware implication |
|-----------|----------------|---------------------|
| VAD | Small classifier/stateful network, often always-on | L4: MCU/DSP, measured always-on power budget |
| STT (streaming) | Encoder-decoder, chunked | L1/L3: streaming inference, low latency |
| TTS (neural) | Autoregressive, flow/diffusion, or parallel synthesis | L1/L3: profile each stage; compute/memory bottlenecks vary |
| Keyword spotting | Small CNN/RNN, always-on | L4: TinyML, dedicated audio NPU |
| Speaker verification | Embedding model, one-shot | L1: inference + vector similarity |
| Noise suppression | U-Net or RNNoise, real-time | L4/L6: DSP or FPGA, strict latency budget |

---

## 1. Speech-to-Text (STT / ASR)

### How Modern STT Works

```
Audio Input → Feature Extraction → Encoder → Decoder → Text Output
             (mel spectrogram)    (transformer)  (CTC/attention)
```

### Key Models and Architectures

| Model | Architecture | Strengths | Use case |
|-------|-------------|-----------|----------|
| **Whisper** (OpenAI) | Encoder-decoder transformer | Multi-language, robust, open-source | General-purpose STT |
| **Wav2Vec 2.0** (Meta) | Self-supervised encoder + CTC | Pre-trained on unlabeled audio | Low-resource languages |
| **Conformer** (Google) | Convolution + transformer | Convolution and attention for speech modeling | Production ASR |
| **DeepSpeech** (Mozilla) | RNN + CTC | Simple, lightweight | Legacy / embedded |
| **Whisper.cpp** | GGML quantized Whisper | Runs on CPU/edge without GPU | On-device STT |
| **Faster-Whisper** | CTranslate2 backend | Alternative runtime; compare matched model, beam, precision, and hardware | Production serving |

### Audio Feature Extraction

```python
import torchaudio

# Requires matching torch/torchaudio and decoder dependencies in an isolated environment.
waveform, sample_rate = torchaudio.load("speech.wav")  # [channels, samples]
waveform = waveform.mean(dim=0, keepdim=True)          # declared mono downmix
if sample_rate != 16000:
    waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    sample_rate = 16000
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate, n_fft=400, hop_length=160, n_mels=80
)
mel = mel_transform(waveform)  # [1, 80, time_frames]
```

This is a **generic power mel spectrogram**, not automatically a pretrained model's input. Whisper needs its exact log-mel normalization, padding/window convention, and model-specific mel count (some checkpoints use 128, not 80). Use the model's processor, and compare preprocessing against its reference implementation. `MelSpectrogram` also uses centred windows by default, which entails padding/lookahead semantics relevant to streaming. Current `torchaudio.load` uses TorchCodec from TorchAudio 2.9 onward; install the matching decoder stack or use the documented TorchCodec API. [TorchAudio documentation](https://docs.pytorch.org/audio/stable/generated/torchaudio.load.html), [Whisper audio implementation](https://github.com/openai/whisper/blob/main/whisper/audio.py).

**Worked audio contract:** 16,000 samples at 16 kHz are 1 second. A 400-sample analysis window spans 25 ms; a 160-sample hop spans 10 ms. Relabelling 48,000 captured samples as 16 kHz produces 3 seconds and changes pitch—it is not resampling. Preserve capture sample index, sample rate, channel policy, amplitude convention, and resampler delay.



**Why this matters for hardware:**
- Mel spectrogram computation is an FFT + filterbank — can be accelerated on DSP or FPGA (L6)
- The encoder processes the mel spectrogram — this is the compute-heavy part (matrix multiply)
- Streaming STT processes chunks (e.g., 1 second at a time) — requires careful state management

### Streaming vs Batch STT

| Mode | Latency | Accuracy | Use case |
|------|---------|----------|----------|
| **Batch** | Process a complete recording | Depends on model/context/decoding | Transcription, subtitles |
| **Streaming** | Incremental output under a declared lookahead budget | Compare with the same task and transcript normalization | Live conversation, voice assistant |

Streaming requires:
- Chunked input (e.g., 1-second windows with overlap)
- Architecture-specific state caching, or repeated window inference with overlap reconciliation
- Partial hypothesis output with correction and a clear finalization policy

Whisper is not inherently a causal streaming encoder. Sliding windows can provide incremental transcripts, but repeated context, overlap deduplication, timestamp alignment, and endpointing must be designed and measured. Do not assume arbitrary encoder states can be reused across windows.

### Recognition evaluation with a worked example

WER = `(substitutions + deletions + insertions) / reference_word_count`, using minimum edit distance and a declared normalization policy. Reference “move to bay two” versus hypothesis “move bay three” has one deletion and one substitution: WER=2/4=0.5. Added words can make WER exceed 1. For an empty reference, do not divide by zero: report silence false-transcription counts/rate separately. Aggregate corpus edit counts and reference words; averaging utterance WER gives a different weighting.

For a robot, also score intent, slot/unit correctness, negation, and rejection. Losing “not” may change only one word while reversing an action. Split evaluation by speaker/session before tuning. Save errors for silence, machinery noise, accent/language, overlapping speakers, and short commands.

**Exercise:** 20 reference words with 2 substitutions, 1 deletion, and 3 insertions gives WER=0.30. A 4-second clip processed in 1 second has real-time factor `RTF=processing/audio=0.25`; this says nothing about endpoint delay or time to first partial transcript.

### Projects

1. **Run Whisper** on a sample audio file. Measure inference time and WER (word error rate).
2. **Whisper on Jetson** — deploy Whisper with TensorRT. Measure latency and compare with CPU.
3. **Streaming STT** — implement chunked Whisper inference. Measure time-to-first-word.
4. **Whisper.cpp on edge** — compare supported quantized formats against an unquantized reference on the same audio, decoder settings, and hardware. Quantization labels and support differ by runtime.

---

## 2. Text-to-Speech (TTS)

### How Modern TTS Works

```
Text Input → Text Analysis → Acoustic Model → Vocoder → Audio Output
             (phonemes,       (mel spectrogram   (waveform
              prosody)         generation)         synthesis)
```

### Key Models and Architectures

| Model | Type | Quality | Speed | Use case |
|-------|------|---------|-------|----------|
| **VITS** | End-to-end (text → audio) | High | Fast | Production TTS |
| **Bark** (Suno) | GPT-style autoregressive | Very high, expressive | Slow | Creative, multi-language |
| **Tortoise TTS** | Autoregressive + diffusion | Historical expressive synthesis example | Measure actual configuration | Voice synthesis |
| **Piper** | VITS-based ONNX voices | Voice/model-dependent | Measure CPU RTF and first audio | On-device, embedded |
| **Coqui TTS** | Multiple architectures | High | Medium | Open-source toolkit |
| **F5-TTS** | Flow matching | High, zero-shot | Fast | Voice cloning, multilingual |
| **XTTS** (Coqui) | Autoregressive acoustic tokens + waveform decoder | Voice/model-dependent | Measure selected checkpoint | Multi-speaker, multi-language |

### TTS Pipeline Components

**Text analysis:**
- Text normalization (numbers, abbreviations, dates → words)
- Grapheme-to-phoneme (G2P) conversion
- Prosody prediction (duration, pitch, energy)

**Acoustic model (mel generation):**
- Generates mel spectrogram from phoneme sequence
- Autoregressive (Tacotron 2) or non-autoregressive (FastSpeech 2, VITS)
- Parallel synthesis can reduce sequential work; actual speed depends on model size, output duration, runtime, and hardware.

**Vocoder (waveform synthesis):**
- Converts mel spectrogram → audio waveform
- HiFi-GAN: fast, high-quality, lightweight
- WaveGlow and WaveNet illustrate different waveform-generation approaches; quality and speed require matched measurements.

```python
# Current OHF-Voice Piper API; requires a separately downloaded voice and its .onnx.json.
import wave
from piper import PiperVoice

voice = PiperVoice.load("en_US-lessac-medium.onnx")
with wave.open("response.wav", "wb") as output:
    voice.synthesize_wav("The inspection task is ready.", output)
```

The maintained project is [OHF-Voice/piper1-gpl](https://github.com/OHF-Voice/piper1-gpl), with [Python API/download instructions](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_PYTHON.md). Its `synthesize` streaming method yields audio chunks; assigning the iterator to a variable does not write a waveform. Record the voice/model revision and license separately from the runtime license. This example is source-checked, not reported as executed here.


### Projects

1. **Run Piper TTS** on CPU. Measure real-time factor (RTF). Report RTF, first playable audio, output duration, and CPU threads; set a task-specific target before optimizing.
2. **VITS on Jetson** — deploy VITS with ONNX Runtime or TensorRT. Measure latency per sentence.
3. **Voice adaptation** — use your own or a permitted speaker sample and a supported model; document the voice rights and evaluate intelligibility and speaker similarity separately.
4. **HiFi-GAN vocoder** — run standalone, benchmark on GPU vs CPU. Understand the mel → waveform bottleneck.

---

## 3. Voice Activity Detection (VAD) & Keyword Spotting

### VAD — Is Someone Speaking?

Always-on, ultra-low-power. Runs continuously to wake up the full STT pipeline.

| Option | Input/state contract to verify | What to measure |
|---|---|---|
| Silero VAD | Supported sample rate, frame length, recurrent-state/reset behavior | Missed speech, false speech, endpoint latency, CPU time |
| WebRTC VAD | Supported PCM rate and 10/20/30 ms frames for the chosen binding | Same metrics plus sensitivity mode |
| Custom MCU model | Actual operator/memory support and preprocessing | Board-measured always-on power and wake failures |

Published CPU frame timings do not establish MCU milliwatts. Model file size is not peak runtime memory, and compute time is not the full audio-context/endpoint delay.

```python
# Silero VAD — production-quality, lightweight
import torch
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
model = load_silero_vad()  # install/pin the package in a separate speech environment

wav = read_audio('speech.wav', sampling_rate=16000)
timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)
# start/end are sample indices by default: divide by 16000 for seconds.
# Preserve the original capture offset when mapping chunk-local indices to a stream.
```

### Keyword Spotting — "Hey [Device]"

Always-on wake word detection. Choose a power budget from battery capacity, duty cycle, and required standby time; no universal 1 mW requirement applies.

- **Models:** Small CNNs (DS-CNN), RNNs, or attention-based
- **Training:** Typically custom-trained on the specific wake word
- **Deployment:** TFLite Micro on Cortex-M, dedicated audio DSP
- **Connection to L4/L5:** This is a workload you'd design custom silicon for (always-on NPU)

### Noise Suppression / Enhancement

Real-time audio cleanup before STT.

| Model | Approach | Engineering question |
|---|---|---|
| RNNoise | Recurrent network with signal-processing features | Does suppression improve task WER without clipping speech onset? |
| DTLN | Dual-signal transformation LSTM network | What frame/lookahead and recurrent state does the exported model require? |
| DeepFilterNet | Deep filtering with perceptual frequency-band processing | What end-to-end audio latency and quality does this configuration achieve? |

DTLN is not a transformer architecture. Noise reduction can damage speech; score recognition with and without it on identical recordings. Source implementations: [DTLN](https://github.com/breizhn/DTLN), [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet).

### Projects

1. **Silero VAD** — run on a continuous audio stream. Measure detection latency and false positive rate.
2. **Keyword spotter on MCU** — train a small CNN for a custom wake word. Deploy with TFLite Micro on Cortex-M.
3. **RNNoise on FPGA** — implement the GRU-based noise suppression in HLS (connection to Phase 4A).

---

## 4. End-to-End Voice Pipeline

### Architecture for Edge Voice Assistant

```
┌─────────────────────────────────────────────────────────┐
│                   Edge Device (Jetson / MCU+NPU)         │
│                                                          │
│  ┌─────────┐   ┌──────────┐   ┌──────────┐            │
│  │  VAD    │──▶│   STT    │──▶│  NLU /   │            │
│  │(always  │   │(Whisper  │   │  LLM     │            │
│  │  on)    │   │ or       │   │(on-device│            │
│  └─────────┘   │Conformer)│   │ or cloud)│            │
│                └──────────┘   └────┬─────┘            │
│                                    │                    │
│                                    ▼                    │
│                              ┌──────────┐              │
│  ┌─────────┐                │   TTS    │              │
│  │ Speaker │◀───────────────│  (VITS/  │              │
│  │         │                │  Piper)  │              │
│  └─────────┘                └──────────┘              │
└─────────────────────────────────────────────────────────┘
```

**Illustrative design budget, not achieved measurements:**

| Stage | Target | Model |
|-------|--------|-------|
| VAD detection | < 50ms | Silero VAD |
| STT (streaming) | < 300ms to first word | Whisper / Conformer |
| NLU / LLM response | < 500ms | On-device small LLM or cloud |
| TTS synthesis | < 200ms to first audio | VITS / Piper |
| **End-to-end** | **Measure from a declared start event to first audible response** | Stages overlap; endpoint silence and audio buffering also count |

A timestamped command adapter should receive a finalized transcript, explicit intent/slots, capture/end times, and an expiry. Reject ambiguous units, unsupported actions, and expired commands before requesting a robot task. Test “do not move,” background speech, and silence; do not wire raw transcript strings directly to motion. For interruption, cancel queued playback and pending work using explicit task IDs.

### Projects

1. **Full pipeline on Jetson** — VAD → Whisper STT → simple NLU → Piper TTS. Measure end-to-end latency.
2. **Optimize for latency** — quantize STT to INT8, use streaming chunked inference, pre-warm TTS. Measure the deadline-miss rate against your declared interaction target.
3. **Compare cloud vs edge** — same pipeline on Jetson vs cloud API. Measure latency, accuracy, and privacy trade-off.

---

## 5. Voice AI for Hardware Design Context

| Voice workload | Compute pattern | Why hardware engineers care |
|---------------|----------------|-----------------------------|
| Mel spectrogram | FFT + filterbank | Can be accelerated in DSP/FPGA — fixed-function vs general compute trade-off |
| Encoder (Conformer) | Attention + conv | Same matmul-heavy compute as vision — systolic arrays apply |
| Autoregressive decode | Sequential token generation | Memory-bound like LLM decode — HBM bandwidth matters |
| Vocoder (HiFi-GAN) | Transposed conv, upsampling | Unique compute pattern — not well-served by standard matmul accelerators |
| VAD / keyword | Tiny CNN/RNN | Target for always-on NPU design (< 1mW) — Phase 5F AI Chip Design |
| Noise suppression | GRU / filterbank | Real-time streaming with strict latency — FPGA or dedicated DSP |

---

## Resources

| Resource | What it covers |
|----------|---------------|
| [Whisper](https://github.com/openai/whisper) | Open-source STT model |
| [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) | CTranslate2-based fast Whisper |
| [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) | C/C++ port for edge deployment |
| [Piper](https://github.com/OHF-Voice/piper1-gpl) | Fast on-device TTS |
| [Coqui TTS](https://github.com/coqui-ai/TTS) | Open-source TTS toolkit |
| [Silero VAD](https://github.com/snakers4/silero-vad) | Production-quality VAD |
| [RNNoise](https://github.com/xiph/rnnoise) | Real-time noise suppression |
| [ESPnet](https://github.com/espnet/espnet) | End-to-end speech processing toolkit |
| [SpeechBrain](https://github.com/speechbrain/speechbrain) | PyTorch speech toolkit |

---

## Completion gate

Submit at least 20 held-out, human-checked utterances from more than one session, plus silence/noise controls, with dataset rights/provenance and exact transcript normalization. Report corpus WER, intent/slot errors, silence false transcriptions, RTF, endpoint delay, first response audio, and peak memory. Include a sample-rate mismatch, negation, and expired-command test. Keep real speaker recordings out of train/test overlap. A successful single utterance is a smoke test, not an accuracy estimate.

**Diagnostic answers:** resampling changes sample count while preserving duration; header relabelling changes duration. VAD finds speech regions, not speaker identity or permission to act. RTF<1 is necessary for some sustained workloads but does not prove low interaction latency.

## Next

→ [**Module 5A — Edge AI & Model Optimization**](../6.%20Edge%20AI%20and%20Model%20Optimization/Guide.md) — quantize and deploy these voice models on edge hardware.


## Carry this course into the physical-AI system

**Start from:** Recorded audio, sample rate, and a small set of task intents.

**Build and challenge:** Measure buffering, endpointing, recognition, and dispatch separately. Test silence, noise, and an expired command; a transcript must pass an intent adapter before a robot task is requested.

**Pass forward:** An optional language-command interface with latency, expiry, and rejection behavior. Continue with [the connected course](../../Track%20B%20-%20Agentic%20AI%20and%20ML%20Engineering/3.%20Agentic%20AI%20and%20GenAI/Guide.md) and keep the artifact in the [Physical AI Design Studio](../../../Physical-AI-Design-Studio.md).
