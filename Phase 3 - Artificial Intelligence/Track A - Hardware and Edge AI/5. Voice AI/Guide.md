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

## Why Voice AI Matters for Hardware Engineers

Voice is one of the most latency-sensitive AI workloads. Users notice 200ms+ delay in conversation. This creates hard requirements for hardware:

- **STT (Speech-to-Text):** Real-time transcription needs streaming inference with <100ms latency per chunk
- **TTS (Text-to-Speech):** Natural voice synthesis requires autoregressive generation (like LLMs) or fast diffusion models
- **On-device voice:** Privacy-critical applications (medical, military, automotive) need inference without cloud — your edge chip must handle it
- **VAD (Voice Activity Detection):** Always-on, ultra-low-power — runs on MCU or dedicated DSP (L4/L5 hardware)

| Voice task | Compute pattern | Hardware implication |
|-----------|----------------|---------------------|
| VAD | Tiny CNN, always-on | L4: MCU/DSP, < 1mW power budget |
| STT (streaming) | Encoder-decoder, chunked | L1/L3: streaming inference, low latency |
| TTS (neural) | Autoregressive or diffusion | L1/L3: memory-bound generation, like LLM decode |
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
| **Conformer** (Google) | Convolution + transformer | Best accuracy on benchmarks | Production ASR |
| **DeepSpeech** (Mozilla) | RNN + CTC | Simple, lightweight | Legacy / embedded |
| **Whisper.cpp** | GGML quantized Whisper | Runs on CPU/edge without GPU | On-device STT |
| **Faster-Whisper** | CTranslate2 backend | 4x faster than original Whisper | Production serving |

### Audio Feature Extraction

```python
import torchaudio

# Load audio
waveform, sample_rate = torchaudio.load("speech.wav")

# Convert to mel spectrogram (the "image" that the model sees)
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,        # FFT window size
    hop_length=160,   # 10ms hop (160 samples at 16kHz)
    n_mels=80         # 80 mel frequency bins
)
mel = mel_transform(waveform)
# Shape: [1, 80, T] — 80 frequency bins x T time frames
# This is the input to the encoder
```

**Why this matters for hardware:**
- Mel spectrogram computation is an FFT + filterbank — can be accelerated on DSP or FPGA (L6)
- The encoder processes the mel spectrogram — this is the compute-heavy part (matrix multiply)
- Streaming STT processes chunks (e.g., 1 second at a time) — requires careful state management

### Streaming vs Batch STT

| Mode | Latency | Accuracy | Use case |
|------|---------|----------|----------|
| **Batch** | Process entire audio file at once | Highest | Transcription, subtitles |
| **Streaming** | Process chunks in real-time | Slightly lower | Live conversation, voice assistant |

Streaming requires:
- Chunked input (e.g., 1-second windows with overlap)
- Encoder state caching between chunks
- Partial hypothesis output with correction

### Projects

1. **Run Whisper** on a sample audio file. Measure inference time and WER (word error rate).
2. **Whisper on Jetson** — deploy Whisper with TensorRT. Measure latency and compare with CPU.
3. **Streaming STT** — implement chunked Whisper inference. Measure time-to-first-word.
4. **Whisper.cpp on edge** — run quantized Whisper on Raspberry Pi or Jetson. Benchmark INT8 vs FP16.

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
| **Tortoise TTS** | Autoregressive + diffusion | Highest quality | Very slow | Voice cloning |
| **Piper** | VITS-based, optimized | Good | Very fast | On-device, embedded |
| **Coqui TTS** | Multiple architectures | High | Medium | Open-source toolkit |
| **F5-TTS** | Flow matching | High, zero-shot | Fast | Voice cloning, multilingual |
| **XTTS** (Coqui) | GPT + VITS | High, voice cloning | Medium | Multi-speaker, multi-language |

### TTS Pipeline Components

**Text analysis:**
- Text normalization (numbers, abbreviations, dates → words)
- Grapheme-to-phoneme (G2P) conversion
- Prosody prediction (duration, pitch, energy)

**Acoustic model (mel generation):**
- Generates mel spectrogram from phoneme sequence
- Autoregressive (Tacotron 2) or non-autoregressive (FastSpeech 2, VITS)
- Non-autoregressive is faster — better for edge deployment

**Vocoder (waveform synthesis):**
- Converts mel spectrogram → audio waveform
- HiFi-GAN: fast, high-quality, lightweight
- WaveGlow / WaveNet: higher quality, much slower

```python
# Piper TTS — fast on-device TTS
import piper

voice = piper.PiperVoice.load("en_US-lessac-medium.onnx")
audio = voice.synthesize("Hello, I am running on edge hardware.")
# Runs on CPU, ~10x real-time on Raspberry Pi 4
```

### Projects

1. **Run Piper TTS** on CPU. Measure real-time factor (RTF). Target: RTF < 0.1 (10x faster than real-time).
2. **VITS on Jetson** — deploy VITS with ONNX Runtime or TensorRT. Measure latency per sentence.
3. **Voice cloning** — use XTTS or F5-TTS to clone a voice from a 10-second sample.
4. **HiFi-GAN vocoder** — run standalone, benchmark on GPU vs CPU. Understand the mel → waveform bottleneck.

---

## 3. Voice Activity Detection (VAD) & Keyword Spotting

### VAD — Is Someone Speaking?

Always-on, ultra-low-power. Runs continuously to wake up the full STT pipeline.

| Model | Size | Latency | Power | Platform |
|-------|------|---------|-------|----------|
| **Silero VAD** | 1.5 MB | <1ms per frame | ~10mW on MCU | CPU, edge |
| **WebRTC VAD** | <100 KB | <0.1ms | ~1mW | Any CPU |
| **Custom CNN VAD** | 50–500 KB | <1ms | <5mW | MCU, DSP |

```python
# Silero VAD — production-quality, lightweight
import torch
model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad')
(get_speech_timestamps, _, read_audio, _, _) = utils

wav = read_audio('speech.wav', sampling_rate=16000)
timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)
# Returns: [{'start': 1000, 'end': 15000}, ...] — speech segments
```

### Keyword Spotting — "Hey [Device]"

Always-on wake word detection. Must run at < 1mW for battery-powered devices.

- **Models:** Small CNNs (DS-CNN), RNNs, or attention-based
- **Training:** Typically custom-trained on the specific wake word
- **Deployment:** TFLite Micro on Cortex-M, dedicated audio DSP
- **Connection to L4/L5:** This is a workload you'd design custom silicon for (always-on NPU)

### Noise Suppression / Enhancement

Real-time audio cleanup before STT.

| Model | Approach | Latency | Quality |
|-------|----------|---------|---------|
| **RNNoise** | GRU-based, handcrafted features | <5ms | Good |
| **DTLN** | Dual-signal transformer | ~10ms | High |
| **DeepFilterNet** | Attention-based filterbank | ~10ms | Very high |
| **NSNet2** | Dense + GRU | <5ms | Good |

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

**Latency budget for natural conversation:**

| Stage | Target | Model |
|-------|--------|-------|
| VAD detection | < 50ms | Silero VAD |
| STT (streaming) | < 300ms to first word | Whisper / Conformer |
| NLU / LLM response | < 500ms | On-device small LLM or cloud |
| TTS synthesis | < 200ms to first audio | VITS / Piper |
| **Total round-trip** | **< 1 second** | |

### Projects

1. **Full pipeline on Jetson** — VAD → Whisper STT → simple NLU → Piper TTS. Measure end-to-end latency.
2. **Optimize for latency** — quantize STT to INT8, use streaming chunked inference, pre-warm TTS. Target < 800ms round-trip.
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
| [Piper](https://github.com/rhasspy/piper) | Fast on-device TTS |
| [Coqui TTS](https://github.com/coqui-ai/TTS) | Open-source TTS toolkit |
| [Silero VAD](https://github.com/snakers4/silero-vad) | Production-quality VAD |
| [RNNoise](https://github.com/xiph/rnnoise) | Real-time noise suppression |
| [ESPnet](https://github.com/espnet/espnet) | End-to-end speech processing toolkit |
| [SpeechBrain](https://github.com/speechbrain/speechbrain) | PyTorch speech toolkit |

---

## Next

→ [**Module 5A — Edge AI & Model Optimization**](../6.%20Edge%20AI%20and%20Model%20Optimization/Guide.md) — quantize and deploy these voice models on edge hardware.
