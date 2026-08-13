# Lecture 1: AI-Driven Wireless Communication — Neural PHY, Smart Radios, and Learned Spectrum

## Overview

The classical wireless stack — channel coding, modulation, channel estimation, equalization, decoding — is built from **hand-derived mathematical blocks** (LDPC, Polar, MMSE, Viterbi) that have been tuned over decades against tractable channel models (AWGN, Rayleigh, 3GPP TDL). In real deployments the channel is none of these: it is a moving stew of **multipath, hardware impairments, blockers, interference, and traffic**. AI-driven wireless replaces or augments these classical blocks with **learned models** that fit the *actual* channel and *actual* hardware in front of them.

This lecture is written for the AI **hardware** engineer, not the communications theorist. The question is not "which loss function should we use" — it is **"where in the radio chain does an NPU/DSP/FPGA actually sit, what data crosses that boundary, and how tight is the latency budget?"** We cover three layers:

* **Part A — PHY-layer ML:** neural receivers, channel estimation, beamforming, autoencoder-based air interfaces. The "1 ms TTI, billion-MAC-per-slot" world.
* **Part B — RAN-level ML:** O-RAN's RIC architecture, scheduling, MIMO management, energy savings. Seconds-to-minutes loop, GPU/CPU in the basement.
* **Part C — SDR + DL:** modulation classification, RF fingerprinting, anomaly detection, GNU Radio + PyTorch pipelines you can actually run on a HackRF / USRP / Jetson.

By the end you should be able to **size the silicon** for a neural receiver, **place** ML inference correctly across UE / DU / CU / RIC, and **build** a small end-to-end demo on commodity SDR hardware.

---

## The Wireless Chain and Where AI Plugs In

```
   Tx bits ──► Source ──► Channel ──► Modulator ──► Pulse shaping ──► RF ──► Antenna
              coding     coding      (QAM/OFDM)     + DAC          front-end   array
                                                                                 │
                                                                                 ▼
                                                                        Wireless channel
                                                                        (multipath, fading,
                                                                         interference, noise)
                                                                                 │
   Rx bits ◄── Source ◄── Channel ◄── Demod ◄── Equalize ◄── Channel ◄── ADC ◄── RF ◄────┘
              decode     decode      (soft       + sync       est.       front-end
                                      LLR)
                ▲          ▲            ▲          ▲           ▲
                │          │            │          │           │
            (NN decoder)(NN belief    (NN demap)(NN equal.)(NN channel
                        prop)                              est. — the
                                                           biggest win)
```

The blocks where ML has produced **deployed or near-deployed** wins, in order of maturity:

| Block | Classical method | ML replacement | Status (2026) |
|---|---|---|---|
| Channel estimation | LS / LMMSE on pilots | CNN / U-Net on pilot grid | **Deployed** (Qualcomm, Samsung modems) |
| Demapping / soft LLRs | Closed-form QAM demap | MLP demapper per RE | **Field trials**, 3GPP Rel-18 study |
| MIMO detection | MMSE / sphere decoder | DetNet / unfolded MMSE-Net | Research → trials |
| Channel decoding (LDPC/Polar) | BP / SC-L | NN-aided BP, neural list decoder | Research |
| Beam management | Codebook sweep + RSRP | CNN on CSI / vision-aided beam pred. | **3GPP Rel-19 WI** |
| End-to-end autoencoder PHY | n/a | Tx & Rx co-trained NNs | Research only (no air-interface standard) |
| MAC scheduling | Proportional fair, etc. | RL / contextual bandits | **O-RAN xApps deployed** |
| RF impairment compensation | Polynomial DPD | NN-based DPD | **Deployed in flagship 5G PAs** |

The key insight: **the closer to the antenna, the harder the latency budget**. A neural channel estimator running per OFDM slot has ~125 µs (5G numerology 1) end-to-end including DMA. A RAN energy-saving xApp has minutes.

---

## Part A — PHY-Layer ML

### A.1 Neural channel estimation — the canonical case study

In 5G NR the UE/gNB inserts known **DMRS (Demodulation Reference Signal)** pilots into the resource grid. The classical estimator does **least-squares** on the pilot REs, then interpolates (linear, MMSE, Wiener) onto the data REs. The MMSE estimator is optimal if you know the channel covariance — which you don't, so deployed receivers use simplified Wiener filters tuned for "average" 3GPP channel profiles.

A learned estimator treats the pilot-grid → full-grid problem as **image super-resolution**:

```
Input  : tensor[num_pilots_freq × num_pilots_time × 2]   (real, imag)
                + noise variance estimate
Output : tensor[num_subcarriers × num_symbols × 2]       (estimated channel)
```

A reference architecture (ChannelNet / SRCNN-style):

```python
import torch
import torch.nn as nn

class ChannelEstNet(nn.Module):
    """
    Input  : LS estimate on pilot grid, zero-filled to full grid.
             Shape: [B, 2, F, T]   (real/imag, subcarriers, symbols)
    Output : Refined channel estimate over the full grid.
    """
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(2, channels, 9, padding=4)
        self.conv2 = nn.Conv2d(channels, channels // 2, 1)
        self.conv3 = nn.Conv2d(channels // 2, 2, 5, padding=2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):                  # x: [B, 2, F, T]
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return self.conv3(x)
```

**Hardware sizing — back of the envelope.**

For 5G NR, 100 MHz, numerology 1: 273 PRBs × 12 = 3276 subcarriers, 14 symbols per slot, 0.5 ms slot. The DMRS pattern (Type 1, 1-symbol) gives ~6552 pilot REs per slot.

With the network above and `channels=64`:
- conv1: 2·64·9·9 = 10,368 MACs/output × 3276·14 outputs ≈ 475 M MACs
- conv2: 64·32·1·1 = 2,048 MACs/output × 3276·14 ≈ 94 M MACs
- conv3: 32·2·5·5 = 1,600 MACs/output × 3276·14 ≈ 73 M MACs

→ **≈ 640 M MACs per slot, × 2000 slots/s = 1.28 TMACs/s = 2.56 TOPS** for a single 100 MHz carrier on a single UE.

A Qualcomm Hexagon NPU (X75 modem class) delivers ~10–15 TOPS at INT8 in a phone power envelope. So this estimator fits — but only with **INT8, structured pruning, and tile-friendly layouts**. A naive FP16 deployment would not.

**Where this lives on silicon:**
- **UE side:** modem NPU (Hexagon Tensor, Samsung's Exynos NPU, Apple's RF baseband neural blocks). Tightly coupled to the LDPC/Polar decoder via on-die SRAM.
- **gNB side:** baseband DU. **Nvidia Aerial cuPHY** runs this on an L4/L40S/A100 GPU. **Intel FlexRAN** runs it on Xeon AVX-512 / AMX with optional FPGA offload.

### A.2 Neural demapping — the cheap, high-ROI block

Soft demappers convert equalized symbols into **log-likelihood ratios (LLRs)** for the channel decoder. The exact LLR for QAM-256 in colored interference is messy; classical receivers use a Gaussian approximation that loses 0.3–0.8 dB in realistic conditions.

A neural demapper is tiny — usually an **MLP** with two hidden layers of 64–128 units, run **per resource element**. It is highly parallel (no temporal dependency), maps cleanly onto vector engines, and is the most common "first real ML PHY block" deployed.

```python
class NeuralDemapper(nn.Module):
    """Map (equalized_symbol_re, equalized_symbol_im, noise_var) -> LLRs"""
    def __init__(self, bits_per_symbol=8, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, bits_per_symbol),
        )
    def forward(self, x):  # x: [N, 3] -> [N, bits_per_symbol]
        return self.net(x)
```

This is a **per-RE pointwise op**. Throughput is gated by memory bandwidth from the equalizer output, not by compute. On a Hexagon NPU it sustains 100s of millions of demaps/sec at INT8 — comfortable for a 100 MHz carrier.

### A.3 Beam management — the Rel-19 story

mmWave (FR2) and emerging FR3 systems use **hundreds** of narrow beams. Classical beam acquisition does an exhaustive sweep of an SSB codebook — slow and energy-hungry. ML approaches predict the best beam from:

* a **partial sweep** of a few beams (spatial-domain beam prediction), or
* **historical beams** (temporal-domain prediction), or
* **side channels**: camera, GPS, environment map (sensor-aided / "Synesthesia of Machines").

3GPP Rel-19 has a normative work item for **AI/ML beam management** with two sub-use-cases (spatial and temporal) — the first time a learned model gets explicit signaling support in the standard.

A reference architecture is a small CNN over the RSRP grid of swept beams:

```
Input  : RSRP grid [num_swept_beams_az × num_swept_beams_el]
Hidden : 2 conv layers + 1 FC
Output : softmax over the full codebook (e.g., 64 or 256 beams)
```

The hardware question is **where the camera/sensor features fuse with the RF features** — typically the AP's NPU (Snapdragon Auto, Jetson) rather than the modem itself.

### A.4 End-to-end autoencoder PHY (research only — but important)

In this paradigm, the Tx (encoder) and Rx (decoder) are **co-trained neural networks**. The "channel" is a **differentiable layer** (AWGN noise, multipath impulse response, possibly hardware impairments). The constellation, pulse shape, and receiver are all learned jointly.

```
bits ──► [NN encoder] ──► waveform ──► [channel] ──► [NN decoder] ──► bits
              ▲                                            ▲
              └───── jointly trained via BCE / BLER loss ───┘
```

The result is a **non-standard air interface**. Constellations look weird (not QAM), pilots disappear, and there is no longer a clean PHY/MAC split. This is incompatible with 3GPP signaling — so it lives in **non-cellular** niches: underwater acoustic, free-space optical, LEO ISL, military waveforms.

It is also the cleanest demonstration of **hardware-software co-design** in radio: the encoder/decoder topology is fixed at silicon-design time, weights ship in firmware, and updates are model retrains.

### A.5 Hardware reality check — what actually ships

| Vendor / Platform | AI compute available to the PHY | Used for |
|---|---|---|
| **Qualcomm Snapdragon X75/X80 modem** | Hexagon Tensor + 2nd-gen "AI Processor" inside the modem | Channel est., demapping, link adaptation, PA DPD |
| **Samsung Exynos Modem 5400** | NPU shared with AP via tightly coupled L2 | Channel est., predictive scheduling hints |
| **MediaTek M90 (Dimensity)** | APU + DSP cluster | Beam mgmt, link adaptation |
| **Nvidia Aerial cuPHY** | A100 / L40S / H100 GPU per DU | Full inline GPU PHY: FFT, ch.est., MIMO, demap, LDPC, all CUDA kernels |
| **Intel FlexRAN** | Xeon AVX-512/AMX + optional vRAN FPGA (ACC100/200) | Software PHY with FPGA offload for FEC |
| **Marvell OCTEON 10 Fusion** | DPU + ML accelerator | Inline DU PHY for O-RAN |
| **AMD T2 (Pensando)** | Adaptive SoC + ML engines | Vendor-specific O-RAN DU |

> **Key Insight:** Inline PHY ML is hardest where you'd expect — the **UE modem**, with tens of milliwatts to spare and microsecond budgets. The basement (DU/gNB) has effectively unlimited compute by comparison; the engineering problem there is **scheduling latency and determinism**, not throughput.

---

## Part B — RAN-Level ML and the O-RAN RIC

The 3GPP PHY processes a slot in **microseconds**. The MAC scheduler runs every 1 ms. RRM (resource management) decisions happen on tens of milliseconds. Above that — load balancing, energy savings, mobility, anomaly detection — there is a **seconds-to-minutes** loop that is the natural home of "infrastructure ML".

O-RAN formalized this with the **RIC (RAN Intelligent Controller)**:

```
   ┌──────────────────────────────────────────────────────────┐
   │              Non-RT RIC  (in SMO, > 1 s)                  │
   │   rApps: training pipelines, policy generation, A1 mgmt   │
   │                  GPU/CPU servers, off-RAN                 │
   └──────────────────────────┬───────────────────────────────┘
                              │ A1 (policy/intent)
                              ▼
   ┌──────────────────────────────────────────────────────────┐
   │             Near-RT RIC  (10 ms – 1 s)                    │
   │   xApps: traffic steering, QoS, anomaly det., handover    │
   │             CPU + GPU/FPGA accelerator                    │
   └──────────────────────────┬───────────────────────────────┘
                              │ E2 (per-cell telemetry + actions)
                              ▼
   ┌──────────────────────────────────────────────────────────┐
   │   O-CU  ◄────►  O-DU  ◄────►  O-RU                        │
   │                  ▲                                        │
   │              (μs–ms PHY/MAC, inline ML here)              │
   └──────────────────────────────────────────────────────────┘
```

### B.1 What runs as an xApp

* **Traffic steering** — pick which cell / carrier a UE should be on, given load and QoS class.
* **MIMO mode selection** — SU- vs MU-MIMO, rank adaptation, codebook selection.
* **Energy savings** — switch off cells / carriers / antenna ports under low load (3GPP TR 38.864).
* **Anomaly / intrusion detection** — flag rogue base stations, jammers, IMSI catchers.
* **QoS prediction** — predict throughput on a route (e.g., for V2X).

These are mostly **contextual bandits** or **lightweight DRL**. The decision interval is comfortable, but the action space can be large (think: "for each of 1000 UEs, pick one of 8 cells, every 100 ms"), so model size is gated by inference latency × cell count, not by accuracy.

### B.2 What runs as an rApp

The Non-RT RIC handles **training**. An rApp typically:

1. Pulls aggregated KPIs and per-UE traces from the SMO data lake.
2. Trains a model (often offline, often on GPUs in a colo).
3. Pushes the model artifact + a **policy** to one or more xApps via A1.

This is where ML pipelines look most like conventional MLOps — Kubeflow, Airflow, MLflow, model registry. The interesting wrinkle is that the **deployment target** (an xApp on a near-RT RIC) has hard latency constraints that the training job must respect.

### B.3 ML for energy savings — the load-bearing use case

Operators care about wireless ML mostly because it **reduces opex**. Cell carriers consume **40–60% of RAN energy**. ML-driven cell shutdown decisions, based on predicted traffic and mobility, are deployed at scale (Vodafone, DT, NTT, KDDI all have running production rApps).

The model is usually unimpressive — gradient-boosted trees or a small LSTM on per-cell traffic timeseries. The **systems** problem is non-trivial:
- Latency for waking a cell back up is 100–500 ms (RF PA warmup, sync).
- A bad prediction = dropped session or coverage hole.
- The policy must be **explainable** to operations staff and **bounded** (e.g., "never shut off more than N cells in this tracking area").

This is a perfect case where the AI hardware engineer's job is **not** to add TOPS — it is to fit the explainability + safety harness onto a deployment that already works.

---

## Part C — SDR + Deep Learning

**Software-defined radio (SDR)** is the right teaching platform: it lets you put a real PyTorch model on real **IQ samples**. The minimum viable bench:

| Hardware | Sample rate | Use |
|---|---|---|
| **RTL-SDR ($30)** | ≤ 2.4 MS/s | Modulation classification, FM/AIS demod |
| **HackRF One** | 20 MS/s | Modulation classification, wideband scan |
| **Adalm-Pluto** | 60 MS/s, TX+RX | Loopback experiments, end-to-end NN PHY |
| **USRP B210 / B205mini** | 56 MS/s, 2×2 MIMO | Serious experiments |
| **USRP X310 / N310** | 200 MS/s | Production-grade |
| **Jetson Orin + USRP** | depends | Edge ML on IQ |

### C.1 Modulation classification — the "hello world" of RF ML

Given a snippet of IQ samples, classify the modulation (BPSK / QPSK / 16QAM / 64QAM / GFSK / OFDM / AM / FM / …). DeepSig's **RML2018.01a** dataset is the standard benchmark.

A reference CNN (O'Shea / DeepSig "ResNet-style"):

```python
class ModClassNet(nn.Module):
    def __init__(self, n_classes=24):
        super().__init__()
        # Input: [B, 2, 1024]  (I/Q over 1024 samples)
        self.features = nn.Sequential(
            nn.Conv1d(2, 64, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(64, n_classes)
    def forward(self, x):
        return self.head(self.features(x).squeeze(-1))
```

On RML2018.01a, this kind of model hits ~80% top-1 above 10 dB SNR (vs ~60% for handcrafted cumulant-based classifiers).

### C.2 GNU Radio + PyTorch loopback

GNU Radio flowgraphs are how SDR work actually happens. The PyTorch interface is a custom Python block:

```python
# torch_classifier_block.py — GNU Radio sink that runs a PyTorch model
import numpy as np
import torch
from gnuradio import gr

class TorchClassifier(gr.sync_block):
    def __init__(self, model_path, window=1024):
        gr.sync_block.__init__(
            self,
            name="torch_classifier",
            in_sig=[np.complex64],
            out_sig=None,
        )
        self.window = window
        self.model = torch.jit.load(model_path).eval().cuda()
        self.buf = np.zeros(window, dtype=np.complex64)

    def work(self, input_items, output_items):
        x = input_items[0][: self.window]
        if len(x) < self.window:
            return len(x)
        iq = np.stack([x.real, x.imag], axis=0)             # [2, W]
        t = torch.from_numpy(iq).float().unsqueeze(0).cuda()
        with torch.no_grad():
            logits = self.model(t)
        # publish via stream tags / ZMQ / gr.tag_t — flow-graph specific
        return len(x)
```

The realistic pitfalls:

* **Sample-rate mismatch** — train at the *same* sample rate (or with the *same* resampling) you'll deploy at, otherwise the IQ statistics change.
* **DC offset / IQ imbalance** — every SDR has them; either calibrate or include them in training augmentations.
* **Frequency offset** — randomize during training; classifiers that assume perfectly synced carriers fall apart at SNRs they were trained on.
* **Frame boundary** — for protocols with packet structure (Wi-Fi, BLE), align the input window with the preamble or accept the alignment as random and train accordingly.

### C.3 RF fingerprinting — the security-adjacent use case

Two transmitters running the *same* standard produce subtly different IQ tails because of **unique RF impairments** (PA nonlinearity, oscillator phase noise, mixer leakage). Models trained on **raw IQ** can distinguish individual devices — useful for authentication ("is this really my device?") and adversarial ("is this device the one that was advertised?").

The hardware story: this works well on **Jetson Orin Nano** sitting next to an SDR, because the classifier is small and the **bottleneck is RF, not compute**. It is a clean roadmap project that exercises the entire embedded-AI stack.

### C.4 Where ML helps vs where it doesn't (in SDR)

| Task | ML helps? | Why |
|---|---|---|
| Modulation classification (low SNR) | **Yes** | Beats hand-engineered features |
| Carrier / timing sync | Rarely | Mature classical algorithms, hard to beat |
| FFT / channelization | **No** | DSP wins on power and predictability |
| Spectrum sensing (cognitive radio) | **Yes** | Captures non-Gaussian interference |
| Direction finding (DoA) | Sometimes | NN-MUSIC competitive in low snapshot regimes |
| Decoding known protocols (Wi-Fi, BLE) | **No** | Standards-compliant decoders are optimal and free |
| Decoding **unknown** protocols | **Yes** | The whole point of "RF reverse engineering" |

---

## Where Inference Actually Lives — A Latency Map

```
   Budget       Block                            Hardware
   ───────────────────────────────────────────────────────────────
   ~ns          Per-sample IQ processing         RFIC / dedicated DSP
                (filters, mixers, FFT)
   ~µs          Per-OFDM-symbol                  FPGA / hard accel
                (timing sync, FFT, eq.)
   ~10 µs       Per-RE demap, LLR                NPU / GPU tensor core
   ~100 µs      Per-slot channel est.,           NPU / GPU
                MIMO detect, decode
   ~1 ms        Per-TTI MAC scheduling           CPU + small NPU hints
   ~10 ms       Link adaptation, BSR/PHR proc.   CPU
   ~100 ms      Mobility prediction, beam mgmt   xApp on near-RT RIC
   ~1 s         Cell on/off, slice allocation    xApp / rApp
   ~min         Capacity planning, fault triage  rApp / SMO ML platform
   ~hour+       Network design, model retraining offline GPU clusters
```

This is the hardware engineer's map. The "where does ML belong on this radio?" question reduces to: **which row of this table am I in?** Each row dictates a different accelerator, a different programming model, a different deployment cadence, and a different risk profile when the model is wrong.

---

## Hands-On Exercises

1. **Neural channel estimator on synthetic 3GPP data.** Generate a TDL-C channel realization (Python: `py3gpp` or `sionna`). Compute LS pilot estimates, train the `ChannelEstNet` from §A.1 to denoise/interpolate to the full grid. Compare MSE vs LMMSE at SNRs from −5 to 25 dB. Then quantize to INT8 (PTQ) and re-measure — expect <0.5 dB degradation; if you see more, your activation range calibration is wrong.

2. **Neural demapper drop-in.** Build a 64-QAM transmitter and an OFDM receiver in Sionna (NVIDIA's GPU-accelerated link-level simulator). Replace its analytic demapper with the `NeuralDemapper` from §A.2. Measure BLER vs SNR with and without the neural block. Profile inference latency on the host GPU and on a Jetson Orin (TensorRT FP16 and INT8).

3. **Modulation classifier on real IQ.** Capture 30 minutes of IQ in a busy ISM band with an RTL-SDR or HackRF. Slice into 1024-sample windows. Train the `ModClassNet` from §C.1 on **synthetic** GNU Radio data (so labels are correct). Run inference on the captured real data; compare predictions to what you can identify by hand on a waterfall. Document where the model is over-confident.

4. **GNU Radio + PyTorch pipeline.** Build a flowgraph: `osmocom_source` → `low_pass_filter` → `torch_classifier_block` → `file_sink`. Run on Jetson Orin Nano with TensorRT-converted weights. Measure end-to-end latency from RF tap to classification publish, and identify the dominant cost (IQ DMA, host↔device copy, inference, post-processing).

5. **Hardware sizing exercise.** Pick a target air interface (Wi-Fi 7, 5G FR1 100 MHz, 5G FR2 400 MHz, LEO Ka-band). Choose **one** PHY block to neuralize. Derive:
   - MACs/sec and bytes/sec at the block's input and output,
   - the implied SRAM working-set,
   - the silicon area at 5 nm with a reasonable MAC/mm² figure,
   - whether the block fits in the modem's existing NPU or needs a dedicated tile.

   Compare to public information about a deployed modem (e.g., Snapdragon X80 reference docs). Identify where your estimate is off and why.

6. **Pick an O-RAN xApp and read the E2 spec.** Skim O-RAN.WG3.E2SM-KPM and pick a target use case (e.g., traffic steering). Sketch the data flow from per-UE KPI report → xApp model inference → A1 / RIC control message. Note the **action latency** budget (typically 10–100 ms) and identify what model size that lets you ship.

---

## Key Takeaways

| Takeaway | Why it matters for AI hardware |
|---|---|
| Channel estimation is the most mature deployed PHY-ML block | If you're designing a modem NPU, the workload profile starts here |
| The latency ladder dictates the accelerator | µs jobs go to FPGA/NPU, ms jobs to CPU+NPU, seconds+ go to GPU servers |
| Inline PHY ML is bandwidth-bound, not compute-bound | Layout, tiling, and SRAM design dominate; raw TOPS is a vanity metric |
| O-RAN's RIC is the canonical home for "slow ML" in wireless | Standardized telemetry (E2/A1) means MLOps tools transfer cleanly |
| End-to-end autoencoder PHY exists but doesn't ship in 3GPP | Useful for non-cellular niches; doesn't drive cellular silicon roadmaps yet |
| Beam management is the next standardized ML block (3GPP Rel-19) | First normative ML in cellular standards — silicon vendors will react |
| SDR + a Jetson is the right teaching bench | You can build a complete neural-receiver demo with off-the-shelf parts |

---

## Resources

* **[NVIDIA Sionna](https://nvlabs.github.io/sionna/):** GPU-accelerated link-level simulator with first-class PyTorch / JAX integration. The reference platform for PHY-ML research.
* **[NVIDIA Aerial / cuPHY](https://developer.nvidia.com/aerial):** Production GPU-resident PHY for 5G/6G gNB DU.
* **[Intel FlexRAN Reference Architecture](https://networkbuilders.intel.com/solutionslibrary/flexran):** CPU+FPGA reference for software RAN — useful counterpoint to Aerial.
* **[O-RAN Alliance Specifications](https://www.o-ran.org/specifications):** The canonical place to read about RIC, E2, A1, and ML use cases. Start with WG2 (Non-RT RIC) and WG3 (Near-RT RIC).
* **[3GPP TR 38.843 — AI/ML for the air interface](https://www.3gpp.org/dynareport/38843.htm):** The study report behind the Rel-19 normative work; covers CSI feedback, beam management, positioning.
* **[DeepSig RadioML datasets](https://www.deepsig.ai/datasets):** RML2016 / RML2018 — the standard benchmarks for modulation classification.
* **["An Introduction to Deep Learning for the Physical Layer," O'Shea & Hoydis (2017)](https://arxiv.org/abs/1702.00832):** The foundational autoencoder-PHY paper. Read this once.
* **["Machine Learning at the Wireless Edge," Park et al.](https://ieeexplore.ieee.org/document/9145080):** Survey covering edge-ML hardware constraints in wireless.
* **[Qualcomm Wireless AI Research](https://www.qualcomm.com/research/artificial-intelligence/wireless-ai):** Publications and white papers from the team behind Hexagon-resident PHY ML.
* **[GNU Radio](https://www.gnuradio.org/) + [gr-torch](https://github.com/gnuradio):** SDR framework + community PyTorch integrations.
* **[Adalm-Pluto Getting Started](https://wiki.analog.com/university/tools/pluto):** Cheapest TX-capable SDR — good for end-to-end NN PHY loopback labs.
