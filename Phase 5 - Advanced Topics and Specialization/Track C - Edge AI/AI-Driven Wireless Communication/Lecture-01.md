# AI-driven wireless: from noisy samples to a measured receiver

**Reviewed 18 October 2026.** Start with Python, complex numbers, probability, and gradient descent. The first experiment needs only Python 3.10+ and runs on a CPU. Allow 60–90 minutes. Its output is a receiver-quality report and a sample-buffer test, with no radio hardware or transmitted signal.

You will learn to distinguish three tasks: estimating bits from received samples, meeting a radio pipeline's timing budget, and choosing slower network-control policies. A neural network can help one task while making another worse.

## 1. Follow one bit through a radio

A transmitter maps bits into symbols, shapes them into a waveform, and sends that waveform through a channel. A receiver samples it, corrects timing/frequency errors, estimates the channel, and produces soft bit evidence for a channel decoder. A complex baseband sample `I + jQ` holds two real numbers: in-phase and quadrature components. Neither component alone is a complete arbitrary modulated signal.

In an OFDM receiver, a **resource element** is one subcarrier in one OFDM symbol. A **pilot** is a known symbol used to estimate the channel. At a pilot with nonzero transmitted value `x`, least squares gives `h_hat = y/x`. Interpolating those estimates across time and frequency is a separate problem. LMMSE is optimal among linear estimators under its covariance assumptions; general MMSE optimality needs a suitable distribution model too.

A learned receiver might refine channel estimates, predict bit likelihoods, or compensate a hardware impairment. It still needs a defined sample rate, tensor layout, output convention, and deadline. Learning only a receiver does not require changing the transmitted waveform. Jointly learning a new waveform may change interoperability and must be evaluated against the relevant interface requirements.

## 2. Work through a receiver you can solve exactly

Use binary phase-shift keying (BPSK) with this explicit convention:

\[
x=2b-1,\qquad y=x+d+n,\qquad n\sim\mathcal N(0,\sigma^2).
\]

Here `b` is 0 or 1, `d` is a constant receiver bias, and `n` is real Gaussian noise. Symbol/bit energy is 1, there is one bit per symbol, and there is no coding. If `gamma = 10**(EbN0_dB/10)`, then `sigma² = 1/(2 gamma)`. This real-noise variance differs from the total variance of a complex noise sample.

For equal bit priors, expand the two Gaussian squared errors:

\[
\ell(y)=\log\frac{P(b=1\mid y)}{P(b=0\mid y)}
=\frac{(y-d+1)^2-(y-d-1)^2}{2\sigma^2}
=\frac{2(y-d)}{\sigma^2}.
\]

Positive LLR means bit **1** in this lesson. The hard decision threshold is `y=d`. Other mappers can use the opposite bit convention; check before connecting a demapper to a decoder.

At `Eb/N0 = 2 dB`, `gamma ≈ 1.585`, `sigma² ≈ 0.3155`, and `d=0.6`. For `y=0.4`, the correct LLR is about `−1.268`, so choose bit 0. A receiver that incorrectly assumes zero bias chooses bit 1.

Now learn `ell_hat = w*y + c` from labeled samples. Binary cross entropy is

\[
L=\operatorname{softplus}(\ell_{hat})-b\ell_{hat},\quad
\frac{\partial L}{\partial w}=(\operatorname{sigmoid}(\ell_{hat})-b)y,\quad
\frac{\partial L}{\partial c}=\operatorname{sigmoid}(\ell_{hat})-b.
\]

This two-parameter model is enough for the specified channel. A larger neural network is unnecessary here. The matched analytic receiver is the reference against which learning should be judged.

## 3. Run, then explain, the CPU experiment

From the repository root:

```bash
cd 'Phase 5 - Advanced Topics and Specialization/Track C - Edge AI/AI-Driven Wireless Communication'
python3 labs/receiver.py
python3 -m unittest discover -s labs -p 'test_*.py' -v
```

Read [receiver.py](labs/receiver.py), [the tests](labs/test_receiver.py), and the [recorded result](labs/receiver-reference.json). Training uses 2,000 samples with seed 11; the two evaluations use 20,000 new samples each with seeds 22 and 33. Test samples never update the parameters.

Observed on Python 3.12.3:

| Receiver | Same bias `d=+0.6` | Changed bias `d=−0.6` |
|---|---:|---:|
| Incorrect zero-threshold assumption | 12.210% BER | 12.060% BER |
| Analytic receiver with the actual bias | 3.740% BER | 3.755% BER |
| Learned receiver, trained only at `+0.6` | 3.780% BER | 31.525% BER |

The learned threshold is about 0.586. It nearly matches 0.6 on the training distribution, then fails after the bias changes. The analytic reference is an **oracle** because it receives the true bias; a deployed receiver must estimate that value. Learning has not beaten a matched Gaussian detector in this experiment.

The theoretical matched BER is

\[
P_b=Q(\sqrt{2\gamma})=\tfrac12\operatorname{erfc}(\sqrt\gamma)
\approx 0.037506.
\]

Finite samples vary around that value. The regression test compares Monte Carlo error to a five-standard-error tolerance rather than demanding exact equality. BER counts incorrect **bits**. BLER counts **blocks** containing errors after the relevant decoding procedure; the lab has no code or decoder and makes no BLER claim. Hard-decision BER also cannot establish whether LLR magnitudes are calibrated; compare BCE and decoder performance separately.

**Try it:** increase the number of gradient steps. Does BER move as much as BCE? Then train across several biases while withholding one bias and one SNR. State which parameters the receiver observes; a model cannot infer an unobserved channel change from a single ambiguous sample without additional information or assumptions.

## 4. Preserve samples across callback boundaries

A streaming scheduler can supply 3 samples, then 8, then 1. A classifier needing windows of 4 must keep the first 3 until another sample arrives. Returning “consumed 3” while discarding them changes the signal.

`IQWindows.push()` buffers partial input and emits complete windows with absolute sample offsets. For the 11 samples `0…10`, supplied as chunks of 3 and 8, it emits windows starting at 0 and 4. Three samples remain; `finish()` explicitly reports that trailing partial window as discarded. Irregular chunks produce exactly the same windows as one complete capture. Nonfinite input is rejected before buffer mutation.

At sample rate 2 MS/s, collecting 1,024 samples takes **512 microseconds**. This acquisition time exists before inference begins. An overlapping classifier with stride 512 would need a new result every 256 microseconds in steady state, while its first window still needs 512 microseconds to arrive. The supplied assembler uses non-overlapping windows; overlap is an extension.

For GNU Radio 3.x integration, a `sync_block` with no output stream is a **sink**. A synchronous sink consumes the number of items returned by `work`; preserve every consumed sample in the buffer or record a deliberate drop. Publish classification results through a message port or a separate typed output design. To record raw IQ, branch the upstream stream to a file sink. A zero-output classifier sink cannot connect onward to a stream file sink. See the [official block types](https://wiki.gnuradio.org/index.php/Types_of_Blocks).

The portable lab tests buffering logic. It does not execute GNU Radio, a classifier checkpoint, or an SDR. A hardware extension must record timestamp origin, sample format, scaling, discontinuities, dropped windows, bounded queues, and capture configuration. Split evaluation by capture/session/device so overlapping windows from one recording do not leak into both training and test sets. Modulation recognition alone does not decode an unknown protocol.

## 5. Calculate the radio budget before choosing hardware

For normal cyclic prefix, NR has 14 OFDM symbols per slot and `2**mu` slots per 1 ms subframe. Therefore:

| Numerology `mu` | Subcarrier spacing | Average slot duration |
|---|---:|---:|
| 0 | 15 kHz | 1,000 µs |
| 1 | 30 kHz | 500 µs |
| 2 | 60 kHz | 250 µs |
| 3 | 120 kHz | 125 µs |

The duration follows [3GPP TS 38.211 v19.1.0, §4.2 and Table 4.3.2-1](https://www.etsi.org/deliver/etsi_TS/138200_138299/138211/19.01.00_60/ts_138211v190100p.pdf). A slot duration is not the whole estimator's available execution time: scheduling, buffering, channel decoding, transfer, and protocol deadlines also matter. Numerology 1 is **500 µs**, not 125 µs.

Consider an illustrative full-grid estimator with 273 PRBs, 12 subcarriers per PRB, and 14 symbols: 45,864 positions. Three same-resolution convolutions use channel/kernel tuples `(2→64,9×9)`, `(64→32,1×1)`, and `(32→2,5×5)`. Ignore bias and activation operations for this count.

\[
\text{MACs}=45864(2\cdot64\cdot81+64\cdot32+32\cdot2\cdot25)
=642,829,824.
\]

At 2,000 slots/s this requires **1.28566 TMAC/s**, or 2.57132 TOPS when one multiply plus one add counts as two operations. This is an arithmetic demand for this hypothetical network, not measured accelerator throughput. The first FP16 activation alone occupies 5,870,592 bytes, about 5.60 MiB. Multi-antenna streams, additional carriers, scratch storage, precision, and tiling change the cost.

A 30.72 MS/s stream with signed 16-bit I and 16-bit Q needs `30.72e6 × 4 = 122.88 MB/s` per antenna before transport overhead. Complex float32 doubles that payload. Peak TOPS alone cannot establish that the estimator fits: measure supported kernels, useful bandwidth, transfers, workspace, queueing, thermal conditions, and completion-time tails.

DMRS counts depend on mapping type, configuration type, symbols, ports, and CDM groups. Do not multiply the full grid by an unexplained “pilot percentage.” A network also needs to distinguish an absent pilot from a valid zero estimate, for example with an explicit pilot mask.

## 6. Put network intelligence in the right control loop

The PHY chain and an O-RAN controller serve different purposes. [O-RAN's architecture](https://docs.o-ran-sc.org/en/latest/architecture/architecture.html) places high PHY/MAC/RLC in the O-DU and low PHY/RF functions in the O-RU, subject to the split. The near-real-time RIC uses **E2** for telemetry and supported control actions; xApps extend that controller. The non-real-time RIC provides longer-timescale policies and model-management functions. **A1** carries policy guidance between non-real-time and near-real-time RIC components; it is not a replacement for E2 control of a radio node. See the [official A1 project](https://docs.o-ran-sc.org/projects/o-ran-sc-nonrtric/en/latest/overview.html).

A scheduler policy experiment needs a concrete observation, action, and objective. For example, observations might be per-user queue lengths and estimated rates; an action selects a supported allocation; the objective trades throughput against delay and fairness. Compare to a defined baseline using identical traffic traces. Include signaling delay and stale observations in the environment. A policy score cannot establish PHY deadline compliance or a universal energy-saving percentage.

For a receiver, report BER/BLER by channel and SNR. For a beam predictor, report selection error and resulting link performance with measurement overhead. For a RIC policy, report the actual network objective and constraints. One classification-accuracy number cannot substitute for all three.

## 7. Current optional simulator route

As reviewed on 18 October 2026, **Sionna 2.2.0** uses **PyTorch** for PHY/SYS; Sionna RT uses Mitsuba 3 and Dr.Jit. Older TensorFlow Sionna tutorials target a different major version. Start from the [current documentation](https://nvlabs.github.io/sionna/) and match installed package versions before copying a tutorial.

A second supplied exercise, [sionna_cpu.py](labs/sionna_cpu.py), was actually run with Sionna 2.2.0 and PyTorch 2.14.0+cpu. From this lesson directory, create an optional environment:

```bash
python3 -m venv .venv-sionna
.venv-sionna/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
.venv-sionna/bin/python -m pip install -r labs/requirements-sionna.txt
.venv-sionna/bin/python -m pip check
.venv-sionna/bin/python labs/sionna_cpu.py
```

This dependency set is substantially larger than the standard-library exercise. The recorded environment reused an existing CPU PyTorch installation and installed Sionna plus its dependencies in an isolated environment; `pip check` passed. The [observed result](labs/sionna-reference.json) is 3,914 errors in 100,000 bits (3.914%), within five standard errors of the 3.751% analytic value. It also verifies noiseless decisions, tensor dimensions, unit symbol energy, and finite nonzero gradients through the demapper. Random samples may differ across dependency/platform versions.

Here QPSK carries two bits per symbol, so with unit symbol energy `Eb=1/2`. Sionna's `AWGN(...)(symbols, no)` takes **total complex noise variance** `no=N0=1/(2 gamma)` in this example. Its real and imaginary noise components each have variance `N0/2`. The PAM mapper in this Sionna version maps bit 0 to +1, the opposite BPSK symbol labeling from §2; its demapper still returns positive LLR for bit 1. The supplied simulator example uses mapper and demapper together and asserts that convention.

The CPU exercise above intentionally exposes the equations without a simulator dependency. A next simulator experiment can add coding, QAM, fading, OFDM, and pilots one at a time. At each step preserve a classical baseline, assert tensor shapes and noise conventions, and rerun quality measurements. Record simulator version, channel parameters, random seeds, model/config hashes, test captures, and execution backend. Synthetic channel success establishes behavior under that simulator, not an over-the-air deployment result.

## Check your understanding

1. Why does changing the bias from +0.6 to −0.6 break the learned receiver? **Its learned decision threshold remains near +0.6; retraining or an observed/estimated channel parameter is needed.**
2. Does a fourfold faster accelerator fix a 1,024-sample acquisition delay? **It reduces compute time, not the 512 µs collection time at 2 MS/s.**
3. Why is `0.5 ms / network_runtime` insufficient evidence for a valid NR receiver? **It omits the other processing stages, scheduling and protocol deadline.**
4. If BER is unchanged after quantization, is the decoder necessarily unchanged? **No. Soft reliability magnitudes may change even when the hard decisions do not.**
5. Can a near-real-time xApp replace a per-symbol detector? **Their interfaces and timing roles differ; define the local PHY pipeline separately.**

**Deliverable:** submit the JSON receiver report, passing tests, a channel-shift plot or table, and a one-page bandwidth/timing calculation. Explain one failed hypothesis using the measured result.
