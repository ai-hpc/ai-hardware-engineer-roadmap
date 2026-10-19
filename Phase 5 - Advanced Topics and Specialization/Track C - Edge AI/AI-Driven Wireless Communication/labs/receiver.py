"""CPU teaching receiver: explicit channel, LLR sign, held-out data and sample order.

Python standard library only. This is a synthetic real-valued BPSK experiment,
not an NR receiver, RF capture, channel decoder or hardware benchmark.
"""
import json
import math
import random
from dataclasses import dataclass


def awgn_bits(count, ebn0_db, bias, seed):
    """Equiprobable bits, x=2b-1, real noise variance N0/2; Eb=1."""
    if count <= 0 or not all(map(math.isfinite, (ebn0_db, bias))):
        raise ValueError("positive count and finite channel parameters required")
    variance = 1.0 / (2.0 * 10 ** (ebn0_db / 10.0))
    rng = random.Random(seed)
    return [(2 * (bit := rng.randrange(2)) - 1 + bias
             + rng.gauss(0, math.sqrt(variance)), bit) for _ in range(count)]


def sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1 + exp_value)


def loss_and_gradient(rows, weight, intercept):
    """BCE with logit weight*y+intercept; mean gradient over supplied rows."""
    if not rows:
        raise ValueError("training batch cannot be empty")
    loss = dw = dc = 0.0
    for sample, bit in rows:
        logit = weight * sample + intercept
        loss += max(logit, 0) - bit * logit + math.log1p(math.exp(-abs(logit)))
        residual = sigmoid(logit) - bit
        dw += residual * sample
        dc += residual
    return tuple(value / len(rows) for value in (loss, dw, dc))


def fit(rows, epochs=400, learning_rate=0.5):
    weight = intercept = 0.0
    for _ in range(epochs):
        _, dw, dc = loss_and_gradient(rows, weight, intercept)
        weight -= learning_rate * dw
        intercept -= learning_rate * dc
    return weight, intercept


def metrics(rows, weight, intercept):
    errors = sum((weight * sample + intercept >= 0) != bit for sample, bit in rows)
    return {"bits": len(rows), "errors": errors, "ber": errors / len(rows),
            "bce": loss_and_gradient(rows, weight, intercept)[0]}


@dataclass(frozen=True)
class Window:
    start_sample: int
    samples: tuple


class IQWindows:
    """Non-overlapping IQ windows; preserve partial input across scheduler calls.

    start_sample counts samples from the first push, never callback invocations.
    A stream discontinuity must create a new instance, not silently concatenate.
    """
    def __init__(self, size):
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("window size must be a positive integer")
        self.size = size
        self.pending = []
        self.next_start = 0
        self.finished = False

    def push(self, samples):
        if self.finished:
            raise ValueError("capture is finished; create a new assembler for a new stream")
        values = list(samples)
        if any(not math.isfinite(complex(x).real) or not math.isfinite(complex(x).imag)
               for x in values):
            raise ValueError("IQ samples must be finite")
        # Validate the full batch before mutating state.
        self.pending.extend(values)
        windows = []
        complete = len(self.pending) // self.size
        for index in range(complete):
            start = index * self.size
            windows.append(Window(self.next_start + start,
                                  tuple(self.pending[start:start + self.size])))
        self.next_start += complete * self.size
        del self.pending[:complete * self.size]
        return windows

    def finish(self):
        """End of finite capture: return explicitly discarded trailing sample count."""
        count = len(self.pending)
        self.pending.clear()
        self.finished = True
        return count


def nr_slot_us(numerology):
    if isinstance(numerology, bool) or not isinstance(numerology, int) or not 0 <= numerology <= 6:
        raise ValueError("numerology must be an integer from 0 to 6")
    return 1000 / 2 ** numerology


def receiver_budget():
    positions = 273 * 12 * 14
    layer_macs = [positions * 2 * 64 * 9 * 9,
                  positions * 64 * 32,
                  positions * 32 * 2 * 5 * 5]
    return {"mu1_slot_us": nr_slot_us(1), "mu3_slot_us": nr_slot_us(3),
            "conv_macs_per_slot": layer_macs, "total_macs_per_slot": sum(layer_macs),
            "required_tmac_per_second": sum(layer_macs) * 2000 / 1e12,
            "first_activation_fp16_bytes": positions * 64 * 2,
            "one_antenna_complex_int16_bytes_per_second": 30_720_000 * 4,
            "window_1024_at_2msps_us": 1024 / 2_000_000 * 1e6}


def experiment():
    snr_db, bias = 2.0, 0.6
    variance = 1 / (2 * 10 ** (snr_db / 10))
    training = awgn_bits(2000, snr_db, bias, 11)
    weight, intercept = fit(training)
    results = {}
    for name, actual_bias, seed in (("matched", bias, 22), ("shifted", -bias, 33)):
        test = awgn_bits(20000, snr_db, actual_bias, seed)
        results[name] = {
            "actual_bias": actual_bias,
            "zero_threshold": metrics(test, 2 / variance, 0),
            "matched_oracle": metrics(test, 2 / variance, -2 * actual_bias / variance),
            "learned": metrics(test, weight, intercept)}
    stream = IQWindows(4)
    windows = stream.push([complex(i, -i) for i in range(3)])
    windows += stream.push([complex(i, -i) for i in range(3, 11)])
    return {"experiment": "synthetic real BPSK; no RF or NR execution",
            "ebn0_db": snr_db, "train_bits": len(training),
            "learned_weight": weight, "learned_intercept": intercept,
            "learned_threshold": -intercept / weight,
            "analytic_matched_ber": 0.5 * math.erfc(math.sqrt(10 ** (snr_db / 10))),
            "results": results, "budget": receiver_budget(),
            "window_start_samples": [w.start_sample for w in windows],
            "explicit_trailing_samples_discarded": stream.finish()}


if __name__ == "__main__":
    print(json.dumps(experiment(), indent=2))
