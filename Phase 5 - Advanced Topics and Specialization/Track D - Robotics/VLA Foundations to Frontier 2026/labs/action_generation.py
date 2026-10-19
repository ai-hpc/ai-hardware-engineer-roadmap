"""Train scalar conditional epsilon/velocity models; optional CPU PyTorch.

Run from the course directory: python labs/action_generation.py --output /tmp/action-run
The categories are supplied labels, not a learned language encoder. No image,
robot dynamics, ACT architecture, or upstream VLA implementation runs here.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import random
import statistics
import time

import torch
from torch import nn
from torch.nn import functional as F

from action_math import (LABELS, SIGMA, alpha_schedule, ddim_indices,
                         distribution_metrics, oracle_sample)


class ActionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(13, 64), nn.SiLU(),
                                    nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, x, t, labels):
        phase = t * torch.tensor([1, 2, 4, 8], dtype=x.dtype, device=x.device) * math.pi
        features = torch.cat((x, t, phase.sin(), phase.cos(),
                              F.one_hot(labels, 3).to(x.dtype)), dim=1)
        return self.layers(features)


def training_batch(count, generator, method):
    if count < 1 or method not in ("flow", "ddim"):
        raise ValueError("positive batch size and method flow/ddim required")
    labels = torch.randint(3, (count,), generator=generator)
    random_side = 2 * torch.randint(2, (count,), generator=generator) - 1
    center = torch.where(labels == 0, -1, torch.where(labels == 1, 1, random_side))
    clean = center.float().unsqueeze(1) + SIGMA * torch.randn(count, 1, generator=generator)
    noise = torch.randn(count, 1, generator=generator)
    if method == "flow":
        t = torch.rand(count, 1, generator=generator)
        return (1 - t) * noise + t * clean, t, labels, clean - noise
    indices = torch.randint(1, 65, (count, 1), generator=generator)
    signal = torch.tensor(alpha_schedule())[indices]
    return signal.sqrt() * clean + (1 - signal).sqrt() * noise, indices.float() / 64, labels, noise


def fit(method: str, seed: int, updates: int):
    if updates < 1:
        raise ValueError("updates must be positive")
    torch.manual_seed(seed)
    model = ActionNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_rng = torch.Generator().manual_seed(seed + 1000)
    validation = training_batch(4096, torch.Generator().manual_seed(2026), method)

    def loss_on_validation():
        with torch.no_grad():
            return F.mse_loss(model(*validation[:3]), validation[3]).item()

    initial = loss_on_validation()
    history = []
    start = time.perf_counter()
    for step in range(1, updates + 1):
        x, t, labels, target = training_batch(256, train_rng, method)
        loss = F.mse_loss(model(x, t, labels), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 500 == 0 or step == updates:
            history.append({"update": step, "validation_loss": loss_on_validation()})
    model.eval()
    return model, {"seed": seed, "updates": updates, "batch_size": 256,
                   "train_generator_seed": seed + 1000, "validation_seed": 2026,
                   "initial_validation_loss": initial, "history": history,
                   "training_seconds": time.perf_counter() - start}


@torch.no_grad()
def sample(model, noise, labels, method, evaluations, *, reverse_sign=False):
    if method not in ("flow", "ddim") or type(evaluations) is not int or evaluations < 1:
        raise ValueError("method flow/ddim and positive integer evaluations required")
    if noise.ndim != 2 or noise.shape[1] != 1 or labels.shape != (noise.shape[0],):
        raise ValueError("noise must be (batch,1) and labels (batch,)")
    if not torch.isfinite(noise).all() or not ((labels >= 0) & (labels <= 2)).all():
        raise ValueError("finite noise and labels in [0,2] required")
    x = noise.clone()
    if method == "flow":
        for i in range(evaluations):
            t = torch.full_like(x, i / evaluations)
            x += (-1 if reverse_sign else 1) * model(x, t, labels) / evaluations
    else:
        schedule = alpha_schedule()
        levels = ddim_indices(64, evaluations)
        for current, following in zip(levels, levels[1:]):
            alpha, next_alpha = schedule[current], schedule[following]
            epsilon = model(x, torch.full_like(x, current / 64), labels)
            clean = (x - math.sqrt(1 - alpha) * epsilon) / math.sqrt(alpha)
            # Explicit numerical choice: avoid large epsilon error amplification
            # at the almost-noise endpoint. This changes the sampler; record it.
            clean = clean.clamp(-2.0, 2.0)
            x = math.sqrt(next_alpha) * clean + math.sqrt(1 - next_alpha) * epsilon
    return x


def evaluate(model, method, seed, counts=(8, 32, 64), samples=2048):
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(samples, 1, generator=generator)
    rows = []
    for evaluations in counts:
        for label_id, label in enumerate(LABELS):
            labels = torch.full((samples,), label_id, dtype=torch.long)
            sample(model, noise, labels, method, evaluations)  # warm-up
            timings = []
            for _ in range(3):
                start = time.perf_counter()
                generated = sample(model, noise, labels, method, evaluations)
                timings.append(1000 * (time.perf_counter() - start))
            rows.append({"method": method, "evaluations": evaluations, "label": label,
                         "warm_batch_ms_median_of_3": statistics.median(timings),
                         **distribution_metrics(generated[:, 0].tolist(), label)})
    labels = torch.zeros(samples, dtype=torch.long)
    wrong_label = sample(model, noise, labels + 1, method, 32)[:, 0].tolist()
    controls = {"right_label_scored_as_left": distribution_metrics(wrong_label, "left")}
    if method == "flow":
        reversed_values = sample(model, noise, labels, method, 32, reverse_sign=True)[:, 0].tolist()
        controls["reversed_flow_scored_as_left"] = distribution_metrics(reversed_values, "left")
    return rows, controls, noise


@torch.no_grad()
def reload_verified(model, checkpoint, probe):
    """Reject a corrupted or incompatible export before scoring its samples.

    Exact equality is the deliberately narrow contract here: same CPU, dtype,
    architecture and evaluation mode. It is not a cross-device tolerance rule.
    """
    restored = ActionNet().eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True)["state_dict"])
    expected, actual = model(*probe[:3]), restored(*probe[:3])
    if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
        raise ValueError("checkpoint reload probe produced nonfinite output")
    error = (expected - actual).abs().max().item()
    if not torch.equal(expected, actual):
        raise ValueError(f"checkpoint reload parity failed: maximum absolute error {error}")
    return restored, error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=4000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 31])
    args = parser.parse_args()
    if args.updates < 1:
        parser.error("--updates must be positive")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"scope": "synthetic scalar action distributions; no VLA/robot benchmark",
              "python": platform.python_version(), "torch": torch.__version__,
              "threads": torch.get_num_threads(), "action_stddev": SIGMA,
              "diffusion_alpha_terminal": alpha_schedule()[-1],
              "ddim_eta": 0, "ddim_clean_clip": [-2, 2], "sampling_seed": 991,
              "training": [], "metrics": [], "negative_controls": []}
    for method in ("flow", "ddim"):
        for seed in args.seeds:
            model, training = fit(method, seed, args.updates)
            checkpoint = args.output / f"{method}-{seed}.pt"
            torch.save({"method": method, "seed": seed, "state_dict": model.state_dict(),
                        "schema": "scalar-action-v1", "labels": LABELS,
                        "sigma": SIGMA, "ddim_clean_clip": [-2, 2]}, checkpoint)
            probe = training_batch(32, torch.Generator().manual_seed(817), method)
            restored, training["reload_max_abs_error"] = reload_verified(model, checkpoint, probe)
            rows, controls, _ = evaluate(restored, method, 991)
            result["training"].append({"method": method, **training})
            result["metrics"].extend({"training_seed": seed, **row} for row in rows)
            result["negative_controls"].append({"method": method, "training_seed": seed, **controls})
            print(method, seed, training["history"][-1], flush=True)
    rng = random.Random(991)
    noises = [rng.gauss(0, 1) for _ in range(2048)]
    result["oracle_metrics"] = []
    for method in ("flow", "ddim"):
        for count in (8, 32, 64):
            for label in LABELS:
                values = [oracle_sample(label, z, count, method) for z in noises]
                result["oracle_metrics"].append({"method": method, "evaluations": count, "label": label,
                                                  **distribution_metrics(values, label)})
    result["conditional_mean_baseline"] = [
        {"label": label, **distribution_metrics([mean] * 2048, label)}
        for label, mean in zip(LABELS, (-1.0, 1.0, 0.0))]
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
