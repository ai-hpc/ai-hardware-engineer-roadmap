#!/usr/bin/env python3
"""Dependency-free synthetic behavior-cloning lab. No robot hardware is controlled."""

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import sys
import time


@dataclass(frozen=True)
class Sample:
    episode: int
    error: float
    velocity: float
    action: float


def make_dataset(seed=7, episodes=36, steps=30):
    """Related observations within each synthetic demonstration; noisy labels."""
    rng = random.Random(seed)
    rows = []
    for episode in range(episodes):
        position, velocity = rng.uniform(-1.0, 1.0), rng.uniform(-0.4, 0.4)
        for _ in range(steps):
            error = -position
            action = 0.8 * error - 0.6 * velocity
            rows.append(Sample(episode, error, velocity, action + rng.gauss(0, 0.02)))
            velocity += action * 0.1
            position += velocity * 0.1
    return rows


def validate_splits(parts):
    seen = set()
    for part in parts:
        if not part:
            raise ValueError("Every split must contain observations")
        ids = {row.episode for row in part}
        if ids & seen:
            raise ValueError("Episode leakage: a demonstration occurs in multiple splits")
        seen.update(ids)


def split_episodes(rows, seed=11):
    ids = sorted({row.episode for row in rows})
    if len(ids) < 6:
        raise ValueError("Need at least six episodes for three nonempty splits")
    random.Random(seed).shuffle(ids)
    n_train, n_valid = len(ids) * 2 // 3, len(ids) // 6
    groups = [set(ids[:n_train]), set(ids[n_train:n_train + n_valid]), set(ids[n_train + n_valid:])]
    parts = [[row for row in rows if row.episode in group] for group in groups]
    validate_splits(parts)
    return parts


def fit_scaler(train):
    if not train:
        raise ValueError("Cannot fit preprocessing on an empty dataset")
    columns = [[row.error for row in train], [row.velocity for row in train]]
    means = [statistics.fmean(column) for column in columns]
    scales = [max(statistics.pstdev(column), 1e-12) for column in columns]
    scales = [1.0 if value <= 1e-12 else value for value in scales]
    return means, scales


def features(error, velocity, scaler):
    means, scales = scaler
    return [(error - means[0]) / scales[0], (velocity - means[1]) / scales[1], 1.0]


def predict(weights, error, velocity, scaler):
    return sum(w * x for w, x in zip(weights, features(error, velocity, scaler)))


def loss_and_gradient(weights, rows, scaler):
    if not rows:
        raise ValueError("Cannot compute loss on an empty dataset")
    loss, grad = 0.0, [0.0] * 3
    for row in rows:
        xs = features(row.error, row.velocity, scaler)
        residual = sum(w * x for w, x in zip(weights, xs)) - row.action
        loss += residual * residual
        for j in range(3):
            grad[j] += 2.0 * residual * xs[j]
    return loss / len(rows), [g / len(rows) for g in grad]


def gradient_error(weights, rows, scaler, epsilon=1e-6):
    _, analytical = loss_and_gradient(weights, rows, scaler)
    errors = []
    for j in range(3):
        plus, minus = list(weights), list(weights)
        plus[j] += epsilon
        minus[j] -= epsilon
        numerical = (loss_and_gradient(plus, rows, scaler)[0]
                     - loss_and_gradient(minus, rows, scaler)[0]) / (2 * epsilon)
        errors.append(abs(numerical - analytical[j]))
    return max(errors)


def train_model(rows, scaler, rate, epochs):
    weights = [0.0] * 3
    for _ in range(epochs):
        _, grad = loss_and_gradient(weights, rows, scaler)
        weights = [w - rate * g for w, g in zip(weights, grad)]
        if not all(math.isfinite(w) for w in weights):
            raise ValueError("Training diverged; inspect scaling and learning rate")
    return weights


def evaluate(rows, policy):
    if not rows:
        raise ValueError("Cannot evaluate an empty dataset")
    grouped = {}
    for row in rows:
        error = policy(row.error, row.velocity) - row.action
        grouped.setdefault(row.episode, []).append(error * error)
    per_episode = [{"episode": episode, "frames": len(errors),
                    "action_rmse_m_s2": math.sqrt(statistics.fmean(errors))}
                   for episode, errors in sorted(grouped.items())]
    return {
        "pooled_action_rmse_m_s2": math.sqrt(statistics.fmean(
            error for errors in grouped.values() for error in errors)),
        "mean_episode_action_rmse_m_s2": statistics.fmean(
            row["action_rmse_m_s2"] for row in per_episode),
        "per_episode": per_episode,
    }


def rollout(policy, seed=101, episodes=12, error_bias=0.0,
            observation_noise_std=0.0, noise_seed=203):
    """Paired initial conditions; semi-implicit Euler; deterministic toy dynamics."""
    if not math.isfinite(observation_noise_std) or observation_noise_std < 0:
        raise ValueError("Observation noise standard deviation must be finite and nonnegative")
    rng, results = random.Random(seed), []
    noise_rng = random.Random(noise_seed)
    for episode in range(episodes):
        position, velocity = rng.uniform(-1.4, 1.4), rng.uniform(-0.3, 0.3)
        initial = [position, velocity]
        clipped = 0
        for _ in range(200):
            observed_error = -position + error_bias + noise_rng.gauss(0, observation_noise_std)
            raw = policy(observed_error, velocity)
            if not math.isfinite(raw):
                raise ValueError("Policy produced a non-finite action")
            acceleration = max(-1.0, min(1.0, raw))
            clipped += acceleration != raw
            velocity += acceleration * 0.1
            position += velocity * 0.1
        results.append({"episode": episode, "initial_position_m": initial[0],
                        "initial_velocity_m_s": initial[1], "final_position_m": position,
                        "final_velocity_m_s": velocity, "clipped_actions": clipped,
                        "success": abs(position) < 0.1 and abs(velocity) < 0.1})
    return {"successes": sum(row["success"] for row in results),
            "episodes": episodes, "per_episode": results}


def run(seed=7, epochs=200, observation_noise_std=0.0):
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if not math.isfinite(observation_noise_std) or observation_noise_std < 0:
        raise ValueError("Observation noise standard deviation must be finite and nonnegative")
    started = time.perf_counter()
    rows = make_dataset(seed)
    train, valid, test = split_episodes(rows)
    scaler = fit_scaler(train)
    error = gradient_error([0.1, -0.2, 0.05], train[:10], scaler)
    if error > 1e-7:
        raise ValueError("Finite-difference gradient check failed")
    try:
        validate_splits([train, valid + [train[0]], test])
    except ValueError:
        leakage_caught = True
    else:
        raise ValueError("Deliberate episode leakage was not detected")
    candidates = []
    for rate in (0.01, 0.05, 0.2):
        weights = train_model(train, scaler, rate, epochs)
        score = evaluate(valid, lambda e, v: predict(weights, e, v, scaler))
        candidates.append({"learning_rate": rate, "weights": weights,
                           "validation_rmse_m_s2": score["pooled_action_rmse_m_s2"]})
    best = min(candidates, key=lambda item: item["validation_rmse_m_s2"])
    policy = lambda e, v: predict(best["weights"], e, v, scaler)
    train_mean = statistics.fmean(row.action for row in train)
    noise_rng = random.Random(202)
    noisy_test = [Sample(row.episode, row.error + noise_rng.gauss(0, observation_noise_std),
                         row.velocity, row.action) for row in test]
    data_bytes = json.dumps([row.__dict__ for row in rows], sort_keys=True).encode()
    return {
        "execution_status": "measured CPU calculation on a synthetic fixture",
        "python": sys.version.split()[0], "platform": platform.platform(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dataset_sha256": hashlib.sha256(data_bytes).hexdigest(),
        "dataset_seed": seed, "split_seed": 11, "rollout_seed": 101,
        "epochs": epochs, "split_episode_ids": {
            name: sorted({row.episode for row in part})
            for name, part in zip(("train", "validation", "test"), (train, valid, test))},
        "scaler_training_mean": scaler[0], "scaler_training_std": scaler[1],
        "max_gradient_absolute_error": error, "leakage_fault_caught": leakage_caught,
        "candidates": candidates, "selected_learning_rate": best["learning_rate"],
        "weights_normalized_inputs": best["weights"],
        "test": evaluate(test, policy),
        "test_zero_baseline": evaluate(test, lambda e, v: 0.0),
        "test_training_mean_baseline": evaluate(test, lambda e, v: train_mean),
        "test_error_bias_0_6_m": evaluate(test, lambda e, v: policy(e + 0.6, v)),
        "observation_noise_std_m": observation_noise_std,
        "observation_noise_seeds": {"held_out": 202, "rollout": 203},
        "test_observation_noise": evaluate(noisy_test, policy),
        "rollout_contract": {"dt_s": 0.1, "steps": 200,
                             "acceleration_limit_m_s2": 1.0,
                             "final_position_tolerance_m": 0.1,
                             "final_velocity_tolerance_m_s": 0.1},
        "rollout": rollout(policy), "rollout_zero_baseline": rollout(lambda e, v: 0.0),
        "rollout_error_bias_0_6_m": rollout(policy, error_bias=0.6),
        "rollout_observation_noise": rollout(policy, observation_noise_std=observation_noise_std),
        "whole_lab_wall_seconds": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--observation-noise-std-m", type=float, default=0.0,
                        help="Evaluation-only Gaussian error-observation noise in meters")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    if not 1 <= args.epochs <= 2000:
        parser.error("--epochs must be between 1 and 2000 for this bounded lab")
    if not math.isfinite(args.observation_noise_std_m) or args.observation_noise_std_m < 0:
        parser.error("--observation-noise-std-m must be finite and nonnegative")
    report = run(args.seed, args.epochs, args.observation_noise_std_m)
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
