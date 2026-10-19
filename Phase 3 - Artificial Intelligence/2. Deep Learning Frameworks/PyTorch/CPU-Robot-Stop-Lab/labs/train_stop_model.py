"""Train, select, test, serialize, export, and challenge a small CPU classifier."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import torch
from torch import nn

from data import FEATURES, dataset_sha256, fit_scaler, make_dataset, sensor_shift, split_episodes, stopping_distance


class StopModel(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.network = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, observations):
        return self.network((observations - self.mean) / self.std)


def tensors(rows):
    return torch.tensor([r.features() for r in rows], dtype=torch.float32), torch.tensor([r.stop_required for r in rows], dtype=torch.long)


def metrics(labels, predicted, probabilities=None):
    if not labels or len(labels) != len(predicted):
        raise ValueError("nonempty aligned labels/predictions required")
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predicted))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predicted))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predicted))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predicted))
    result = {"examples": len(labels), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
              "accuracy": (tp + tn) / len(labels), "stop_recall": tp / (tp + fn) if tp + fn else None,
              "stop_precision": tp / (tp + fp) if tp + fp else None,
              "cost_5fn_plus_fp_per_example": (5 * fn + fp) / len(labels)}
    if probabilities is not None:
        if len(probabilities) != len(labels) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
            raise ValueError("aligned finite probabilities in [0,1] required")
        result["brier_score"] = sum((p-y)**2 for p, y in zip(probabilities, labels)) / len(labels)
    return result


def choose_threshold(labels, probabilities):
    candidates = []
    for threshold in [i/20 for i in range(1, 20)]:
        score = metrics(labels, [int(p >= threshold) for p in probabilities], probabilities)
        candidates.append({"threshold": threshold, **score})
    selected = min(candidates, key=lambda s: (s["cost_5fn_plus_fp_per_example"], s["fn"], s["threshold"]))
    return selected["threshold"], candidates


@torch.inference_mode()
def evaluate(model, rows, threshold):
    model.eval()
    x, y = tensors(rows)
    logits = model(x)
    probs = logits.softmax(dim=-1)[:, 1].tolist()
    predicted = [int(p >= threshold) for p in probs]
    result = metrics(y.tolist(), predicted, probs)
    result["cross_entropy_nats"] = nn.functional.cross_entropy(logits, y).item()
    result["slices"] = {}
    for name, selected in {
        "high_speed_ge_3_m_s": [i for i,r in enumerate(rows) if r.speed_m_s >= 3],
        "lower_speed": [i for i,r in enumerate(rows) if r.speed_m_s < 3],
    }.items():
        if selected:
            result["slices"][name] = metrics([rows[i].stop_required for i in selected], [predicted[i] for i in selected])
    result["errors"] = [{"episode": r.episode, "features": r.features(), "true": r.stop_required,
                         "predicted": pred, "stop_probability": probability}
                        for r,pred,probability in zip(rows, predicted, probs) if pred != r.stop_required]
    return result


def export_and_check(model, x, output):
    model.eval()
    batch = torch.export.Dim("batch", min=1, max=1024)
    exported = torch.export.export(model, (x[:8],), dynamic_shapes=({0: batch},))
    path = output / "stop_model.pt2"
    torch.export.save(exported, path)
    loaded = torch.export.load(path).module()
    parity = {}
    with torch.inference_mode():
        for size in (1, 7, 32):
            error = (loaded(x[:size]) - model(x[:size])).abs().max().item()
            if error > 1e-6:
                raise AssertionError("export changed logits")
            parity[str(size)] = error
    try:
        loaded(torch.zeros(2, 5))
    except (RuntimeError, ValueError, AssertionError):
        wrong_shape_rejected = True
    else:
        raise AssertionError("export accepted an invalid feature dimension")
    return {"batch_logit_max_abs_error": parity, "wrong_feature_shape_rejected": wrong_shape_rejected,
            "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "contract": "float32 CPU [batch,4], batch 1..1024; export graph, not optimized deployment binary"}


def run(output, seed=31, epochs=80, distance_bias=0.8):
    if not 1 <= epochs <= 300 or not math.isfinite(distance_bias):
        raise ValueError("epochs must be 1..300 and distance bias finite")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    rows = make_dataset()
    training, validation, test = split_episodes(rows)
    mean, std = fit_scaler(training)
    model = StopModel(mean, std)
    x, y = tensors(training)
    vx, vy = tensors(validation)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.0001)
    history, best_loss, best_epoch, best_state = [], math.inf, None, None
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(training))
        total_loss = 0.0
        for indices in order.split(128):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x[indices]), y[indices])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)
        model.eval()
        with torch.inference_mode():
            validation_loss = nn.functional.cross_entropy(model(vx), vy).item()
        history.append({"epoch": epoch + 1, "train_ce": total_loss / len(training), "validation_ce": validation_loss})
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch + 1
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    with torch.inference_mode():
        threshold, candidates = choose_threshold(vy.tolist(), model(vx).softmax(-1)[:, 1].tolist())
    # Test rows first enter model evaluation only after epoch and threshold selection.
    test_report = evaluate(model, test, threshold)
    shifted = sensor_shift(test, distance_bias)
    shifted_report = evaluate(model, shifted, threshold)
    majority = int(sum(r.stop_required for r in training) >= len(training)/2)
    labels = [r.stop_required for r in test]
    rule = lambda r: int(r.distance_m <= stopping_distance(r.speed_m_s, r.deceleration_m_s2, r.latency_s))
    baselines = {"training_majority": metrics(labels, [majority]*len(test)),
                 "analytic_rule_clean": metrics(labels, [rule(r) for r in test]),
                 "analytic_rule_biased_sensor": metrics(labels, [rule(r) for r in shifted])}
    weight_path = output / "state_dict.pt"
    torch.save(model.state_dict(), weight_path)
    restored = StopModel([0]*4, [1]*4)
    restored.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
    restored.eval()
    tx, _ = tensors(test)
    with torch.inference_mode():
        reload_error = (restored(tx) - model(tx)).abs().max().item()
        swapped = tx[:, [1, 0, 2, 3]]
        swapped_error = (restored(swapped) - model(tx)).abs().max().item()
    if reload_error > 1e-7 or swapped_error < 0.1:
        raise AssertionError("reload parity or swapped-column negative check failed")
    export_report = export_and_check(restored, tx, output)
    report = {
        "fixture": "synthetic constant-deceleration stopping-distance classification; no real robot validation",
        "feature_order": list(FEATURES), "torch": torch.__version__, "python": sys.version.split()[0],
        "platform": platform.platform(), "threads": torch.get_num_threads(), "device": "cpu",
        "training_seed": seed, "dataset_seed": 17, "split_seed": 23,
        "dataset_sha256": dataset_sha256(rows),
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ("data.py", "train_stop_model.py")},
        "split_episode_ids": {name: sorted({r.episode for r in part}) for name,part in zip(("training", "validation", "test"), (training,validation,test))},
        "normalization": {"training_mean": mean, "training_std": std},
        "parameters": sum(p.numel() for p in model.parameters()), "epochs": epochs,
        "selected_epoch": best_epoch, "selected_threshold": threshold,
        "threshold_cost": "5 * missed stops + unnecessary stops; pedagogical costs, chosen before evaluation",
        "history": history, "validation_threshold_candidates": candidates,
        "test": test_report, "distance_sensor_bias_m": distance_bias, "shifted_test": shifted_report,
        "baselines": baselines, "reload_max_abs_logit_error": reload_error,
        "swapped_column_max_abs_logit_error": swapped_error, "export": export_report,
        "weights_sha256": hashlib.sha256(weight_path.read_bytes()).hexdigest(),
        "whole_experiment_wall_seconds": time.perf_counter() - started,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--distance-bias-m", type=float, default=.8)
    args = parser.parse_args()
    report = run(args.output, args.seed, args.epochs, args.distance_bias_m)
    print(json.dumps({key: report[key] for key in ("torch", "parameters", "selected_epoch", "selected_threshold", "reload_max_abs_logit_error", "export", "whole_experiment_wall_seconds")}, indent=2))
    for name in ("test", "shifted_test"):
        print(name, {key: report[name][key] for key in ("accuracy", "stop_recall", "fp", "fn", "cross_entropy_nats")})


if __name__ == "__main__":
    main()
