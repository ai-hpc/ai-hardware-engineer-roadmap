"""Export/profile the actually trained scalar action model on CPU.

Requires the full course checkout and its optional CPU PyTorch environment.
Measured module calls, profiler events, and exported graph are kept separate
from analytical operation counts and hypothetical external-memory traffic.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[4]
ACTION_LABS = (REPO / "Phase 5 - Advanced Topics and Specialization" /
               "Track D - Robotics/VLA Foundations to Frontier 2026/labs")
sys.path.insert(0, str(ACTION_LABS))
from action_generation import ActionNet, sample  # noqa: E402
from action_math import LABELS, SIGMA  # noqa: E402


def load_model(checkpoint: Path):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (saved.get("schema") != "scalar-action-v1" or saved.get("method") != "flow"
            or tuple(saved.get("labels", ())) != LABELS or saved.get("sigma") != SIGMA):
        raise ValueError("checkpoint must match the scalar-action-v1 flow contract")
    model = ActionNet().eval()
    model.load_state_dict(saved["state_dict"], strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("checkpoint contains nonfinite parameters")
    return model


def inputs(batch: int):
    if type(batch) is not int or batch < 1:
        raise ValueError("positive integer batch required")
    return (torch.linspace(-1.25, 1.25, batch).reshape(batch, 1),
            torch.zeros(batch, 1), torch.arange(batch, dtype=torch.long) % 3)


def trace_linear_calls(model, batch: int, evaluations: int):
    """Observe actual module invocations, without treating hooks as a full graph."""
    x, _, labels = inputs(batch)
    rows, handles = [], []

    def hook_for(name):
        def hook(module, arguments, output):
            value = arguments[0]
            m = math.prod(value.shape[:-1])
            k, n = module.in_features, module.out_features
            rows.append({"invocation": len(rows), "module": name,
                         "input_shape": list(value.shape), "output_shape": list(output.shape),
                         "dtype": str(value.dtype), "mnk": [m, n, k],
                         "derived_multiply_add_ops": 2 * m * n * k,
                         "derived_bias_adds": m * n if module.bias is not None else 0,
                         "weight_identity": name + ".weight", "weight_logical_bytes": module.weight.numel() * module.weight.element_size(),
                         "bias_identity": name + ".bias", "bias_logical_bytes": module.bias.numel() * module.bias.element_size() if module.bias is not None else 0,
                         "output_logical_bytes": output.numel() * output.element_size()})
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(hook_for(name)))
    try:
        output = sample(model, x, labels, "flow", evaluations)
    finally:
        for handle in handles:
            handle.remove()
    return rows, output


def tile_fraction(rows, tile_m, tile_n, tile_k):
    """Shape-derived useful/fully padded arithmetic, not measured utilization."""
    if any(type(v) is not int or v < 1 for v in (tile_m, tile_n, tile_k)):
        raise ValueError("positive integer tile sizes required")
    useful = padded = 0
    for row in rows:
        m, n, k = row["mnk"]
        useful += m * n * k
        padded += ((m + tile_m - 1) // tile_m * tile_m *
                   ((n + tile_n - 1) // tile_n * tile_n) *
                   ((k + tile_k - 1) // tile_k * tile_k))
    if not padded:
        raise ValueError("nonempty invocation rows required")
    return useful / padded


def export_model(model, batch, output_dir=None):
    arguments = inputs(batch)
    with torch.no_grad():
        exported = torch.export.export(model, arguments)
    module = exported.module()
    loaded_module = None
    if output_dir:
        path = output_dir / f"flow-batch-{batch}.pt2"
        torch.export.save(exported, path)
        loaded_module = torch.export.load(path).module()
    max_error, reload_max_error, probe_count = 0.0, 0.0, 0
    # Vary inputs independently, including every category at batch size one.
    # A label roll alone would leave the batch-one example's label unchanged.
    with torch.no_grad():
        for time_value, x_offset in ((0.0, 0.0), (0.0, 0.25), (0.3, 0.0), (0.9, -0.4)):
            for label_offset in range(len(LABELS)):
                x, t, labels = arguments
                probe = (x + x_offset, t + time_value, (labels + label_offset) % len(LABELS))
                reference, candidate = model(*probe), module(*probe)
                torch.testing.assert_close(candidate, reference, atol=1e-6, rtol=1e-5)
                max_error = max(max_error, (reference - candidate).abs().max().item())
                if loaded_module is not None:
                    reloaded = loaded_module(*probe)
                    torch.testing.assert_close(reloaded, reference, atol=1e-6, rtol=1e-5)
                    reload_max_error = max(reload_max_error, (reference - reloaded).abs().max().item())
                probe_count += 1
    graph = []
    for node in exported.graph.nodes:
        value = node.meta.get("val")
        graph.append({"name": node.name, "kind": node.op, "target": str(node.target),
                      "inputs": [source.name for source in node.all_input_nodes],
                      "shape": list(value.shape) if isinstance(value, torch.Tensor) else None,
                      "dtype": str(value.dtype) if isinstance(value, torch.Tensor) else None})
    return {"scope": "one field evaluation, static batch; Euler loop remains Python",
            "static_batch": batch, "max_export_probe_error": max_error,
            "export_probe_count": probe_count,
            "serialized_reload_probe_count": probe_count if loaded_module is not None else 0,
            "max_serialized_reload_probe_error": reload_max_error if loaded_module is not None else None,
            "graph": graph}


def profile_query(model, batch, evaluations, trace_path=None):
    x, _, labels = inputs(batch)
    sample(model, x, labels, "flow", evaluations)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                record_shapes=True) as profiler:
        sample(model, x, labels, "flow", evaluations)
    if trace_path:
        profiler.export_chrome_trace(str(trace_path))
    events = [{"operator": e.key, "input_shapes": e.input_shapes, "calls": e.count,
               "instrumented_self_cpu_time_us": e.self_cpu_time_total}
              for e in profiler.key_averages(group_by_input_shape=True)]
    durations = []
    # Hooks and profiler are both absent for these separate wall measurements.
    for _ in range(11):
        start = time.perf_counter_ns()
        sample(model, x, labels, "flow", evaluations)
        durations.append((time.perf_counter_ns() - start) / 1e6)
    return {"events": events, "warm_wall_ms": durations,
            "warm_wall_ms_median": statistics.median(durations),
            "warm_wall_ms_max_observed": max(durations),
            "timing_scope": "11 warm CPU batches including sampler validation and Euler loop; no profiler/hooks"}


def build_report(model, batch, evaluations, output_dir=None):
    rows, sampled = trace_linear_calls(model, batch, evaluations)
    allocations = {r["weight_identity"]: r["weight_logical_bytes"] for r in rows}
    allocations.update({r["bias_identity"]: r["bias_logical_bytes"] for r in rows})
    profile = profile_query(model, batch, evaluations,
                            output_dir / f"cpu-b{batch}-s{evaluations}.json" if output_dir else None)
    counts = Counter()
    for event in profile["events"]:
        counts[event["operator"]] += event["calls"]
    expected = len(rows)
    if counts.get("aten::linear") != expected or counts.get("aten::addmm") != expected:
        raise RuntimeError("observed CPU operator lowering changed; inspect before accepting this report")
    return {"batch": batch, "evaluations": evaluations, "observed_linear_calls": len(rows),
            "module_calls": rows, "profiler": profile,
            "logical_parameter_allocations": allocations,
            "distinct_logical_parameter_bytes": sum(allocations.values()),
            "derived_multiply_add_ops": sum(r["derived_multiply_add_ops"] for r in rows),
            "derived_bias_adds": sum(r["derived_bias_adds"] for r in rows),
            "hypothetical_parameter_bytes_if_streamed_each_call": sum(r["weight_logical_bytes"] + r["bias_logical_bytes"] for r in rows),
            "padding_fraction_16_16_16": tile_fraction(rows, 16, 16, 16),
            "padding_fraction_1_16_16": tile_fraction(rows, 1, 16, 16),
            "sampled_outputs": sampled[:, 0].tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = load_model(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"scope": "actual CPU trace and static export of course-trained scalar model; no accelerator measurement",
              "python": platform.python_version(), "torch": torch.__version__, "threads": 1,
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "source_hashes": {name: hashlib.sha256((ACTION_LABS / name).read_bytes()).hexdigest()
                                for name in ("action_generation.py", "action_math.py")},
              "parameters": sum(p.numel() for p in model.parameters()),
              "excluded_from_operation_estimate": ["SiLU", "time sin/cos", "one-hot and concatenation", "Euler updates", "validation and scalar bookkeeping"],
              "not_measured": ["DRAM traffic", "allocator live/reserved peak", "SRAM ports", "power", "silicon throughput", "robot behavior"],
              "exports": [export_model(model, batch, args.output) for batch in (1, 7)],
              "queries": [build_report(model, batch, steps, args.output)
                          for batch in (1, 7) for steps in (8, 32)]}
    (args.output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"parameters": result["parameters"],
                      "exports": [{k: v for k, v in e.items() if k != "graph"} for e in result["exports"]],
                      "queries": [{k: q[k] for k in ("batch", "evaluations", "observed_linear_calls", "distinct_logical_parameter_bytes", "derived_multiply_add_ops", "padding_fraction_16_16_16")} for q in result["queries"]]}, indent=2))


if __name__ == "__main__":
    main()
