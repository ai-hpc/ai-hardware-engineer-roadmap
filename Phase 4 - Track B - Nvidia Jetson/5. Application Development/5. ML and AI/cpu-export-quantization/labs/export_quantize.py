"""Actual PyTorch -> ONNX -> CPU Runtime -> static INT8 QDQ experiment.

Reuses the original synthetic training fixture. No model/dataset downloads.
This is an export/calibration exercise, not a physical robot controller.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Curriculum-Authoring-Guide.md").exists())
TRAINING = ROOT / "Phase 3 - Artificial Intelligence/2. Deep Learning Frameworks/PyTorch/CPU-Robot-Stop-Lab/labs"
sys.path.insert(0, str(TRAINING))
import train_stop_model as training
from data import make_dataset, split_episodes, sensor_shift


class Reader(CalibrationDataReader):
    def __init__(self, samples):
        self.samples = iter(samples)

    def get_next(self):
        x = next(self.samples, None)
        return None if x is None else {"observations": x[None].astype(np.float32)}


def session(path):
    settings = ort.SessionOptions()
    settings.intra_op_num_threads = 1
    settings.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), settings, providers=["CPUExecutionProvider"])


def predict(runtime, observations):
    x = np.asarray(observations)
    if x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != 4 or not 1 <= len(x) <= 1024:
        raise ValueError("expected float32 [1..1024,4] observations")
    if not np.isfinite(x).all():
        raise ValueError("nonfinite observations")
    return runtime.run(["logits"], {"observations": x})[0]


def score(runtime, rows, threshold):
    x, y = training.tensors(rows)
    logits = predict(runtime, x.numpy())
    p = torch.from_numpy(logits).softmax(-1)[:, 1].tolist()
    return {**training.metrics(y.tolist(), [int(v >= threshold) for v in p], p),
            "cross_entropy_nats": torch.nn.functional.cross_entropy(torch.from_numpy(logits), y).item()}


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    original = training.run(output / "training")
    model = training.StopModel([0] * 4, [1] * 4)
    model.load_state_dict(torch.load(output / "training/state_dict.pt", map_location="cpu", weights_only=True))
    model.eval()
    train_rows, valid_rows, test_rows = split_episodes(make_dataset())
    x, _ = training.tensors(train_rows)
    raw_path, fp_path = output / "export.onnx", output / "preprocessed.onnx"
    torch.onnx.export(model, (x[:8],), str(raw_path), dynamo=True, opset_version=21,
                      input_names=["observations"], output_names=["logits"], external_data=False,
                      dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=1024)},))
    onnx.checker.check_model(onnx.load(raw_path))
    quant_pre_process(str(raw_path), str(fp_path), skip_optimization=False)
    fp = session(fp_path)
    parity = {}
    with torch.inference_mode():
        for count in (1, 7, 32):
            parity[str(count)] = float(np.max(np.abs(predict(fp, x[:count].numpy()) - model(x[:count]).numpy())))
            if parity[str(count)] > 1e-5:
                raise AssertionError("FP32 export parity failed")
    # Calibration uses training inputs only, fixed before all evaluation.
    indices = random.Random(57).sample(range(len(train_rows)), 256)
    representative = x[indices].numpy()
    # Deliberately degenerate calibration, using one training observation 256 times.
    repeated = np.repeat(x[:1].numpy(), len(indices), axis=0)
    cases = {}
    threshold = original["selected_threshold"]
    for name, samples in (("representative", representative), ("repeated_one_row", repeated)):
        target = output / f"{name}.onnx"
        quantize_static(str(fp_path), str(target), Reader(samples), quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                        per_channel=True, op_types_to_quantize=["MatMul", "Gemm"])
        onnx.checker.check_model(onnx.load(target))
        runtime = session(target)
        tx, _ = training.tensors(test_rows)
        result = predict(runtime, tx.numpy())
        counts = Counter(n.op_type for n in onnx.load(target).graph.node)
        if not counts["QuantizeLinear"] or not counts["DequantizeLinear"]:
            raise AssertionError("quantization produced no QDQ nodes")
        cases[name] = {
            "calibration_rows": len(samples), "unique_rows": len(np.unique(samples, axis=0)),
            "calibration_episode_ids": sorted({train_rows[i].episode for i in (indices if name == "representative" else [0])}),
            "operator_counts": dict(counts), "artifact_bytes": target.stat().st_size,
            "artifact_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "test": score(runtime, test_rows, threshold),
            "distance_bias_0_8m": score(runtime, sensor_shift(test_rows, .8), threshold),
            "max_abs_test_logit_error": float(np.max(np.abs(result - predict(fp, tx.numpy())))),
            "fresh_session_reload_max_abs_error": float(np.max(np.abs(result - predict(session(target), tx.numpy())))),
        }
    tx, _ = training.tensors(test_rows)
    swapped = tx[:, [1, 0, 2, 3]].contiguous().numpy()
    report = {
        "date_utc": datetime.now(timezone.utc).date().isoformat(), "device": "CPUExecutionProvider", "python": platform.python_version(),
        "torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__,
        "opset": 21, "threshold_from_validation": threshold,
        "training_dataset_sha256": original["dataset_sha256"],
        "split_episode_ids": original["split_episode_ids"],
        "source_sha256": {"export_quantize.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), **original["source_sha256"]},
        "fp32_export_max_abs_errors": parity, "fp32_artifact_bytes": fp_path.stat().st_size,
        "fp32_test": score(fp, test_rows, threshold),
        "fp32_distance_bias_0_8m": score(fp, sensor_shift(test_rows, .8), threshold),
        "swapped_feature_max_abs_error": float(np.max(np.abs(predict(fp, swapped) - predict(fp, tx.numpy())))),
        "quantization_cases": cases,
        "limitations": ["synthetic analytical labels", "no Jetson/GPU/TensorRT run", "QDQ presence is not proof of integer kernel coverage", "no speedup claim"],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))
