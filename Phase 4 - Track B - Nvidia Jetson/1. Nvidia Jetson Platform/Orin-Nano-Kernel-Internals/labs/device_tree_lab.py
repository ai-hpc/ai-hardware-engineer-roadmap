#!/usr/bin/env python3
"""Compile and apply a synthetic DT overlay to files; never install a live DT."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent
REQUIRED = ("dtc", "fdtoverlay", "fdtget")
SENSOR = "/i2c@1000/sensor@48"


def run(*args):
    return subprocess.run([str(arg) for arg in args], check=True, capture_output=True,
                          text=True, timeout=10).stdout.strip()


def check_tools():
    missing = [tool for tool in REQUIRED if shutil.which(tool) is None]
    if missing:
        raise RuntimeError("Install device-tree-compiler; missing " + ", ".join(missing))


def compile_tree(source, output, *, symbols=True):
    args = ["dtc"] + (["-@"] if symbols else [])
    return run(*args, "-I", "dts", "-O", "dtb", "-o", output, source)


def get(blob, node, property_name, kind="s"):
    return run("fdtget", "-t", kind, blob, node, property_name)


def validate_candidate(blob):
    """Small fixture-specific policy; NOT general DT schema/binding validation."""
    actual = {"controller_status": get(blob, "/i2c@1000", "status"),
              "sensor_status": get(blob, SENSOR, "status"),
              "compatible": get(blob, SENSOR, "compatible"),
              "address_hex": get(blob, SENSOR, "reg", "x"),
              "sample_rate_hz": int(get(blob, SENSOR, "sample-rate-hz", "u"))}
    if actual != {"controller_status": "okay", "sensor_status": "okay",
                  "compatible": "course,synthetic-sensor", "address_hex": "48",
                  "sample_rate_hz": 200}:
        raise ValueError(f"synthetic device contract failed: {actual}")
    return actual


def build(directory, base=HERE/"board.dts", overlay=HERE/"sensor-overlay.dts", *, symbols=True):
    check_tools()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    base_dtb = directory / "board.dtb"
    overlay_dtb = directory / "sensor.dtbo"
    merged = directory / "merged.dtb"
    compile_tree(base, base_dtb, symbols=symbols)
    compile_tree(overlay, overlay_dtb)
    run("fdtoverlay", "-i", base_dtb, "-o", merged, overlay_dtb)
    return merged


def demonstration(directory):
    merged = build(directory)
    return {"evidence_kind": "real dtc/libfdt compilation and merge; synthetic board, no hardware",
            "dtc_version": run("dtc", "--version"),
            "before": {"controller_status": get(Path(directory)/"board.dtb", "/i2c@1000", "status"),
                       "sample_rate_hz": int(get(Path(directory)/"board.dtb", SENSOR, "sample-rate-hz", "u"))},
            "after": validate_candidate(merged),
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (HERE/"board.dts", HERE/"sensor-overlay.dts")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir:
        report = demonstration(args.output_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="course-dt-") as directory:
            report = demonstration(directory)
    print(json.dumps(report, indent=2))
