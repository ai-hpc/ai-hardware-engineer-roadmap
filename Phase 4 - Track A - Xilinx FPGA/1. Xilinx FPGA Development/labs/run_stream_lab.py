#!/usr/bin/env python3
"""Compile and run the actual RTL; optionally prove a deliberate fault is caught."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
FAULTS = {
    "overwrite-stall": (
        "end else if (s_ready) begin", "end else if (s_valid || s_ready) begin", "STALL"
    ),
    "wrap-overflow": (
        "(s_data == 8'hff) ? 8'hff : s_data + 8'd1", "s_data + 8'd1", "DATA"
    ),
    "retain-on-reset": ("full <= 1'b0;", "full <= (full === 1'b1);", "VALID"),
}


def tool_paths() -> tuple[str, str]:
    paths = []
    for name in ("iverilog", "vvp"):
        executable = os.environ.get(name.upper(), name)
        path = shutil.which(executable)
        if path is None:
            raise RuntimeError(f"Missing {name}; install Icarus Verilog or set {name.upper()} to its executable.")
        paths.append(path)
    return paths[0], paths[1]


def run(seed: int = 1, fault: str | None = None, vcd: Path | None = None) -> dict:
    if not 1 <= seed <= 0xffffffff:
        raise ValueError("seed must be in [1, 4294967295]")
    if fault is not None and fault not in FAULTS:
        raise ValueError(f"unknown fault: {fault}")
    iverilog, vvp = tool_paths()
    source = (HERE / "stream_increment.sv").read_text()
    if fault:
        original, broken, _ = FAULTS[fault]
        if source.count(original) != 1:
            raise RuntimeError("Fault injection anchor changed; inspect the RTL before updating it.")
        source = source.replace(original, broken)
    with tempfile.TemporaryDirectory(prefix="course-rtl-") as directory:
        work = Path(directory)
        rtl = work / "stream_increment.sv"
        rtl.write_text(source)
        compiled = work / "stream.vvp"
        compile_result = subprocess.run(
            [iverilog, "-g2012", "-Wall", "-s", "tb_stream_increment", "-o", str(compiled),
             str(rtl), str(HERE / "tb_stream_increment.sv")],
            text=True, capture_output=True, timeout=30,
        )
        if compile_result.returncode:
            raise RuntimeError("RTL compilation failed:\n" + compile_result.stdout + compile_result.stderr)
        command = [vvp, str(compiled), f"+SEED={seed}"]
        if vcd:
            vcd = vcd.resolve()
            vcd.parent.mkdir(parents=True, exist_ok=True)
            command.append(f"+VCD={vcd}")
        simulation = subprocess.run(command, text=True, capture_output=True, timeout=30)
    transcript = simulation.stdout + simulation.stderr
    if fault:
        expected_assertion = FAULTS[fault][2]
        if simulation.returncode == 0 or f"{expected_assertion}:" not in transcript:
            raise RuntimeError(f"Fault {fault!r} was not caught by the expected check:\n{transcript}")
        return {"status": "fault_detected", "fault": fault, "assertion": expected_assertion,
                "seed": seed, "simulation_exit_code": simulation.returncode, "transcript": transcript}
    if simulation.returncode:
        raise RuntimeError("RTL simulation failed:\n" + transcript)
    match = re.search(r"^RESULT (.+)$", transcript, re.MULTILINE)
    if not match:
        raise RuntimeError("Simulator exited without the scoreboard completion record:\n" + transcript)
    metrics = {name: int(value) for name, value in re.findall(r"(\w+)=(\d+)", match.group(1))}
    return {"status": "passed", "seed": seed, **metrics, "transcript": transcript}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--inject-fault", choices=FAULTS)
    parser.add_argument("--vcd", type=Path, help="Optional waveform destination; generated outside source tree by default.")
    parser.add_argument("--output", type=Path, help="Optional JSON report destination.")
    args = parser.parse_args()
    try:
        result = run(args.seed, args.inject_fault, args.vcd)
    except (RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"{error}\n")
    serialized = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
