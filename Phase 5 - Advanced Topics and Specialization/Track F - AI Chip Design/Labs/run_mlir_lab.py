#!/usr/bin/env python3
"""Parse, lower, translate, and JIT a complete Linalg example (tested MLIR20.1.2)."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent
PASSES = (
    "--convert-linalg-to-loops", "--lower-affine", "--convert-scf-to-cf",
    "--convert-arith-to-llvm", "--convert-index-to-llvm",
    "--finalize-memref-to-llvm", "--convert-cf-to-llvm",
    "--convert-func-to-llvm", "--reconcile-unrealized-casts",
)


def tool(name: str) -> str:
    override = os.environ.get(name.upper().replace("-", "_"))
    found = (shutil.which(override) if override else
             shutil.which(name + "-20") or shutil.which(name))
    if not found:
        raise FileNotFoundError(f"{name} unavailable; install MLIR20 tools or set "
                                f"{name.upper().replace('-', '_')}")
    return found


def run(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError(f"command failed: {' '.join(command)}\n{result.stderr}")
    return result.stdout.strip()


def execute(source: Path, output_dir: Path) -> dict[str, str]:
    opt, translate, runner = map(tool, ("mlir-opt", "mlir-translate", "mlir-runner"))
    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep stages visible: success parsing is different from success executing.
    parsed = output_dir / "01-verified.mlir"
    loops = output_dir / "02-loops.mlir"
    lowered = output_dir / "03-llvm-dialect.mlir"
    llvm_ir = output_dir / "04-matmul.ll"
    run([opt, str(source), "-o", str(parsed)])
    run([opt, str(parsed), "--convert-linalg-to-loops", "-o", str(loops)])
    run([opt, str(parsed), *PASSES, "-o", str(lowered)])
    run([translate, "--mlir-to-llvmir", str(lowered), "-o", str(llvm_ir)])
    result = run([runner, str(lowered), "-e", "main", "-entry-point-result=i32"])
    if result != "415":
        raise AssertionError(f"matrix validation failed: expected 415, got {result!r}")
    return {"toolchain": run([opt, "--version"]).splitlines()[0],
            "result": result, "output_dir": str(output_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE / "matmul.mlir")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        if args.output_dir:
            report = execute(args.source, args.output_dir)
        else:
            with tempfile.TemporaryDirectory(prefix="course-mlir-") as directory:
                report = execute(args.source, Path(directory))
                report["output_dir"] = "temporary stages removed; use --output-dir to retain"
        for key, value in report.items():
            print(f"{key}: {value}")
    except (FileNotFoundError, RuntimeError, AssertionError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
