#!/usr/bin/env bash
# CPU parser/sampler component timing; this is not an LLM throughput benchmark.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if (( $# > 1 )); then
    echo "Usage: $0 [build-directory]" >&2
    exit 2
fi
build=${1:-"$root/build-cpu"}
if [[ ! -x "$build/benchmark_cpu" ]]; then
    echo "First run scripts/test_plan.sh with this build directory." >&2
    exit 2
fi
python3 "$root/labs/benchmark_cpu.py" "$build/benchmark_cpu"
