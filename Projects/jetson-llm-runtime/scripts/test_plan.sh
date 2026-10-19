#!/usr/bin/env bash
# Native CPU checks. No model download or device configuration changes.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if (( $# > 1 )); then
    echo "Usage: $0 [build-directory]" >&2
    exit 2
fi
build=${1:-"$root/build-cpu"}
cmake -S "$root" -B "$build" -DCMAKE_BUILD_TYPE=Release -DJLLM_BUILD_JETSON=OFF
cmake --build "$build" --parallel 2
ctest --test-dir "$build" --output-on-failure
