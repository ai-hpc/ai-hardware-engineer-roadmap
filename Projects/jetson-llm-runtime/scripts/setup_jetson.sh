#!/usr/bin/env bash
# Inspect prerequisites and build CPU teaching components. No global tuning.
set -euo pipefail
runtime_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${JLLM_BUILD_DIR:-$runtime_root/build-cpu}"
command -v cmake >/dev/null
command -v c++ >/dev/null
command -v python3 >/dev/null
if [[ -f /etc/nv_tegra_release ]]; then
    cat /etc/nv_tegra_release
fi
cmake -S "$runtime_root" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release -DJLLM_BUILD_JETSON=OFF
cmake --build "$build_dir" -j2
ctest --test-dir "$build_dir" --output-on-failure
printf '%s\n' 'CPU course checks complete. See TESTING.md for the optional Jetson source build.'
