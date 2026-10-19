#!/usr/bin/env bash
# Profile an explicitly selected command with an installed Nsight Systems.
set -euo pipefail
if (( $# < 3 )) || [[ "$2" != -- ]]; then
    echo "Usage: $0 <new-output-prefix> -- <executable> [arguments...]" >&2
    exit 2
fi
output=$1
shift 2
command -v nsys >/dev/null
if [[ -e "$output.nsys-rep" || -e "$output.sqlite" ]]; then
    echo "Choose a new output prefix; refusing to overwrite an existing trace." >&2
    exit 2
fi
nsys profile --trace=cuda,nvtx,osrt --output="$output" "$@"
nsys stats "$output.nsys-rep"
