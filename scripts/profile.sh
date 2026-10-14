#!/usr/bin/env bash
set -euo pipefail
BUILD_DIR="${1:-build-gpu}"
if ! command -v ncu >/dev/null; then
  echo 'ncu (Nsight Compute) is required' >&2
  exit 2
fi
ncu --set basic --metrics dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed --target-processes all "${BUILD_DIR}/warpmoe_bench" --shape qwen --batch 1 --iters 3 --warmup 1
