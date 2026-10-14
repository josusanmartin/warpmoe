#!/usr/bin/env bash
set -euo pipefail
BUILD_DIR="${1:-build-gpu}"
"${BUILD_DIR}/warpmoe_bench" --shape tiny --batch 1 --iters 20
"${BUILD_DIR}/warpmoe_bench" --shape tiny --batch 8 --iters 20
if command -v compute-sanitizer >/dev/null; then
  compute-sanitizer --tool memcheck "${BUILD_DIR}/warpmoe_bench" --shape tiny --batch 2 --iters 1 --warmup 1
  compute-sanitizer --tool racecheck "${BUILD_DIR}/warpmoe_bench" --shape tiny --batch 2 --iters 1 --warmup 1
else
  echo 'WARNING: compute-sanitizer unavailable; GPU race and memory checks not performed.' >&2
fi
