# WarpMoE v0.1 build status — 2026-10-13

## Verified in this environment

- CMake configures with the CPU-only fallback when CUDA is missing.
- CPU reference library and tests compile with GCC.
- `ctest` passes CPU reference tests and Python roofline tests.
- `scripts/roofline.py --batch 1 --bandwidth-gbs 800 --baseline-us 40` reports 18 MiB of expert weights and a 23.593 us theoretical streaming floor.

## Not yet verified

- `nvcc` compilation of `cuda/kernels.cu` and `src/cuda_bench.cu`.
- Actual CUDA results, cross-CTA readiness protocol under Compute Sanitizer, Nvidia graph captures, or GPU performance.
- Practical RTX 3090 bandwidth, optimized Marlin/vLLM baseline, or claimed latency speedup.

The CUDA scaffold uses **INT4 weights and FP32 activations** for tractable correctness testing; BF16 activations/intermediates and an optimized Marlin-compatible layout are not implemented yet.

The current machine lacks `nvcc` and NVIDIA GPU driver/device access. **Do not describe the CUDA executable as tested or fast.**

## First commands to run on an RTX 3090 workstation

```bash
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release \
    -DWARPMOE_REQUIRE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-gpu -j
ctest --test-dir build-gpu --output-on-failure
scripts/gpu_smoke.sh build-gpu
./build-gpu/warpmoe_bandwidth --mib 512 --iters 30
./build-gpu/warpmoe_bench --shape qwen --batch 1 --iters 40
```

Before drawing any performance conclusions, fix any compiler, sanitizer or correctness errors, then capture the actual optimized (graph-based) MoE baseline for comparison.

## GitHub readiness

Added `.github/workflows/ci.yml` with CPU tests and a CUDA toolkit compile-only job, and an opt-in `scripts/publish_github.sh` that creates a private repository after confirmation. This workflow file has not yet run on GitHub Actions.