# WarpMoE

**Research prototype**: a model-parameterized, GPU-resident executor for quantized Mixture-of-Experts decoding on NVIDIA Ampere. The project begins with correctness tests and a two-kernel vs persistent comparison, before attempting competitive INT4 MoE kernels.

**Status:** CPU reference and initial CUDA execution path supplied. The CUDA path has **not** been compiled or tested on NVIDIA hardware in this environment. Nothing here is yet demonstrated to outperform Marlin, FlashInfer, or vLLM. Read [SPEC.md](SPEC.md) before interpreting a benchmark.

## Quick start

```bash
# Any Linux/macOS host with CMake 3.24+ and C++17:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure

# On a Linux host with NVIDIA CUDA Toolkit and a compatible GPU:
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release \
  -DWARPMOE_REQUIRE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-gpu -j
ctest --test-dir build-gpu --output-on-failure
./build-gpu/warpmoe_bench --shape tiny --batch 1 --iters 100
./build-gpu/warpmoe_bench --shape qwen --batch 1 --iters 40
./build-gpu/warpmoe_bandwidth --mib 512 --iters 30
```

Run `scripts/profile.sh build-gpu` to collect kernel profiling metrics (requires Nsight Compute, `ncu` in PATH). On consumer GPUs, profiling counters may need driver permission configuration.

## What is implemented?

- A deterministic CPU oracle, INT4 packed-weight format, group scales, route generator, reference MoE forward, and test vectors.
- Two CUDA code paths using the **same** naive GEMV computation: (a) separate gate/up and down kernels, (b) one persistent kernel with topological tickets and release/acquire readiness flags.
- CUDA event timings, a correctness comparison, basic CSV output, and a separate streaming-read bandwidth probe.
- Synthetic layer shapes: small correctness fixture and Qwen3-30B-A3B-sized MoE layer. Real model weights and optimized Marlin layouts are **not** included.

## Handing off to another coding agent

Start with **[AGENTS.md](AGENTS.md)** for the receiving agent's instructions, then follow the prioritized **[ROADMAP.md](ROADMAP.md)**, the technical **[SPEC.md](SPEC.md)**, and verified **[STATUS.md](STATUS.md)**. Check GitHub Actions on the latest commit before making performance claims.

## Planned next changes

1. Get the CUDA implementation compiling and correctness checked on RTX 3090, then run `compute-sanitizer`.
2. Measure bandwidth and get a CUDA-Graph-captured production baseline (Marlin/vLLM) using *matching* W4A16 layout and semantics.
3. Replace naive W4A32 GEMV with BF16-activation, coalesced/prepermuted W4A16 GEMV, keeping v0.1 worker semantics.
4. Experiment with CTA-local `cp.async` prefetch, separately from fusion.
5. At batch 8–128, select GEMV/GEMM and tile count per expert using live routing distribution and compare under skew.

## Files

- [SPEC.md](SPEC.md) — detailed hypothesis, limits, benchmark protocol, milestones.
- `include/warpmoe/format.hpp` — weights/shapes/routes and task ticket plan.
- `src/reference.cpp` — CPU reference.
- `cuda/kernels.cu` — GPU baseline and persistent executor.
- `src/cuda_bench.cu` — GPU runner.
- `cuda/bandwidth.cu` — sustainable streaming read benchmark.
- `tests/test_reference.cpp` — CPU unit tests.

## Reproducibility policy

All speedups must identify GPU, clocks/power, CUDA driver/toolkit, model shape, batch, quantization, routing data, warmup count, repetition count and comparison backend. Report kernel-only versus end-to-end results separately. Synthetic workloads are not substitutes for real-model tokens/sec.

License: MIT (see LICENSE).

## Repository

Source and research prototype hosted at https://github.com/josusanmartin/warpmoe.

GitHub Actions checks CPU correctness and attempts CUDA `sm_86` compilation without a GPU. GPU correctness and performance remain unverified until tested on NVIDIA hardware.
