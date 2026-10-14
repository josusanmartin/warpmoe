# WarpMoE — Autonomous Agent Handoff

You are taking over the research codebase at https://github.com/josusanmartin/warpmoe.

## What success means
Create a technically rigorous, reproducible CUDA C++ research prototype investigating a GPU-resident quantized-MoE execution runtime on one NVIDIA RTX 3090. The project aims to improve MoE inference latency or throughput *against optimized, graph-captured production kernels*, not just against a toy reference. Its eventual reusable interface should accept multiple MoE configurations.

Read, in order:
1. [ROADMAP.md](ROADMAP.md) — exact prioritized tasks and acceptance gates.
2. [SPEC.md](SPEC.md) — model math, quantization assumptions, synchronization design, theoretical roofline, and experiment policy.
3. [STATUS.md](STATUS.md) — what is verified, what is not.
4. [README.md](README.md) — basic build and test usage.

## Current code (do not assume GPU correctness)
- `include/warpmoe/format.hpp`, `src/reference.cpp`: deterministic packed INT4 + FP32 CPU oracle.
- `cuda/kernels.cu`: first-cut gate/up and down GPU kernels plus persistent ticket scheduler; currently W4A32, naive compute loop.
- `src/cuda_bench.cu`: captures both scaffold paths in CUDA Graphs and compares outputs/timings.
- `cuda/bandwidth.cu`: streaming-read microbenchmark (validate true DRAM traffic with Nsight).
- `tests/`: CPU unit and roofline tests.
- `.github/workflows/ci.yml`: CPU build/test and CUDA **compile-only** checks.

The initial repo CI failed on a mixed-type `auto` declaration in `tests/test_reference.cpp`; a corrective commit was submitted. **Check the newest CI outcome** rather than assuming resolution. The project previously passed local CPU-only tests, but the independently uploaded GitHub version must be checked against CI.

## Required first commands

```bash
git clone https://github.com/josusanmartin/warpmoe.git
cd warpmoe
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure

# Ampere RTX 3090 Linux machine with CUDA toolkit
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release \
  -DWARPMOE_REQUIRE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-gpu -j
ctest --test-dir build-gpu --output-on-failure
bash scripts/gpu_smoke.sh build-gpu
./build-gpu/warpmoe_bandwidth --mib 512 --iters 30
./build-gpu/warpmoe_bench --shape tiny --batch 1 --iters 30
```

The synthetic Qwen-shaped benchmark can be costly and uses a naive GEMV loop; do not run large sweeps until tiny correctness, synchronization, and GPU profiling are validated.

## Research direction
H1: At batch 1–4, persistent fusion and dependency-aware streaming close actual kernel gaps while maintaining memory bandwidth. H2: At batch 8–128, per-expert GEMV/GEMM choice and partitioning improve skew-induced tails. These are separate experiments. Target W4A16 (INT4 packed weights and BF16 activations); current scaffold is W4A32 (FP32 activations). Preserve numeric semantics and CPU oracle.

First reproduce a **strong CUDA-graph-captured Marlin/vLLM or other optimized W4A16 baseline on sm_86**. Profile actual DRAM bytes, kernels, and gap times; do not infer speedup from 20.16 μs theoretical packed-read floor alone.

Then optimize a coalesced prepermuted INT4 GEMV worker in CUDA C++/inline PTX, test static versus persistent with identical math, and measure `cp.async` prefetch separately. Explore runtime per-expert GEMV/GEMM decision only after that.

## Report contract
For each benchmark: GPU model/driver/clocks, CUDA/toolchain, backend commit and flags, model shape, batch, expert selection histogram, quantization, memory traffic, warmup/repetitions, p50/p95/p99, numerical error, and links to exact logs/commit. Track progress honestly in STATUS.md.

## Immediate known risks to inspect
- GPU compile/sanitizer and race/progress correctness not yet validated.
- Shared kernel compute paths are naive and could be orders of magnitude slower than production.
- Deadlock/occupancy risk of waiting CTAs; ticket ordering alone does not prove every launch configuration safe.
- Current format is not Marlin's layout and must be converted or compared under a matched interface.
- Do not push performance claims absent GPU measurements.

## Instruction for the receiving agent
You are the implementation owner. Start by inspecting the **actual current repository and latest CI**, fix concrete failures, and produce passing tests/commits before pursuing optimization. Ask for RTX 3090 execution only where hardware access is truly necessary. Do not prematurely announce successful CUDA execution or performance wins.
