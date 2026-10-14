# WarpMoE — Verified status and blockers

Updated 2026-10-13 (America/Los_Angeles) during agent handoff.

## GitHub repository

Repository: https://github.com/josusanmartin/warpmoe
Default branch: `main`. Source of truth: [SPEC.md](SPEC.md), [ROADMAP.md](ROADMAP.md), and [AGENTS.md](AGENTS.md).

## Confirmed observations

- Local CPU-only CMake build and ctest on the model's working container passed **2/2 tests**.
- GitHub CI run [38502435291](https://github.com/josusanmartin/warpmoe/actions/runs/38502435291) failed in **both** CPU and CUDA compile jobs because `tests/test_reference.cpp` had `auto x=make_inputs(s,9),y=reference_forward(a,x);` (two different types in a single `auto` declaration). CUDA compilation reached `libwarpmoe_cuda.a` before the CPU-test error aborted the CUDA build; this **does not establish a fully passing CUDA build**.
- Corrective commit `5ab5fd4e702afe4cb60ccf852ee09af0bb47ea33` splits that declaration into two statements. The latest CI run must be checked to verify the fix.
- The repository currently implements an initial model-parameterized INT4/FP32 (**W4A32**) CPU oracle, two-stage CUDA scaffold, persistent ticket scheduler, timing harness, and streaming-bandwidth microbenchmark.

## Not yet verified / do not claim

- Passing post-fix CI on latest main (until checked).
- Complete CUDA binary compilation + linking, or GPU execution on real NVIDIA hardware.
- compute-sanitizer memcheck/racecheck of the persistent readiness protocol.
- Marlin/vLLM optimized, CUDA-graph-captured W4A16 baseline on RTX 3090.
- Practical GPU bandwidth, real DRAM bytes, layer p50/p99, end-to-end tokens/s, or any actual speedup.
- BF16 activation path, prepermuted/coalesced weights, `cp.async` prefetch, adaptive GEMV/GEMM worker selection, real-model integration.

## Next actions

1. Review newest GitHub Actions results and address any remaining CPU/CUDA compiler failures.
2. On Linux with RTX 3090, run `cmake`/`ctest` and `bash scripts/gpu_smoke.sh build-gpu`.
3. Establish optimized graph-captured W4A16 baseline and sustained bandwidth **before** developing a more sophisticated scheduler.
4. Follow [ROADMAP.md](ROADMAP.md) sequentially and record exact commands, results, commit hashes in this file.

The initial project is **research scaffolding**, not a validated performant GPU runtime.
