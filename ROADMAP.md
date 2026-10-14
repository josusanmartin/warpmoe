# WarpMoE — Agent Roadmap

**Source of truth:** [SPEC.md](SPEC.md) for mathematical and benchmarking requirements; [STATUS.md](STATUS.md) for evidence of actual progress. This roadmap describes work to do, not completed capabilities.

## Mission

Investigate whether an Ampere-targeted, model-parameterized, GPU-resident runtime can accelerate W4A16 (INT4 weights, BF16 activations) MoE decode against **optimized CUDA-graph-captured production baselines**.

Primary hardware: RTX 3090, sm_86, 24 GiB. Implement in CUDA C++17, optional inline PTX. Do not start by rewriting everything in PTX or implementing a general-purpose GPU OS.

Two distinct hypotheses:
- **H1, batch 1–4:** fusion/persistence and streaming avoid critical-path gaps in bandwidth-limited decoding.
- **H2, batch 8–128:** per-expert GEMV/GEMM worker choice and tile splitting reduce the skew-driven straggler tail.

## Ordered milestones and acceptance gates

### P0 — Unblock builds and validate existing scaffold (G1)
- Inspect the current GitHub Actions results. The initial commit failed due to a mixed-type multi-`auto` declaration in `tests/test_reference.cpp`; a correction was committed on 2026-10-13 (PT).
- Check both CPU and CUDA compile jobs on the **latest** commit. CI CUDA runs compilation without GPU execution.
- On an Ampere GPU: run `scripts/gpu_smoke.sh build-gpu`; use `compute-sanitizer` memcheck/racecheck on tiny test inputs. Exercise B=1,2,8, top-k, nonuniform routing, nondivisible tile endpoints, and multiple worker-block counts.
- **Gate:** All CPU tests pass; CUDA builds; GPU answers match independent CPU reference under documented tolerances; no detectable synchronization/memory bugs. Do not claim this until measured.

### P1 — Establish production baseline & practical bandwidth floor (G2)
- Capture vLLM/Marlin-MoE (or strongest compatible W4A16 backend) **inside CUDA Graphs** on RTX 3090. Record library commits, exact quantization encoding, grouping/scales, device clocks, and inputs. If Marlin not compatible, document why and use next best optimized backend.
- Determine effective compulsory DRAM bytes *from actual Nsight measurements*, versus the ideal packed weight-byte estimate.
- Run `warpmoe_bandwidth`, report its limitations and cross-check real DRAM traffic in Nsight Compute.
- Report layer wall time, each kernel's active time, visible gaps, launch count, registers, DRAM bandwidth, key warp stalls and graph capture path.
- For Qwen-shaped B=1, 8 experts * 2.25 MiB = 18 MiB. Datasheet-only optimistic read floor = 20.16 μs at 936 GB/s; compute a practical floor from measured bandwidth, metadata, and traffic. This is **not** a complete latency lower bound.
- **Gate:** An independently reproducible optimized CUDA-graph benchmark and credible headroom estimate. If recoverable gap appears too small (~15% is only a heuristic), stop/revise H1.

### P2 — Competitive Ampere W4A16 worker (G3 prerequisite)
- Existing `cuda/kernels.cu` is a **naive W4A32/FP32-GEMV scaffold**. Replace/add BF16 activations and intermediates with explicitly defined accumulation and rounding semantics.
- Implement coalesced memory accesses, offline INT4 repacking/prepermutation, efficient dequant/scale loading. Keep separate optimized conventional and persistent variants with *identical compute math*.
- Profile integer dequant issue pressure versus actual bandwidth; consider GEMV worker first. No hand-written whole-kernel PTX or SASS.
- **Gate:** Correct W4A16 numerical semantics, sanitized GPU tests, no unexplained severe performance regressions versus optimized matched-format reference.

### P3 — H1: persistent dependency + prefetch ablation
- Implement/benchmark fused gate/up -> SiLU -> down -> reduction in persistent grid, preserving release/acquire and safe cooperative progress.
- Compare off/on CTA-local `cp.async` stream-at-claim prefetch with identical data layout/work ordering. Prefetch consumes L2/DRAM and shared memory; do not count prefetched bytes as removed traffic.
- Benchmark B=1 and B=4, repeated measurements and multiple clocks; compare production graph baseline, optimized conventional implementation, persistent-without-prefetch, persistent-with-prefetch.
- **Gate:** A reproducible competitive latency result attributable to a specific mechanism, with no correctness regressions.

### P4 — H2: per-expert heterogeneous scheduling (G4)
- Derive expert token counts **on device** from actual/frozen router output; use per-expert counts for a *discrete* GEMV/GEMM worker decision and split count.
- Watch unioned register/shared-memory footprints for a mixed kernel; measure occupancy. Consider separate kernels if unified resources erase gains.
- Use real top-k routing traces and *controlled* synthetic skew. Batch sizes B=8,32,64,128. Compare static GEMV, static GEMM, adaptive, and the fastest production baseline.
- Report tails, throughput, and numerical comparisons, including expert execution skew distribution.
- **Gate:** Same scheduler supports two different MoE shapes and improves targeted skew cases without hiding regressions elsewhere.

### P5 — Real model integration (G5)
- Integrate a real quantized MoE model into a supported serving stack. Confirm exact quant conventions, score normalization, activation dtype, and output tolerance.
- Record token-level correctness, a small evaluation/perplexity test, and end-to-end tokens/sec. Distinguish per-layer benefits from end-to-end gains via Amdahl's law.
- **Gate:** End-to-end reproducible measurement against production configuration.

## Stop rules and integrity constraints
- The repo does not yet demonstrate a performance improvement.
- Never equate synthetic CUDA graph vs naive GPU code with Marlin/vLLM comparisons.
- Never describe a CI compile as GPU verification.
- Do not conflate W4A32 scaffold with W4A16 production math.
- Do not claim all models get faster; different regimes may use different backends or fallbacks.
- Do not assume all kernel time below 90% DRAM BW is recoverable scheduling overhead.
- Do not claim newness for persistent kernels, device-side routing, readiness counters, or prefetch. Compare MonoMoE, FlashMoE, Mirage MPK, Cohere, RaMP, AutoMegaKernel and related implementations.

## First action for the next agent

1. Read `SPEC.md`, `STATUS.md`, then inspect GitHub Actions on `main`.
2. Fix failing CI before touching performance optimizations.
3. Run GPU correctness/sanitizer on the RTX 3090 and store exact build versions/results in `STATUS.md`.
4. Establish optimized Marlin/vLLM graph baseline and practical memory floor.
5. Only then implement higher-performance workers and scheduling.

At each milestone update `STATUS.md` with **observed** results, exact command/output, and commit links.
