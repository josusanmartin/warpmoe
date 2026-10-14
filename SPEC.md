# WarpMoE — engineering specification

Status: **v0.1 / research prototype** · Target: NVIDIA Ampere `sm_86` (RTX 3090, 24 GiB) · Language: CUDA C++17; optional inline PTX.

## Thesis

A persistent GPU executor can accelerate **small-batch, weight-only-INT4 MoE inference** by avoiding dependent kernel boundaries; at moderate decode batches it may additionally benefit from routing-aware work assignment. We will test these as **separate** hypotheses before investing in a general runtime.

This is not a claim of research novelty. Persistent MoE megakernels, device-side task scheduling, routing-aware dispatch and weight prefetch exist in MPK, FlashMoE, MonoMoE, MegaMoE, RaMP, and other work. A defensible eventual contribution requires a measured improvement against optimized equivalents on a defined hardware/workload regime.

## Scope / non-goals

### Target workloads
- **First:** Batch-1 MoE decode, Ampere W4A16, one GPU, one layer, 128 experts, top-8, hidden=2048, expert intermediate=768 (Qwen3-30B-A3B-shaped).
- **Next:** Batch 4/8/32/64/128 and real routing traces; quantify skew and straggler tails.
- **Then:** One end-to-end integration with a real model, using its true routing, quantization, activation and numerical conventions.
- Generality means shape- and model-parameterized interfaces. It does **not** mean one universal fastest kernel for all GPUs, quantizations, or batch sizes.

### Out of scope for v0.1
- Training/backpropagation, multi-GPU expert parallelism, scheduling unrelated CUDA kernels.
- Untrusted-program isolation, generalized GPU processes, preemptive GPU OS features.
- Whole-kernel handwritten PTX or SASS; claims of greater performance than vendor kernels before measuring.
- A full Qwen3 deployment, tokenizer, attention, KV cache, or model-serving stack.

## Model math

Inputs: activations `X[B,H]`, top-k selected experts `expert_id[B,K]`, router scores `score[B,K]` (routing computed upstream in v0.1).

For each `(b,k)` with expert `e`:

```
g = X[b] @ W_gate[e].T
u = X[b] @ W_up[e].T
z[b,k] = SiLU(g) * u
y[b,k] = z[b,k] @ W_down[e].T
out[b] += score[b,k] * y[b,k]
```

No bias in the v0.1 core; use **FP32 activations, accumulation and output** as a correctness scaffold. This is W4A32 execution, **not yet the target W4A16/BF16 implementation**. Real integration must match Qwen3's score normalization, types and tolerances. Routing decisions remain frozen while comparing implementations.

### Initial reference quantization format (scaffold: W4A32)
- Weight-only 4-bit symmetric *signed-in-storage-by-offset*: nibble values 0..15 encode signed values -8..7 as `q - 8`.
- Packed nibble order: lower nibble = even scalar index, upper nibble = odd scalar index.
- Gate and up row-major `[I,H]`, down row-major `[H,I]`, expert-major contiguous storage.
- Per-output-row / per-input-group scale as FP32 for initial correctness; `group_size` divides both H and I.
- **Important:** activation values are currently float32. For the targeted W4A16 path, implement FP16/BF16 input/intermediates (Qwen3 specifies BF16), match rounding semantics, and revalidate. This is **not** Marlin's optimized layout and not an AWQ/GPTQ-compatible external file format. Conversion to an optimized, scale-comparable format is required before an apples-to-apples Marlin benchmark. FP16/BF16 scales and offline weight permutation are planned upgrades.

## Current prototype (v0.1)

1. **CPU reference**: deterministic shape/routing/weight generators, exact nibble pack/unpack convention, MoE computation, traffic-floor estimator, unit tests.
2. **Conventional GPU implementation**: one gate+up kernel then one dependent down+weighted-reduce kernel, with both using identical compute loops to the persistent version.
3. **Persistent GPU implementation**: same compute loops, one kernel, resident blocks claim topologically ordered tickets using a global atomic counter; a device-scope release/acquire completion counter gates each down projection until its gate/up tiles finish.
4. **Streaming memory bandwidth probe**: measure a practical read-bandwidth denominator on the target GPU; do not assume datasheet peak is achievable.
5. **Timing harness**: GPU events, warmups, baseline-vs-persistent correctness, machine-readable CSV, sample Nsight scripts.

**What v0.1 does not yet include:** on-device top-k generation; shared-memory `cp.async` prefetch; optimized/coalesced INT4 dequant GEMV; tensor-core GEMM worker; heterogeneous per-expert worker selection; Marlin/vLLM comparison; CUDA Graph capture. The first working kernel is a correctness and synchronization scaffold, not a competitive inference kernel. It intentionally has no claim of speedup.

## Execution architecture

```
Host (C++): weights, X, routing -> GPU buffers
           |            
           +--> conventional two-kernel reference
           |
           +--> persistent kernel on fixed resident grid
                   +--> atomic task ticket
                   +--> gate/up tiles (all first in ticket space)
                   +--> per-pair readiness counter (release)
                   +--> down tiles wait for readiness (acquire)
                   +--> FP32 accumulation into output
```

- v0.1 tickets are **implicit**, not a dynamically allocated task descriptor heap. This keeps the execution invariant testable.
- Static ticket order places **all gate/up work before any down work**. The readiness counters are per `(batch,slot)` pair.
- A ready count is published only after every thread in the gate/up tile completed its writes (`__syncthreads` before device-scope `release`).
- A consumer loads with device-scope `acquire` and waits for the number of gate tiles associated with its pair.
- Resident blocks repeatedly execute one task, without grid-wide barriers. The producer/consumer order is chosen to avoid waiting on an unclaimed later ticket. This is **cooperative progress**, not hardware preemption.
- Because blocks are persistent, spinning consumers cost residency and bandwidth; this requires profiling.

## Benchmarks: honest comparisons

### Phase 0 (now): sanity/performance diagnostics
- CPU reference vs both CUDA implementations on tiny deterministic cases.
- Conventional CUDA vs persistent CUDA using **identical** compute routines and quant layout; tells us whether removing one kernel boundary helps the current scaffold. It cannot establish superiority to production MoE kernels.
- Use `ncu` to inspect DRAM throughput, long-scoreboard stalls, registers, SM occupancy, and launch overhead. Do not treat low DRAM throughput as pure scheduling overhead.

### Phase 1: strong baselines
- GPU benchmark of a **CUDA-Graph-captured** optimized W4A16 backend: Marlin-MoE (when supported by GPU / format), CUTLASS or vLLM's best compatible kernel. Record library commit, CUDA version, quantization conventions, and tensor shapes.
- Correct for confounders: same weights/activations/top-k, dtype, scales, output and numerical tolerances; include any layout conversion cost if done on the critical path.
- DRAM traffic must be measured with Nsight; cached bytes and quant metadata change the denominator.

### Phase 2: model-independent improvements
- Replace v0.1 FP32-activation GEMV loop with **BF16-activation**, offline prepermuted INT4 / coalesced GEMV worker; optimize registers and loads.
- Add `cp.async` prefetch as a **separate measured ablation**; consumer prefetches its own tiles, not another CTA's shared memory.
- Add device-side routing-dependent task generation; if already available in a fused router, do not duplicate computation.
- At batch >= 8, introduce a discrete per-expert GEMV vs GEMM choice and per-expert tile splitting. Use routing traces and synthetic skew. A mixed-worker kernel must not silently reduce occupancy through unioned resource footprints.

### Report separately
- Layer latency (us), p50/p95/p99 across many iterations, effective traffic / measured sustained BW and a *clearly labelled optimistic* roofline.
- GPU DRAM bytes/read throughput, achieved tensor-pipe throughput, warp stalls, register pressure/occupancy.
- Batch-1 latency and batch-32+ throughput; uniform and skewed routing.
- End-to-end tokens/s only after an actual model integration; no extrapolated demo claims.

## Initial analytical roofline

For Qwen3-shaped one layer, each expert has `3 * 2048 * 768 = 4,718,592` weights. At packed INT4 that is 2.25 MiB/expert. Batch 1 with eight distinct experts therefore requires **18 MiB**, or **20.16 us at 936 GB/s** (theoretical datasheet ideal, not practical throughput). Across 48 layers the selected expert matrices total **864 MiB** per token. Router weights, activations, scales and redundant reads add to this figure.

At batch N with top-8 and E=128, a simple uniform independent approximation yields `E * (1 - (1 - 8/E)^N)` distinct experts, **not a realistic routing-skew model**. Static batching can reuse each active expert's weights; whether a particular kernel does so must be checked.

**Go/no-go:** measure practical bandwidth on the target GPU, then the optimized baseline's elapsed layer latency and actual DRAM bytes. If it is already close to the attainable memory floor and the remaining gap is not recoverable launch/synchronization time, abandon the persistent path before building elaborate scheduling abstractions. A 15% estimated recoverable fraction is a decision heuristic, not a proven upper bound.

## Acceptance gates

| Gate | Required evidence |
|---|---|
| G0 — CPU | Unit tests passing for nibble packing, routing, MoE output and roofline math |
| G1 — CUDA | Builds on `sm_86`; tiny CPU-vs-GPU and two-kernel-vs-persistent outputs agree under stated tolerances; compute-sanitizer reports no races |
| G2 — Baseline | Measured sustained streaming BW, optimized CUDA Graph baseline, Nsight trace and complete numerical comparison |
| G3 — Competitive decode | Improvement versus a strong optimized baseline across repeated runs, without numerics/regressions |
| G4 — Transfer | At least two distinct MoE shape configurations use same scheduling core; robust skew-sweep data |
| G5 — Real model | Actual inference correctness and tokens/s on a real quantized model |

We should not call the project "faster" until G3 is demonstrated on real hardware. Research novelty also requires a deeper comparison with FlashMoE, MonoMoE, MPK, Cohere, RaMP and AutoMegaKernel.

## Risks to disprove early

- A naive row-per-warp GEMV may be dominated by dequant/issue pressure rather than DRAM bandwidth. Optimize compute and weight layout before drawing conclusions about scheduling.
- One mixed GEMV/GEMM kernel may lower occupancy because max registers/shared memory are reserved for all paths.
- `cp.async` prefetch can contend with gate/up for bandwidth and cannot bypass L2; additional shared-memory residency can be counterproductive.
- A waiting CTA can reduce progress. Topological tickets require a finite cooperative job and correct cross-CTA synchronization.
- Two-stage reference and persistent implementations share bugs; compare against an independent CPU oracle.
- Different scales/quant formats, routing or temperature make performance comparisons invalid.

## Sources / prior art

- Qwen3 model config: https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json
- FlashMoE: https://github.com/osayamenja/FlashMoE
- Mirage/MPK: https://github.com/mirage-project/mirage
- MonoMoE: https://arxiv.org/abs/2609.04244
- RaMP: https://arxiv.org/abs/2604.26039
- Cohere megakernels: https://cohere.com/blog/megakernels
- NVIDIA CUDA memory model: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html
- NVIDIA Nsight Compute: https://docs.nvidia.com/nsight-compute/ProfilingGuide/

## CI and collaboration

The repository includes GitHub Actions CPU tests and a compile-only CUDA job using an NVIDIA CUDA development container targeting `sm_86`. Compilation in CI validates syntax and linking but **not** GPU memory safety, numerical correctness or performance. Running `scripts/gpu_smoke.sh` and Nsight on real hardware is required for gates G1–G5. A separate publishing script will create a *private* GitHub repository when explicitly executed locally with an authenticated GitHub CLI.