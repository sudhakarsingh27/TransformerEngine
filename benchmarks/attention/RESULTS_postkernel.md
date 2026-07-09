# Post-kernel results — PR 2829 (fused thd_reorder + thd_valid_copy)

Date: 2026-06-01. TE `2.17.0.dev0+24a95ab8` (branch `cp_thd_swa_with_ag`), editable from the
worktree. Hardware: 8× H100 80GB HBM3, cuDNN 9.21, NCCL 2.29.7, bf16, qkv_format=thd.

This documents benchmarking of the two sync-free kernel commits on top of the existing
[`RESULTS.md`](RESULTS.md) / [`README.md`](README.md) baseline:
- `2dc5c15c` — fused `thd_reorder` (+ sync-free CP THD reorder)
- `24a95ab8` — sync-free `thd_valid_copy` for AllGather CP THD fwd/bwd

## 1. Correctness regression found + fixed: FA3 + all_gather race

Benchmarking surfaced a **crash regression**: `kernel_backend=FlashAttention`,
`cp_comm_type=all_gather`, `qkv_format=thd` → intermittent `cudaErrorIllegalAddress` on
larger-token workloads (bucket32k/64k/128k, mixed32k); small ones (rl16k, outlier64k) pass.
FusedAttention AG and FA3 a2a are unaffected. Pre-kernel TE (`29785a09`, RESULTS.md 2026-05-08)
ran these FA3 AG configs fine, so the two kernel commits introduced it.

**Root cause** (forward pass, confirmed): a cross-stream **caching-allocator reuse race**. The
fused `tex.thd_reorder` runs on the main stream; its faster, sync-free execution lets the host
run further ahead into the 2-stream per-step loop (step 1 on `cp_stream`), and PyTorch's
allocator recycles an in-flight block. Confirmed three ways, all of which avoid the crash:
`CUDA_LAUNCH_BLOCKING=1`, `PYTORCH_NO_CUDA_MEMORY_CACHING=1`, and a hard stream sync after the
reorder. The faulting block is not reachable from Python (gather outputs are contiguous and
kept-alive; reorder outputs live through the loop; `record_stream` on both has no effect; the
kernel/binding are correct in isolation) — most likely FlashAttention's own scratch, exposed by
the tighter overlap. Full analysis: `ag_thd_swa/mb_runs/FA3_AG_RECORDSTREAM_RACE.md`.

**Interim fix** (`context_parallel.py`, env-gated `AG_REORDER_SYNC`, default on):
`torch.cuda.current_stream().synchronize()` after the forward AG reorder, before the per-step
loop. Quantified cost (FA3 AG cp2, serial, vs fix off):

| config | no-fix (ms) | with-fix (ms) | cost |
|---|---:|---:|---:|
| rl16k causal | 24.12 | 24.40 | +1.2% |
| outlier64k causal | 109.90 | 110.44 | +0.5% |
| rl16k_swa512 | 9.48 | 9.89 | +4.3% |

Far below the 11–30% e2e the kernels recovered — it's one sync per forward, placed *before* the
compute loop, so the cp_stream-vs-main per-step overlap is intact (unlike the old per-segment
`.item()` D2H stalls inside the copy loops). A zero-cost allocator/event fix is a PR follow-up
(needs a CUDA memory-snapshot trace to pin the block).

## 2. Clean cp2 numbers (serial, with fix), ms/iter fwd+bwd, benchmark=30

> Methodology note: collected **serially** (one job at a time). The earlier 4-wide-concurrent
> sweep produced 2–6× inflated times from NVLink contention across simultaneous all-gather jobs
> and is NOT used here. These serial numbers are clean but not directly comparable to the
> 4-wide baseline tables in README.md/RESULTS.md; treat as a fresh same-conditions snapshot.

| workload | mask | FusedAttn AG | FusedAttn a2a | FA3 AG | FA3 a2a |
|---|---|---:|---:|---:|---:|
| rl16k | causal | 31.40 | 27.98 | 24.55 | 24.34 |
| rl16k | W=512 | 27.13 | 10.39 | 9.89 | 10.19 |
| mixed32k | causal | 163.65 | 139.95 | 120.46 | 120.33 |
| mixed32k | W=512 | 133.68 | 24.58 | 22.47 | 24.78 |

**Takeaways (cp2):** the fix makes **FA3 all_gather functional** (was crashing) and it is the
**fastest AG path** at both workloads:
- causal: FA3 AG vs FusedAttn AG — rl16k 24.55 vs 31.40 (−22%), mixed32k 120.46 vs 163.65 (−26%).
- W=512: FA3 AG vs FusedAttn AG — rl16k 9.89 vs 27.13 (**2.7×**), mixed32k 22.47 vs 133.68 (**6.0×**).
  FusedAttn AG barely speeds up under SWA (133.68 vs 163.65 causal) because cuDNN's AG SWA
  backward does not honor the window; FA3 AG drops to ~the a2a level.
- a2a is competitive across backends and is the best non-FA3 option for SWA.

(bucket128k and cp8 were not reached before the node time limit; the resumable drivers remain.)


## 3. Status of the full sweep

The full 288-run sweep (AG+a2a × {FusedAttn,FA3} × 6 workloads × 4 masks × cp{2,4,8}) was
started but not completed this session (node time limits). Driver + partial data:
`TransformerEngine/benchmark_results/run_postkernel_sweep.sh` and `postkernel/` (resumable via
`done_postkernel.txt`). FA3 AG cells require the fix above to run crash-free.

## 4. THD helper-kernel microbenchmarks

Date: 2026-06-03. Script: `benchmarks/attention/thd_kernel_bench.py`. Shape:
batch=16, seqlen=4096, heads=16, dim=64, bf16, 5 warmup + 20 timed iterations, single H100.
Wall-clock timings include host overhead; this is intentional because the old `thd_valid_copy`
reference contains `.item()` calls that synchronize the host with the device.

| cp | path | legacy wall ms | kernel wall ms | speedup | kernel event ms |
|---:|---|---:|---:|---:|---:|
| 2 | `thd_reorder` contiguous->rank | 18.6232 | 0.1090 | 170.84x | 0.0948 |
| 2 | `thd_reorder` rank->contiguous | 19.2267 | 0.1094 | 175.76x | 0.0951 |
| 2 | `thd_valid_copy` | 9.4218 | 0.1052 | 89.59x | 0.0904 |
| 4 | `thd_reorder` contiguous->rank | 37.4912 | 0.1083 | 346.08x | 0.0942 |
| 4 | `thd_reorder` rank->contiguous | 11.0553 | 0.0966 | 114.43x | 0.0958 |
| 4 | `thd_valid_copy` | 0.7070 | 0.0911 | 7.76x | 0.0904 |
| 8 | `thd_reorder` contiguous->rank | 11.6829 | 0.0954 | 122.42x | 0.0944 |
| 8 | `thd_reorder` rank->contiguous | 17.2532 | 0.0970 | 177.87x | 0.0962 |
| 8 | `thd_valid_copy` | 0.7075 | 0.0911 | 7.76x | 0.0904 |

Kernel-only helper timings from the same runs were small: `thd_read_half_tensor` was ~0.05 ms for
`[t,h,d]` and ~0.09 ms for `[2,t,h,d]`, `thd_read_second_half_lse` was ~0.009-0.026 ms, and
`thd_get_partitioned_indices` was ~0.005-0.017 ms depending on cp size.

Reference policy: reorder and valid-copy use the older non-optimized Python implementations from
the PR history (`89b1066d` and `24a95ab8^`). `thd_read_half_tensor` and
`thd_read_second_half_lse` did not have a prior Python reference in the branch history, so the
script reports kernel-only timings for those helpers and the unit tests use fixed semantic
expected tensors.
