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

**Takeaways (rl16k cp2):** the fix makes **FA3 all_gather functional** (was crashing) and it is the **fastest AG path** — 24.55 ms causal vs FusedAttn AG 31.40 ms (FA3 -22%), and 9.89 vs 27.13 ms at W=512 (FA3 2.7x, since cuDNN AG SWA backward does not honor the window). a2a is competitive across backends. (mixed32k/larger workloads were not finished before the node time limit.)


## 3. Status of the full sweep

The full 288-run sweep (AG+a2a × {FusedAttn,FA3} × 6 workloads × 4 masks × cp{2,4,8}) was
started but not completed this session (node time limits). Driver + partial data:
`TransformerEngine/benchmark_results/run_postkernel_sweep.sh` and `postkernel/` (resumable via
`done_postkernel.txt`). FA3 AG cells require the fix above to run crash-free.
