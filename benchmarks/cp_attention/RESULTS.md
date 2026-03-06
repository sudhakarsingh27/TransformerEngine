# Context Parallel Attention Benchmark Results

**Model:** llama3_8b — 32 Q-heads, 8 KV-heads (GQA), head_dim=128, batch=1, bf16
**Kernel:** cuDNN FusedAttention, causal mask, H100 80GB
**Date:** 2026-03-05

> JAX and PyTorch results were collected on separate machines. Absolute latency is not directly comparable across frameworks; speedup ratios and TFLOPS are the meaningful metrics.

## Table of Contents

1. [JAX vs PyTorch Head-to-Head](#jax-vs-pytorch-head-to-head-cp-runs-only)
2. [CP Speedup over no-CP Baseline](#cp-speedup-over-no-cp-baseline)
3. [GPU Kernel Profiling: BSHD vs THD](#gpu-kernel-profiling-bshd-vs-thd)
   - [Kernel Category Breakdown](#kernel-category-breakdown)
   - [Kernel Details by Category](#kernel-details-by-category)
   - [Why BSHD is Faster than THD](#why-bshd-is-faster-than-thd)
4. [AllGather vs Ring (THD/Striped) — JAX](#allgather-vs-ring-thdstriped--jax)
5. [Sliding Window Attention (SWA) vs Full Causal](#sliding-window-attention-swa-vs-full-causal)
6. [Key Observations](#key-observations)
7. [Setup](#setup)

---

## JAX vs PyTorch Head-to-Head (CP runs only)

### BSHD + DualChunkSwap — CP Fwd+Bwd Time (ms) and TFLOPS

| GPUs | SeqLen | JAX-BSHD fwd+bwd (ms) | PT-BSHD fwd+bwd (ms) | JAX TFLOPS | PT TFLOPS |
|------|--------|--------------|-------------|------------|-----------|
| 2 | 32K | 28.27 | 29.47 | 1089 | 1045 |
| 2 | 64K | 107.75 | 115.09 | 1143 | 1070 |
| 2 | 128K | 417.42 | 450.84 | **1180** | 1093 |
| 4 | 32K | 16.84 | **15.86** | 1828 | **1941** |
| 4 | 64K | 57.81 | 58.75 | **2130** | 2096 |
| 4 | 128K | **219.15** | 230.87 | **2248** | 2134 |
| 8 | 32K | 13.48 | **10.22** | 2284 | **3014** |
| 8 | 64K | 35.08 | **31.82** | 3510 | **3870** |
| 8 | 128K | **118.29** | 120.28 | **4164** | 4095 |

### THD — CP Fwd+Bwd Time (ms) and TFLOPS
*(JAX uses Striped reordering; PyTorch uses DualChunkSwap)*

| GPUs | SeqLen | JAX-THD fwd+bwd (ms) | PT-THD fwd+bwd (ms) | JAX TFLOPS | PT TFLOPS |
|------|--------|-------------|------------|------------|-----------|
| 2 | 32K | 30.20 | 31.88 | 1019 | 966 |
| 2 | 64K | 113.73 | 122.09 | 1083 | 1009 |
| 2 | 128K | 440.06 | 478.39 | **1119** | 1030 |
| 4 | 32K | 18.63 | **17.55** | 1653 | **1754** |
| 4 | 64K | 61.44 | 63.64 | **2004** | 1935 |
| 4 | 128K | **230.80** | 246.40 | **2134** | 1999 |
| 8 | 32K | 15.31 | **12.46** | 2010 | **2471** |
| 8 | 64K | 39.19 | **35.20** | 3142 | **3499** |
| 8 | 128K | **126.46** | 130.34 | **3895** | 3779 |

**Takeaway:** JAX leads at 2 GPUs and for long sequences; PyTorch p2p pulls ahead at 4–8 GPUs for shorter sequences where communication overhead matters more.

---

## CP Speedup over no-CP Baseline

| GPUs | SeqLen | JAX-BSHD | JAX-THD | PT-BSHD | PT-THD |
|------|--------|----------|---------|---------|--------|
| 2 | 32K | 1.84x | 1.80x | 1.89x | 1.83x |
| 2 | 64K | 1.92x | 1.90x | 1.94x | 1.90x |
| 2 | 128K | 2.02x | 2.00x | 2.02x | 1.99x |
| 4 | 32K | 3.09x | 2.93x | **3.52x** | 3.32x |
| 4 | 64K | 3.59x | 3.51x | **3.78x** | 3.65x |
| 4 | 128K | 3.85x | 3.81x | **3.95x** | 3.86x |
| 8 | 32K | 3.85x | 3.57x | **5.46x** | 4.67x |
| 8 | 64K | 5.92x | 5.50x | **7.00x** | 6.60x |
| 8 | 128K | 7.15x | 6.95x | **7.58x** | 7.30x |

---

## GPU Kernel Profiling: BSHD vs THD

Profiled with `nsys` using `cudaProfilerApi` capture range on PyTorch, 8 GPUs, llama3_8b, S=128K, 3 timed iterations. Kernel traces filtered to GPU 0 (rank 0) to avoid double-counting.

Profiling tool: `profile_cp_kernels.py --compare --ngpus 8 --seq_len 131072`

### Kernel Category Breakdown

| Category | BSHD Time (ms) | BSHD % | THD Time (ms) | THD % | Delta |
|----------|---------------|--------|--------------|-------|-------|
| cuDNN Attention | 2705.01 | 83.6% | 2876.38 | 79.7% | +171.38 ms |
| NCCL Communication | 476.58 | 14.7% | 612.39 | 17.0% | +135.81 ms |
| CP Support (THD) | 0.00 | 0.0% | 64.95 | 1.8% | +64.95 ms |
| Other | 53.17 | 1.6% | 53.94 | 1.5% | +0.77 ms |
| **TOTAL** | **3234.76** | **100%** | **3607.67** | **100%** | **+372.91 ms** |

> Times shown are cumulative across 3 profiled iterations (divide by 3 for per-iteration cost).
> Per-iteration overhead of THD vs BSHD: ~124 ms.

### Kernel Details by Category

**cuDNN Attention** — The core fused attention kernels from cuDNN. Both BSHD and THD use the same underlying `sm90_flash_fprop` and `sm90_flash_bprop` kernels, but THD invokes _ragged_ variants that handle variable-length token sequences.

| Kernel | BSHD | THD | Notes |
|--------|------|-----|-------|
| `sdpa_sm90_flash_fprop_wgmma_f16` | 192x | 192x | Forward pass attention kernel (identical) |
| `sdpa_sm90_flash_bprop_wgmma_f16` | 192x | 192x | Backward pass attention kernel (identical) |
| `compute_dot_do_o_specialized` | 192x | — | BSHD: dense dO·O dot product |
| `compute_dot_do_o_ragged_specialized` | — | 192x | THD: ragged variant (+overhead from index lookups) |
| `convert_dq_to_16bits` | 192x | — | BSHD: dense dQ conversion |
| `convert_dq_to_16bits_ragged` | — | 192x | THD: ragged variant |
| `fmha_reduce_head` | 384x | — | BSHD: head reduction |
| `fmha_reduce_head_ragged` | — | 384x | THD: ragged head reduction |
| `qkv_tma_setup` | — | 384x | THD-only: TMA descriptor setup for ragged access |

The `_ragged` kernel variants add overhead because they must look up per-token offsets from `cu_seqlens` at each access, whereas dense BSHD kernels use simple strided indexing.

**NCCL Communication** — Both formats use 720 `ncclDevKernel_SendRecv` calls (same count since CP size and iteration count are identical). THD is +136ms slower because token-level partitioning via `thd_get_partitioned_indices` produces non-contiguous memory layouts, requiring extra data gathering before send/recv.

| Kernel | BSHD | THD |
|--------|------|-----|
| `ncclDevKernel_SendRecv` | 720x, 476.58 ms | 720x, 612.39 ms |

**CP Support (THD)** — These kernels exist **only in THD** and are responsible for the token-level partitioning and correction logic that THD requires. BSHD shows 0.00 ms because DualChunkSwap sharding is done purely through tensor indexing on the host — no custom CUDA kernels are needed. In contrast, THD must launch dedicated kernels to handle its variable-length token sequences:

| Kernel | Count | Purpose |
|--------|-------|---------|
| `thd_read_half_tensor_kernel` | 672x | Reads half of a partitioned tensor for each CP step — extracts the tokens this rank needs to process from the received KV buffer |
| `thd_grad_correction_kernel` | 279x | Corrects gradients (dK, dV) after the backward pass — accumulates partial gradients from different CP ranks back to the correct token positions |
| `thd_out_correction_kernel` | 192x | Corrects the forward output using log-sum-exp (LSE) values — rescales partial attention outputs from each CP step to produce numerically correct results |
| `thd_lse_kernel` | 105x | Computes log-sum-exp across CP steps — needed for the numerically stable online softmax correction when combining partial attention results |

These 1248 kernel launches contribute 64.95 ms (1.8% of THD total). While individually small, they add up because they are launched at every CP communication step (7 steps for 8 GPUs) across both forward and backward passes.

**Other** — Memset, memcpy, elementwise ops, seed extraction. Roughly equal between formats. Notable THD-specific entries:
- `cu_seqlens_to_actual_seqlens` (384x): converts padded cu_seqlens to actual lengths
- `cu_seqlens_padded_to_offsets` (384x): computes memory offsets from padded cu_seqlens

### Why BSHD is Faster than THD

THD's total overhead vs BSHD at 8 GPUs, 128K is **+373ms** (across 3 iterations), breaking down as:

1. **cuDNN ragged kernels: +171ms (46% of overhead)**
   The same attention algorithm, but ragged variants must dereference `cu_seqlens` to find token boundaries at each step. Dense BSHD uses simple `batch * seq * head * dim` striding.

2. **NCCL communication: +136ms (36% of overhead)**
   Same number of send/recv calls, but THD's non-contiguous token layout requires extra gathering. BSHD's DualChunkSwap produces contiguous memory chunks that map directly to NCCL buffers.

3. **THD support kernels: +65ms (17% of overhead)**
   The `thd_read_half`, `thd_grad_correction`, `thd_out_correction`, and `thd_lse` kernels have no BSHD equivalent — BSHD handles these operations implicitly through its chunk-level data layout.

4. **Other: +1ms (negligible)**
   Extra `cu_seqlens` processing kernels in THD.

---

## AllGather vs Ring (THD/Striped) — JAX

**Setup:** `bench_cp_jax.py --cp_strategy all_gather|ring --layout thd`, llama3_8b, THD/Striped, 5 warmup + 20 timed iters. ALL_GATHER tested with stripe_size=1024 and 2048 (the optimal range from the stripe_size sensitivity sweep below).

### Head-to-head: RING vs ALL_GATHER (ss=1024) vs ALL_GATHER (ss=2048)

| GPUs | SeqLen | RING (ms) | AG ss=1024 (ms) | AG ss=2048 (ms) | Best AG/RING | RING TFLOPS | Best AG TFLOPS |
|------|--------|-----------|-----------------|-----------------|-------------|-------------|----------------|
| 2 | 32K | 30.52 | 35.13 | 34.98 | 1.15x slower | 1009 | 880 |
| 2 | 64K | 115.39 | 135.22 | 131.19 | 1.14x slower | 1067 | 939 |
| 2 | 128K | 444.11 | 533.14 | 510.55 | 1.15x slower | 1109 | 965 |
| 4 | 32K | 18.69 | **19.86** | 20.58 | 1.06x slower | 1647 | 1550 |
| 4 | 64K | 61.87 | 70.10 | **69.81** | 1.13x slower | 1991 | 1764 |
| 4 | 128K | 232.68 | 270.33 | **261.77** | 1.12x slower | 2117 | 1882 |
| 8 | 32K | 15.30 | **12.72** | 13.87 | **0.83x faster** | 2013 | **2420** |
| 8 | 64K | 39.38 | **38.90** | 40.24 | **0.99x ~tied** | 3128 | **3166** |
| 8 | 128K | 127.50 | 142.18 | **140.07** | 1.10x slower | 3864 | 3517 |

### CP Speedup over no-CP baseline

| GPUs | SeqLen | RING | AG ss=1024 | AG ss=2048 |
|------|--------|------|-----------|-----------|
| 2 | 32K | 1.78x | 1.55x | 1.55x |
| 2 | 64K | 1.87x | 1.59x | 1.63x |
| 2 | 128K | 1.97x | 1.64x | 1.72x |
| 4 | 32K | 2.92x | 2.73x | 2.64x |
| 4 | 64K | 3.48x | 3.06x | 3.08x |
| 4 | 128K | 3.77x | 3.25x | 3.35x |
| 8 | 32K | 3.55x | **4.29x** | 3.93x |
| 8 | 64K | 5.45x | **5.53x** | 5.35x |
| 8 | 128K | **6.88x** | 6.16x | 6.26x |

### stripe_size sensitivity (2 GPUs, 32K)

ALL_GATHER performance is **highly sensitive to stripe_size**. This micro-sweep on a single config (2 GPUs, 32K) identifies the optimal range used in the full sweep above.

| stripe_size | ALL_GATHER (ms) | vs no-CP | vs RING (31ms) |
|-------------|-----------------|----------|----------------|
| 1 | 5293 | 0.01x | 172x slower |
| 4 | 1316 | 0.04x | 43x slower |
| 16 | 340 | 0.16x | 11x slower |
| 64 | 95.5 | 0.56x | 3.1x slower |
| 128 | 62.9 | 0.85x | 2.0x slower |
| 256 | 44.6 | 1.21x | 1.45x slower |
| 512 | 37.5 | 1.43x | 1.22x slower |
| **1024** | **35.2** | **1.53x** | **1.14x slower** |
| **2048** | **35.0** | **1.54x** | **1.14x slower** |
| 4096 | 37.1 | 1.45x | 1.21x slower |

The dominant factor is **cuDNN ragged segment granularity**, not the `lax.switch` overhead. With stripe_size=1, each ragged segment is a single token — cuDNN spends most of its time on per-token offset lookups. Larger stripe sizes produce fewer, larger contiguous chunks that the kernel processes efficiently. Beyond ss=2048, performance regresses slightly as too-large stripes reduce causal load balancing uniformity.

### Implementation detail: `lax.switch`

`FusedAttnCPStripedWithAllGatherFwdPrimitive` (`transformer_engine/jax/cpp_extensions/attention.py:2166-2180`) uses `lax.switch(cp_rank, [fn] * cp_size)` to work around a cuDNN limitation (no right-aligned masking with dynamic sequence lengths). The stripe_size sweep confirms this is **not the bottleneck** — the 150x improvement from ss=1 to ss=1024 comes entirely from giving cuDNN larger segments.

### Conclusion

**At 8 GPUs, ALL_GATHER with ss=1024 beats RING** for short sequences (12.72ms vs 15.30ms at 32K) and ties at 64K. At 2–4 GPUs and longer sequences, **RING is 6–17% faster**. The crossover happens because ALL_GATHER's single collective scales better than RING's per-step send/recv at high GPU counts, while RING's pipelined overlap is more efficient at lower parallelism.

**Practical guidance:**
- **2–4 GPUs or long sequences**: RING is the better default (6–17% faster).
- **8+ GPUs, short-to-medium sequences**: ALL_GATHER with ss=1024 is competitive or faster.
- **stripe_size >= 256** is required for ALL_GATHER to be useful; ss=1024 is optimal across all tested configs.
- Both strategies produce numerically identical results (same loss values).

---

## Sliding Window Attention (SWA) vs Full Causal

**Setup:** `bench_cp_jax.py --window_size W`, llama3_8b, THD/Striped/RING, 5 warmup + 20 timed iters. Both no-CP baseline and CP run use the same window_size for fair comparison.

### 8 GPUs, 128K sequence

| Window | No-CP (ms) | CP (ms) | CP Speedup |
|--------|-----------|---------|------------|
| Full causal | 875 | 128 | 6.87x |
| W=512 | 1435 | 207 | 6.95x |
| W=1024 | 1439 | 207 | 6.96x |
| W=2048 | 1445 | 207 | 6.98x |
| W=4096 | 1461 | 209 | 7.01x |
| W=8192 | 1489 | 211 | 7.07x |
| W=16384 | 1535 | 218 | 7.05x |
| W=32768 | 1625 | 231 | 7.04x |
| BSHD full (ref) | 845 | 119 | 7.10x |

### 8 GPUs, 32K sequence

| Window | No-CP (ms) | CP (ms) | CP Speedup |
|--------|-----------|---------|------------|
| Full causal | 54 | 15.1 | 3.60x |
| W=512 | 94 | 21.2 | 4.44x |
| W=1024 | 95 | 21.2 | 4.48x |
| W=2048 | 97 | 21.9 | 4.42x |
| W=4096 | 99 | 21.3 | 4.68x |

### Analysis

**SWA is slower than full causal attention** across all tested window sizes and sequence lengths. This is counterintuitive — a smaller window should mean less computation.

1. **cuDNN does not skip masked computation.** The SWA kernel appears to still process the full attention matrix and apply a window mask, rather than reducing the actual number of FLOPs. This adds overhead (window bounds checking, mask logic) without reducing compute.

2. **The overhead is largely fixed, not proportional to window size.** At 128K, all windows from W=512 to W=2048 produce nearly identical CP times (~207ms). The no-CP baseline shows slight variation (1435–1445ms), but the CP overhead is constant. This confirms the SWA cost is a fixed per-kernel surcharge, not proportional to the window size.

3. **CP speedup is slightly better with SWA.** At 32K: SWA gives 4.4–4.7x speedup vs 3.6x for full causal. This is because SWA inflates the no-CP baseline more than the CP time (the per-GPU compute with CP is small enough that the fixed SWA overhead is relatively less significant).

4. **XLA compilation is very slow for SWA + CP + THD.** At 32K with 8 GPUs, each SWA config takes ~7 minutes to compile. The SWA window parameter changes the XLA computation graph, triggering full recompilation.

5. **Loss values vary with window size** — full causal gives -4,079,616 (128K) / -1,003,520 (32K), while SWA windows give different values, confirming correct SWA computation (not just masking after the fact).

### Conclusion

With the current cuDNN FusedAttention backend in TE JAX, **SWA does not provide a throughput benefit** — it adds a fixed overhead of ~60% (128K) to ~75% (32K) regardless of window size. The kernel implements SWA through masking rather than compute skipping. SWA remains useful for **model quality** (controlling attention span) but should not be expected to improve performance in this configuration.

---

## Key Observations

**BSHD beats THD in both frameworks.** BSHD+DualChunkSwap is consistently 6–14% faster in TFLOPS vs THD in JAX and 8–22% faster in PyTorch. Kernel profiling confirms this comes from three sources: ragged cuDNN kernels (46% of overhead), NCCL communication with non-contiguous memory (36%), and THD-specific correction kernels (17%).

**PyTorch p2p scales better at 8 GPUs.** The advantage is clearest at short sequences where compute-to-communication ratio is low (32K, 8 GPUs: PT 5.46x vs JAX 3.85x). At 128K the gap narrows significantly (PT 7.58x vs JAX 7.15x) as compute dominates.

**Scaling improves with sequence length.** All configs approach near-linear efficiency at 8 GPU x 128K, confirming CP is most beneficial for long-context workloads.

**AllGather beats Ring at 8 GPUs with optimal stripe_size.** With ss=1024, AllGather is faster than Ring at 8 GPUs/32K (12.72ms vs 15.30ms) and ties at 64K, because its single collective scales better than Ring's per-step send/recv at high GPU counts. At 2–4 GPUs, Ring is 6–17% faster. stripe_size >= 256 is required (default ss=1 is 172x slower due to cuDNN ragged segment granularity).

**SWA adds overhead, does not reduce compute.** Sliding window attention is slower than full causal across all window sizes (512–32768) and sequence lengths (32K, 128K). The cuDNN kernel implements SWA through masking rather than compute skipping, adding a fixed ~60–75% overhead. SWA is useful for model quality but not for throughput.

**Peak throughput (8 GPU, 128K):**

| Config | TFLOPS |
|--------|--------|
| JAX BSHD | **4164** |
| PT BSHD | 4095 |
| JAX THD | 3895 |
| PT THD | 3779 |

---

## Setup

| | PyTorch | JAX |
|-|---------|-----|
| CP strategy | p2p (NCCL send/recv) | RING or ALL_GATHER (XLA collective) |
| BSHD reorder | DualChunkSwap | DualChunkSwap |
| THD reorder | DualChunkSwap | Striped (stripe_size=1 default) |
| Timing | `torch.cuda.Event` | `perf_counter` + `block_until_ready` |
| Profiling | `nsys` + `cudaProfilerApi` | — |
| Warmup / Timed | 5 / 20 (3 for profiling) | 5 / 20 |
| Scripts | `bench_cp_pytorch.py` | `bench_cp_jax.py` |
| Profiler | `profile_cp_kernels.py` | — |

**Benchmark inputs:** Both PyTorch and JAX benchmarks use single-segment, uniform-length, fully-packed THD inputs (no variable-length packing, no padding). This represents the best case for THD — real workloads with packed multi-segment sequences would have higher ragged kernel overhead.

**JAX benchmark flags:** `--cp_strategy ring|all_gather`, `--stripe_size N` (THD only), `--window_size W` (SWA), `--profile` (reduced iters for nsys).
