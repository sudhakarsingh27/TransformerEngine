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
4. [AllGather vs Ring (THD/Striped) — JAX Investigation](#allgather-vs-ring-thdstriped--jax-investigation)
5. [Key Observations](#key-observations)
6. [Setup](#setup)

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

## AllGather vs Ring (THD/Striped) — JAX Investigation

**Setup:** `bench_cp_jax.py --cp_strategy all_gather|ring --layout thd`, llama3_8b, stripe_size=1, 5 warmup + 20 timed iters.

**Short answer:** ALL_GATHER produces **numerically correct** results but is **unusably slow** with THD/Striped due to a known implementation limitation in the TE JAX backend. This is not a performance regression in our benchmark — it is a property of the current implementation.

### What we observed

| GPUs | SeqLen | RING (ms) | ALL_GATHER (ms) | Ratio |
|------|--------|-----------|-----------------|-------|
| 2 | 32K | 30.18 | 5290.92 | **175x slower** |

Loss values match (`-1003520.0000` for both), confirming correctness.

### Root cause

`FusedAttnCPStripedWithAllGatherFwdPrimitive` (`transformer_engine/jax/cpp_extensions/attention.py:2166-2180`) uses `lax.switch` to handle per-rank computation differences:

```python
functions = [partial(_cross_attn, q, k_ag, v_ag, ...) for _ in range(cp_size)]
return lax.switch(cp_rank, functions)   # cp_rank differs per GPU at runtime
```

In XLA's SPMD model, `lax.switch` with a **data-dependent index** (cp_rank varies per GPU) must compile **all `cp_size` branches** into the program and evaluate them all at runtime, keeping only the selected result. This multiplies the effective computation by `cp_size`. Furthermore, each branch operates on the **fully all-gathered KV** (length S, not S/N), so the per-branch compute is already larger than RING's per-step compute.

The TE source acknowledges this with an explicit TODO:
> *"cuDNN does not support right-aligned masking with dynamic sequence length padding. Therefore we must explicitly instantiate each CP rank slicing and use a runtime switch... TODO: When cuDNN supports, we should be able to avoid this unrolled loop."*

### RING results from the same sweep (valid)

| GPUs | SeqLen | No-CP (ms) | RING (ms) | Speedup | TFLOPS |
|------|--------|-----------|-----------|---------|--------|
| 2 | 32K | 54.14 | 30.18 | 1.79x | 1020 |
| 2 | 64K | 215.69 | 113.49 | 1.90x | 1085 |
| 2 | 128K | 874.95 | 438.66 | 1.99x | 1123 |
| 4 | 32K | 54.38 | 18.64 | 2.92x | 1652 |
| 4 | 64K | 216.25 | 61.50 | 3.52x | 2002 |
| 4 | 128K | 875.71 | 231.08 | 3.79x | 2132 |
| 8 | 32K | 54.58 | 15.44 | 3.53x | 1994 |
| 8 | 64K | 215.72 | 39.32 | 5.49x | 3132 |
| 8 | 128K | 875.76 | 127.22 | 6.88x | 3872 |

These are consistent with the earlier THD sweeps — RING with THD/Striped scales well and the numbers are valid.

### Conclusion

A fair AllGather vs Ring comparison is **not possible with THD/Striped** in the current TE JAX release. To compare the two strategies, use **BSHD layout** (`--layout bshd`), which routes to `FusedAttnCPWithAllGatherFwdPrimitive` — a different implementation that does not use `lax.switch` and should not have this overhead.

---

## Key Observations

**BSHD beats THD in both frameworks.** BSHD+DualChunkSwap is consistently 6–14% faster in TFLOPS vs THD in JAX and 8–22% faster in PyTorch. Kernel profiling confirms this comes from ragged cuDNN overhead, extra NCCL data movement, and THD-specific correction kernels.

**PyTorch p2p scales better at 8 GPUs.** The advantage is clearest at short sequences where compute-to-communication ratio is low (32K, 8 GPUs: PT 5.46x vs JAX 3.85x). At 128K the gap narrows significantly (PT 7.58x vs JAX 7.15x) as compute dominates.

**Scaling improves with sequence length.** All configs approach near-linear efficiency at 8 GPU x 128K, confirming CP is most beneficial for long-context workloads.

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
| CP strategy | p2p (NCCL send/recv) | RING (XLA collective) |
| BSHD reorder | DualChunkSwap | DualChunkSwap |
| THD reorder | DualChunkSwap | Striped (stripe=1) |
| Timing | `torch.cuda.Event` | `perf_counter` + `block_until_ready` |
| Profiling | `nsys` + `cudaProfilerApi` | `nsys` (external wrap) |
| Warmup / Timed | 5 / 20 (3 for profiling) | 5 / 20 (3 for profiling) |
| Scripts | `bench_cp_pytorch.py` | `bench_cp_jax.py` |
| Profiler | `profile_cp_kernels.py` | — |
