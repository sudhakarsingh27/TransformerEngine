# Context Parallel Attention Benchmark Results

**Model:** llama3_8b — 32 Q-heads, 8 KV-heads (GQA), head_dim=128, batch=1, bf16
**Kernel:** cuDNN FusedAttention, causal mask, H100 80GB
**Date:** 2026-03-05

> JAX and PyTorch results were collected on separate machines. Absolute latency is not directly comparable across frameworks; speedup ratios and TFLOPS are the meaningful metrics.

---

## JAX vs PyTorch Head-to-Head (CP runs only)

### BSHD + DualChunkSwap — CP Latency (ms) and TFLOPS

| GPUs | SeqLen | JAX-BSHD (ms) | PT-BSHD (ms) | JAX TFLOPS | PT TFLOPS |
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

### THD — CP Latency (ms) and TFLOPS
*(JAX uses Striped reordering; PyTorch uses DualChunkSwap)*

| GPUs | SeqLen | JAX-THD (ms) | PT-THD (ms) | JAX TFLOPS | PT TFLOPS |
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

## Key Observations

**BSHD beats THD in both frameworks.** BSHD+DualChunkSwap is consistently 6–14% faster in TFLOPS vs THD in JAX and 8–22% faster in PyTorch. THD carries additional no-CP overhead (~5–10ms) from token padding/indexing.

**PyTorch p2p scales better at 8 GPUs.** The advantage is clearest at short sequences where compute-to-communication ratio is low (32K, 8 GPUs: PT 5.46x vs JAX 3.85x). At 128K the gap narrows significantly (PT 7.58x vs JAX 7.15x) as compute dominates.

**Scaling improves with sequence length.** All configs approach near-linear efficiency at 8 GPU × 128K, confirming CP is most beneficial for long-context workloads.

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
| Warmup / Timed | 5 / 20 | 5 / 20 |
| Scripts | `bench_cp_pytorch.py` | `bench_cp_jax.py` |
