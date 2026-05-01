# Context-Parallel Attention Benchmarks

Benchmark and profile suite for THD context-parallel attention with three communication backends: **p2p** (ring), **all_gather** (full KV gather), and **a2a** (all-to-all head redistribution).

## Quick Start

All commands run from the `tests/pytorch/attention/` directory. The runner (`run_attention_with_cp.py`) accepts `key=value` CLI args after the torch.distributed launcher.

### Single benchmark run

```bash
cd tests/pytorch/attention

# Benchmark: 50 timed iterations on 2 GPUs, bucket32k workload, a2a backend
python -m torch.distributed.launch --nproc-per-node=2 \
    run_attention_with_cp.py \
    dtype=bf16 model=bucket32k qkv_format=thd \
    kernel_backend=FusedAttention cp_comm_type=a2a \
    benchmark=50 log_level=WARNING \
    thd_seqlen_pattern="24576,28672,30720,32768"
```

### Single profile run (nsys)

```bash
# Profile: 5 iterations, rank-0 only capture
NSYS_OUT=my_profile torchrun --nproc-per-node=4 --no-python \
    /path/to/nsys_rank0_only.sh \
    python run_attention_with_cp.py \
    dtype=bf16 model=mixed32k qkv_format=thd \
    kernel_backend=FusedAttention cp_comm_type=p2p \
    benchmark=5 log_level=WARNING \
    thd_seqlen_pattern="16384,24576,32768,8192,28672,32768,20480,16384"
```

The `nsys_rank0_only.sh` wrapper runs rank 0 under `nsys profile` and other ranks bare.

### SWA (Sliding Window Attention)

SWA configs append `_swa<W>` to the model name. p2p does not support SWA — use all_gather or a2a.

```bash
python -m torch.distributed.launch --nproc-per-node=8 \
    run_attention_with_cp.py \
    dtype=bf16 model=mixed32k_swa512 qkv_format=thd \
    kernel_backend=FusedAttention cp_comm_type=a2a \
    benchmark=50 log_level=WARNING \
    thd_seqlen_pattern="16384,24576,32768,8192,28672,32768,20480,16384"
```

## Runner Parameters

| Parameter | Default | Description |
|---|---|---|
| `dtype` | — | `bf16`, `fp16`, or `fp8` |
| `model` | — | Config name from `benchmark_cp.py` (e.g. `bucket32k`, `mixed32k_swa1024`) |
| `qkv_format` | `bshd` | `bshd`, `sbhd`, or `thd` (variable-length packed) |
| `kernel_backend` | `FlashAttention` | `FusedAttention` (cuDNN) or `FlashAttention` |
| `cp_comm_type` | `p2p` | `p2p`, `all_gather`, or `a2a` |
| `benchmark` | `0` | Number of timed iterations (0 = correctness-only, no timing) |
| `thd_seqlen_pattern` | `random` | Comma-separated per-sequence lengths, or `random`/`max`/`half`/`linear`/`alternating` |
| `log_level` | `WARNING` | Python logging level |
| `is_training` | `True` | Run backward pass |
| `deterministic` | `False` | Force deterministic cuDNN algorithms |

## Available Configs

Configs are defined in `benchmark_cp.py` and auto-merged into the runner's config dict.

### Uniform THD (constant seqlen)

| Config | B | S | H | g | d | mask |
|---|---:|---:|---:|---:|---:|---|
| bench_8k | 2 | 8192 | 32 | 8 | 128 | causal |
| bench_16k | 1 | 16384 | 32 | 8 | 128 | causal |
| bench_32k | 1 | 32768 | 32 | 8 | 128 | causal |
| cp_thd_0 | 8 | 8192 | 12 | 12 | 128 | causal |
| cp_thd_1 | 8 | 8192 | 12 | 12 | 128 | non-causal |
| cp_thd_2 | 16 | 4096 | 12 | 12 | 128 | causal |
| cp_thd_3 | 8 | 8192 | 12 | 2 | 128 | causal |

### Variable-length training workloads (Llama3-8B-shaped: H=32, g=8, d=128)

| Workload | B | S_max | thd_seqlen_pattern |
|---|---:|---:|---|
| rl16k | 8 | 16384 | 4096,6144,6144,8192,8192,10240,12288,16384 |
| bucket32k | 4 | 32768 | 24576,28672,30720,32768 |
| mixed32k | 8 | 32768 | 16384,24576,32768,8192,28672,32768,20480,16384 |
| outlier64k | 4 | 65536 | 8192,8192,8192,65536 |
| bucket64k | 4 | 65536 | 57344,61440,63488,65536 |
| bucket128k | 3 | 131072 | 114688,122880,131072 |

SWA variants: append `_swa512`, `_swa1024`, or `_swa2048` to any training workload name (e.g. `mixed32k_swa1024`). Window is `(W, 0)` — left-only sliding window with causal mask.

### Skip rules

- **a2a**: requires `num_heads % cp_size == 0` AND `num_gqa_groups % cp_size == 0`
- **p2p + SWA**: not supported (p2p ring protocol cannot express windowed attention)

## Benchmark Results

Hardware: 8× H100 80GB HBM3, NCCL, bf16, FusedAttention (cuDNN ≥ 9.22).
Iters: 50 timed (after 10 warmup). Values in ms/iter (fwd+bwd).

### Full causal — training workloads

| Workload | cp=2 p2p | cp=2 AG | cp=2 a2a | cp=4 p2p | cp=4 AG | cp=4 a2a | cp=8 p2p | cp=8 AG | cp=8 a2a |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rl16k | 26.50 | 32.63 | **28.03** | 16.31 | 18.59 | **14.50** | 12.57 | 12.97 | **7.99** |
| bucket32k | **103.82** | 124.23 | 105.37 | 56.27 | 62.49 | **53.58** | 33.17 | 34.65 | **27.40** |
| mixed32k | **140.22** | 167.77 | 142.02 | 76.71 | 84.49 | **72.55** | 45.30 | 47.39 | **37.16** |
| outlier64k | **130.87** | 157.70 | 131.88 | 68.99 | 77.18 | **66.86** | 38.68 | 41.03 | **34.23** |
| bucket64k | **435.59** | 524.93 | 439.71 | 227.24 | 252.45 | **220.08** | 123.44 | 131.01 | **111.55** |
| bucket128k | **1253.48** | 1710.59 | 1293.13 | **640.02** | 729.16 | 640.68 | 337.82 | 360.72 | **324.56** |

**Bold = fastest.** a2a wins at cp≥4; p2p ties at cp=2 (network-bottlenecked).

### SWA — training workloads (all_gather vs a2a)

**cp=2**

| Workload | W=512 AG | W=512 a2a | W=1024 AG | W=1024 a2a | W=2048 AG | W=2048 a2a |
|---|---:|---:|---:|---:|---:|---:|
| rl16k | 28.74 | **10.44** | 29.19 | **11.98** | 30.02 | **14.84** |
| bucket32k | 99.83 | **16.60** | 100.67 | **19.26** | 102.16 | **24.93** |
| mixed32k | 135.73 | **24.53** | 136.25 | **28.61** | 139.33 | **37.76** |
| outlier64k | 124.48 | **13.01** | 124.99 | **15.09** | 125.68 | **19.25** |
| bucket64k | 412.27 | **33.53** | 416.65 | **39.41** | 419.81 | **52.62** |
| bucket128k | 1369.76 | **49.45** | 1408.04 | **58.71** | 1415.66 | **78.19** |

**cp=4**

| Workload | W=512 AG | W=512 a2a | W=1024 AG | W=1024 a2a | W=2048 AG | W=2048 a2a |
|---|---:|---:|---:|---:|---:|---:|
| rl16k | 17.46 | **6.25** | 17.74 | **7.00** | 18.09 | **8.45** |
| bucket32k | 50.81 | **9.63** | 51.44 | **11.02** | 52.37 | **13.68** |
| mixed32k | 69.47 | **14.27** | 70.26 | **16.40** | 71.65 | **20.47** |
| outlier64k | 61.28 | **7.61** | 61.23 | **8.69** | 61.88 | **10.80** |
| bucket64k | 196.76 | **19.32** | 197.83 | **22.25** | 201.31 | **28.24** |
| bucket128k | 547.60 | **27.87** | FAIL* | **32.39** | FAIL* | **41.71** |

**cp=8**

| Workload | W=512 AG | W=512 a2a | W=1024 AG | W=1024 a2a | W=2048 AG | W=2048 a2a |
|---|---:|---:|---:|---:|---:|---:|
| rl16k | 12.51 | **3.89** | 12.56 | **4.31** | 12.73 | **5.01** |
| bucket32k | 29.48 | **5.64** | 29.58 | **6.32** | 29.93 | **7.62** |
| mixed32k | 40.79 | **8.04** | 40.86 | **9.12** | 41.44 | **11.18** |
| outlier64k | 33.44 | **4.60** | 33.52 | **5.06** | 33.70 | **6.10** |
| bucket64k | 102.93 | **10.80** | 103.33 | **12.27** | 104.03 | **15.21** |
| bucket128k | FAIL* | **15.56** | FAIL* | **17.76** | FAIL* | **22.22** |

*bucket128k SWA + all_gather crashes with `cudaErrorIllegalInstruction` at cp≥4. See known issues below.

### Key takeaway: use a2a for SWA

all_gather gathers the full KV tensor regardless of window size — SWA only reduces compute, not communication. a2a redistributes Q heads so both communication and compute shrink with the window. The speedup ranges from **2× (rl16k)** to **28× (bucket128k)** depending on seqlen.

## Known Issues

**bucket128k SWA + all_gather at cp≥4**: crashes with `cudaErrorIllegalInstruction`. Only affects the AG path — a2a and full causal AG pass. Likely a cuDNN edge case with SWA masking on very large gathered KV tensors (131072 × 2×cp_size tokens). Workaround: use a2a (also 5–28× faster).

## Correctness Tests

```bash
# Run all CP benchmark configs through correctness checks (2 GPU)
pytest benchmark_cp.py -k "test_cp_benchmark_configs" -x -v

# Cross-backend consistency (compare p2p/all_gather/a2a outputs)
pytest benchmark_cp.py -k "test_cp_thd_cross_backend_consistency" -x -v
```
