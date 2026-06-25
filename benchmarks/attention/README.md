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

bf16, FusedAttention, 50 timed iterations after 10 warmup. Values are ms/iter
(fwd+bwd). Runs below were executed serially, one config at a time.

### Full causal — H100

| Workload | cp=2 p2p | cp=2 AG | cp=2 a2a | cp=4 p2p | cp=4 AG | cp=4 a2a | cp=8 p2p | cp=8 AG | cp=8 a2a |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rl16k | 26.15 | 30.47 | 27.72 | 16.17 | 16.99 | 14.41 | 12.73 | 11.02 | **7.79** |
| bucket32k | 101.98 | 120.56 | 104.28 | 55.23 | 60.66 | 53.36 | 32.52 | 33.02 | **27.18** |
| mixed32k | 137.62 | 162.24 | 140.76 | 75.12 | 81.97 | 72.16 | 44.59 | 45.39 | **36.82** |
| outlier64k | 128.53 | 152.95 | 130.36 | 67.87 | 74.73 | 66.74 | 37.97 | 39.14 | **33.75** |
| bucket64k | 428.08 | 512.41 | 436.26 | 224.03 | 247.81 | 219.89 | 120.60 | 127.15 | **111.34** |
| bucket128k | 1232.55 | 1717.97 | 1279.64 | 631.15 | 713.87 | 640.01 | 330.05 | 356.20 | **322.68** |

### Full causal — B200

| Workload | cp=2 p2p | cp=2 AG | cp=2 a2a | cp=4 p2p | cp=4 AG | cp=4 a2a | cp=8 p2p | cp=8 AG | cp=8 a2a |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rl16k | 14.69 | 14.41 | 15.80 | 9.38 | 8.76 | 8.37 | 8.88 | 6.18 | **4.79** |
| bucket32k | 55.58 | 53.36 | 56.00 | 30.67 | 29.85 | 28.32 | 18.57 | 17.10 | **14.33** |
| mixed32k | 75.28 | 72.48 | 75.01 | 41.84 | 40.45 | 38.28 | 24.97 | 23.62 | **19.49** |
| outlier64k | 69.63 | 66.45 | 68.65 | 37.40 | 36.08 | 34.71 | 21.05 | 20.04 | **17.56** |
| bucket64k | 246.87 | 223.89 | 226.78 | 122.98 | 120.00 | 113.55 | 66.64 | 64.65 | **56.36** |
| bucket128k | 713.40 | 639.87 | 645.05 | 344.90 | 339.30 | 321.85 | 180.12 | 176.64 | **157.90** |

**Bold = fastest.** a2a at cp=8 is the fastest point for every workload on
both systems. The absolute timings are hardware- and environment-specific.

### Scaling efficiency — H100

Ideal would be 4× for cp=2 → cp=8.

| Workload | p2p scale | AG scale | a2a scale |
|---|---:|---:|---:|
| rl16k | 2.05× | 2.76× | **3.56×** |
| bucket32k | 3.14× | 3.65× | **3.84×** |
| mixed32k | 3.09× | 3.57× | **3.82×** |
| outlier64k | 3.39× | **3.91×** | 3.86× |
| bucket64k | 3.55× | **4.03×** | 3.92× |
| bucket128k | 3.73× | **4.82×** | 3.97× |

### Scaling efficiency — B200

| Workload | p2p scale | AG scale | a2a scale |
|---|---:|---:|---:|
| rl16k | 1.65× | 2.33× | **3.30×** |
| bucket32k | 2.99× | 3.12× | **3.91×** |
| mixed32k | 3.01× | 3.07× | **3.85×** |
| outlier64k | 3.31× | 3.32× | **3.91×** |
| bucket64k | 3.70× | 3.46× | **4.02×** |
| bucket128k | 3.96× | 3.62× | **4.09×** |

### SWA — training workloads (all_gather vs a2a)

**cp=2**

| Workload | W=512 AG | W=512 a2a | W=1024 AG | W=1024 a2a | W=2048 AG | W=2048 a2a |
|---|---:|---:|---:|---:|---:|---:|
| rl16k | 22.76 | **8.93** | 23.09 | **10.18** | 23.83 | **12.46** |
| bucket32k | 39.43 | **9.33** | 39.57 | **10.73** | 40.22 | **13.39** |
| mixed32k | 60.31 | **15.44** | 60.97 | **17.81** | 62.55 | **22.45** |
| outlier64k | 121.29 | **14.98** | 121.88 | **17.36** | 123.41 | **22.11** |
| bucket64k | 121.32 | **14.96** | 121.88 | **17.34** | 123.35 | **22.11** |
| bucket128k | 253.68 | **19.71** | 254.55 | **23.05** | 256.89 | **30.42** |

**cp=4**

| Workload | W=512 AG | W=512 a2a | W=1024 AG | W=1024 a2a | W=2048 AG | W=2048 a2a |
|---|---:|---:|---:|---:|---:|---:|
| rl16k | 14.29 | **5.38** | 14.59 | **6.01** | 15.07 | **7.15** |
| bucket32k | 21.52 | **5.56** | 21.56 | **6.23** | 21.96 | **7.51** |
| mixed32k | 33.16 | **8.91** | 33.60 | **10.11** | 34.24 | **12.38** |
| outlier64k | 60.28 | **8.59** | 60.84 | **9.83** | 61.41 | **12.21** |
| bucket64k | 60.35 | **8.57** | 60.65 | **9.83** | 61.37 | **12.16** |
| bucket128k | 121.71 | **11.47** | 122.28 | **13.17** | 123.56 | **16.53** |

**cp=8**

| Workload | W=512 AG | W=512 a2a | W=1024 AG | W=1024 a2a | W=2048 AG | W=2048 a2a |
|---|---:|---:|---:|---:|---:|---:|
| rl16k | 10.67 | **3.51** | 10.77 | **3.91** | 10.91 | **4.38** |
| bucket32k | 13.80 | **3.71** | 13.99 | **4.00** | 14.17 | **4.65** |
| mixed32k | 21.17 | **5.25** | 21.29 | **5.83** | 21.64 | **6.95** |
| outlier64k | 33.36 | **5.14** | 33.87 | **5.69** | 33.89 | **6.86** |
| bucket64k | 33.44 | **5.06** | 33.55 | **5.73** | 33.90 | **6.83** |
| bucket128k | 64.09 | **6.61** | 64.40 | **7.39** | 64.65 | **9.11** |

### Key takeaway: use a2a for SWA

all_gather gathers the full KV tensor regardless of window size — SWA only reduces compute, not communication. a2a redistributes Q heads so both communication and compute shrink with the window. The AG-vs-a2a speedup ranges from **~2× (rl16k)** to **~13× (bucket128k W=512)** depending on seqlen and window size.

### a2a vs all_gather speedup with SWA (AG/a2a ratio)

| Workload | cp=2 W=512 | cp=2 W=1024 | cp=2 W=2048 | cp=4 W=512 | cp=4 W=1024 | cp=4 W=2048 | cp=8 W=512 | cp=8 W=1024 | cp=8 W=2048 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rl16k | 2.5× | 2.3× | 1.9× | 2.7× | 2.4× | 2.1× | 3.0× | 2.8× | 2.5× |
| bucket32k | 4.2× | 3.7× | 3.0× | 3.9× | 3.5× | 2.9× | 3.7× | 3.5× | 3.0× |
| mixed32k | 3.9× | 3.4× | 2.8× | 3.7× | 3.3× | 2.8× | 4.0× | 3.7× | 3.1× |
| outlier64k | 8.1× | 7.0× | 5.6× | 7.0× | 6.2× | 5.0× | 6.5× | 6.0× | 4.9× |
| bucket64k | 8.1× | 7.0× | 5.6× | 7.0× | 6.2× | 5.0× | 6.6× | 5.9× | 5.0× |
| bucket128k | 12.9× | 11.0× | 8.4× | 10.6× | 9.3× | 7.5× | 9.7× | 8.7× | 7.1× |

## Known Issues

**SWA + all_gather rare `cudaErrorIllegalInstruction`**: a small number of SWA AG runs at cp=2 with 4-wide parallel-batch execution crashed intermittently. The same configs pass cleanly when run alone or with cp≥4. The crash signature matches an earlier stream-race fix (`cp_stream.wait_stream(...)` after the THD reorder, commit `611d876e`), suggesting another asynchronous race only exposed under heavy concurrent driver load. Workaround: use a2a (always faster anyway), or run cp=2 SWA AG configs serially.

## Correctness Tests

```bash
# Run all CP benchmark configs through correctness checks (2 GPU)
pytest benchmark_cp.py -k "test_cp_benchmark_configs" -x -v

# Cross-backend consistency (compare p2p/all_gather/a2a outputs)
pytest benchmark_cp.py -k "test_cp_thd_cross_backend_consistency" -x -v
```
