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

bf16, 50 timed iterations after 10 warmup. Values are ms/iter (fwd+bwd).
Runs below were executed serially, one config at a time. Timings are
hardware-, backend-, and environment-specific.

### Full causal — FusedAttention — H100

| Workload | cp=2 p2p | cp=2 AG | cp=2 a2a | cp=4 p2p | cp=4 AG | cp=4 a2a | cp=8 p2p | cp=8 AG | cp=8 a2a |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rl16k | 26.15 | 30.47 | 27.72 | 16.17 | 16.99 | 14.41 | 12.73 | 11.02 | **7.79** |
| bucket32k | 101.98 | 120.56 | 104.28 | 55.23 | 60.66 | 53.36 | 32.52 | 33.02 | **27.18** |
| mixed32k | 137.62 | 162.24 | 140.76 | 75.12 | 81.97 | 72.16 | 44.59 | 45.39 | **36.82** |
| outlier64k | 128.53 | 152.95 | 130.36 | 67.87 | 74.73 | 66.74 | 37.97 | 39.14 | **33.75** |
| bucket64k | 428.08 | 512.41 | 436.26 | 224.03 | 247.81 | 219.89 | 120.60 | 127.15 | **111.34** |
| bucket128k | 1232.55 | 1717.97 | 1279.64 | 631.15 | 713.87 | 640.01 | 330.05 | 356.20 | **322.68** |

### Full causal — FusedAttention — B200

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

### Scaling efficiency — FusedAttention — H100

Ideal would be 4× for cp=2 → cp=8.

| Workload | p2p scale | AG scale | a2a scale |
|---|---:|---:|---:|
| rl16k | 2.05× | 2.76× | **3.56×** |
| bucket32k | 3.14× | 3.65× | **3.84×** |
| mixed32k | 3.09× | 3.57× | **3.82×** |
| outlier64k | 3.39× | **3.91×** | 3.86× |
| bucket64k | 3.55× | **4.03×** | 3.92× |
| bucket128k | 3.73× | **4.82×** | 3.97× |

### Scaling efficiency — FusedAttention — B200

| Workload | p2p scale | AG scale | a2a scale |
|---|---:|---:|---:|
| rl16k | 1.65× | 2.33× | **3.30×** |
| bucket32k | 2.99× | 3.12× | **3.91×** |
| mixed32k | 3.01× | 3.07× | **3.85×** |
| outlier64k | 3.31× | 3.32× | **3.91×** |
| bucket64k | 3.70× | 3.46× | **4.02×** |
| bucket128k | 3.96× | 3.62× | **4.09×** |

### Full causal — FlashAttention — H100

| Workload | cp=2 p2p | cp=2 AG | cp=2 a2a | cp=4 p2p | cp=4 AG | cp=4 a2a | cp=8 p2p | cp=8 AG | cp=8 a2a |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rl16k | 23.06 | 24.14 | 24.27 | 14.55 | 14.96 | 12.53 | 11.74 | 11.02 | **6.74** |
| bucket32k | 87.56 | 88.89 | 89.34 | 48.49 | 49.30 | 45.54 | 29.41 | 29.24 | **23.42** |
| mixed32k | 118.71 | 120.55 | 121.01 | 66.36 | 67.36 | 62.30 | 40.57 | 40.74 | **31.90** |
| outlier64k | 109.96 | 110.87 | 110.66 | 59.29 | 59.27 | 56.43 | 33.95 | 33.51 | **28.53** |
| bucket64k | 367.18 | 368.55 | 368.57 | 194.66 | 193.22 | 186.38 | 106.79 | 106.89 | **94.95** |
| bucket128k | 1057.24 | 1054.31 | 1052.07 | 545.23 | 540.03 | 528.22 | 289.27 | 284.24 | **267.81** |

### SWA — FlashAttention — H100

p2p does not support SWA, so only all_gather and a2a are shown.

**cp=2**

| Workload | W=512 AG | W=512 a2a | W=1024 AG | W=1024 a2a | W=2048 AG | W=2048 a2a |
|---|---:|---:|---:|---:|---:|---:|
| rl16k | **9.57** | 10.17 | 11.02 | 11.33 | 13.89 | 13.59 |
| bucket32k | **14.79** | 16.67 | 16.96 | 18.26 | 21.65 | 22.70 |
| mixed32k | **22.29** | 24.77 | 25.65 | 27.41 | 33.05 | 34.21 |
| outlier64k | **12.32** | 12.98 | 14.11 | 14.37 | 17.60 | 17.56 |
| bucket64k | **31.36** | 33.78 | 35.90 | 37.39 | 45.67 | 47.54 |
| bucket128k | **46.60** | 49.92 | 52.84 | 55.64 | 67.18 | 71.10 |

**cp=4**

| Workload | W=512 AG | W=512 a2a | W=1024 AG | W=1024 a2a | W=2048 AG | W=2048 a2a |
|---|---:|---:|---:|---:|---:|---:|
| rl16k | 7.65 | **5.99** | 8.42 | 6.56 | 10.05 | 7.68 |
| bucket32k | 11.69 | **9.59** | 12.85 | 10.51 | 15.20 | 12.52 |
| mixed32k | 17.51 | **14.33** | 19.30 | 15.64 | 22.94 | 18.76 |
| outlier64k | 9.54 | **7.53** | 10.51 | 8.23 | 12.32 | 9.84 |
| bucket64k | 23.60 | **19.39** | 25.93 | 21.23 | 30.74 | 25.80 |
| bucket128k | 35.19 | **28.32** | 38.77 | 30.89 | 45.71 | 38.23 |

**cp=8**

| Workload | W=512 AG | W=512 a2a | W=1024 AG | W=1024 a2a | W=2048 AG | W=2048 a2a |
|---|---:|---:|---:|---:|---:|---:|
| rl16k | 6.78 | **3.60** | 7.21 | 3.91 | 8.07 | 4.48 |
| bucket32k | 10.20 | **5.52** | 10.83 | 5.96 | 12.08 | 6.98 |
| mixed32k | 15.26 | **8.05** | 16.20 | 8.76 | 18.11 | 10.34 |
| outlier64k | 8.31 | **4.44** | 8.90 | 4.76 | 9.86 | 5.57 |
| bucket64k | 20.28 | **10.90** | 21.56 | 11.82 | 24.08 | 14.09 |
| bucket128k | 29.70 | **15.73** | 31.51 | 17.13 | 35.14 | 20.56 |

### SWA — FusedAttention — H100

p2p does not support SWA, so only all_gather and a2a are shown.

**cp=2**

| Workload | W=512 AG | W=512 a2a | W=1024 AG | W=1024 a2a | W=2048 AG | W=2048 a2a |
|---|---:|---:|---:|---:|---:|---:|
| rl16k | 26.76 | **10.41** | 27.22 | 11.92 | 28.07 | 14.77 |
| bucket32k | 99.17 | **16.52** | 100.05 | 19.13 | 101.43 | 24.78 |
| mixed32k | 133.84 | **24.50** | 134.64 | 28.55 | 137.54 | 37.78 |
| outlier64k | 123.14 | **12.92** | 123.52 | 14.93 | 124.70 | 19.21 |
| bucket64k | 412.00 | **33.58** | 412.96 | 39.37 | 416.78 | 52.74 |
| bucket128k | 1405.52 | **49.36** | 1414.60 | 58.67 | 1411.75 | 78.22 |

**cp=4**

| Workload | W=512 AG | W=512 a2a | W=1024 AG | W=1024 a2a | W=2048 AG | W=2048 a2a |
|---|---:|---:|---:|---:|---:|---:|
| rl16k | 15.41 | **6.13** | 15.58 | 6.88 | 16.03 | 8.27 |
| bucket32k | 49.45 | **9.53** | 49.63 | 10.90 | 50.31 | 13.48 |
| mixed32k | 67.53 | **14.20** | 68.14 | 16.18 | 69.26 | 20.21 |
| outlier64k | 59.47 | **7.50** | 59.88 | 8.48 | 60.18 | 10.54 |
| bucket64k | 195.60 | **19.22** | 196.20 | 22.05 | 197.80 | 27.92 |
| bucket128k | 544.28 | **27.92** | 545.33 | 32.14 | 547.58 | 41.46 |

**cp=8**

| Workload | W=512 AG | W=512 a2a | W=1024 AG | W=1024 a2a | W=2048 AG | W=2048 a2a |
|---|---:|---:|---:|---:|---:|---:|
| rl16k | 10.52 | **3.77** | 10.68 | 4.07 | 10.84 | 4.80 |
| bucket32k | 27.95 | **5.49** | 28.19 | 6.16 | 28.50 | 7.47 |
| mixed32k | 38.85 | **8.05** | 39.09 | 9.08 | 39.68 | 11.08 |
| outlier64k | 31.92 | **4.46** | 32.01 | 4.92 | 32.30 | 5.93 |
| bucket64k | 101.71 | **10.82** | 102.19 | 12.23 | 102.98 | 15.10 |
| bucket128k | 272.90 | **15.59** | 271.73 | 17.74 | 274.35 | 22.00 |

### SWA — FusedAttention — B200

p2p does not support SWA, so only all_gather and a2a are shown.

**cp=2**

| Workload | W=512 AG | W=512 a2a | W=1024 AG | W=1024 a2a | W=2048 AG | W=2048 a2a |
|---|---:|---:|---:|---:|---:|---:|
| rl16k | **5.89** | 7.18 | 6.68 | 7.93 | 8.43 | 9.37 |
| bucket32k | **8.98** | 11.23 | 10.30 | 12.54 | 13.36 | 15.19 |
| mixed32k | **13.32** | 15.85 | 15.41 | 17.88 | 20.14 | 22.15 |
| outlier64k | **7.11** | 8.80 | 8.14 | 9.89 | 10.45 | 11.93 |
| bucket64k | **17.91** | 21.53 | 20.80 | 24.42 | 27.78 | 30.99 |
| bucket128k | **26.29** | 31.71 | 31.16 | 36.38 | 41.46 | 46.31 |

**cp=4**

| Workload | W=512 AG | W=512 a2a | W=1024 AG | W=1024 a2a | W=2048 AG | W=2048 a2a |
|---|---:|---:|---:|---:|---:|---:|
| rl16k | 4.72 | **4.08** | 5.12 | 4.45 | 5.93 | 5.16 |
| bucket32k | 7.06 | **6.14** | 7.76 | 6.76 | 9.10 | 8.07 |
| mixed32k | 10.47 | **8.89** | 11.54 | 9.98 | 13.58 | 11.92 |
| outlier64k | 5.69 | **4.91** | 6.22 | 5.46 | 7.17 | 6.36 |
| bucket64k | 14.02 | **11.92** | 15.39 | 13.35 | 18.24 | 16.17 |
| bucket128k | 20.35 | **16.95** | 22.38 | 19.11 | 26.87 | 23.71 |

**cp=8**

| Workload | W=512 AG | W=512 a2a | W=1024 AG | W=1024 a2a | W=2048 AG | W=2048 a2a |
|---|---:|---:|---:|---:|---:|---:|
| rl16k | 4.20 | **2.72** | 4.50 | 2.87 | 4.81 | 3.13 |
| bucket32k | 6.14 | **3.61** | 6.55 | 3.93 | 7.20 | 4.60 |
| mixed32k | 8.98 | **5.05** | 9.53 | 5.59 | 10.64 | 6.50 |
| outlier64k | 4.98 | **3.00** | 5.25 | 3.26 | 5.72 | 3.74 |
| bucket64k | 12.03 | **6.62** | 12.74 | 7.32 | 14.20 | 8.69 |
| bucket128k | 17.28 | **9.31** | 18.33 | 10.28 | 20.43 | 12.41 |


### SWA communication trends

SWA reduces the attention compute window, but the communication tradeoff is backend
and CP-size dependent. In these runs, all_gather is often competitive or faster at
cp=2 for FlashAttention H100 and FusedAttention B200, while a2a is consistently
faster at cp=4 and cp=8. FusedAttention H100 is the strongest a2a case: the
measured AG/a2a speedup ranges from 1.9x to 28.5x,
with the largest gaps on long-sequence bucket128k all_gather rows.

## Correctness Tests

```bash
# Run all CP benchmark configs through correctness checks (2 GPU)
pytest benchmark_cp.py -k "test_cp_benchmark_configs" -x -v

# Cross-backend consistency (compare p2p/all_gather/a2a outputs)
pytest benchmark_cp.py -k "test_cp_thd_cross_backend_consistency" -x -v
```
