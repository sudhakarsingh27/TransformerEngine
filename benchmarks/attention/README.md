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

bf16, `qkv_format=thd`, training mode, 50 timed iterations after warmup. Values are ms/iter (fwd+bwd). Tables use generic H100/B200 labels only. `AG` means `all_gather`; bold marks the fastest entry in each workload row.

These FA4 tables were produced through the `FlashAttention` backend with FlashAttention 4 selected by the runtime environment. The TSVs still record `kernel_backend=FlashAttention`, so the section headings use FA4 to avoid conflating them with earlier FlashAttention runs.

### Causal - FA4 - H100

| Workload | cp=2 p2p | cp=2 AG | cp=2 a2a | cp=4 p2p | cp=4 AG | cp=4 a2a | cp=8 p2p | cp=8 AG | cp=8 a2a |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rl16k | 23.27 | 24.89 | 24.98 | 14.67 | 15.09 | 13.02 | 11.38 | 10.86 | **7.01** |
| bucket32k | 88.98 | 93.55 | 92.11 | 48.81 | 51.49 | 47.37 | 29.09 | 29.73 | **24.14** |
| mixed32k | 119.90 | 125.90 | 124.79 | 66.98 | 69.91 | 64.38 | 40.15 | 41.31 | **32.91** |
| outlier64k | 111.04 | 114.78 | 114.00 | 59.60 | 61.48 | 58.69 | 33.80 | 34.34 | **29.57** |
| bucket64k | 369.77 | 384.63 | 379.41 | 196.91 | 205.02 | 194.41 | 106.67 | 112.20 | **98.32** |
| bucket128k | 1082.18 | 1093.04 | 1082.35 | 552.78 | 568.59 | 551.29 | 289.38 | 300.60 | **277.54** |

### Causal - FA4 - B200

| Workload | cp=2 p2p | cp=2 AG | cp=2 a2a | cp=4 p2p | cp=4 AG | cp=4 a2a | cp=8 p2p | cp=8 AG | cp=8 a2a |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rl16k | 12.85 | 13.73 | 14.02 | 8.10 | 8.79 | 7.32 | 7.98 | 6.71 | **4.20** |
| bucket32k | 45.26 | 47.81 | 48.07 | 25.26 | 26.88 | 24.45 | 15.87 | 16.30 | **12.44** |
| mixed32k | 61.11 | 65.21 | 65.11 | 34.58 | 37.03 | 33.47 | 21.74 | 22.77 | **17.03** |
| outlier64k | 55.69 | 58.35 | 58.55 | 30.01 | 31.51 | 29.49 | 17.59 | 18.33 | **14.98** |
| bucket64k | 185.27 | 194.61 | 192.56 | 97.24 | 104.30 | 97.47 | 54.65 | 58.68 | **49.20** |
| bucket128k | 526.94 | 548.85 | 543.44 | 269.17 | 284.27 | 272.47 | 142.54 | 152.89 | **137.20** |

### SWA - FA4 - H100

p2p does not support SWA, so only all_gather and a2a are shown.

**cp=2**

| Workload | W=512 AG | W=512 a2a | W=1024 AG | W=1024 a2a | W=2048 AG | W=2048 a2a |
|---|---:|---:|---:|---:|---:|---:|
| rl16k | 10.17 | **10.13** | 11.46 | 11.30 | 14.15 | 13.74 |
| bucket32k | 17.22 | **16.72** | 19.23 | 18.33 | 23.21 | 22.94 |
| mixed32k | 25.43 | **24.80** | 28.46 | 27.34 | 35.04 | 34.46 |
| outlier64k | 13.95 | **12.86** | 15.61 | 14.23 | 19.06 | 17.61 |
| bucket64k | 37.66 | **33.89** | 41.89 | 37.58 | 50.78 | 48.24 |
| bucket128k | 55.37 | **49.84** | 61.80 | 55.73 | 75.09 | 71.97 |

**cp=4**

| Workload | W=512 AG | W=512 a2a | W=1024 AG | W=1024 a2a | W=2048 AG | W=2048 a2a |
|---|---:|---:|---:|---:|---:|---:|
| rl16k | 8.33 | **5.95** | 9.01 | 6.56 | 10.38 | 7.73 |
| bucket32k | 13.83 | **9.61** | 14.89 | 10.51 | 16.75 | 12.65 |
| mixed32k | 20.36 | **14.33** | 21.80 | 15.57 | 24.96 | 18.92 |
| outlier64k | 11.28 | **7.42** | 11.98 | 8.09 | 13.58 | 9.86 |
| bucket64k | 30.17 | **19.49** | 32.13 | 21.29 | 36.10 | 26.14 |
| bucket128k | 45.17 | **28.31** | 48.30 | 31.06 | 54.75 | 38.79 |

**cp=8**

| Workload | W=512 AG | W=512 a2a | W=1024 AG | W=1024 a2a | W=2048 AG | W=2048 a2a |
|---|---:|---:|---:|---:|---:|---:|
| rl16k | 7.45 | **3.62** | 7.79 | 3.88 | 8.43 | 4.49 |
| bucket32k | 12.14 | **5.51** | 12.59 | 5.97 | 13.61 | 7.01 |
| mixed32k | 17.83 | **8.04** | 18.60 | 8.78 | 20.05 | 10.39 |
| outlier64k | 10.19 | **4.37** | 10.53 | 4.73 | 11.19 | 5.56 |
| bucket64k | 27.17 | **10.91** | 28.10 | 11.83 | 30.02 | 14.16 |
| bucket128k | 40.12 | **15.77** | 41.42 | 17.17 | 44.12 | 20.66 |

### SWA - FA4 - B200

p2p does not support SWA, so only all_gather and a2a are shown.

**cp=2**

| Workload | W=512 AG | W=512 a2a | W=1024 AG | W=1024 a2a | W=2048 AG | W=2048 a2a |
|---|---:|---:|---:|---:|---:|---:|
| rl16k | **6.42** | 7.04 | 7.15 | 7.65 | 8.51 | 8.81 |
| bucket32k | **9.69** | 11.03 | 10.83 | 12.07 | 13.03 | 14.31 |
| mixed32k | **14.44** | 15.52 | 16.14 | 17.18 | 19.99 | 20.85 |
| outlier64k | **7.73** | 8.67 | 8.61 | 9.58 | 10.39 | 11.30 |
| bucket64k | **19.55** | 21.13 | 21.81 | 23.59 | 26.72 | 29.26 |
| bucket128k | **28.62** | 30.97 | 32.07 | 34.82 | 39.94 | 43.60 |

**cp=4**

| Workload | W=512 AG | W=512 a2a | W=1024 AG | W=1024 a2a | W=2048 AG | W=2048 a2a |
|---|---:|---:|---:|---:|---:|---:|
| rl16k | 5.31 | **4.05** | 5.63 | 4.32 | 6.40 | 4.91 |
| bucket32k | 7.85 | **6.01** | 8.43 | 6.52 | 9.63 | 7.69 |
| mixed32k | 11.67 | **8.74** | 12.57 | 9.63 | 14.33 | 11.40 |
| outlier64k | 6.29 | **4.90** | 6.76 | 5.28 | 7.63 | 6.09 |
| bucket64k | 15.63 | **11.80** | 16.74 | 12.99 | 19.16 | 15.50 |
| bucket128k | 22.70 | **16.76** | 24.44 | 18.59 | 27.86 | 22.69 |

**cp=8**

| Workload | W=512 AG | W=512 a2a | W=1024 AG | W=1024 a2a | W=2048 AG | W=2048 a2a |
|---|---:|---:|---:|---:|---:|---:|
| rl16k | 4.83 | **2.52** | 5.06 | 2.73 | 5.58 | 3.05 |
| bucket32k | 6.96 | **3.63** | 7.25 | 3.89 | 7.89 | 4.42 |
| mixed32k | 10.28 | **5.03** | 10.79 | 5.45 | 11.73 | 6.29 |
| outlier64k | 5.70 | **3.01** | 5.85 | 3.25 | 6.33 | 3.60 |
| bucket64k | 13.77 | **6.61** | 14.36 | 7.16 | 15.55 | 8.44 |
| bucket128k | 19.84 | **9.21** | 20.69 | 10.19 | 22.39 | 12.01 |

### Prior Reference Results

The following previously collected FusedAttention and FlashAttention tables are kept as reference baselines. The FlashAttention H100 tables are the prior FA3/FlashAttention reference. These were not collected in the same matrix as the FA4 refresh above, so use them directionally unless rerun under the same environment.

#### Full causal - FusedAttention - H100

| Workload | cp=2 p2p | cp=2 AG | cp=2 a2a | cp=4 p2p | cp=4 AG | cp=4 a2a | cp=8 p2p | cp=8 AG | cp=8 a2a |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rl16k | 26.15 | 30.47 | 27.72 | 16.17 | 16.99 | 14.41 | 12.73 | 11.02 | **7.79** |
| bucket32k | 101.98 | 120.56 | 104.28 | 55.23 | 60.66 | 53.36 | 32.52 | 33.02 | **27.18** |
| mixed32k | 137.62 | 162.24 | 140.76 | 75.12 | 81.97 | 72.16 | 44.59 | 45.39 | **36.82** |
| outlier64k | 128.53 | 152.95 | 130.36 | 67.87 | 74.73 | 66.74 | 37.97 | 39.14 | **33.75** |
| bucket64k | 428.08 | 512.41 | 436.26 | 224.03 | 247.81 | 219.89 | 120.60 | 127.15 | **111.34** |
| bucket128k | 1232.55 | 1717.97 | 1279.64 | 631.15 | 713.87 | 640.01 | 330.05 | 356.20 | **322.68** |

#### Full causal - FusedAttention - B200

| Workload | cp=2 p2p | cp=2 AG | cp=2 a2a | cp=4 p2p | cp=4 AG | cp=4 a2a | cp=8 p2p | cp=8 AG | cp=8 a2a |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rl16k | 14.69 | 14.41 | 15.80 | 9.38 | 8.76 | 8.37 | 8.88 | 6.18 | **4.79** |
| bucket32k | 55.58 | 53.36 | 56.00 | 30.67 | 29.85 | 28.32 | 18.57 | 17.10 | **14.33** |
| mixed32k | 75.28 | 72.48 | 75.01 | 41.84 | 40.45 | 38.28 | 24.97 | 23.62 | **19.49** |
| outlier64k | 69.63 | 66.45 | 68.65 | 37.40 | 36.08 | 34.71 | 21.05 | 20.04 | **17.56** |
| bucket64k | 246.87 | 223.89 | 226.78 | 122.98 | 120.00 | 113.55 | 66.64 | 64.65 | **56.36** |
| bucket128k | 713.40 | 639.87 | 645.05 | 344.90 | 339.30 | 321.85 | 180.12 | 176.64 | **157.90** |

#### Scaling efficiency - FusedAttention - H100

Ideal would be 4x for cp=2 to cp=8.

| Workload | p2p scale | AG scale | a2a scale |
|---|---:|---:|---:|
| rl16k | 2.05x | 2.76x | **3.56x** |
| bucket32k | 3.14x | 3.65x | **3.84x** |
| mixed32k | 3.09x | 3.57x | **3.82x** |
| outlier64k | 3.39x | **3.91x** | 3.86x |
| bucket64k | 3.55x | **4.03x** | 3.92x |
| bucket128k | 3.73x | **4.82x** | 3.97x |

#### Scaling efficiency - FusedAttention - B200

| Workload | p2p scale | AG scale | a2a scale |
|---|---:|---:|---:|
| rl16k | 1.65x | 2.33x | **3.30x** |
| bucket32k | 2.99x | 3.12x | **3.91x** |
| mixed32k | 3.01x | 3.07x | **3.85x** |
| outlier64k | 3.31x | 3.32x | **3.91x** |
| bucket64k | 3.70x | 3.46x | **4.02x** |
| bucket128k | 3.96x | 3.62x | **4.09x** |

#### Full causal - FlashAttention - H100

| Workload | cp=2 p2p | cp=2 AG | cp=2 a2a | cp=4 p2p | cp=4 AG | cp=4 a2a | cp=8 p2p | cp=8 AG | cp=8 a2a |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rl16k | 23.06 | 24.14 | 24.27 | 14.55 | 14.96 | 12.53 | 11.74 | 11.02 | **6.74** |
| bucket32k | 87.56 | 88.89 | 89.34 | 48.49 | 49.30 | 45.54 | 29.41 | 29.24 | **23.42** |
| mixed32k | 118.71 | 120.55 | 121.01 | 66.36 | 67.36 | 62.30 | 40.57 | 40.74 | **31.90** |
| outlier64k | 109.96 | 110.87 | 110.66 | 59.29 | 59.27 | 56.43 | 33.95 | 33.51 | **28.53** |
| bucket64k | 367.18 | 368.55 | 368.57 | 194.66 | 193.22 | 186.38 | 106.79 | 106.89 | **94.95** |
| bucket128k | 1057.24 | 1054.31 | 1052.07 | 545.23 | 540.03 | 528.22 | 289.27 | 284.24 | **267.81** |

#### SWA - FlashAttention - H100

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

#### SWA - FusedAttention - H100

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

#### SWA - FusedAttention - B200

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

#### Prior SWA communication trends

SWA reduces the attention compute window, but the communication tradeoff is backend
and CP-size dependent. In these prior runs, all_gather is often competitive or
faster at cp=2 for FlashAttention H100 and FusedAttention B200, while a2a is
consistently faster at cp=4 and cp=8. FusedAttention H100 is the strongest a2a
case: the measured AG/a2a speedup ranges from 1.9x to 28.5x, with the largest
gaps on long-sequence bucket128k all_gather rows.

### FA4 Trends

H100 causal FA4 was close to the prior H100 FlashAttention table: -3.1% to 6.1% row deltas, with a 3.0% median delta. Treat this as a directional comparison because the runs were not collected in the same matrix.

For H100 SWA, FA4 a2a stayed essentially flat versus the prior H100 FlashAttention SWA table (-1.7% to 1.5% deltas, 0.2% median), while all_gather was slower (1.9% to 35.1% deltas, 14.5% median).

### Uniform THD Results

Uniform THD runs use `thd_seqlen_pattern=max`. Uniform SWA uses the `cp_thd_swa_*` configs below; p2p is unsupported for SWA. `fail` means the supported config failed after retry, so no timing is reported.

### Uniform THD - FA4 - H100

| Config | cp=2 p2p | cp=2 AG | cp=2 a2a | cp=4 p2p | cp=4 AG | cp=4 a2a | cp=8 p2p | cp=8 AG | cp=8 a2a |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bench_8k | 5.03 | 5.62 | 5.24 | 4.51 | 4.09 | 3.22 | 5.94 | 3.52 | **2.55** |
| bench_16k | 8.05 | 8.51 | 8.15 | 5.39 | 5.41 | 4.51 | 7.30 | 4.00 | **3.31** |
| bench_32k | 28.09 | 29.23 | 28.83 | 15.70 | 16.10 | 14.68 | 10.43 | 9.33 | **7.60** |
| cp_thd_0 | 7.74 | 10.62 | 8.47 | 6.07 | 8.49 | **4.91** | 7.37 | 7.52 | - |
| cp_thd_1 | 12.06 | 14.58 | 12.68 | 7.83 | 10.65 | **6.80** | 9.19 | 8.77 | - |
| cp_thd_2 | 5.74 | 8.49 | 6.41 | 5.70 | 7.51 | **3.91** | 7.41 | 7.08 | - |
| cp_thd_3 | 6.99 | 7.36 | 7.37 | 5.28 | 4.86 | - | 6.40 | **3.88** | - |

### Uniform THD SWA - FA4 - H100

| Config | cp=2 AG | cp=2 a2a | cp=4 AG | cp=4 a2a | cp=8 AG | cp=8 a2a |
|---|---:|---:|---:|---:|---:|---:|
| cp_thd_swa_0 | 7.04 | 4.92 | 6.75 | **3.20** | 6.58 | - |
| cp_thd_swa_1 | 3.90 | 4.05 | 3.51 | - | **3.29** | - |
| cp_thd_swa_2 | 6.99 | 4.89 | 6.76 | **3.14** | 6.62 | - |
| cp_thd_swa_3 | 3.90 | 3.75 | 3.47 | - | **3.34** | - |

### Uniform THD - FA4 - B200

| Config | cp=2 p2p | cp=2 AG | cp=2 a2a | cp=4 p2p | cp=4 AG | cp=4 a2a | cp=8 p2p | cp=8 AG | cp=8 a2a |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bench_8k | 3.17 | 3.69 | 3.38 | 3.64 | 3.12 | 2.33 | 5.42 | 3.22 | **2.10** |
| bench_16k | 4.50 | 5.01 | 4.69 | 4.03 | 3.65 | 2.91 | 6.17 | 3.37 | **2.33** |
| bench_32k | 14.32 | 15.05 | 14.96 | 8.19 | 8.72 | 7.63 | 7.66 | 5.72 | **4.37** |
| cp_thd_0 | 5.51 | 6.31 | 5.44 | 4.40 | 5.35 | **3.21** | 5.99 | 4.85 | - |
| cp_thd_1 | 7.39 | 8.41 | 7.35 | 4.95 | 6.48 | **4.16** | 7.67 | 5.65 | - |
| cp_thd_2 | 4.52 | 5.42 | 4.47 | 4.17 | 4.84 | **2.69** | 6.21 | 4.64 | - |
| cp_thd_3 | 4.04 | 4.61 | 4.52 | 4.07 | 3.50 | - | 5.58 | **3.35** | - |

### Uniform THD SWA - FA4 - B200

| Config | cp=2 AG | cp=2 a2a | cp=4 AG | cp=4 a2a | cp=8 AG | cp=8 a2a |
|---|---:|---:|---:|---:|---:|---:|
| cp_thd_swa_0 | 4.61 | 3.81 | 4.43 | **2.34** | 4.35 | - |
| cp_thd_swa_1 | 2.99 | 2.93 | **2.84** | - | 3.01 | - |
| cp_thd_swa_2 | 4.61 | 3.82 | 4.39 | **2.34** | 4.32 | - |
| cp_thd_swa_3 | 3.09 | 2.91 | 2.87 | - | **2.84** | - |

### Uniform THD - FusedAttention - H100

| Config | cp=2 p2p | cp=2 AG | cp=2 a2a | cp=4 p2p | cp=4 AG | cp=4 a2a | cp=8 p2p | cp=8 AG | cp=8 a2a |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bench_8k | 5.49 | 6.46 | 5.83 | 5.14 | 4.36 | 3.60 | 7.45 | 3.59 | **2.72** |
| bench_16k | 8.99 | 10.63 | 9.20 | 6.23 | 6.01 | 5.22 | 7.27 | 4.25 | **3.35** |
| bench_32k | 31.87 | 38.08 | 32.68 | 17.80 | 19.21 | 16.93 | 12.65 | 10.39 | **8.69** |
| cp_thd_0 | 8.52 | 11.96 | 9.33 | 6.74 | 9.02 | **5.43** | 8.06 | 7.71 | - |
| cp_thd_1 | 13.41 | 15.80 | 14.23 | 8.79 | 11.13 | **7.57** | 12.52 | 8.97 | - |
| cp_thd_2 | 6.22 | 9.19 | 6.86 | 5.91 | 7.72 | **4.24** | 8.16 | 7.18 | - |
| cp_thd_3 | 7.63 | 8.77 | 8.16 | 5.94 | 5.38 | - | 7.24 | **3.97** | - |

### Uniform THD SWA - FusedAttention - H100

| Config | cp=2 AG | cp=2 a2a | cp=4 AG | cp=4 a2a | cp=8 AG | cp=8 a2a |
|---|---:|---:|---:|---:|---:|---:|
| cp_thd_swa_0 | 11.63 | 5.17 | 8.75 | **3.34** | 7.61 | - |
| cp_thd_swa_1 | 8.34 | **3.99** | 5.36 | - | 4.16 | - |
| cp_thd_swa_2 | 11.64 | 5.21 | 8.80 | **3.27** | 7.64 | - |
| cp_thd_swa_3 | 8.27 | **3.96** | 5.39 | - | 4.20 | - |

### Uniform THD - FusedAttention - B200

| Config | cp=2 p2p | cp=2 AG | cp=2 a2a | cp=4 p2p | cp=4 AG | cp=4 a2a | cp=8 p2p | cp=8 AG | cp=8 a2a |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bench_8k | 3.63 | 3.76 | 3.84 | 4.05 | 3.09 | 2.69 | 6.86 | 2.88 | **2.35** |
| bench_16k | 5.24 | 5.36 | 5.63 | 4.75 | 3.77 | 3.52 | 7.91 | 3.30 | **2.64** |
| bench_32k | 17.64 | 17.04 | 17.69 | 9.81 | 9.44 | 9.00 | 8.50 | 6.09 | **5.34** |
| cp_thd_0 | 5.67 | 6.07 | 5.89 | 4.82 | 4.96 | **3.59** | 7.16 | 4.54 | - |
| cp_thd_1 | 7.74 | 8.32 | 7.70 | 5.30 | 6.09 | **4.30** | 7.16 | fail | - |
| cp_thd_2 | 4.54 | 5.05 | 4.73 | 4.32 | 4.48 | **2.89** | 6.67 | 4.20 | - |
| cp_thd_3 | 4.21 | 4.58 | 5.06 | 4.45 | 3.41 | - | 6.02 | **3.15** | - |

### Uniform THD SWA - FusedAttention - B200

| Config | cp=2 AG | cp=2 a2a | cp=4 AG | cp=4 a2a | cp=8 AG | cp=8 a2a |
|---|---:|---:|---:|---:|---:|---:|
| cp_thd_swa_0 | 4.26 | 3.80 | 4.08 | **2.38** | 4.05 | - |
| cp_thd_swa_1 | 2.86 | 2.94 | **2.71** | - | 2.79 | - |
| cp_thd_swa_2 | 4.28 | 3.80 | 4.08 | **2.46** | 4.00 | - |
| cp_thd_swa_3 | 2.80 | 2.98 | **2.55** | - | 2.76 | - |

### Uniform THD - FA3 - H100

| Config | cp=2 p2p | cp=2 AG | cp=2 a2a | cp=4 p2p | cp=4 AG | cp=4 a2a | cp=8 p2p | cp=8 AG | cp=8 a2a |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bench_8k | 4.91 | 5.62 | 5.10 | 4.84 | 4.29 | 3.22 | 7.41 | 3.78 | **2.63** |
| bench_16k | 7.85 | 8.29 | 7.94 | 6.34 | 5.45 | 4.53 | 15.60 | 4.23 | **3.15** |
| bench_32k | 27.66 | 28.06 | 27.96 | 15.58 | 15.55 | 14.20 | 17.95 | 9.29 | **7.44** |
| cp_thd_0 | 7.54 | 10.56 | 8.21 | 6.45 | 8.62 | **4.84** | 9.86 | 7.70 | - |
| cp_thd_1 | 11.98 | 14.53 | 12.61 | 8.40 | 10.55 | **6.77** | 10.04 | 8.76 | - |
| cp_thd_2 | 5.69 | 8.61 | 6.28 | 5.86 | 7.75 | **3.83** | 16.50 | 7.30 | - |
| cp_thd_3 | 6.84 | 7.30 | 7.13 | 5.62 | 5.04 | - | 14.76 | **4.40** | - |

### Uniform THD SWA - FA3 - H100

| Config | cp=2 AG | cp=2 a2a | cp=4 AG | cp=4 a2a | cp=8 AG | cp=8 a2a |
|---|---:|---:|---:|---:|---:|---:|
| cp_thd_swa_0 | 6.92 | 4.89 | 6.63 | **3.12** | 6.58 | - |
| cp_thd_swa_1 | 3.90 | 3.82 | 4.01 | - | **3.30** | - |
| cp_thd_swa_2 | 6.91 | 4.88 | 6.69 | **3.22** | 6.57 | - |
| cp_thd_swa_3 | 3.71 | 3.83 | **3.44** | - | 3.47 | - |

### Uniform THD Trends

On H100, FA3 and FA4 are nearly identical for most Uniform THD rows: median FA3/FA4 is 1.00 for non-SWA and 0.99 for SWA. The main exception is cp=8 p2p on several non-SWA rows, where FA3 is much slower than FA4.

On H100, FusedAttention is slower than FA4 on this Uniform matrix: median FusedAttention/FA4 is 1.11 for non-SWA and 1.26 for SWA. The largest SWA gaps are all_gather rows; a2a remains the fastest supported SWA mode in the FusedAttention table.

On B200, FusedAttention is close to FA4 for Uniform SWA (median FusedAttention/FA4 0.94) and modestly slower for non-SWA (median 1.07). One supported B200 FusedAttention row, `cp_thd_1 all_gather cp8`, failed twice with a CUDA illegal-instruction abort and is reported as `fail`.

## Correctness Tests

```bash
# Run all CP benchmark configs through correctness checks (2 GPU)
pytest benchmark_cp.py -k "test_cp_benchmark_configs" -x -v

# Cross-backend consistency (compare p2p/all_gather/a2a outputs)
pytest benchmark_cp.py -k "test_cp_thd_cross_backend_consistency" -x -v
```
