# SDPA backward reproducers

Requires a CUDA GPU, PyTorch and a matching Transformer Engine build with cuDNN. FA3 references additionally require FlashAttention 3. The split-shape Flash path uses version-sensitive TE private wrappers; the reported BF16 measurements used TE `5d4673da00226be3772cdb135db4cdd0e7b8edeb`, cuDNN 9.27 and frontend 1.30.0. The FP8 memory results used TE `ffcb3c0013dbdead74a144c5aef890f62be86d2a` and cuDNN 9.25.

- `repro_sdpa_shape_proportions.py`: single-GPU THD split geometry. CUDA-event times include complete calls; per-iteration NVTX ranges support primary-kernel profiling. Its printed causal-pair reference ignores the optional SWA bound.
- `cudnn_swa_bwd_repro.py`: historical single-GPU BSHD full/windowed causal comparison; FA3 is required.
- `repro_fp8_thd_bwd_noncp.py`: native FP8/BF16 backward memory, with THD/BSHD controls. Zero inputs measure allocation behavior, not accuracy.

Example causal geometry:

```bash
python repro_sdpa_shape_proportions.py --backend fused --batch 1 \
  --q-seqlen 131072 --kv-seqlens 131072 262144 393216 524288 \
  --cp-geometry 2 --warmup 10 --iters 7
```

Use `--backend fa3` for comparison. `--window-left 1024` selects SWA; this added option has only syntax validation, not a GPU rerun.

Example memory discriminator:

```bash
python repro_fp8_thd_bwd_noncp.py --mode fp8 --qkv-format thd \
  --num-sequences 32 --sequence-length 2048
```

Use `--mode bf16` or `--qkv-format bshd` for controls. Large stock FP8 cases can request workspace exceeding GPU memory, as described in the issue.
