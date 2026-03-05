# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Context Parallel Attention Benchmark - PyTorch (DualChunkSwap / p2p)

Supports both BSHD and THD formats. Optionally runs a no-CP baseline on each
rank for comparison (--baseline). Use --verify to check CP vs no-CP correctness.

Launch:
  torchrun --nproc-per-node=N bench_cp_pytorch.py --config llama3_8b --seq_len 32768
  torchrun --nproc-per-node=N bench_cp_pytorch.py --config cp_1_0 --seq_len 4096 --baseline
  torchrun --nproc-per-node=N bench_cp_pytorch.py --config cp_1_0 --seq_len 4096 --verify
"""

import os
import argparse
import statistics

import torch
import torch.distributed as dist

from transformer_engine.pytorch import DotProductAttention
import transformer_engine_torch as tex

# ---------------------------------------------------------------------------
# Model configs (mirrors test configs but with real-world LLM shapes)
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    # name: (num_heads, num_gqa_groups, head_dim)
    "llama2_7b": (32, 32, 128),   # MHA
    "llama2_70b": (64, 8, 128),   # GQA
    "llama3_8b": (32, 8, 128),    # GQA
    # Test-size configs from test_attention_with_cp.py
    "cp_1_0": (12, 12, 128),      # MHA (matches test cp_1_0)
    "cp_2_0": (12, 2, 128),       # GQA (matches test cp_2_0)
}


def parse_args():
    parser = argparse.ArgumentParser(description="CP Attention Benchmark (PyTorch)")
    parser.add_argument("--config", type=str, default="llama3_8b",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model config name")
    parser.add_argument("--seq_len", type=int, default=8192,
                        help="Total sequence length (before CP split)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=20, help="Timed iterations")
    parser.add_argument("--qkv_format", type=str, default="bshd",
                        choices=["bshd", "thd"],
                        help="QKV format: bshd (DualChunkSwap) or thd")
    parser.add_argument("--baseline", action="store_true",
                        help="Also run a no-CP single-GPU baseline for comparison")
    parser.add_argument("--verify", action="store_true",
                        help="Verify CP output matches no-CP output (correctness check)")
    return parser.parse_args()


def setup_distributed():
    """Initialize torch.distributed and return rank, world_size, cp_group, cp_ranks."""
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        device_count = torch.cuda.device_count()
        torch.cuda.set_device(rank % device_count)

    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    cp_ranks = list(range(world_size))
    cp_group = dist.new_group(cp_ranks, backend="nccl")

    return rank, world_size, cp_group, cp_ranks


def shard_bshd(x, batch_size, seq_len, head_dim, world_size, rank):
    """DualChunkSwap sharding for BSHD: pick chunks [rank] and [2*ws - rank - 1]."""
    local_seq = seq_len // world_size
    x = x.view(batch_size, 2 * world_size, seq_len // (2 * world_size), x.shape[2], head_dim)
    seq_idx = torch.tensor([rank, 2 * world_size - rank - 1], device=x.device)
    x = x.index_select(1, seq_idx)
    return x.reshape(batch_size, local_seq, x.shape[3], head_dim).contiguous()


def unshard_bshd(x, batch_size, world_size, rank):
    """Inverse of shard_bshd: reshape local [rank, 2*ws-rank-1] chunks back for comparison."""
    x = x.view(batch_size, 2, x.shape[1] // 2, x.shape[2], x.shape[3])
    return x


def make_inputs_bshd(batch_size, seq_len, num_heads, num_gqa_groups, head_dim, world_size, rank,
                     dtype):
    """Generate per-rank Q/K/V/dout for BSHD format with DualChunkSwap sharding."""
    assert seq_len % (2 * world_size) == 0, (
        f"seq_len={seq_len} must be divisible by 2*cp_size={2*world_size}"
    )

    q_full = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device="cuda")
    k_full = torch.randn(batch_size, seq_len, num_gqa_groups, head_dim, dtype=dtype, device="cuda")
    v_full = torch.randn(batch_size, seq_len, num_gqa_groups, head_dim, dtype=dtype, device="cuda")
    dout_full = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device="cuda")

    q = shard_bshd(q_full, batch_size, seq_len, head_dim, world_size, rank).requires_grad_(True)
    k = shard_bshd(k_full, batch_size, seq_len, head_dim, world_size, rank).requires_grad_(True)
    v = shard_bshd(v_full, batch_size, seq_len, head_dim, world_size, rank).requires_grad_(True)
    dout_local = shard_bshd(dout_full, batch_size, seq_len, head_dim, world_size, rank)
    # DotProductAttention output is (B, S, H*D), reshape dout to match
    local_seq = seq_len // world_size
    dout = dout_local.reshape(batch_size, local_seq, num_heads * head_dim)

    return q, k, v, dout, None, None, None, None, q_full, k_full, v_full, dout_full


def make_inputs_thd(batch_size, seq_len, num_heads, num_gqa_groups, head_dim, world_size, rank,
                    dtype):
    """Generate per-rank Q/K/V/dout for THD format.

    Key: cu_seqlens and cu_seqlens_padded are the FULL (not per-rank) cumulative lengths.
    The CP kernel uses them to understand the global sequence structure. Only Q/K/V data
    is partitioned via thd_get_partitioned_indices.
    """
    assert seq_len % (2 * world_size) == 0, (
        f"seq_len={seq_len} must be divisible by 2*cp_size={2*world_size}"
    )

    # For uniform-length benchmarking, seqlens == seqlens_padded since seq_len is already
    # divisible by 2*world_size
    seqlens = torch.full([batch_size], seq_len, dtype=torch.int32)
    seqlens_padded = (seqlens + 2 * world_size - 1) // (world_size * 2) * (world_size * 2)
    cu_seqlens_padded = torch.cat([
        torch.zeros([1], dtype=torch.int32),
        seqlens_padded.cumsum(0, dtype=torch.int32),
    ]).cuda()
    # For FusedAttention, cu_seqlens reflects actual (non-padded) lengths
    cu_seqlens = torch.clone(cu_seqlens_padded)
    cu_seqlens[1:] = seqlens.cumsum(0, dtype=torch.int32).cuda()

    total_tokens = int(cu_seqlens_padded[-1].item())

    q_full = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device="cuda")
    k_full = torch.randn(total_tokens, num_gqa_groups, head_dim, dtype=dtype, device="cuda")
    v_full = torch.randn(total_tokens, num_gqa_groups, head_dim, dtype=dtype, device="cuda")
    dout_full = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device="cuda")

    # Partition only the data, not the cu_seqlens
    seq_idx = tex.thd_get_partitioned_indices(cu_seqlens_padded, total_tokens, world_size, rank)
    q = q_full.index_select(0, seq_idx).contiguous().requires_grad_(True)
    k = k_full.index_select(0, seq_idx).contiguous().requires_grad_(True)
    v = v_full.index_select(0, seq_idx).contiguous().requires_grad_(True)
    dout_local = dout_full.index_select(0, seq_idx).contiguous()
    # DotProductAttention output is (T, H*D), reshape dout to match
    dout = dout_local.reshape(-1, num_heads * head_dim)

    # Pass the FULL cu_seqlens to the CP kernel (not per-rank)
    return (q, k, v, dout, cu_seqlens, cu_seqlens, cu_seqlens_padded, cu_seqlens_padded,
            q_full, k_full, v_full, dout_full)


def make_inputs_no_cp_bshd(batch_size, seq_len, num_heads, num_gqa_groups, head_dim, dtype):
    """Generate Q/K/V/dout for single-GPU no-CP baseline (BSHD format)."""
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype,
                    device="cuda").requires_grad_(True)
    k = torch.randn(batch_size, seq_len, num_gqa_groups, head_dim, dtype=dtype,
                    device="cuda").requires_grad_(True)
    v = torch.randn(batch_size, seq_len, num_gqa_groups, head_dim, dtype=dtype,
                    device="cuda").requires_grad_(True)
    # DotProductAttention output is (B, S, H*D)
    dout = torch.randn(batch_size, seq_len, num_heads * head_dim, dtype=dtype, device="cuda")
    return q, k, v, dout, None, None, None, None


def make_inputs_no_cp_thd(batch_size, seq_len, num_heads, num_gqa_groups, head_dim, dtype):
    """Generate Q/K/V/dout for single-GPU no-CP baseline (THD format)."""
    total_tokens = batch_size * seq_len
    seqlens = torch.full([batch_size], seq_len, dtype=torch.int32)
    cu_seqlens = torch.cat([
        torch.zeros([1], dtype=torch.int32),
        seqlens.cumsum(0, dtype=torch.int32),
    ]).cuda()

    q = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype,
                    device="cuda").requires_grad_(True)
    k = torch.randn(total_tokens, num_gqa_groups, head_dim, dtype=dtype,
                    device="cuda").requires_grad_(True)
    v = torch.randn(total_tokens, num_gqa_groups, head_dim, dtype=dtype,
                    device="cuda").requires_grad_(True)
    # DotProductAttention output is (T, H*D)
    dout = torch.randn(total_tokens, num_heads * head_dim, dtype=dtype, device="cuda")
    return q, k, v, dout, cu_seqlens, cu_seqlens, cu_seqlens, cu_seqlens


def compute_flops(batch_size, seq_len, num_heads, head_dim):
    """Compute causal fwd+bwd FLOPs."""
    fwd_flops = 4 * batch_size * num_heads * seq_len * seq_len * head_dim
    fwd_bwd_flops = 3.5 * fwd_flops
    causal_flops = 0.5 * fwd_bwd_flops
    return causal_flops


def benchmark_attn(core_attn, q, k, v, dout, cu_sq, cu_skv, cu_sq_pad, cu_skv_pad,
                   warmup, iters):
    """Run warmup + timed iterations, return list of times in ms."""
    for _ in range(warmup):
        if q.grad is not None:
            q.grad = None
            k.grad = None
            v.grad = None
        out = core_attn(
            q, k, v,
            cu_seqlens_q=cu_sq,
            cu_seqlens_kv=cu_skv,
            cu_seqlens_q_padded=cu_sq_pad,
            cu_seqlens_kv_padded=cu_skv_pad,
        )
        out.backward(dout)
        torch.cuda.synchronize()

    times_ms = []
    for _ in range(iters):
        if q.grad is not None:
            q.grad = None
            k.grad = None
            v.grad = None

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        out = core_attn(
            q, k, v,
            cu_seqlens_q=cu_sq,
            cu_seqlens_kv=cu_skv,
            cu_seqlens_q_padded=cu_sq_pad,
            cu_seqlens_kv_padded=cu_skv_pad,
        )
        out.backward(dout)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    return times_ms


def verify_cp_correctness(rank, world_size, cp_group, cp_ranks,
                          num_heads, num_gqa_groups, head_dim,
                          batch_size, seq_len, qkv_format, dtype):
    """Run no-CP on rank 0 with full data, then CP with same data, compare outputs."""
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Verification: CP vs no-CP correctness ({qkv_format})")
        print(f"{'='*60}")

    attn_mask_type = "padding_causal" if qkv_format == "thd" else "causal"

    # --- Generate shared inputs (same seed on all ranks) ---
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    if qkv_format == "bshd":
        (q_cp, k_cp, v_cp, dout_cp, cu_sq, cu_skv, cu_sq_pad, cu_skv_pad,
         q_full, k_full, v_full, dout_full) = make_inputs_bshd(
            batch_size, seq_len, num_heads, num_gqa_groups, head_dim,
            world_size, rank, dtype,
        )
    else:
        (q_cp, k_cp, v_cp, dout_cp, cu_sq, cu_skv, cu_sq_pad, cu_skv_pad,
         q_full, k_full, v_full, dout_full) = make_inputs_thd(
            batch_size, seq_len, num_heads, num_gqa_groups, head_dim,
            world_size, rank, dtype,
        )

    # --- No-CP reference on each rank (full sequence) ---
    nocp_attn = DotProductAttention(
        num_heads, head_dim,
        num_gqa_groups=num_gqa_groups,
        attention_dropout=0.0,
        qkv_format=qkv_format,
        attn_mask_type=attn_mask_type,
    ).cuda()

    q_ref = q_full.clone().requires_grad_(True)
    k_ref = k_full.clone().requires_grad_(True)
    v_ref = v_full.clone().requires_grad_(True)

    if qkv_format == "thd":
        seqlens = torch.full([batch_size], seq_len, dtype=torch.int32)
        cu_seqlens_full = torch.cat([
            torch.zeros([1], dtype=torch.int32),
            seqlens.cumsum(0, dtype=torch.int32),
        ]).cuda()
        ref_cu = cu_seqlens_full
    else:
        ref_cu = None

    # DotProductAttention outputs (B, S, H*D) for bshd or (T, H*D) for thd
    # Reshape dout to match for backward
    hidden = num_heads * head_dim
    if qkv_format == "bshd":
        dout_ref = dout_full.reshape(batch_size, seq_len, hidden)
        dout_cp_reshaped = dout_cp.reshape(batch_size, -1, hidden)
    else:
        dout_ref = dout_full.reshape(-1, hidden)
        dout_cp_reshaped = dout_cp.reshape(-1, hidden)

    out_ref = nocp_attn(
        q_ref, k_ref, v_ref,
        cu_seqlens_q=ref_cu, cu_seqlens_kv=ref_cu,
        cu_seqlens_q_padded=ref_cu, cu_seqlens_kv_padded=ref_cu,
    )
    out_ref.backward(dout_ref)
    torch.cuda.synchronize()

    # --- CP run ---
    cp_attn = DotProductAttention(
        num_heads, head_dim,
        num_gqa_groups=num_gqa_groups,
        attention_dropout=0.0,
        qkv_format=qkv_format,
        attn_mask_type=attn_mask_type,
    ).cuda()
    cp_attn.set_context_parallel_group(cp_group, cp_ranks, torch.cuda.Stream(), "p2p")

    out_cp = cp_attn(
        q_cp, k_cp, v_cp,
        cu_seqlens_q=cu_sq, cu_seqlens_kv=cu_skv,
        cu_seqlens_q_padded=cu_sq_pad, cu_seqlens_kv_padded=cu_skv_pad,
    )
    out_cp.backward(dout_cp_reshaped)
    torch.cuda.synchronize()

    # --- Compare: extract this rank's slice from the reference ---
    # out_ref shape: (B, S, H*D) for bshd, (T, H*D) for thd
    local_seq = seq_len // world_size
    if qkv_format == "bshd":
        out_ref_chunks = out_ref.view(
            batch_size, 2 * world_size, seq_len // (2 * world_size), hidden,
        )
        seq_idx = torch.tensor([rank, 2 * world_size - rank - 1], device=out_ref.device)
        out_ref_local = out_ref_chunks.index_select(1, seq_idx).reshape(
            batch_size, local_seq, hidden,
        )

        dq_ref_chunks = q_ref.grad.view(
            batch_size, 2 * world_size, seq_len // (2 * world_size),
            q_ref.grad.shape[2], q_ref.grad.shape[3],
        )
        dq_ref_local = dq_ref_chunks.index_select(1, seq_idx).reshape(
            batch_size, local_seq, q_ref.grad.shape[2], q_ref.grad.shape[3],
        )
    else:
        # THD: use same partitioning indices
        seqlens_full = torch.full([batch_size], seq_len, dtype=torch.int32)
        seqlens_padded = (seqlens_full + 2 * world_size - 1) // (world_size * 2) * (world_size * 2)
        cu_padded = torch.cat([
            torch.zeros([1], dtype=torch.int32),
            seqlens_padded.cumsum(0, dtype=torch.int32),
        ]).cuda()
        total_tokens = int(cu_padded[-1].item())
        thd_idx = tex.thd_get_partitioned_indices(cu_padded, total_tokens, world_size, rank)
        out_ref_local = out_ref.index_select(0, thd_idx)
        dq_ref_local = q_ref.grad.index_select(0, thd_idx)

    # Compare forward output (both same shape now)
    out_cp_flat = out_cp.detach()
    out_ref_flat = out_ref_local.detach()

    fwd_max_diff = (out_cp_flat - out_ref_flat).abs().max().item()
    fwd_mean_diff = (out_cp_flat - out_ref_flat).abs().mean().item()

    # Compare backward (dQ)
    dq_cp = q_cp.grad.detach()
    dq_ref_l = dq_ref_local.detach()
    bwd_max_diff = (dq_cp - dq_ref_l).abs().max().item()
    bwd_mean_diff = (dq_cp - dq_ref_l).abs().mean().item()

    # bf16 tolerance
    atol = 2.5e-2

    if rank == 0:
        print(f"  Forward  - max_diff: {fwd_max_diff:.6f}, mean_diff: {fwd_mean_diff:.6f}")
        print(f"  Backward (dQ) - max_diff: {bwd_max_diff:.6f}, mean_diff: {bwd_mean_diff:.6f}")
        print(f"  Tolerance (atol): {atol}")
        fwd_pass = fwd_max_diff < atol
        bwd_pass = bwd_max_diff < atol
        print(f"  Forward:  {'PASS' if fwd_pass else 'FAIL'}")
        print(f"  Backward: {'PASS' if bwd_pass else 'FAIL'}")
        if fwd_pass and bwd_pass:
            print("  >> VERIFICATION PASSED <<")
        else:
            print("  >> VERIFICATION FAILED <<")
        print(f"{'='*60}")

    # Cleanup
    del nocp_attn, cp_attn, q_ref, k_ref, v_ref, q_full, k_full, v_full
    del q_cp, k_cp, v_cp, dout_cp, dout_full
    torch.cuda.empty_cache()


def report_results(label, times_ms, batch_size, seq_len, num_heads, head_dim):
    """Compute and return a dict of metrics."""
    median_ms = statistics.median(times_ms)
    mean_ms = statistics.mean(times_ms)
    std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    flops = compute_flops(batch_size, seq_len, num_heads, head_dim)
    tflops = flops / (median_ms / 1000.0) / 1e12
    tokens_per_sec = batch_size * seq_len / (median_ms / 1000.0)
    return {
        "label": label,
        "median_ms": median_ms,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "tflops": tflops,
        "tokens_per_sec": tokens_per_sec,
    }


def print_results(results_list):
    """Print a comparison table from a list of result dicts."""
    print(f"\n{'='*72}")
    print(f"{'Mode':<20} {'Median(ms)':>10} {'Mean(ms)':>10} {'Std(ms)':>10} "
          f"{'TFLOPS':>8} {'Tok/s':>12}")
    print(f"{'-'*72}")
    for r in results_list:
        print(f"{r['label']:<20} {r['median_ms']:>10.2f} {r['mean_ms']:>10.2f} "
              f"{r['std_ms']:>10.2f} {r['tflops']:>8.1f} {r['tokens_per_sec']:>12.0f}")

    if len(results_list) == 2:
        baseline = results_list[0]["median_ms"]
        cp = results_list[1]["median_ms"]
        speedup = baseline / cp
        print(f"{'-'*72}")
        print(f"  Speedup (CP vs no-CP): {speedup:.2f}x")
    print(f"{'='*72}")


def main():
    # Force cuDNN FusedAttention
    os.environ["NVTE_FUSED_ATTN"] = "1"
    os.environ["NVTE_FLASH_ATTN"] = "0"

    args = parse_args()
    rank, world_size, cp_group, cp_ranks = setup_distributed()

    num_heads, num_gqa_groups, head_dim = MODEL_CONFIGS[args.config]
    dtype = torch.bfloat16

    if args.qkv_format == "thd":
        attn_mask_type = "padding_causal"
    else:
        attn_mask_type = "causal"

    if rank == 0:
        print(f"Config: {args.config} | Format: {args.qkv_format} | "
              f"B={args.batch_size}, S={args.seq_len}, H={num_heads}, "
              f"KV_groups={num_gqa_groups}, D={head_dim}")
        print(f"CP size: {world_size} | Baseline: {args.baseline} | Verify: {args.verify}")
        print(f"Warmup: {args.warmup} | Timed iters: {args.iters}")

    # ---- Verification ----
    if args.verify:
        verify_cp_correctness(
            rank, world_size, cp_group, cp_ranks,
            num_heads, num_gqa_groups, head_dim,
            args.batch_size, args.seq_len, args.qkv_format, dtype,
        )

    results = []

    # ---- No-CP baseline (runs on each rank independently, full seq_len) ----
    if args.baseline:
        if rank == 0:
            print(f"\n--- No-CP baseline (single-GPU, full S={args.seq_len}) ---")

        baseline_attn = DotProductAttention(
            num_heads,
            head_dim,
            num_gqa_groups=num_gqa_groups,
            attention_dropout=0.0,
            qkv_format=args.qkv_format,
            attn_mask_type=attn_mask_type,
        ).cuda()

        if args.qkv_format == "bshd":
            bq, bk, bv, bdout, bcu_sq, bcu_skv, bcu_sq_pad, bcu_skv_pad = \
                make_inputs_no_cp_bshd(
                    args.batch_size, args.seq_len, num_heads, num_gqa_groups, head_dim, dtype,
                )
        else:
            bq, bk, bv, bdout, bcu_sq, bcu_skv, bcu_sq_pad, bcu_skv_pad = \
                make_inputs_no_cp_thd(
                    args.batch_size, args.seq_len, num_heads, num_gqa_groups, head_dim, dtype,
                )

        baseline_times = benchmark_attn(
            baseline_attn, bq, bk, bv, bdout,
            bcu_sq, bcu_skv, bcu_sq_pad, bcu_skv_pad,
            args.warmup, args.iters,
        )
        results.append(report_results(
            f"no-CP ({args.qkv_format})", baseline_times,
            args.batch_size, args.seq_len, num_heads, head_dim,
        ))

        # Free baseline tensors
        del baseline_attn, bq, bk, bv, bdout
        torch.cuda.empty_cache()

    # ---- CP run ----
    if rank == 0:
        print(f"\n--- CP (p2p, {args.qkv_format}, {world_size} GPUs) ---")

    core_attn = DotProductAttention(
        num_heads,
        head_dim,
        num_gqa_groups=num_gqa_groups,
        attention_dropout=0.0,
        qkv_format=args.qkv_format,
        attn_mask_type=attn_mask_type,
    ).cuda()

    core_attn.set_context_parallel_group(
        cp_group,
        cp_ranks,
        torch.cuda.Stream(),
        "p2p",
    )

    if args.qkv_format == "bshd":
        q, k, v, dout, cu_sq, cu_skv, cu_sq_pad, cu_skv_pad, _, _, _, _ = make_inputs_bshd(
            args.batch_size, args.seq_len, num_heads, num_gqa_groups, head_dim,
            world_size, rank, dtype,
        )
    else:
        q, k, v, dout, cu_sq, cu_skv, cu_sq_pad, cu_skv_pad, _, _, _, _ = make_inputs_thd(
            args.batch_size, args.seq_len, num_heads, num_gqa_groups, head_dim,
            world_size, rank, dtype,
        )

    if rank == 0:
        print(f"  Local Q shape: {list(q.shape)}")

    cp_times = benchmark_attn(
        core_attn, q, k, v, dout,
        cu_sq, cu_skv, cu_sq_pad, cu_skv_pad,
        args.warmup, args.iters,
    )
    results.append(report_results(
        f"CP-p2p ({args.qkv_format})", cp_times,
        args.batch_size, args.seq_len, num_heads, head_dim,
    ))

    # ---- Print comparison ----
    if rank == 0:
        print_results(results)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
