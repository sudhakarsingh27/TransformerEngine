#!/usr/bin/env python3
"""Measure the rank-maximum allocated peak for THD P2P attention.

This deliberately avoids the test runner's no-CP reference graph.  The probe
creates full tensors only to use Transformer Engine's CP partitioner, releases
them before resetting allocator peaks, and then measures rank-local P2P work.
Forward and end-to-end peaks are separate because P2P forward retention is the
behavior under test.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist

from transformer_engine.pytorch import DotProductAttention
from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import (
    get_batch_on_this_cp_rank,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=262144)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-gqa-groups", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--backend", choices=("fused", "flash"), default="fused")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=1)
    return parser.parse_args()


def barrier(device: int) -> None:
    try:
        dist.barrier(device_ids=[device])
    except TypeError:
        dist.barrier()


def main() -> None:
    args = parse_args()
    if args.warmup < 1:
        raise ValueError("--warmup must be at least one so the final warmup can be measured")
    if args.iters < 0:
        raise ValueError("--iters must be non-negative")
    if args.num_heads % args.num_gqa_groups:
        raise ValueError("--num-heads must be divisible by --num-gqa-groups")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=local_rank)
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", local_rank)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    # Keep backend selection explicit and identical between baseline and PR.
    os.environ["NVTE_FUSED_ATTN"] = "1" if args.backend == "fused" else "0"
    os.environ["NVTE_FLASH_ATTN"] = "1" if args.backend == "flash" else "0"

    cp_group = dist.new_group(list(range(world_size)), backend="nccl")
    stream = torch.cuda.Stream(device=device)
    attn = DotProductAttention(
        args.num_heads,
        args.head_dim,
        num_gqa_groups=args.num_gqa_groups,
        attention_dropout=0.0,
        qkv_format="thd",
        attn_mask_type="padding_causal",
    ).cuda()
    attn.train()
    attn.set_context_parallel_group(cp_group, list(range(world_size)), stream, "p2p")

    # This mirrors the supplied reference benchmark's padded THD setup.
    seqlens = torch.full((args.batch_size,), args.seq_len, dtype=torch.int32, device=device)
    padded = (seqlens + 2 * world_size - 1) // (2 * world_size) * (2 * world_size)
    cu_seqlens_padded = torch.cat(
        (torch.zeros(1, dtype=torch.int32, device=device), padded.cumsum(0, dtype=torch.int32))
    )
    cu_seqlens = cu_seqlens_padded.clone()
    if args.backend == "fused":
        cu_seqlens[1:] = seqlens.cumsum(0, dtype=torch.int32)
    total_tokens = int(cu_seqlens_padded[-1].item())

    torch.manual_seed(1234 + rank)
    q_global = torch.randn(
        total_tokens, args.num_heads, args.head_dim, dtype=dtype, device=device
    )
    k_global = torch.randn(
        total_tokens, args.num_gqa_groups, args.head_dim, dtype=dtype, device=device
    )
    v_global = torch.randn_like(k_global)
    dout_global = torch.randn(
        total_tokens, args.num_heads * args.head_dim, dtype=dtype, device=device
    )
    q_local, k_local, v_local = get_batch_on_this_cp_rank(
        cu_seqlens_padded, q_global, k_global, v_global, cp_group, qvk_format="thd"
    )
    dout_local, _, _ = get_batch_on_this_cp_rank(
        cu_seqlens_padded, dout_global, dout_global, dout_global, cp_group, qvk_format="thd"
    )

    # The globals are intentionally outside the measured lifetime.
    del q_global, k_global, v_global, dout_global
    torch.cuda.empty_cache()
    barrier(local_rank)

    base_allocated = forward_peak = e2e_peak = None
    timed_ms = []
    total_iters = args.warmup + args.iters
    for iteration in range(total_iters):
        # The final warmup is measured: instrumentation never disturbs a timed iteration.
        measure = iteration == args.warmup - 1
        if measure:
            torch.cuda.synchronize(device)
            base_allocated = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)

        q = q_local.detach().clone().requires_grad_(True)
        k = k_local.detach().clone().requires_grad_(True)
        v = v_local.detach().clone().requires_grad_(True)
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        if iteration >= args.warmup:
            start.record()
        out = attn(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
        )
        if isinstance(out, tuple):
            out = out[0]
        if measure:
            torch.cuda.synchronize(device)
            forward_peak = torch.cuda.max_memory_allocated(device)
        out.backward(dout_local)
        if iteration >= args.warmup:
            end.record()
        torch.cuda.synchronize(device)
        if measure:
            e2e_peak = torch.cuda.max_memory_allocated(device)
        if iteration >= args.warmup:
            timed_ms.append(start.elapsed_time(end))
        del out, q, k, v

    assert base_allocated is not None and forward_peak is not None and e2e_peak is not None
    metrics = torch.tensor(
        [base_allocated, forward_peak, e2e_peak], dtype=torch.float64, device=device
    )
    dist.all_reduce(metrics, op=dist.ReduceOp.MAX, group=cp_group)
    if rank == 0:
        mib = 1024.0 * 1024.0
        base, forward, e2e = (float(value.item()) / mib for value in metrics)
        print(
            "CP_P2P_MEMORY "
            + json.dumps(
                {
                    "backend": args.backend,
                    "batch_size": args.batch_size,
                    "cp_size": world_size,
                    "dtype": args.dtype,
                    "e2e_delta_mib": e2e - base,
                    "e2e_peak_mib": e2e,
                    "forward_delta_mib": forward - base,
                    "forward_peak_mib": forward,
                    "head_dim": args.head_dim,
                    "num_gqa_groups": args.num_gqa_groups,
                    "num_heads": args.num_heads,
                    "rank_max_base_mib": base,
                    "seq_len": args.seq_len,
                    "timed_mean_ms": sum(timed_ms) / len(timed_ms) if timed_ms else None,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    barrier(local_rank)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
