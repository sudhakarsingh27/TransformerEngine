#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Microbenchmarks for PR 2829 THD context-parallel helper kernels."""

import argparse
import itertools
import time

import torch
import transformer_engine  # pylint: disable=unused-import
import transformer_engine_torch as tex


def legacy_reorder_thd_to_rank_sharded(x, cu_seqlens, cp_size, seq_dim=0):
    total_slices_of_any_sequence = 2 * cp_size
    slice_sizes = (cu_seqlens[1:] - cu_seqlens[:-1]) // total_slices_of_any_sequence

    indices = [
        (
            torch.arange(
                seq_start + (cp_rank * slice_size),
                seq_start + ((cp_rank + 1) * slice_size),
                device=cu_seqlens.device,
            ),
            torch.arange(
                seq_start + ((total_slices_of_any_sequence - cp_rank - 1) * slice_size),
                seq_start + ((total_slices_of_any_sequence - cp_rank) * slice_size),
                device=cu_seqlens.device,
            ),
        )
        for cp_rank in range(cp_size)
        for slice_size, seq_start in zip(slice_sizes, cu_seqlens[:-1])
    ]

    indices = list(itertools.chain(*indices))
    indices = torch.cat(indices)
    return x.index_select(seq_dim, indices)


def legacy_reorder_thd_to_contiguous(x, cu_seqlens, seq_chunk_ids, cp_size, seq_dim=0):
    max_cum_seqlen_per_cp_rank = cu_seqlens[-1] // cp_size
    cu_seqlens_on_any_cp_rank = cu_seqlens // cp_size

    indices = [
        torch.arange(
            (
                start + max_cum_seqlen_per_cp_rank * (chunk_id // 2)
                if loc < cp_size
                else (start + end) // 2 + max_cum_seqlen_per_cp_rank * (chunk_id // 2)
            ),
            (
                (start + end) // 2 + max_cum_seqlen_per_cp_rank * (chunk_id // 2)
                if loc < cp_size
                else end + max_cum_seqlen_per_cp_rank * (chunk_id // 2)
            ),
            device=cu_seqlens.device,
        )
        for start, end in zip(
            cu_seqlens_on_any_cp_rank[:-1], cu_seqlens_on_any_cp_rank[1:]
        )
        for loc, chunk_id in enumerate(seq_chunk_ids)
    ]

    indices = torch.cat(indices)
    return x.index_select(seq_dim, indices)


def legacy_valid_copy(out, inp, cu_seqlens_padded, cu_seqlens):
    batch_size = cu_seqlens.shape[0] - 1
    for b in range(batch_size):
        s = cu_seqlens_padded[b].item()
        sz = (cu_seqlens[b + 1] - cu_seqlens[b]).item()
        if sz > 0:
            out[s : s + sz].copy_(inp[s : s + sz])


def seq_chunk_ids_before_attn(cp_size, device):
    chunk_ids = torch.empty(2 * cp_size, dtype=torch.int32, device=device)
    for rank in range(cp_size):
        chunk_ids[rank] = 2 * rank
        chunk_ids[rank + cp_size] = 2 * cp_size - 2 * rank - 1
    return chunk_ids


def benchmark(fn, warmup, iters, profile_nvtx_name=None):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    if profile_nvtx_name is not None:
        torch.cuda.nvtx.range_push(profile_nvtx_name)
    start_wall = time.perf_counter()
    start_event.record()
    for _ in range(iters):
        fn()
    end_event.record()
    torch.cuda.synchronize()
    end_wall = time.perf_counter()
    if profile_nvtx_name is not None:
        torch.cuda.nvtx.range_pop()

    return {
        "wall_ms": (end_wall - start_wall) * 1000.0 / iters,
        "event_ms": start_event.elapsed_time(end_event) / iters,
    }


def make_cu_seqlens(batch, seqlen, device):
    return torch.arange(batch + 1, dtype=torch.int32, device=device) * seqlen


def verify_equal(name, actual, expected):
    if not torch.equal(actual, expected):
        raise RuntimeError(f"{name} correctness check failed")


def run_reorder(args, dtype, device):
    seqlen = args.seqlen
    if seqlen % (2 * args.cp_size) != 0:
        raise ValueError("seqlen must be divisible by 2 * cp_size")
    total_tokens = args.batch * seqlen
    cu_seqlens = make_cu_seqlens(args.batch, seqlen, device)
    x = torch.randn(total_tokens, args.heads, args.dim, device=device, dtype=dtype)
    seq_chunk_ids = seq_chunk_ids_before_attn(args.cp_size, device)

    rank_kernel = tex.thd_reorder(x, cu_seqlens, args.cp_size, False, total_tokens)
    rank_legacy = legacy_reorder_thd_to_rank_sharded(x, cu_seqlens, args.cp_size)
    verify_equal("thd_reorder rank-sharded", rank_kernel, rank_legacy)

    contig_kernel = tex.thd_reorder(
        rank_kernel, cu_seqlens, args.cp_size, True, total_tokens
    )
    contig_legacy = legacy_reorder_thd_to_contiguous(
        rank_kernel, cu_seqlens, seq_chunk_ids, args.cp_size
    )
    verify_equal("thd_reorder contiguous", contig_kernel, contig_legacy)
    verify_equal("thd_reorder round trip", contig_kernel, x)

    return [
        (
            "thd_reorder contiguous->rank",
            benchmark(
                lambda: legacy_reorder_thd_to_rank_sharded(x, cu_seqlens, args.cp_size),
                args.warmup,
                args.iters,
                "legacy_thd_reorder_rank" if args.profile_nvtx else None,
            ),
            benchmark(
                lambda: tex.thd_reorder(
                    x, cu_seqlens, args.cp_size, False, total_tokens
                ),
                args.warmup,
                args.iters,
                "kernel_thd_reorder_rank" if args.profile_nvtx else None,
            ),
        ),
        (
            "thd_reorder rank->contiguous",
            benchmark(
                lambda: legacy_reorder_thd_to_contiguous(
                    rank_kernel, cu_seqlens, seq_chunk_ids, args.cp_size
                ),
                args.warmup,
                args.iters,
                "legacy_thd_reorder_contig" if args.profile_nvtx else None,
            ),
            benchmark(
                lambda: tex.thd_reorder(
                    rank_kernel, cu_seqlens, args.cp_size, True, total_tokens
                ),
                args.warmup,
                args.iters,
                "kernel_thd_reorder_contig" if args.profile_nvtx else None,
            ),
        ),
    ]


def run_valid_copy(args, dtype, device):
    padded_seqlen = args.seqlen
    valid_seqlen = args.valid_seqlen or max(1, padded_seqlen - args.valid_padding)
    total_tokens = args.batch * padded_seqlen
    cu_seqlens_padded = make_cu_seqlens(args.batch, padded_seqlen, device)
    cu_seqlens = make_cu_seqlens(args.batch, valid_seqlen, device)
    inp = torch.randn(total_tokens, args.heads, args.dim, device=device, dtype=dtype)

    expected = torch.zeros_like(inp)
    actual = torch.zeros_like(inp)
    legacy_valid_copy(expected, inp, cu_seqlens_padded, cu_seqlens)
    tex.thd_valid_copy(actual, inp, cu_seqlens_padded, cu_seqlens)
    verify_equal("thd_valid_copy", actual, expected)

    legacy_out = torch.empty_like(inp)
    kernel_out = torch.empty_like(inp)
    return [
        (
            "thd_valid_copy",
            benchmark(
                lambda: legacy_valid_copy(
                    legacy_out, inp, cu_seqlens_padded, cu_seqlens
                ),
                args.warmup,
                args.iters,
                "legacy_thd_valid_copy" if args.profile_nvtx else None,
            ),
            benchmark(
                lambda: tex.thd_valid_copy(
                    kernel_out, inp, cu_seqlens_padded, cu_seqlens
                ),
                args.warmup,
                args.iters,
                "kernel_thd_valid_copy" if args.profile_nvtx else None,
            ),
        )
    ]


def run_kernel_only(args, dtype, device):
    total_tokens = args.batch * args.seqlen
    cu_seqlens = make_cu_seqlens(args.batch, args.seqlen, device)
    x = torch.randn(total_tokens, args.heads, args.dim, device=device, dtype=dtype)
    kv = torch.randn(2, total_tokens, args.heads, args.dim, device=device, dtype=dtype)
    lse = torch.randn(
        args.batch, args.heads, args.seqlen, device=device, dtype=torch.float32
    )
    packed_lse = torch.randn(
        args.heads, total_tokens, device=device, dtype=torch.float32
    )

    return [
        (
            "thd_read_half_tensor [t,h,d]",
            None,
            benchmark(
                lambda: tex.thd_read_half_tensor(x, cu_seqlens, 1),
                args.warmup,
                args.iters,
                "kernel_thd_read_half_tensor_q" if args.profile_nvtx else None,
            ),
        ),
        (
            "thd_read_half_tensor [2,t,h,d]",
            None,
            benchmark(
                lambda: tex.thd_read_half_tensor(kv, cu_seqlens, 1),
                args.warmup,
                args.iters,
                "kernel_thd_read_half_tensor_kv" if args.profile_nvtx else None,
            ),
        ),
        (
            "thd_read_second_half_lse batch-major",
            None,
            benchmark(
                lambda: tex.thd_read_second_half_lse(
                    lse, cu_seqlens, False, args.seqlen // 2
                ),
                args.warmup,
                args.iters,
                "kernel_thd_read_lse_batch" if args.profile_nvtx else None,
            ),
        ),
        (
            "thd_read_second_half_lse packed",
            None,
            benchmark(
                lambda: tex.thd_read_second_half_lse(
                    packed_lse, cu_seqlens, True, total_tokens // 2
                ),
                args.warmup,
                args.iters,
                "kernel_thd_read_lse_packed" if args.profile_nvtx else None,
            ),
        ),
        (
            "thd_get_partitioned_indices",
            None,
            benchmark(
                lambda: tex.thd_get_partitioned_indices(
                    cu_seqlens, total_tokens, args.cp_size, args.cp_rank
                ),
                args.warmup,
                args.iters,
                "kernel_thd_get_partitioned_indices" if args.profile_nvtx else None,
            ),
        ),
    ]


def print_results(rows, args):
    print(
        f"# THD kernel microbench: batch={args.batch} seqlen={args.seqlen} "
        f"cp={args.cp_size} heads={args.heads} dim={args.dim} dtype={args.dtype} "
        f"iters={args.iters}"
    )
    print("| path | legacy wall ms | kernel wall ms | speedup | kernel event ms |")
    print("|---|---:|---:|---:|---:|")
    for name, legacy, kernel in rows:
        if legacy is None:
            legacy_wall = "--"
            speedup = "--"
        else:
            legacy_wall = f"{legacy['wall_ms']:.4f}"
            speedup = f"{legacy['wall_ms'] / kernel['wall_ms']:.2f}x"
        print(
            f"| {name} | {legacy_wall} | {kernel['wall_ms']:.4f} | "
            f"{speedup} | {kernel['event_ms']:.4f} |"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seqlen", type=int, default=8192)
    parser.add_argument("--valid-seqlen", type=int, default=None)
    parser.add_argument("--valid-padding", type=int, default=128)
    parser.add_argument("--cp-size", type=int, default=8)
    parser.add_argument("--cp-rank", type=int, default=0)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--profile-nvtx", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.seqlen % 2 != 0:
        raise ValueError("seqlen must be even")
    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda")
    torch.cuda.set_device(0)

    rows = []
    rows.extend(run_reorder(args, dtype, device))
    rows.extend(run_valid_copy(args, dtype, device))
    rows.extend(run_kernel_only(args, dtype, device))
    print_results(rows, args)


if __name__ == "__main__":
    main()
