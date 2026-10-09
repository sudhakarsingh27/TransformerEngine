# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fixed-token context-parallel attention benchmark configurations."""

import json
import time

import torch
import torch.distributed as dist
from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import (
    get_thd_partitioned_indices,
)

from utils import ModelConfig


FIXED_TOKEN_SHAPES = (
    ("uniform_1x512k", 1, 524288),
    ("uniform_2x256k", 2, 262144),
    ("uniform_4x128k", 4, 131072),
    ("uniform_8x64k", 8, 65536),
    ("uniform_16x32k", 16, 32768),
    ("uniform_32x16k", 32, 16384),
    ("uniform_64x8k", 64, 8192),
    ("uniform_128x4k", 128, 4096),
    ("uniform_256x2k", 256, 2048),
)

# Keep these out of the correctness catalog: their 512K-token footprint is
# intended for rank-local benchmark mode, not the full-input reference path.
model_configs = {
    name: ModelConfig(
        batch_size,
        sequence_length,
        32,
        128,
        num_gqa_groups=8,
        dropout_p=0.0,
        attn_mask_type="causal",
    )
    for name, batch_size, sequence_length in FIXED_TOKEN_SHAPES
}


def run_fixed_token_benchmark(
    *,
    core_attn,
    config,
    model,
    kernel_backend,
    cp_comm_type,
    cp_comm_group,
    load_balancing_strategy,
    pad_between_seqs,
    cu_seqlens,
    input_shapes,
    benchmark_iters,
):
    """Time rank-local CP forward/backward and report per-iteration rank maxima."""
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    cu_seqlens_q, cu_seqlens_kv, cu_seqlens_q_padded, cu_seqlens_kv_padded = cu_seqlens
    q_input_shape, k_input_shape, v_input_shape, attn_output_shape = input_shapes
    seq_idx_q = get_thd_partitioned_indices(
        cu_seqlens_q_padded,
        int(q_input_shape[0]),
        world_size,
        rank,
        device="cuda",
        load_balancing_strategy=load_balancing_strategy,
    )
    seq_idx_kv = get_thd_partitioned_indices(
        cu_seqlens_kv_padded,
        int(k_input_shape[0]),
        world_size,
        rank,
        device="cuda",
        load_balancing_strategy=load_balancing_strategy,
    )
    local_shapes = [
        (seq_idx_q.numel(), *q_input_shape[1:]),
        (seq_idx_kv.numel(), *k_input_shape[1:]),
        (seq_idx_kv.numel(), *v_input_shape[1:]),
        (seq_idx_q.numel(), *attn_output_shape[1:]),
    ]
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    q, k, v, dout = [
        torch.clamp(torch.randn(shape, dtype=torch.bfloat16), min=-1, max=1).cuda()
        for shape in local_shapes
    ]
    q, k, v, dout = [tensor.contiguous() for tensor in (q, k, v, dout)]
    q, k, v = [tensor.requires_grad_() for tensor in (q, k, v)]

    core_attn.set_context_parallel_group(
        cp_comm_group,
        range(world_size),
        torch.cuda.Stream(),
        cp_comm_type,
        load_balancing_strategy,
    )

    warmup_iters = 10
    local_latencies_ms = []
    for iteration in range(warmup_iters + benchmark_iters):
        dist.barrier(group=cp_comm_group)
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = core_attn(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            cu_seqlens_q_padded=cu_seqlens_q_padded,
            cu_seqlens_kv_padded=cu_seqlens_kv_padded,
            pad_between_seqs=pad_between_seqs,
        )
        if isinstance(out, tuple):
            out = out[0]
        out.backward(dout)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        if iteration >= warmup_iters:
            local_latencies_ms.append(elapsed_ms)
        q.grad = k.grad = v.grad = None
        del out

    iteration_rank_max_ms = torch.tensor(
        local_latencies_ms, dtype=torch.float64, device=q.device
    )
    dist.reduce(iteration_rank_max_ms, dst=0, op=dist.ReduceOp.MAX, group=cp_comm_group)
    if rank == 0:
        rank_max_samples = iteration_rank_max_ms.cpu().tolist()
        result = {
            "model": model,
            "backend": kernel_backend,
            "communication_type": cp_comm_type,
            "qkv_format": "thd",
            "dtype": "bf16",
            "cp_size": world_size,
            "is_training": True,
            "batch_size": config.batch_size,
            "sequence_length": config.max_seqlen_q,
            "total_tokens": config.batch_size * config.max_seqlen_q,
            "num_heads": config.num_heads,
            "num_gqa_groups": config.num_gqa_groups,
            "head_dim": config.head_dim_qk,
            "dropout": config.dropout_p,
            "warmup_iterations": warmup_iters,
            "timed_iterations": benchmark_iters,
            "statistic": "mean_of_iteration_rank_max_ms",
            "latency_ms": sum(rank_max_samples) / benchmark_iters,
            "iteration_rank_max_ms": rank_max_samples,
        }
        print(f"CP_BENCH_RESULT {json.dumps(result, sort_keys=True)}", flush=True)
