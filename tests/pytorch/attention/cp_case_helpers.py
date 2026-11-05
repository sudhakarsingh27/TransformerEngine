# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Shared data and operation helpers for one BF16 THD CP2 Case."""

import copy

import torch
import torch.distributed as dist
from transformer_engine.pytorch import CPLoadBalancingStrategy, DotProductAttention

from run_attention_with_cp import get_thd_partitioned_indices, get_tols
from test_attention_with_cp import model_configs_fused_attn
from utils import compare_and_assert


def reset_leaves(state):
    """Complete leaf replacement before the next timed operation."""
    state["leaves"] = [x.detach().clone().requires_grad_() for x in state["bases"]]
    torch.cuda.synchronize()


def build_point(state, benchmark_only):
    rank, world, evidence = state["rank"], state["world"], state["evidence"]
    state.clear()
    state.update(rank=rank, world=world, evidence=evidence)
    cfg = copy.deepcopy(model_configs_fused_attn["cp_2_0"])
    cfg.attn_mask_type = "padding_causal"
    strategy = CPLoadBalancingStrategy.DUAL_CHUNK_SWAP
    # Uniform lengths make allocation and reference projection explicit without
    # changing the existing correctness worker's randomized THD input policy.
    assert cfg.max_seqlen_q == cfg.max_seqlen_kv and cfg.max_seqlen_q % (2 * world) == 0
    cu = torch.arange(cfg.batch_size + 1, dtype=torch.int32, device="cuda") * cfg.max_seqlen_q
    tokens = cfg.batch_size * cfg.max_seqlen_q
    global_shapes = [
        (tokens, cfg.num_heads, cfg.head_dim_qk),
        (tokens, cfg.num_gqa_groups, cfg.head_dim_qk),
        (tokens, cfg.num_gqa_groups, cfg.head_dim_v),
        (tokens, cfg.num_heads * cfg.head_dim_v),
    ]
    idx = get_thd_partitioned_indices(
        cu, tokens, world, rank, device="cuda", load_balancing_strategy=strategy
    )
    local_shapes = [(idx.numel(), *shape[1:]) for shape in global_shapes]
    # Benchmark mode chooses local shapes before allocating any input tensor.
    generator = torch.Generator(device="cuda").manual_seed(1234 + (rank if benchmark_only else 0))
    tensors = [
        torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator).clamp_(-1, 1)
        for shape in (local_shapes if benchmark_only else global_shapes)
    ]
    if benchmark_only:
        local = tensors
    else:
        state.update(full_bases=tensors[:3], full_dout=tensors[3])
        local = [x.index_select(0, idx) for x in tensors]

    def module():
        return DotProductAttention(
            cfg.num_heads,
            (cfg.head_dim_qk, cfg.head_dim_v),
            num_gqa_groups=cfg.num_gqa_groups,
            attention_dropout=0.0,
            qkv_format="thd",
            attn_mask_type=cfg.attn_mask_type,
            window_size=cfg.window_size,
        ).cuda()

    attn, stream = module(), torch.cuda.Stream()
    attn.set_context_parallel_group(
        dist.group.WORLD, list(range(world)), stream, "all_gather", strategy
    )
    if not benchmark_only:
        # The runner evaluates CP first, so the reference must have its own module.
        state["reference_attn"] = module()
    state.update(
        bases=local[:3],
        dout=local[3],
        attn=attn,
        stream=stream,
        cfg=cfg,
        idx=idx,
        lengths=dict(
            cu_seqlens_q=cu, cu_seqlens_kv=cu, cu_seqlens_q_padded=cu, cu_seqlens_kv_padded=cu
        ),
    )
    evidence.update(
        full_reference_allocated=not benchmark_only,
        global_shapes=global_shapes,
        local_shapes=local_shapes,
    )
    reset_leaves(state)
    return state


def run_fwd_bwd(attn, leaves, dout, lengths, retain_result):
    out = attn(*leaves, **lengths, pad_between_seqs=True)
    out.backward(dout)
    if retain_result:
        # Own the result storage across reset and the subsequent reference call.
        return tuple(x.detach().clone() for x in (out, *(leaf.grad for leaf in leaves)))
    return None


def project_reference(state):
    leaves = [x.detach().clone().requires_grad_() for x in state["full_bases"]]
    result = run_fwd_bwd(
        state["reference_attn"], leaves, state["full_dout"], state["lengths"], retain_result=True
    )
    return tuple(x.index_select(0, state["idx"]) for x in result)


def verify_results(actual, expected, cfg):
    atol, rtol, rmse = get_tols(cfg, "bf16")
    errors = {}
    for name, a, e in zip(("out", "dq", "dk", "dv"), actual, expected):
        assert torch.isfinite(a).all().item(), f"{name}_cp is not finite"
        assert torch.isfinite(e).all().item(), f"{name}_reference is not finite"
        compare_and_assert(e, a, name + "_reference", name + "_cp", atol, rtol, rmse, False)
        errors[name] = (a.float() - e.float()).abs().max().item()
    return errors
