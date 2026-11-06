# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import copy
import os
import sys
import logging
from contextlib import nullcontext
from unittest.mock import patch
import torch
import pytest
import torch.distributed as dist
from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import (
    get_cu_seqlens_on_cp_rank,
    get_thd_partitioned_indices,
    flash_attn_a2a_communicate,
)
from transformer_engine.pytorch.attention.dot_product_attention.utils import (
    combine_and_quantize,
    get_thd_padding_mask,
)
from transformer_engine.pytorch import DType
from test_attention_with_cp import (
    model_configs_flash_attn,
    model_configs_fused_attn,
)
from transformer_engine.pytorch import (
    autocast,
    CPLoadBalancingStrategy,
    DotProductAttention,
    MultiheadAttention,
    Float8Quantizer,
    Float8CurrentScalingQuantizer,
    MXFP8Quantizer,
)
from transformer_engine.common.recipe import (
    DelayedScaling,
    Float8CurrentScaling,
    MXFP8BlockScaling,
    Format,
)
from utils import ModelConfig, compare_and_assert

# Pool mode (NVTE_CP_POOL_PG=1) only: shared CP collective groups, created once
# per pool by run_attention_with_cp_pool.main() and reused across every case in
# that pool. world_size and the rank set don't change per case, so re-creating
# these per call would be wasted NCCL setup (~50-100 ms each). Single-shot
# subprocess mode leaves these None / [] and run_dpa_with_cp creates/destroys
# its own groups inline.
_pool_cp_comm_group = None
_pool_cp_comm_sub_groups: list = []

dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8": torch.bfloat16}


def generate_input_shapes(
    qkv_format: str,
    config: ModelConfig,
    world_size: int,
    kernel_backend: str,
    fa_pad_between_seqs: str = "False",
    load_balancing_strategy=CPLoadBalancingStrategy.DUAL_CHUNK_SWAP,
    no_load_balance_case="",
):
    if qkv_format == "bshd":
        q_input_shape = (
            config.batch_size,
            config.max_seqlen_q,
            config.num_heads,
            config.head_dim_qk,
        )
        k_input_shape = (
            config.batch_size,
            config.max_seqlen_kv,
            config.num_gqa_groups,
            config.head_dim_qk,
        )
        v_input_shape = (
            config.batch_size,
            config.max_seqlen_kv,
            config.num_gqa_groups,
            config.head_dim_v,
        )
        attn_output_shape = (
            config.batch_size,
            config.max_seqlen_q,
            config.num_heads * config.head_dim_v,
        )
        cu_seqlens_q = None
        cu_seqlens_kv = None
        cu_seqlens_q_padded = None
        cu_seqlens_kv_padded = None
    elif qkv_format == "sbhd":
        q_input_shape = (
            config.max_seqlen_q,
            config.batch_size,
            config.num_heads,
            config.head_dim_qk,
        )
        k_input_shape = (
            config.max_seqlen_kv,
            config.batch_size,
            config.num_gqa_groups,
            config.head_dim_qk,
        )
        v_input_shape = (
            config.max_seqlen_kv,
            config.batch_size,
            config.num_gqa_groups,
            config.head_dim_v,
        )
        attn_output_shape = (
            config.max_seqlen_q,
            config.batch_size,
            config.num_heads * config.head_dim_v,
        )
        cu_seqlens_q = None
        cu_seqlens_kv = None
        cu_seqlens_q_padded = None
        cu_seqlens_kv_padded = None
    elif qkv_format == "thd":
        if no_load_balance_case:
            # Odd local slabs exercise the absence of dual-chunk divisibility.
            total = 5 * world_size
            if no_load_balance_case == "crossing":
                seqlens_q = torch.tensor([3, total - 7, 4], dtype=torch.int32)
                seqlens_q_padded = seqlens_q
            else:
                assert no_load_balance_case == "padding_only"
                assert kernel_backend == "FusedAttention"
                seqlens_q = torch.tensor([3, 2, 1], dtype=torch.int32)
                # With CP=4, ranks 2 and 3 own only physical padding.
                seqlens_q_padded = torch.tensor([5, 4, total - 9], dtype=torch.int32)
        elif load_balancing_strategy is CPLoadBalancingStrategy.NO_LOAD_BALANCE:
            assert config.batch_size == 2
            if kernel_backend == "FlashAttention" and fa_pad_between_seqs == "False":
                seqlens_q = torch.tensor(
                    [config.max_seqlen_q - 2, config.max_seqlen_q], dtype=torch.int32
                )
                assert seqlens_q.sum().item() % world_size == 0
                seqlens_q_padded = seqlens_q
            else:
                # Exercise both document padding and a CP chunk boundary inside a document.
                seqlens_q = torch.tensor(
                    [config.max_seqlen_q - 2, config.max_seqlen_q - 1],
                    dtype=torch.int32,
                )
                padded_total = 2 * config.max_seqlen_q
                assert padded_total % world_size == 0
                seqlens_q_padded = torch.tensor(
                    [config.max_seqlen_q - 1, config.max_seqlen_q + 1],
                    dtype=torch.int32,
                )
        else:
            seqlens_q = torch.randint(0, config.max_seqlen_q + 1, [config.batch_size]).to(
                torch.int32
            )
            seqlens_q_padded = (
                (seqlens_q + 2 * world_size - 1) // (world_size * 2) * (world_size * 2)
            )
        cu_seqlens_q_padded = torch.cat(
            [
                torch.zeros([1], dtype=torch.int32),
                seqlens_q_padded.cumsum(0, dtype=torch.int32),
            ]
        ).cuda()
        cu_seqlens_q = torch.clone(cu_seqlens_q_padded)

        # Generate padded data (cu_seqlens_q reflects non-padded lengths, so it
        # differs from cu_seqlens_q_padded) for FusedAttention always, and for
        # FlashAttention only when its test param requests it. DPA auto-detects
        # pad_between_seqs downstream from the cu_seqlens_q vs cu_seqlens_q_padded
        # mismatch.
        if kernel_backend == "FusedAttention" or fa_pad_between_seqs == "True":
            cu_seqlens_q[1:] = seqlens_q.cumsum(0, dtype=torch.int32).cuda()

        # NOTE: In case of Cross-Attention, `cu_seqlens_kv` and `cu_seqlens_kv_padded`
        # will not be the same as `cu_seqlens_q` and `cu_seqlens_q_padded` respectively.
        cu_seqlens_kv = cu_seqlens_q
        cu_seqlens_kv_padded = cu_seqlens_q_padded

        total_tokens = cu_seqlens_q_padded[-1]

        q_input_shape = (
            total_tokens,
            config.num_heads,
            config.head_dim_qk,
        )
        k_input_shape = (
            total_tokens,
            config.num_gqa_groups,
            config.head_dim_qk,
        )
        v_input_shape = (
            total_tokens,
            config.num_gqa_groups,
            config.head_dim_v,
        )
        attn_output_shape = (
            total_tokens,
            config.num_heads * config.head_dim_v,
        )
    else:
        assert False, f"{qkv_format=} is not supported!"

    return (
        q_input_shape,
        k_input_shape,
        v_input_shape,
        attn_output_shape,
        cu_seqlens_q,
        cu_seqlens_kv,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
    )


def get_tols(config, dtype):
    if dtype == "bf16":
        if config.num_heads == config.num_gqa_groups:
            atol = 2.5e-2
            rtol = 2.5e-2
        else:
            atol = 3.5e-2
            rtol = 3.5e-2
        rmse_tol = 0.01
    elif dtype == "fp16":
        atol = 5e-3
        rtol = 5e-3
        rmse_tol = 0.01
    elif dtype == "fp8":
        atol = 5e-1
        rtol = 5e-1
        rmse_tol = 0.15
    else:
        assert False, f"{dtype=} is not supported!"

    return atol, rtol, rmse_tol


def assert_matches_independent_reference(out, q, k, v, config, qkv_format, dtype, softmax_scale):
    """Check the non-CP output against a float64 reference instead of against another run.

    The CP suite compares a context-parallel run with a non-CP run *of the same backend*
    (``tensors_no_cp`` vs ``tensors_cp``). That validates the context-parallel plumbing and nothing
    about the kernel beneath it: any error affecting both sides equally cancels and the comparison
    passes. It is not a theoretical gap -- a backend was found returning an all-zero output while
    two of three ``cp_comm_type`` values reported PASS, because zeros matched zeros. The same
    structure then reported a partial *fix* as a regression, since correcting one side made the two
    disagree.

    This anchors one (batch, head) slice of the non-CP output to an independently computed
    reference, so a systematically wrong kernel cannot cancel itself out. It is a sanity bound, not
    a precision test -- the CP/no-CP comparison already covers precision. It is meant to catch
    catastrophic wrongness: zeros, NaNs, a mask applied in the wrong place.

    float64 rather than float32: torch computes fp32 matmuls in TF32 on Ampere and newer, whose
    significand is 11 bits -- the same as fp16 -- so an fp32 reference cannot judge a bf16 kernel.

    Configurations the reference does not model (fp8, bias, non-vanilla softmax, thd/ragged,
    padding masks) are skipped rather than approximated, so this never fails for a reason it cannot
    explain.
    """
    if qkv_format not in ("bshd", "sbhd"):
        return
    if dtype == "fp8" or config.attn_bias_type != "no_bias" or config.softmax_type != "vanilla":
        return
    # Softcap rescales the scores through tanh before the softmax. Modelling it here would mean
    # duplicating that formula in the reference, which is how a reference starts reimplementing the
    # thing it is supposed to check independently. Skipped instead.
    if getattr(config, "softcap", 0.0):
        return
    if "padding" in config.attn_mask_type:
        return
    if out is None or q is None:
        return
    # DotProductAttention returns [..., h * d_v], i.e. 3D. Restore the head axis from q and v --
    # without this the guard below would return early and the whole check would silently do
    # nothing, which is the failure mode this function exists to catch in the first place.
    if out.dim() == 3:
        out = out.view(*out.shape[:2], q.shape[2], v.shape[-1])
    if out.dim() != 4:
        return

    seq_dim = 1 if qkv_format == "bshd" else 0
    # One (batch, head) slice: the full float64 score matrix would be gigabytes.
    take = (
        lambda t, head: (t[0, :, head] if qkv_format == "bshd" else t[:, 0, head]).detach().double()
    )
    qhead, kvhead = 0, 0  # GQA groups are contiguous, so q head 0 belongs to kv group 0
    qs, ks, vs = take(q, qhead), take(k, kvhead), take(v, kvhead)
    got = take(out, qhead)
    sq, skv = qs.shape[0], ks.shape[0]

    scores = (qs @ ks.transpose(-1, -2)) * softmax_scale
    window = getattr(config, "window_size", None)
    left, right = window if window is not None else (-1, -1)
    if "causal" in config.attn_mask_type:
        right = 0
    # Bottom-right alignment when the lengths differ; identical to top-left when they do not.
    offset = (skv - sq) if "bottom_right" in config.attn_mask_type else 0
    blocked = torch.zeros(sq, skv, dtype=torch.bool, device=qs.device)
    if right != -1:
        blocked |= torch.ones(sq, skv, dtype=torch.bool, device=qs.device).triu(offset + right + 1)
    if left != -1:
        blocked |= torch.ones(sq, skv, dtype=torch.bool, device=qs.device).tril(offset - left - 1)
    scores = scores.masked_fill(blocked, float("-inf"))
    reference = torch.softmax(scores, dim=-1) @ vs

    # Reported separately from the tolerance: an all-zero or non-finite output is the specific
    # failure this exists to catch, and it should be named rather than shown as a large error.
    assert torch.isfinite(out).all(), "non-CP output is not finite"
    assert out.abs().max() > 0, (
        "non-CP output is all zeros -- the backend computed nothing. The CP/no-CP comparison"
        " cannot see this, because both sides are equally zero."
    )
    denom = reference.abs().max().clamp_min(1e-12)
    error = float((got - reference).abs().max() / denom)
    assert error < 5e-2, (
        "non-CP output disagrees with an independent float64 reference: relative error"
        f" {error:.3e}. This is a loose sanity bound, so exceeding it points at a wrong mask or a"
        " wrong kernel rather than a precision issue."
    )


def assert_no_load_balance_a2a_exchange(group, cu_seqlens_padded):
    """Check token and head placement independently in each exchange direction."""
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    total = int(cu_seqlens_padded[-1])
    local = total // size
    stream = torch.cuda.Stream()
    for names, before in (
        (["q", "k", "v"], True),
        (["dout"], True),
        (["out"], False),
        (["dq", "dk", "dv"], False),
    ):
        inputs, expected = [], []
        for index, name in enumerate(names):
            heads = 2 * size if name in ("q", "dq", "out", "dout") else size
            full = torch.arange(total * heads * 2, device="cuda").view(total, heads, 2)
            full = full + index * 10000
            slab = full[rank * local : (rank + 1) * local].contiguous()
            shard = full[:, rank * (heads // size) : (rank + 1) * (heads // size)].contiguous()
            inputs.append(slab if before else shard)
            expected.append(shard if before else slab)
        actual = flash_attn_a2a_communicate(
            inputs,
            None,
            0,
            size,
            group,
            stream,
            before,
            qkv_format="thd",
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
            a2a_input_names=names,
            load_balancing_strategy=CPLoadBalancingStrategy.NO_LOAD_BALANCE,
        )
        if len(names) == 1:
            actual = [actual]
        for output, reference in zip(actual, expected):
            torch.testing.assert_close(output, reference, rtol=0, atol=0)


def assert_no_load_balance_mha(group, cu_seqlens, cu_seqlens_padded, dtype):
    """Compare MHA outputs, input/weight gradients and independent document positions."""
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    total, hidden = int(cu_seqlens_padded[-1]), 512
    use_rope = torch.equal(cu_seqlens, cu_seqlens_padded)
    local = total // size
    torch.manual_seed(4321)
    reference = MultiheadAttention(
        hidden,
        8,
        kv_channels=64,
        num_gqa_groups=4,
        attention_dropout=0.0,
        qkv_format="thd",
        attn_mask_type="padding_causal",
        params_dtype=dtypes[dtype],
        bias=False,
        fuse_qkv_params=True,
    ).cuda()
    parallel = copy.deepcopy(reference)
    parallel.set_context_parallel_group(
        group,
        list(range(size)),
        torch.cuda.Stream(),
        "a2a",
        CPLoadBalancingStrategy.NO_LOAD_BALANCE,
    )
    x = torch.randn(total, hidden, device="cuda", dtype=dtypes[dtype], requires_grad=True)
    grad = torch.randn_like(x)
    valid = ~get_thd_padding_mask(total, cu_seqlens, cu_seqlens_padded)
    grad[~valid] = 0
    x_local = x.detach()[rank * local : (rank + 1) * local].clone().requires_grad_()
    freqs = torch.randn(total, 1, 1, 32, device="cuda", dtype=torch.float32)
    freqs = torch.cat((freqs, freqs), dim=-1)
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    positions = torch.tensor([p for length in lengths for p in range(length)], device="cuda")

    def document_rope(tensor, table, *_args, **_kwargs):
        # Explicit global positions avoid sharing the fragment/offset logic under test.
        angles = table[positions, 0]
        value = tensor.float()
        rotated = torch.cat((-value[..., 32:], value[..., :32]), dim=-1)
        return (value * angles.cos() + rotated * angles.sin()).to(tensor.dtype)

    kwargs = dict(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=total,
        max_seqlen_kv=total,
        rotary_pos_emb=freqs if use_rope else None,
        pad_between_seqs=not use_rope,
    )
    if not use_rope:
        with pytest.raises(AssertionError, match="inter-sequence padding"):
            parallel(x_local, **dict(kwargs, rotary_pos_emb=freqs))
    with patch(
        "transformer_engine.pytorch.attention.multi_head_attention.apply_rotary_pos_emb",
        document_rope,
    ):
        out = reference(x, **kwargs)
    out.backward(grad)
    out_local = parallel(x_local, **kwargs)
    out_local.backward(grad[rank * local : (rank + 1) * local].contiguous())
    atol = rtol = 3.5e-2 if dtype == "bf16" else 5e-3
    local_valid = valid[rank * local : (rank + 1) * local]
    torch.testing.assert_close(
        x_local.grad[~local_valid], torch.zeros_like(x_local.grad[~local_valid])
    )
    for actual, expected, name in (
        (out_local[local_valid], out[rank * local : (rank + 1) * local][local_valid], "MHA output"),
        (x_local.grad, x.grad[rank * local : (rank + 1) * local], "MHA input gradient"),
    ):
        if actual.numel():
            if torch.count_nonzero(expected) == 0:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            else:
                compare_and_assert(expected, actual, "reference", name, atol, rtol, 0.01, False)
    for (name, param), (ref_name, ref_param) in zip(
        parallel.named_parameters(), reference.named_parameters()
    ):
        assert name == ref_name and param.grad is not None and ref_param.grad is not None
        # CP owns tokens, so callers sum each replicated weight gradient exactly once.
        dist.all_reduce(param.grad, group=group)
        compare_and_assert(ref_param.grad, param.grad, "reference", name, atol, rtol, 0.01, False)


def assert_packed_attention_reference(out, q, k, v, dout, cu, padded_cu, config, dtype):
    """Anchor the small no-load-balance cases to explicit per-document attention."""
    q_ref, k_ref, v_ref = [x.detach().double().requires_grad_() for x in (q, k, v)]
    actual, physical = cu.tolist(), padded_cu.tolist()
    outputs, valid = [], []
    repeats = q.shape[1] // k.shape[1]
    for index, start in enumerate(physical[:-1]):
        length = actual[index + 1] - actual[index]
        end = start + length
        qs = q_ref[start:end].transpose(0, 1)
        ks = k_ref[start:end].repeat_interleave(repeats, dim=1).transpose(0, 1)
        vs = v_ref[start:end].repeat_interleave(repeats, dim=1).transpose(0, 1)
        scores = qs @ ks.transpose(-1, -2) / q.shape[-1] ** 0.5
        mask = torch.ones(length, length, device=q.device, dtype=torch.bool).triu(1)
        result = (scores.masked_fill(mask, -torch.inf).softmax(-1) @ vs).transpose(0, 1)
        outputs.append(result.reshape(length, -1))
        valid.extend(range(start, end))
    expected = torch.cat(outputs)
    expected.backward(dout[valid].double())
    atol, rtol, rmse = get_tols(config, dtype)
    for name, actual_tensor, expected_tensor in (
        ("output", out[valid], expected),
        ("dq", q.grad[valid], q_ref.grad[valid]),
        ("dk", k.grad[valid], k_ref.grad[valid]),
        ("dv", v.grad[valid], v_ref.grad[valid]),
    ):
        compare_and_assert(
            expected_tensor.to(actual_tensor.dtype),
            actual_tensor,
            "float64 reference",
            name,
            atol,
            rtol,
            rmse,
            False,
        )


def run_dpa_with_cp(
    dtype="bf16",
    model=None,
    qkv_format="bshd",
    kernel_backend="FlashAttention",
    cp_comm_type="p2p",
    fp8_bwd="True",
    fp8_dpa="False",
    fp8_mha="False",
    scaling_mode="delayed",
    f16_O="False",
    is_training="True",
    fa_pad_between_seqs="False",
    deterministic="False",
    load_balancing_strategy="DUAL_CHUNK_SWAP",
    no_load_balance_case="",
    softcap="0.0",
    log_level=logging.WARNING,
):
    """Test DotProductAttention module with context parallelism"""
    logging.root.setLevel(log_level)
    load_balancing_strategy = CPLoadBalancingStrategy[load_balancing_strategy]
    # When is_training is False, gradient outputs are None.
    is_training = is_training == "True"
    pad_between_seqs = None
    if qkv_format == "thd":
        # Keep this in sync with generate_input_shapes so DPA gets the explicit
        # padding state without a GPU-to-CPU sync.
        pad_between_seqs = kernel_backend == "FusedAttention" or fa_pad_between_seqs == "True"

    # set up environment variables and config
    if deterministic == "True":
        os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
    else:
        os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "1"
    fp8_bwd = fp8_bwd == "True" and dtype == "fp8"
    os.environ["NVTE_FP8_DPA_BWD"] = "1" if fp8_bwd else "0"
    fp8_dpa = fp8_dpa == "True" and dtype == "fp8"
    fp8_mha = fp8_mha == "True" and dtype == "fp8" and scaling_mode != "mxfp8"
    f16_O = dtype == "fp8" and scaling_mode in ["current", "mxfp8"] and f16_O == "True"
    os.environ["NVTE_DPA_FP8CS_O_in_F16"] = "1" if f16_O else "0"
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    if kernel_backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
        # Deep-copy: the module-level dict is shared across pool cases; the
        # THD branch below rewrites attn_mask_type in place, which would
        # otherwise leak into subsequent cases reusing the same model key.
        config = copy.deepcopy(model_configs_flash_attn[model])
    if kernel_backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"
        if model in model_configs_fused_attn:
            config = copy.deepcopy(model_configs_fused_attn[model])
        else:
            assert False, f"{model=} is not a known FusedAttention CP config!"
    assert config.attn_mask_type in [
        "causal",
        "no_mask",
    ], f"{config.attn_mask_type=} is not supported!"
    if qkv_format == "thd":
        if "causal" in config.attn_mask_type:
            config.attn_mask_type = "padding_causal"
        else:
            config.attn_mask_type = "padding"
    config.softcap = float(softcap)
    if no_load_balance_case:
        assert load_balancing_strategy is CPLoadBalancingStrategy.NO_LOAD_BALANCE
        assert cp_comm_type == "a2a" and qkv_format == "thd" and is_training
        config.batch_size = 3
        pad_between_seqs = no_load_balance_case == "padding_only"

    # set up distributed group
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    # When NVTE_CP_POOL_PG=1, the pool runner owns the lifecycle of the main
    # process group across many cases; here we only reuse it.
    _pool_managed_pg = os.getenv("NVTE_CP_POOL_PG", "0") == "1"
    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        device_count = torch.cuda.device_count()
        device = rank % device_count
        torch.cuda.set_device(device)
    logging.info(f"[Rank {rank}] Setup: world_size {world_size}")
    if not _pool_managed_pg:
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)

    # Set up communication group for CP. In pool mode, the pool worker has
    # already pre-created world-scoped and a2a+p2p sub-groups once and stashed
    # them in module-level pointers; we reuse those and the pool destroys them
    # at shutdown. In single-shot mode we create them per call and destroy in
    # the finally below.
    cp_comm_ranks = range(world_size)
    assert rank in cp_comm_ranks
    _reusing_pool_groups = _pool_managed_pg and _pool_cp_comm_group is not None
    cp_comm_group = None
    cp_comm_sub_groups: list = []
    if _reusing_pool_groups:
        cp_comm_group = _pool_cp_comm_group
        cp_comm_sub_groups = _pool_cp_comm_sub_groups if cp_comm_type == "a2a+p2p" else []
    else:
        cp_comm_group = dist.new_group(cp_comm_ranks, backend="nccl")
        if cp_comm_type == "a2a+p2p":
            assert world_size % 2 == 0, (
                "{cp_comm_type=} requires world_size % 2 = 0 as it assumes the a2a level has"
                " cp_size = 2."
            )
            cp_comm_sub_ranks = [range(i * 2, (i + 1) * 2) for i in range(world_size // 2)]
            cp_comm_sub_ranks += [range(i, world_size, 2) for i in range(2)]
            for sub_ranks in cp_comm_sub_ranks:
                sub_group = dist.new_group(sub_ranks, backend="nccl")
                if rank in sub_ranks:
                    cp_comm_sub_groups.append(sub_group)
    if dtype == "fp8":
        if scaling_mode == "delayed":
            fp8_recipe = DelayedScaling(fp8_dpa=fp8_dpa, fp8_mha=fp8_mha)
        if scaling_mode == "current":
            fp8_recipe = Float8CurrentScaling(fp8_dpa=fp8_dpa, fp8_mha=fp8_mha)
        if scaling_mode == "mxfp8":
            fp8_recipe = MXFP8BlockScaling(fp8_format=Format.E4M3, fp8_dpa=fp8_dpa, fp8_mha=fp8_mha)

    # instantiate attention module
    core_attn = DotProductAttention(
        config.num_heads,
        (config.head_dim_qk, config.head_dim_v),
        num_gqa_groups=config.num_gqa_groups,
        attention_dropout=config.dropout_p,
        qkv_format=qkv_format,
        attn_mask_type=config.attn_mask_type,
        window_size=config.window_size,
        softcap=config.softcap,
        softmax_type=config.softmax_type,
        return_max_logit=config.return_max_logit,
    ).cuda()
    if not is_training:
        core_attn.eval()
    if is_training and config.softmax_type != "vanilla":
        core_attn.softmax_offset.requires_grad = True

    # generate attention inputs
    (
        q_input_shape,
        k_input_shape,
        v_input_shape,
        attn_output_shape,
        cu_seqlens_q,
        cu_seqlens_kv,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
    ) = generate_input_shapes(
        qkv_format,
        config,
        world_size,
        kernel_backend,
        fa_pad_between_seqs,
        load_balancing_strategy,
        no_load_balance_case,
    )
    if no_load_balance_case:
        assert_no_load_balance_a2a_exchange(cp_comm_group, cu_seqlens_q_padded)
        if kernel_backend == "FusedAttention":
            assert_no_load_balance_mha(cp_comm_group, cu_seqlens_q, cu_seqlens_q_padded, dtype)
    q_orig = torch.clamp(torch.randn(q_input_shape, dtype=dtypes[dtype]), min=-1, max=1).cuda()
    k_orig = torch.clamp(torch.randn(k_input_shape, dtype=dtypes[dtype]), min=-1, max=1).cuda()
    v_orig = torch.clamp(torch.randn(v_input_shape, dtype=dtypes[dtype]), min=-1, max=1).cuda()
    dout_orig = torch.clamp(
        torch.randn(attn_output_shape, dtype=dtypes[dtype]), min=-1, max=1
    ).cuda()
    if scaling_mode == "delayed":
        qkv_quantizer = Float8Quantizer(
            fp8_dtype=DType.kFloat8E4M3,
            scale=torch.tensor([1], dtype=torch.float32).cuda(),
            amax=torch.tensor([0], dtype=torch.float32).cuda(),
        )
        dout_quantizer = Float8Quantizer(
            fp8_dtype=DType.kFloat8E5M2,
            scale=torch.tensor([1], dtype=torch.float32).cuda(),
            amax=torch.tensor([0], dtype=torch.float32).cuda(),
        )
    if scaling_mode == "current":
        qkv_quantizer = Float8CurrentScalingQuantizer(
            fp8_dtype=DType.kFloat8E4M3,
            device="cuda",
        )
        dout_quantizer = Float8CurrentScalingQuantizer(
            fp8_dtype=DType.kFloat8E5M2,
            device="cuda",
        )
    if scaling_mode == "mxfp8":
        qkv_quantizer = MXFP8Quantizer(
            fp8_dtype=DType.kFloat8E4M3,
            rowwise=True,
            columnwise=True,
        )
        qkv_quantizer.optimize_for_gemm = True
        qkv_quantizer.internal = False
        dout_quantizer = MXFP8Quantizer(
            fp8_dtype=DType.kFloat8E5M2,
            rowwise=True,
            columnwise=True,
        )
        dout_quantizer.optimize_for_gemm = True
        dout_quantizer.internal = False
    qkv_layout = "_".join([qkv_format] * 3)
    q, k, v, dout = [x.clone().detach() for x in [q_orig, k_orig, v_orig, dout_orig]]
    if fp8_mha:
        q, k, v, qkv_layout, _ = combine_and_quantize(qkv_layout, q, k, v, qkv_quantizer)
    for x in [q, k, v]:
        x.requires_grad = True

    if config.attn_bias_type not in ["no_bias", "alibi"]:
        bias_shape_map = {
            "1hss": (1, config.num_heads, config.max_seqlen_q, config.max_seqlen_kv),
            "11ss": (1, 1, config.max_seqlen_q, config.max_seqlen_kv),
            "b1ss": (config.batch_size, 1, config.max_seqlen_q, config.max_seqlen_kv),
            "bhss": (
                config.batch_size,
                config.num_heads,
                config.max_seqlen_q,
                config.max_seqlen_kv,
            ),
            "111s": (1, 1, 1, config.max_seqlen_kv),
        }
        attn_bias_shape = bias_shape_map.get(config.bias_shape)
        if attn_bias_shape is None:
            assert False, f"cuDNN does not support {config.bias_shape=}"
        bias = torch.randn(*attn_bias_shape, dtype=dtypes[dtype]).cuda()
        # cuDNN does not support dbias calculation for 111s as of cuDNN 9.18
        # TODO(KshitijLakhani): Set requires_grad to True for all shapes once 111s is supported
        bias.requires_grad = True if config.bias_shape != "111s" else False
    else:
        bias = None

    ############ run without CP ############
    logging.info(f"[Rank {rank}] Run without context parallelism")
    if dtype == "fp8":
        fp8_context = autocast(enabled=True, recipe=fp8_recipe, amax_reduction_group=cp_comm_group)
    else:
        fp8_context = nullcontext()
    max_logit = None
    with fp8_context:
        # q, k, v, out in FP8; dout in F16
        out = core_attn(
            q,
            k,
            v,
            core_attention_bias_type=config.attn_bias_type,
            core_attention_bias=bias,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            cu_seqlens_q_padded=cu_seqlens_q_padded,
            cu_seqlens_kv_padded=cu_seqlens_kv_padded,
            pad_between_seqs=pad_between_seqs,
            fp8_output=fp8_mha,
        )
        if config.return_max_logit:
            out, max_logit = out
        # Anchor the non-CP result to something other than another run of the same backend, before
        # it is reshaped for the CP comparison. See the function docstring for why this exists.
        assert_matches_independent_reference(
            out, q, k, v, config, qkv_format, dtype, core_attn.softmax_scale
        )
        if is_training:
            if fp8_bwd and fp8_mha:
                dout_fp8 = dout_quantizer(dout)
                out.backward(dout_fp8)
            else:
                out.backward(dout)
    if is_training:
        dq, dk, dv, dbias = q.grad, k.grad, v.grad, bias.grad if bias is not None else None
        d_softmax_offset = (
            core_attn.softmax_offset.grad if config.softmax_type != "vanilla" else None
        )
    else:
        dq, dk, dv, dbias = None, None, None, None
        d_softmax_offset = None

    if no_load_balance_case:
        assert_packed_attention_reference(
            out, q, k, v, dout, cu_seqlens_q, cu_seqlens_q_padded, config, dtype
        )

    ############ run with CP ############
    logging.info(f"[Rank {rank}] Run with context parallelism")

    # set up inputs
    q_, k_, v_, dout_, *rest = [
        x.clone().detach()
        for x in [q_orig, k_orig, v_orig, dout_orig] + ([] if bias is None else [bias])
    ]
    bias_ = rest[0] if len(rest) else None
    if qkv_format in ("bshd", "sbhd"):
        seq_dim = qkv_format.index("s")
        q_, k_, v_, dout_ = [
            x.view(
                *x.shape[:seq_dim],
                2 * world_size,
                x.shape[seq_dim] // (2 * world_size),
                *x.shape[(seq_dim + 1) :],
            )
            for x in [q_, k_, v_, dout_]
        ]
        seq_idx = torch.tensor([rank, 2 * world_size - rank - 1], device=q_.device)
        q_, k_, v_, dout_ = [x.index_select(seq_dim, seq_idx) for x in [q_, k_, v_, dout_]]
        q_, k_, v_, dout_ = [
            x.view(*x.shape[:seq_dim], -1, *x.shape[(seq_dim + 2) :]) for x in [q_, k_, v_, dout_]
        ]
    elif qkv_format == "thd":
        if load_balancing_strategy is CPLoadBalancingStrategy.NO_LOAD_BALANCE:
            # Independent oracle: physical slabs do not depend on document lengths.
            q_tokens, kv_tokens = q_.shape[0] // world_size, k_.shape[0] // world_size
            seq_idx_q = torch.arange(rank * q_tokens, (rank + 1) * q_tokens, device=q_.device)
            seq_idx_kv = torch.arange(rank * kv_tokens, (rank + 1) * kv_tokens, device=k_.device)
        else:
            seq_idx_q = get_thd_partitioned_indices(
                cu_seqlens_q_padded,
                q_.shape[0],
                world_size,
                rank,
                device=q_.device,
                load_balancing_strategy=load_balancing_strategy,
            )
            seq_idx_kv = get_thd_partitioned_indices(
                cu_seqlens_kv_padded,
                k_.shape[0],
                world_size,
                rank,
                device=k_.device,
                load_balancing_strategy=load_balancing_strategy,
            )
        q_, dout_ = [x.index_select(0, seq_idx_q) for x in [q_, dout_]]
        k_, v_ = [x.index_select(0, seq_idx_kv) for x in [k_, v_]]
    else:
        assert False, f"{qkv_format} is an unsupported qkv_format!"
    q_, k_, v_, dout_ = [x.contiguous() for x in [q_, k_, v_, dout_]]
    if scaling_mode == "delayed":
        qkv_quantizer.scale.fill_(1.0)
        qkv_quantizer.amax.fill_(0.0)
        dout_quantizer.scale.fill_(1.0)
        dout_quantizer.amax.fill_(0.0)
    if fp8_mha:
        q_, k_, v_, qkv_layout, _ = combine_and_quantize(qkv_layout, q_, k_, v_, qkv_quantizer)
    if is_training:
        q_, k_, v_ = [x.requires_grad_() for x in [q_, k_, v_]]
    if bias_ is not None:
        ndim = bias_.ndim
        seq_q_dim = ndim - 2
        if qkv_format == "thd":
            bias_seq_idx = seq_idx_q
        else:
            bias_seq_idx = seq_idx
        shape_before_seq = bias_.shape[:seq_q_dim]
        seq_q_size = bias_.shape[seq_q_dim]
        seq_kv_size = bias_.shape[-1]
        if seq_q_size == 1:
            # TODO(KshitijLakhani): Set to True always once cuDNN supports dbias for 111s
            bias_.requires_grad = False
            # Bias is broadcast, no need to partition along sequence dimension
            pass
        else:
            bias_ = bias_.view(
                *shape_before_seq, 2 * world_size, seq_q_size // (2 * world_size), seq_kv_size
            )
            bias_ = bias_.index_select(seq_q_dim, bias_seq_idx)
            bias_ = bias_.view(*shape_before_seq, -1, seq_kv_size)
            bias_.requires_grad = True
    # set up environment
    core_attn.set_context_parallel_group(
        cp_comm_sub_groups if cp_comm_type == "a2a+p2p" else cp_comm_group,
        cp_comm_ranks,
        torch.cuda.Stream(),
        cp_comm_type,
        load_balancing_strategy,
    )
    if is_training and config.softmax_type != "vanilla":
        core_attn.softmax_offset.grad.zero_()
    if dtype == "fp8":
        core_attn.fp8_initialized = False
        core_attn.fp8_meta_tensors_initialized = False
        fp8_context = autocast(enabled=True, recipe=fp8_recipe, amax_reduction_group=cp_comm_group)
    else:
        fp8_context = nullcontext()

    # run attention
    max_logit_ = None
    with fp8_context:
        # q, k, v, out in FP8; dout in F16
        out_ = core_attn(
            q_,
            k_,
            v_,
            core_attention_bias_type=config.attn_bias_type,
            core_attention_bias=bias_,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            cu_seqlens_q_padded=cu_seqlens_q_padded,
            cu_seqlens_kv_padded=cu_seqlens_kv_padded,
            pad_between_seqs=pad_between_seqs,
            fp8_output=fp8_mha,
        )
        if config.return_max_logit:
            out_, max_logit_ = out_
        if no_load_balance_case:
            # Backward must retain forward's distribution even after reconfiguration.
            core_attn.load_balancing_strategy = CPLoadBalancingStrategy.DUAL_CHUNK_SWAP
        if is_training:
            if fp8_bwd and fp8_mha:
                dout_fp8_ = dout_quantizer(dout_)
                out_.backward(dout_fp8_)
            else:
                out_.backward(dout_)
    if is_training:
        dq_, dk_, dv_, dbias_ = (
            q_.grad,
            k_.grad,
            v_.grad,
            bias_.grad if bias_ is not None else None,
        )
        d_softmax_offset_ = (
            core_attn.softmax_offset.grad.clone() if config.softmax_type != "vanilla" else None
        )
    else:
        dq_, dk_, dv_, dbias_ = None, None, None, None
        d_softmax_offset_ = None

    # get outputs
    tensors = [out, dq, dk, dv, dbias, out_, dq_, dk_, dv_, dbias_]
    names = ["out", "dq", "dk", "dv", "dbias", "out_cp", "dq_cp", "dk_cp", "dv_cp", "dbias_cp"]
    if fp8_mha:
        tensors_to_deq = [out, out_] if not fp8_bwd else tensors
        for i, tensor in enumerate(tensors_to_deq):
            # dbias/dbias_ could be None, so skip check for it
            if tensor is not None:
                tensors_to_deq[i] = tensor.dequantize()
        if not fp8_bwd:
            tensors[0], tensors[5] = tensors_to_deq
    for tensor, name in zip(tensors, names):
        # dbias/dbias_ could be None, so skip check for it
        if tensor is not None:
            assert torch.all(~torch.isnan(tensor)), f"{name} has nan values"
            assert torch.all(~torch.isinf(tensor)), f"{name} has inf values"
    out, dq, dk, dv, dbias, out_, dq_, dk_, dv_, dbias_ = tensors

    ############  compare results between CP and no-CP ############
    if qkv_format in ("bshd", "sbhd"):
        if is_training:
            dq, dk, dv, out = [
                x.view(
                    *x.shape[:seq_dim],
                    2 * world_size,
                    x.shape[seq_dim] // (2 * world_size),
                    *x.shape[(seq_dim + 1) :],
                )
                for x in [dq, dk, dv, out]
            ]
            dq, dk, dv, out = [x.index_select(seq_dim, seq_idx) for x in [dq, dk, dv, out]]
            dq_, dk_, dv_, out_ = [
                x.view(*x.shape[:seq_dim], 2, x.shape[seq_dim] // 2, *x.shape[(seq_dim + 1) :])
                for x in [dq_, dk_, dv_, out_]
            ]
            if dbias is not None and dbias_ is not None:
                ndim = dbias.ndim
                # Query seq is at dim -2
                seq_q_dim = ndim - 2
                shape_before_seq = dbias.shape[:seq_q_dim]
                seq_q_size = dbias.shape[seq_q_dim]
                seq_kv_size = dbias.shape[-1]
                # Reshape to split seq_q dimension
                dbias = dbias.view(
                    *shape_before_seq,
                    2 * world_size,
                    seq_q_size // (2 * world_size),
                    seq_kv_size,
                )
                # Index select on the newly created dimension (now at position seq_q_dim)
                dbias = dbias.index_select(seq_q_dim, seq_idx)
                dbias_ = dbias_.view(
                    *shape_before_seq, 2, dbias_.shape[seq_q_dim] // 2, seq_kv_size
                )
        else:
            # Forward-only: reshape only out/out_ for comparison
            out = out.view(
                *out.shape[:seq_dim],
                2 * world_size,
                out.shape[seq_dim] // (2 * world_size),
                *out.shape[(seq_dim + 1) :],
            )
            out = out.index_select(seq_dim, seq_idx)
            out_ = out_.view(
                *out_.shape[:seq_dim], 2, out_.shape[seq_dim] // 2, *out_.shape[(seq_dim + 1) :]
            )

    elif qkv_format == "thd":
        if is_training:
            dq, out = [x.index_select(0, seq_idx_q).contiguous() for x in [dq, out]]
            dk, dv = [x.index_select(0, seq_idx_kv).contiguous() for x in [dk, dv]]
            if load_balancing_strategy is CPLoadBalancingStrategy.DUAL_CHUNK_SWAP:
                cu_seqlens_q_padded = cu_seqlens_q_padded // world_size
                cu_seqlens_q = get_cu_seqlens_on_cp_rank(
                    cu_seqlens_q, cu_seqlens_q_padded, world_size, rank, True, True
                )
                cu_pads_q = cu_seqlens_q_padded - cu_seqlens_q
                num_pads_q = cu_pads_q[1:] - cu_pads_q[:-1]
                cu_seqlens_kv_padded = cu_seqlens_kv_padded // world_size
                cu_seqlens_kv = get_cu_seqlens_on_cp_rank(
                    cu_seqlens_kv, cu_seqlens_kv_padded, world_size, rank, True, True
                )
                num_pads_kv = (cu_seqlens_kv_padded - cu_seqlens_kv)[1:] - (
                    cu_seqlens_kv_padded - cu_seqlens_kv
                )[:-1]
                # FA3/FA4 kernels may leave physically padded rows unwritten. The CP
                # implementation must clean them after partial-output/gradient merges.
                if fa_pad_between_seqs == "True":
                    out_padding_mask = get_thd_padding_mask(
                        out_.shape[0], cu_seqlens_q, cu_seqlens_q_padded
                    )
                    nnz = torch.count_nonzero(out_[out_padding_mask]).item()
                    assert nnz == 0, (
                        f"out_ has {nnz} nonzero values in THD padding — "
                        "context_parallel.py should zero padding positions"
                    )

                    # out_ is a view inside the CP custom autograd Function, so in-place
                    # zeroing is blocked by PyTorch. Clone to break the view relationship.
                    out_ = out_.clone()
                    for x in [out, out_, dq]:
                        for b in range(config.batch_size):
                            x[
                                cu_seqlens_q_padded[b + 1]
                                - num_pads_q[b] : cu_seqlens_q_padded[b + 1]
                            ] = 0.0
                        x[cu_seqlens_q_padded[-1] :] = 0.0
                    for x in [dk, dv]:
                        for b in range(config.batch_size):
                            x[
                                cu_seqlens_kv_padded[b + 1]
                                - num_pads_kv[b] : cu_seqlens_kv_padded[b + 1]
                            ] = 0.0
                        x[cu_seqlens_kv_padded[-1] :] = 0.0
                    # Verify CP backward tensors have clean padding (pre-zeroed in context_parallel.py).
                    for xname, x, cu, np_ in [
                        ("dq_", dq_, cu_seqlens_q_padded, num_pads_q),
                        ("dk_", dk_, cu_seqlens_kv_padded, num_pads_kv),
                        ("dv_", dv_, cu_seqlens_kv_padded, num_pads_kv),
                    ]:
                        nnz = torch.count_nonzero(x[cu[-1] :]).item()
                        assert nnz == 0, (
                            f"{xname} has {nnz} nonzero values in tail padding — "
                            "context_parallel.py should zero padding positions"
                        )
                        for b in range(config.batch_size):
                            if np_[b] > 0:
                                nnz = torch.count_nonzero(x[cu[b + 1] - np_[b] : cu[b + 1]]).item()
                                assert nnz == 0, (
                                    f"{xname} has {nnz} nonzero values in batch {b} padding — "
                                    "context_parallel.py should zero padding positions"
                                )
        else:
            out = out.index_select(0, seq_idx_q).contiguous()
            out_ = out_

    atol, rtol, rmse_tol = get_tols(config, dtype)
    tensors_cp = [out_, dq_, dk_, dv_, dbias_, d_softmax_offset_, max_logit_]
    tensors_no_cp = [out, dq, dk, dv, dbias, d_softmax_offset, max_logit]
    names = ["out", "dq", "dk", "dv", "dbias", "d_softmax_offset", "max_logit"]
    names_cp = [x + "_cp" for x in names]
    names_no_cp = [x + "_no_cp" for x in names]
    is_fp8 = dtype == "fp8"
    for i, t in enumerate(tensors_no_cp):
        if t is not None:
            if "softmax_offset" not in names[i] and "max_logit" not in names[i]:
                if qkv_format == "bshd":
                    # Compare the two sequence chunks separately
                    # Compare dbias
                    if names[i] == "dbias":
                        # Compare the two chunks along dimension 2 (the split sequence dimension)
                        seq_q_dim_bias = 2
                        ndim_bias = t.ndim
                        slice_0 = [slice(None)] * ndim_bias
                        slice_0[seq_q_dim_bias] = 0
                        slice_1 = [slice(None)] * ndim_bias
                        slice_1[seq_q_dim_bias] = 1
                        compare_and_assert(
                            t[tuple(slice_0)],
                            tensors_cp[i][tuple(slice_0)],
                            names_no_cp[i],
                            names_cp[i],
                            atol,
                            rtol,
                            rmse_tol,
                            is_fp8,
                        )
                        compare_and_assert(
                            t[tuple(slice_1)],
                            tensors_cp[i][tuple(slice_1)],
                            names_no_cp[i],
                            names_cp[i],
                            atol,
                            rtol,
                            rmse_tol,
                            is_fp8,
                        )
                    # Compare Q/K/V/out
                    else:
                        #  Compare the two chunks along dimension 1 (the split sequence dimension)
                        compare_and_assert(
                            t[:, 0],
                            tensors_cp[i][:, 0],
                            names_no_cp[i],
                            names_cp[i],
                            atol,
                            rtol,
                            rmse_tol,
                            is_fp8,
                        )
                        compare_and_assert(
                            t[:, 1],
                            tensors_cp[i][:, 1],
                            names_no_cp[i],
                            names_cp[i],
                            atol,
                            rtol,
                            rmse_tol,
                            is_fp8,
                        )
                elif qkv_format == "sbhd":
                    # Compare the two sequence chunks separately
                    # Compare dbias (same as BSHD)
                    if names[i] == "dbias":
                        # Same as bshd: Compare the two chunks along dimension 2 (the split sequence dimension)
                        seq_q_dim_bias = 2
                        ndim_bias = t.ndim
                        slice_0 = [slice(None)] * ndim_bias
                        slice_0[seq_q_dim_bias] = 0
                        slice_1 = [slice(None)] * ndim_bias
                        slice_1[seq_q_dim_bias] = 1
                        compare_and_assert(
                            t[tuple(slice_0)],
                            tensors_cp[i][tuple(slice_0)],
                            names_no_cp[i],
                            names_cp[i],
                            atol,
                            rtol,
                            rmse_tol,
                            is_fp8,
                        )
                        compare_and_assert(
                            t[tuple(slice_1)],
                            tensors_cp[i][tuple(slice_1)],
                            names_no_cp[i],
                            names_cp[i],
                            atol,
                            rtol,
                            rmse_tol,
                            is_fp8,
                        )
                    # Compare Q/K/V/out
                    else:
                        #  Compare the two chunks along dimension 0 (the split sequence dimension)
                        compare_and_assert(
                            t[0],
                            tensors_cp[i][0],
                            names_no_cp[i],
                            names_cp[i],
                            atol,
                            rtol,
                            rmse_tol,
                            is_fp8,
                        )
                        compare_and_assert(
                            t[1],
                            tensors_cp[i][1],
                            names_no_cp[i],
                            names_cp[i],
                            atol,
                            rtol,
                            rmse_tol,
                            is_fp8,
                        )
                elif qkv_format == "thd":
                    if no_load_balance_case and torch.count_nonzero(t) == 0:
                        # A padding-only rank has zero reference range, so RMSE
                        # normalized by the tensor range is undefined.
                        torch.testing.assert_close(tensors_cp[i], t, rtol=0, atol=0)
                    else:
                        compare_and_assert(
                            t,
                            tensors_cp[i],
                            names_no_cp[i],
                            names_cp[i],
                            atol,
                            rtol,
                            rmse_tol,
                            is_fp8,
                        )
            else:
                compare_and_assert(
                    t, tensors_cp[i], names_no_cp[i], names_cp[i], atol, rtol, rmse_tol, is_fp8
                )
            logging.info(f"[Rank {rank}] CP vs no-CP: {names[i]} matches")

    # Teardown on the success path. Pool mode: cp_comm_group / cp_comm_sub_groups
    # point at pool-shared groups owned by the pool runner (which destroys them
    # at pool shutdown), and the main PG is also pool-owned — both branches
    # below are no-ops. Single-shot mode: destroy what we created here. If the
    # body above raises, we skip this — the subprocess dies at function return
    # and NCCL releases the communicators with the process.
    if not _reusing_pool_groups:
        if cp_comm_group is not None:
            try:
                dist.destroy_process_group(cp_comm_group)
            except Exception:
                pass
        for g in cp_comm_sub_groups:
            try:
                dist.destroy_process_group(g)
            except Exception:
                pass
    if not _pool_managed_pg:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def main(**kwargs):
    run_dpa_with_cp(**kwargs)


if __name__ == "__main__":
    kwargs = dict(arg.split("=") for arg in sys.argv[2:])
    main(**kwargs)
