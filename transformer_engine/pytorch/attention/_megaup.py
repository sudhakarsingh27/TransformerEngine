# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Opt-in PyTorch composition of the external MegaUP projection kernels.

This prototype deliberately keeps kernels in the optional external package. It
does not introduce a TE kernel backend or change parameter-gradient reduction.
"""

import os

import torch
import torch.distributed as dist

from transformer_engine.pytorch.quantization import FP8GlobalStateManager
from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage
from transformer_engine.pytorch.cpu_offload import is_cpu_offload_enabled
from transformer_engine.pytorch.constants import CPLoadBalancingStrategy
from transformer_engine.debug.pytorch.debug_state import TEDebugState
from .dot_product_attention import DotProductAttention
from .dot_product_attention.context_parallel import (
    get_seq_chunk_ids_for_reordering_before_attn,
    get_seq_chunk_ids_for_reordering_after_attn,
    thd_cp_rank_order_to_sequence_order,
    thd_sequence_order_to_cp_rank_order,
)
from .rope import apply_rotary_pos_emb


# Packed QKV and attention output use separate fixed-width communication arenas.
# Remember both fixed-capacity arenas so incompatible calls can fall back.
_RUNTIMES = {}


class _PeerProjection(torch.autograd.Function):
    """QKV projection and sequence-to-head exchange, including inverse backward."""

    @staticmethod
    def forward(ctx, x, weight, runtime, cu_seqlens, validate_group, contract):
        from mega_up import peer_write_gemm_bshd, peer_write_gemm_thd

        ctx.save_for_backward(x, weight, cu_seqlens)
        ctx.runtime, ctx.validate_group, ctx.contract = runtime, validate_group, contract
        peer = peer_write_gemm_thd if cu_seqlens is not None else peer_write_gemm_bshd
        # CuTe's DLPack export rejects requires_grad tensors even in no-grad forward.
        raw = peer(x.detach(), weight.detach(), runtime, validate_group=validate_group)
        if cu_seqlens is not None:
            return thd_cp_rank_order_to_sequence_order(raw, cu_seqlens, runtime.up).detach()
        return raw.detach()

    @staticmethod
    def backward(ctx, grad):
        from mega_up import prepare_headshard_pull_gemm_bshd, prepare_headshard_pull_gemm_thd

        x, weight, cu_seqlens = ctx.saved_tensors
        head_dim = x.shape[-1]
        if cu_seqlens is not None:
            grad = thd_sequence_order_to_cp_rank_order(
                grad.contiguous(), cu_seqlens, ctx.runtime.up
            )
            prepare = prepare_headshard_pull_gemm_thd
            dimensions = dict(tokens=x.shape[0])
        else:
            prepare = prepare_headshard_pull_gemm_bshd
            dimensions = dict(batch=x.shape[0], sequence=x.shape[1])
        pull = prepare(
            weight.detach().t().contiguous(),
            ctx.runtime,
            heads=weight.shape[0] // head_dim,
            head_dim=head_dim,
            workspace_epochs=1,
            validate_group=ctx.validate_group,
            **dimensions,
        )
        dx = pull(grad.detach().contiguous(), validate_group=ctx.validate_group)
        # The exchange materializes local full-width dQKV, needed for this rank's dW.
        dw = (
            pull.materialized_input.t() @ x.reshape(-1, weight.shape[1])
            if ctx.needs_input_grad[1]
            else None
        )
        if ctx.validate_group and ctx.contract is not None:
            ctx.contract["ready"] = True
        return dx, dw, None, None, None, None


class _PullProjection(torch.autograd.Function):
    """Head-to-sequence exchange and output projection, with graph-owned saved input."""

    @staticmethod
    def forward(ctx, x, weight, runtime, cu_seqlens, validate_group, contract):
        from mega_up import prepare_headshard_pull_gemm_bshd, prepare_headshard_pull_gemm_thd

        shard_heads, head_dim = x.shape[-2:]
        if cu_seqlens is not None:
            x = thd_sequence_order_to_cp_rank_order(x.contiguous(), cu_seqlens, runtime.up)
            prepare = prepare_headshard_pull_gemm_thd
            dimensions = dict(tokens=x.shape[0] // runtime.up)
        else:
            prepare = prepare_headshard_pull_gemm_bshd
            dimensions = dict(batch=x.shape[0], sequence=x.shape[1] // runtime.up)
        pull = prepare(
            weight.detach(),
            runtime,
            heads=shard_heads * runtime.up,
            head_dim=head_dim,
            workspace_epochs=1,
            validate_group=validate_group,
            **dimensions,
        )
        output = pull(x.detach().contiguous(), validate_group=validate_group)
        # A fresh single-use Pull owns both buffers; outstanding graphs retain them directly.
        ctx.save_for_backward(pull.materialized_input, weight, cu_seqlens)
        ctx.runtime, ctx.validate_group = runtime, validate_group
        # Return an owned alias without view metadata, allowing downstream in-place residuals.
        return output.detach()

    @staticmethod
    def backward(ctx, grad):
        from mega_up import peer_write_gemm_bshd, peer_write_gemm_thd

        materialized, weight, cu_seqlens = ctx.saved_tensors
        peer = peer_write_gemm_thd if cu_seqlens is not None else peer_write_gemm_bshd
        dx = peer(
            grad.detach().contiguous(),
            weight.detach().t().contiguous(),
            ctx.runtime,
            validate_group=ctx.validate_group,
        )
        if cu_seqlens is not None:
            dx = thd_cp_rank_order_to_sequence_order(dx, cu_seqlens, ctx.runtime.up)
        dw = (
            grad.reshape(-1, weight.shape[0]).t() @ materialized
            if ctx.needs_input_grad[1]
            else None
        )
        return dx, dw, None, None, None, None


def _thd_values(cu):
    """Read boundary values; keep this host synchronization explicit."""
    return tuple(cu.tolist())


def _thd_metadata(x, cp, cu_seqlens, max_seqlens):
    """Validate the no-padding-gap subset before any rank enters MegaUP barriers."""
    first, *others = cu_seqlens
    for cu in cu_seqlens:
        if not (
            isinstance(cu, torch.Tensor)
            and cu.device == x.device
            and cu.dtype == torch.int32
            and cu.ndim == 1
            and cu.is_contiguous()
        ):
            raise ValueError("THD boundaries must be contiguous device int32 vectors")
    bounds = _thd_values(first)
    if any(cu is not first and _thd_values(cu) != bounds for cu in others):
        raise ValueError("THD Q/KV logical and physical boundaries must match")
    lengths = tuple(b - a for a, b in zip(bounds, bounds[1:]))
    if not (
        bounds
        and bounds[0] == 0
        and lengths
        and bounds[-1] == cp * x.shape[0]
        and all(length > 0 and length % (2 * cp) == 0 for length in lengths)
    ):
        raise ValueError("THD requires positive, dual-chunk-aligned global lengths")
    maximum = max(lengths)
    maxima = tuple(maximum if value is None else value for value in max_seqlens)
    if any(type(value) is not int or value < maximum for value in maxima):
        raise ValueError("THD maximum sequence lengths must cover every sequence")
    return first, bounds, maxima


def _cached_thd_metadata(mha, x, cp, metadata, max_seqlens):
    """One-entry cache for trusted, normally version-tracked boundary tensors."""
    if not all(isinstance(cu, torch.Tensor) for cu in metadata) or any(
        maximum is not None and type(maximum) is not int for maximum in max_seqlens
    ):
        return _thd_metadata(x, cp, metadata, max_seqlens)
    try:
        key = (
            cp,
            tuple(x.shape),
            x.device,
            max_seqlens,
            tuple(
                (
                    cu._version,
                    tuple(cu.shape),
                    cu.stride(),
                    cu.dtype,
                    cu.device,
                    cu.data_ptr(),
                    cu.storage_offset(),
                )
                for cu in metadata
            ),
        )
    except RuntimeError:
        # Inference tensors have no version counter; reread rather than cache.
        return _thd_metadata(x, cp, metadata, max_seqlens)
    cached = getattr(mha, "_megaup_thd_metadata_cache", None)
    if (
        cached is not None
        and cached[0] == key
        and all(old is new for old, new in zip(cached[1], metadata))
    ):
        return cached[2]
    result = _thd_metadata(x, cp, metadata, max_seqlens)
    # Strong references prevent allocator address reuse from resembling a hit.
    object.__setattr__(mha, "_megaup_thd_metadata_cache", (key, metadata, result))
    return result


def forward_megaup(
    mha,
    x,
    *,
    eligible_call,
    rotary_pos_emb,
    attn_mask_type,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    cu_seqlens_q_padded=None,
    cu_seqlens_kv_padded=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
):
    """Return the specialized MHA result, or None for rank-uniform fallback."""
    trusted = os.getenv("NVTE_MHA_MEGAUP_TRUSTED", "0") == "1"
    contract = getattr(mha, "_megaup_validated_contract", None) if trusted else None
    group = mha.core_attention.cp_group
    if contract is not None and contract["group"] is not group:
        raise RuntimeError("Trusted MegaUP requires the initialized CP group")
    if not isinstance(group, dist.ProcessGroup) or dist.get_world_size(group) == 1:
        return None
    if x.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("NVTE_MHA_MEGAUP does not support CUDA graph capture")

    cp = dist.get_world_size(group)
    heads, head_dim = mha.num_attention_heads, mha.hidden_size_per_attention_head
    width = heads * head_dim
    hidden, groups = mha.proj.out_features, mha.num_gqa_groups
    qkv_width = (heads + 2 * groups) * head_dim
    thd = mha.qkv_format == "thd"
    cu_seqlens, bounds, maxima = None, None, (None, None)
    metadata_valid = True
    metadata = (cu_seqlens_q, cu_seqlens_kv, cu_seqlens_q_padded, cu_seqlens_kv_padded)
    if thd:
        try:
            if trusted:
                cu_seqlens, bounds, maxima = _cached_thd_metadata(
                    mha, x, cp, metadata, (max_seqlen_q, max_seqlen_kv)
                )
            else:
                cu_seqlens, bounds, maxima = _thd_metadata(
                    x, cp, metadata, (max_seqlen_q, max_seqlen_kv)
                )
        except (ValueError, TypeError, RuntimeError, IndexError):
            metadata_valid = False
    else:
        metadata_valid = all(value is None for value in (*metadata, max_seqlen_q, max_seqlen_kv))
    qkv = getattr(mha, "qkv", None)
    weight = getattr(qkv, "weight", None)
    out_weight = getattr(mha.proj, "weight", None)
    valid = (
        eligible_call
        and metadata_valid
        and mha.attention_type == "self"
        and not mha.input_layernorm
        and mha.tp_size == 1
        and not mha.sequence_parallel
        and (mha.qkv_weight_interleaved or groups == heads)
        and mha.q_norm is None
        and mha.k_norm is None
        and mha.softmax_type == "vanilla"
        and mha.core_attention.cp_comm_type == "a2a"
        and mha.core_attention.load_balancing_strategy is CPLoadBalancingStrategy.DUAL_CHUNK_SWAP
        and mha.core_attention.attention_dropout == 0
        and not mha.core_attention.return_max_logit
        and mha.qkv_format in ("bshd", "sbhd", "thd")
        and attn_mask_type in (("padding_causal", "padding") if thd else ("causal", "no_mask"))
        and not FP8GlobalStateManager.is_fp8_enabled()
        and not FP8GlobalStateManager.is_fp8_calibration()
        and not TEDebugState.debug_enabled
        and not is_cpu_offload_enabled()
        and not getattr(qkv, "use_bias", True)
        and not mha.proj.use_bias
        and not getattr(qkv, "fuse_wgrad_accumulation", True)
        and not mha.proj.fuse_wgrad_accumulation
        and not isinstance(x, QuantizedTensorStorage)
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.ndim == (2 if thd else 3)
        and x.shape[-1] == hidden
        and cp in (2, 4, 8)
        and heads % cp == 0
        and groups % cp == 0
        and hidden % head_dim == 0
        and hidden % 256 == 0
        and width % 256 == 0
        and qkv_width % 256 == 0
        and (qkv_width // cp) % 128 == 0
        and (width // cp) % 128 == 0
        and torch.cuda.get_device_capability(x.device)[0] == 10
        and isinstance(weight, torch.Tensor)
        and not isinstance(weight, QuantizedTensorStorage)
        and weight.dtype == torch.bfloat16
        and tuple(weight.shape) == (qkv_width, hidden)
        and weight.device == x.device
        and isinstance(out_weight, torch.Tensor)
        and not isinstance(out_weight, QuantizedTensorStorage)
        and out_weight.dtype == torch.bfloat16
        and tuple(out_weight.shape) == (hidden, width)
        and out_weight.device == x.device
        and out_weight.is_contiguous()
    )
    shape = tuple(x.shape)
    batch, sequence = (
        (1, shape[0])
        if thd and len(shape) == 2
        else (
            (shape[:2] if mha.qkv_format == "bshd" else shape[1::-1]) if len(shape) == 3 else (0, 0)
        )
    )
    runtimes = _RUNTIMES.get(group)
    local = getattr(mha, "_megaup_local_attention", None)
    valid = valid and sequence > 0 and (thd or sequence % 2 == 0) and batch > 0
    valid = valid and (
        local is None
        or (local.num_attention_heads == heads // cp and local.num_gqa_groups == groups // cp)
    )
    if runtimes is not None:
        valid = valid and all(
            runtime.hidden == channels
            and batch * sequence <= runtime.max_rows
            and runtime.device == x.device
            for runtime, channels in zip(runtimes, (qkv_width, width))
        )

    # A local unsupported input must not leave peers entering symmetric barriers.
    signature = (
        bool(valid),
        shape,
        heads,
        groups,
        hidden,
        head_dim,
        mha.qkv_format,
        mha.qkv_weight_interleaved,
        attn_mask_type,
        rotary_pos_emb is not None,
        bounds,
        maxima,
        None if runtimes is None else tuple((r.hidden, r.max_rows) for r in runtimes),
    )
    if contract is not None:
        # Trusted callers keep all ranks on one fixed contract. A local change
        # cannot safely reenter initialization collectives while peers launch payloads.
        if not valid or contract["signature"] != signature:
            raise RuntimeError("Trusted MegaUP contract changed; reset on all CP ranks first")
        validate_group = not contract["ready"]
    else:
        records = [None] * cp
        dist.all_gather_object(records, (signature, trusted), group=group)
        if not all(record == (signature, trusted) for record in records) or not valid:
            return None
        validate_group = True
        import_error = None
        try:
            from mega_up import initialize_runtime
        except ImportError as error:
            import_error = str(error)
        dist.all_gather_object(records, import_error, group=group)
        if any(error is not None for error in records):
            raise ImportError(f"NVTE_MHA_MEGAUP requires the optional mega_up package: {records}")
        if runtimes is None:
            runtimes = tuple(
                initialize_runtime(
                    group, hidden=channels, max_rows=batch * sequence, device=x.device
                )
                for channels in (qkv_width, width)
            )
            _RUNTIMES[group] = runtimes
        if trusted:
            # Publish only after setup succeeds; old graphs retain their own state.
            contract = {
                "signature": signature[:-1] + (tuple((r.hidden, r.max_rows) for r in runtimes),),
                "group": group,
                "ready": False,
            }
            object.__setattr__(mha, "_megaup_validated_contract", contract)
    qkv_runtime, output_runtime = runtimes

    # Each peer receives complete KV groups and their associated query heads. Reordering
    # non-interleaved weights stays in autograd and preserves parameter ownership.
    packed_weight = weight
    if not mha.qkv_weight_interleaved:
        packed_weight = weight.view(3, heads, head_dim, hidden).permute(1, 0, 2, 3)
    packed_weight = packed_weight.reshape(qkv_width, hidden).contiguous()
    if thd:
        raw = _PeerProjection.apply(
            x.contiguous().view(sequence, hidden // head_dim, head_dim),
            packed_weight,
            qkv_runtime,
            cu_seqlens,
            validate_group,
            contract,
        )
        routed = raw.view(cp * sequence, groups // cp, heads // groups + 2, head_dim)
    else:
        x_bshd = (x if mha.qkv_format == "bshd" else x.transpose(0, 1)).contiguous()
        x_bshd = x_bshd.view(batch, sequence, hidden // head_dim, head_dim)
        before = get_seq_chunk_ids_for_reordering_before_attn(cp, x.device)
        after = get_seq_chunk_ids_for_reordering_after_attn(cp, x.device)
        raw = _PeerProjection.apply(
            x_bshd, packed_weight, qkv_runtime, None, validate_group, contract
        )
        chunks = raw.view(batch, 2 * cp, sequence // 2, groups // cp, heads // groups + 2, head_dim)
        routed = chunks.index_select(1, before).reshape(
            batch, cp * sequence, groups // cp, heads // groups + 2, head_dim
        )
    query, key, value = routed.split((heads // groups, 1, 1), dim=-2)
    query = query.flatten(-3, -2)
    key, value = key.squeeze(-2), value.squeeze(-2)
    local_format = "thd" if thd else "bshd"
    if rotary_pos_emb is not None:
        q_pos, k_pos = (
            rotary_pos_emb if isinstance(rotary_pos_emb, tuple) else (rotary_pos_emb,) * 2
        )
        query = apply_rotary_pos_emb(
            query,
            q_pos,
            local_format,
            cu_seqlens=cu_seqlens,
            cp_size=1,
            cp_rank=0,
            fused=True,
            interleaved=mha.rotary_pos_interleaved,
        )
        key = apply_rotary_pos_emb(
            key,
            k_pos,
            local_format,
            cu_seqlens=cu_seqlens,
            cp_size=1,
            cp_rank=0,
            fused=True,
            interleaved=mha.rotary_pos_interleaved,
        )

    if local is None:
        local = DotProductAttention(
            heads // cp,
            head_dim,
            num_gqa_groups=groups // cp,
            attention_dropout=0.0,
            qkv_format="bshd",
            tp_size=1,
            layer_number=mha.layer_number,
            softmax_scale=mha.core_attention.softmax_scale,
        )
        # This BF16-only, parameterless execution cache is not model state.
        # Registering it as a child would add DPA's FP8 _extra_state keys to
        # existing checkpoints. Propagate train/eval explicitly below.
        object.__setattr__(mha, "_megaup_local_attention", local)
    local.cp_group = None
    local.train(mha.training)
    # Lazy construction must preserve the original module's policy even if
    # global deterministic settings changed after MHA was constructed.
    local.deterministic = mha.core_attention.deterministic
    local.flash_attention.deterministic = mha.core_attention.flash_attention.deterministic
    local.fused_attention.deterministic = mha.core_attention.fused_attention.deterministic
    thd_args = (
        dict(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens,
            cu_seqlens_kv_padded=cu_seqlens,
            max_seqlen_q=maxima[0],
            max_seqlen_kv=maxima[1],
            pad_between_seqs=False,
        )
        if thd
        else {}
    )
    attention = local(
        query,
        key,
        value,
        qkv_format=local_format,
        attn_mask_type=attn_mask_type,
        window_size=(-1, 0) if "causal" in attn_mask_type else (-1, -1),
        **thd_args,
    )
    if thd:
        output = _PullProjection.apply(
            attention.view(cp * sequence, heads // cp, head_dim),
            out_weight,
            output_runtime,
            cu_seqlens,
            validate_group,
            contract,
        ).reshape(sequence, hidden)
    else:
        chunks = attention.view(batch, 2 * cp, sequence // 2, heads // cp, head_dim)
        rank_order = chunks.index_select(1, after).reshape(
            batch, cp * sequence, heads // cp, head_dim
        )
        output = _PullProjection.apply(
            rank_order, out_weight, output_runtime, None, validate_group, contract
        ).reshape(batch, sequence, hidden)
    if mha.qkv_format == "sbhd":
        output = output.transpose(0, 1).contiguous()
    # Bias-free TE Linear still returns its empty bias tensor when requested.
    return (output, mha.proj.bias.to(dtype=x.dtype)) if mha.return_bias else output
