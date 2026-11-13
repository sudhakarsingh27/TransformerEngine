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
    thd_cp_rank_order_to_sequence_order,
    thd_sequence_order_to_cp_rank_order,
)
from .rope import apply_rotary_pos_emb


# Packed QKV communicates 3F channels; the output projection communicates F.
# Remember both fixed-capacity arenas so incompatible calls can fall back.
_RUNTIMES = {}


class _THDReorder(torch.autograd.Function):
    """Reuse TE's THD permutation with its explicit inverse in autograd."""

    @staticmethod
    def forward(ctx, x, cu_seqlens, cp, to_sequence):
        ctx.save_for_backward(cu_seqlens)
        ctx.cp, ctx.to_sequence = cp, to_sequence
        reorder = (
            thd_cp_rank_order_to_sequence_order
            if to_sequence
            else thd_sequence_order_to_cp_rank_order
        )
        return reorder(x.contiguous(), cu_seqlens, cp)

    @staticmethod
    def backward(ctx, grad):
        (cu_seqlens,) = ctx.saved_tensors
        reorder = (
            thd_sequence_order_to_cp_rank_order
            if ctx.to_sequence
            else thd_cp_rank_order_to_sequence_order
        )
        return reorder(grad.contiguous(), cu_seqlens, ctx.cp), None, None, None


def _projection_sequences(mha, weight, out_weight, runtimes, batch, sequence):
    """Cache execution wrappers without registering aliases in the model checkpoint."""
    from transformer_engine.pytorch.ops import (
        BasicLinear,
        ContextParallelAllToAll,
        InterleaveQKV,
        Sequential,
    )

    cp = runtimes[0].up
    heads, d = mha.num_attention_heads, mha.hidden_size_per_attention_head
    key = (
        tuple(
            (id(w), w.data_ptr(), tuple(w.shape), w.stride(), w.dtype, w.device)
            for w in (weight, out_weight)
        ),
        id(mha.core_attention.cp_group),
        tuple(id(r) for r in runtimes),
        heads,
        d,
        cp,
        mha.qkv_format,
        mha.qkv_weight_interleaved,
        batch,
        sequence,
    )
    cached = getattr(mha, "_megaup_projection_sequences", None)
    if cached is not None and cached[0] == key:
        return cached[1:]
    linears = []
    for parameter in (weight, out_weight):
        n, k = parameter.shape
        linear = BasicLinear(k, n, device="meta", dtype=parameter.dtype)
        linear.weight = (
            parameter  # Preserve the original Parameter and avoid an initialization allocation.
        )
        linears.append(linear)
    exchanges = [
        ContextParallelAllToAll(
            mha.core_attention.cp_group, direction=direction, megaup_runtime=runtime, head_dim=d
        )
        for direction, runtime in zip(("sequence_to_head", "head_to_sequence"), runtimes)
    ]
    qkv_ops = [linears[0]]
    if not mha.qkv_weight_interleaved:
        qkv_ops.append(InterleaveQKV(heads, d))
    qkv = Sequential(*qkv_ops, exchanges[0])
    output = Sequential(exchanges[1], linears[1])
    to_sequence = to_rank = None
    if mha.qkv_format != "thd":
        # Combine batch/rank transpose and dual-chunk restoration in one data copy.
        rows = torch.arange(cp * batch * sequence, device=weight.device).view(
            cp, batch, 2, sequence // 2
        )
        rows = rows.permute(1, 0, 2, 3).reshape(batch, 2 * cp, sequence // 2)
        before = get_seq_chunk_ids_for_reordering_before_attn(cp, weight.device)
        to_sequence = rows.index_select(1, before).flatten()
        to_rank = torch.argsort(to_sequence)
    # Strong module references retain old weights while outstanding graphs finish.
    cached = (key, qkv, output, to_sequence, to_rank)
    object.__setattr__(mha, "_megaup_projection_sequences", cached)
    return cached[1:]


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
        and mha.num_gqa_groups == heads
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
        and x.shape[-1] == width
        and cp in (2, 4, 8)
        and heads % cp == 0
        and width % 256 == 0
        and (width // cp) % 128 == 0
        and torch.cuda.get_device_capability(x.device)[0] == 10
        and isinstance(weight, torch.Tensor)
        and not isinstance(weight, QuantizedTensorStorage)
        and weight.dtype == torch.bfloat16
        and tuple(weight.shape) == (3 * width, width)
        and weight.device == x.device
        and isinstance(out_weight, torch.Tensor)
        and not isinstance(out_weight, QuantizedTensorStorage)
        and out_weight.dtype == torch.bfloat16
        and tuple(out_weight.shape) == (width, width)
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
    valid = valid and (local is None or local.num_attention_heads == heads // cp)
    if runtimes is not None:
        valid = valid and all(
            runtime.hidden == channels
            and batch * sequence <= runtime.max_rows
            and runtime.device == x.device
            for runtime, channels in zip(runtimes, (3 * width, width))
        )

    # A local unsupported input must not leave peers entering symmetric barriers.
    signature = (
        bool(valid),
        shape,
        heads,
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
                for channels in (3 * width, width)
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
    qkv_sequence, output_sequence, to_sequence, to_rank = _projection_sequences(
        mha, weight, out_weight, runtimes, batch, sequence
    )
    qkv_kwargs = {qkv_sequence[-1]: dict(validate_group=validate_group, validation_state=contract)}
    if thd:
        raw = qkv_sequence(x.contiguous().view(sequence, width), op_kwargs=qkv_kwargs)
        routed = _THDReorder.apply(
            raw.view(cp * sequence, 3 * heads // cp, head_dim), cu_seqlens, cp, True
        ).view(cp * sequence, heads // cp, 3, head_dim)
    else:
        x_bshd = (x if mha.qkv_format == "bshd" else x.transpose(0, 1)).contiguous()
        raw = qkv_sequence(x_bshd.view(batch * sequence, width), op_kwargs=qkv_kwargs)
        routed = raw.index_select(0, to_sequence).view(
            batch, cp * sequence, heads // cp, 3, head_dim
        )
    query, key, value = routed.unbind(-2)
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
            num_gqa_groups=heads // cp,
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
        rank_order = _THDReorder.apply(
            attention.view(cp * sequence, heads // cp, head_dim), cu_seqlens, cp, False
        ).view(cp * sequence, width // cp)
    else:
        rank_order = attention.reshape(batch * cp * sequence, width // cp).index_select(0, to_rank)
    output = output_sequence(
        rank_order, op_kwargs={output_sequence[0]: dict(validate_group=validate_group)}
    ).view((sequence, width) if thd else (batch, sequence, width))
    if mha.qkv_format == "sbhd":
        output = output.transpose(0, 1).contiguous()
    # Bias-free TE Linear still returns its empty bias tensor when requested.
    return (output, mha.proj.bias.to(dtype=x.dtype)) if mha.return_bias else output
