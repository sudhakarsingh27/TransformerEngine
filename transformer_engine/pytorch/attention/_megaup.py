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


# Packed QKV communicates 3F channels; the output projection communicates F.
# Remember both fixed-capacity arenas so incompatible calls can fall back.
_RUNTIMES = {}


def _thd_reorder(x, cu, up, layout, to_sequence):
    """Separate reorder; basic and dual inputs have different ownership contracts."""
    if layout is not None:
        index = layout.to_thd_index if to_sequence else layout.to_source_major_index
        return x.index_select(0, index)
    reorder = (
        thd_cp_rank_order_to_sequence_order if to_sequence else thd_sequence_order_to_cp_rank_order
    )
    return reorder(x, cu, up)


class _PeerProjection(torch.autograd.Function):
    """Linear followed by sequence-to-head exchange, with owned outputs."""

    @staticmethod
    def forward(ctx, x, weight, runtime, cu_seqlens, validate_group, contract):
        from mega_up import peer_write_gemm_bshd, peer_write_gemm_thd

        ctx.save_for_backward(x, weight, cu_seqlens)
        ctx.runtime = runtime
        ctx.validate_group = validate_group
        ctx.contract = contract
        ctx.experiment = contract.get("experiment") if contract is not None else None
        ctx.layout = contract.get("layout") if contract is not None else None
        ctx.indexed = ctx.experiment == "basic_indexed"
        prepared = contract.get("qkv_peer") if contract is not None else None
        # CuTe's DLPack export rejects requires_grad tensors even in no-grad forward.
        if prepared is not None:
            raw = prepared[2](x.detach(), validate_group=validate_group)
        elif cu_seqlens is not None:
            raw = peer_write_gemm_thd(
                x.detach(),
                weight.detach(),
                runtime,
                validate_group=validate_group,
                layout=ctx.layout if ctx.indexed else None,
            )
        else:
            raw = peer_write_gemm_bshd(
                x.detach(), weight.detach(), runtime, validate_group=validate_group
            )
        if cu_seqlens is not None and not ctx.indexed:
            return _thd_reorder(raw, cu_seqlens, runtime.up, ctx.layout, True)
        return raw

    @staticmethod
    def backward(ctx, grad):
        from mega_up import prepare_headshard_pull_gemm_bshd, prepare_headshard_pull_gemm_thd

        x, weight, cu_seqlens = ctx.saved_tensors
        head_dim = x.shape[-1]
        if cu_seqlens is not None:
            if not ctx.indexed:
                grad = _thd_reorder(
                    grad.contiguous(), cu_seqlens, ctx.runtime.up, ctx.layout, False
                )
            prepare = prepare_headshard_pull_gemm_thd
            dimensions = dict(tokens=x.shape[0], layout=ctx.layout if ctx.indexed else None)
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
        if ctx.experiment != "clone_free":
            dx = dx.clone()
        dw = pull.materialized_input.t() @ x.reshape(-1, weight.shape[1])
        # This is the last projection boundary of a complete training step.
        # Mark only this graph's state, never a newer contract on the module.
        if ctx.validate_group and ctx.contract is not None:
            ctx.contract["ready"] = True
        return dx, dw, None, None, None, None


class _PullProjection(torch.autograd.Function):
    """Head-to-sequence exchange followed by linear."""

    @staticmethod
    def forward(ctx, x, weight, runtime, cu_seqlens, validate_group, contract):
        from mega_up import prepare_headshard_pull_gemm_bshd, prepare_headshard_pull_gemm_thd

        ctx.experiment = contract.get("experiment") if contract is not None else None
        ctx.layout = contract.get("layout") if contract is not None else None
        ctx.indexed = ctx.experiment == "basic_indexed"
        shard_heads, head_dim = x.shape[-2:]
        if cu_seqlens is not None:
            if not ctx.indexed:
                x = _thd_reorder(x.contiguous(), cu_seqlens, runtime.up, ctx.layout, False)
            prepare = prepare_headshard_pull_gemm_thd
            dimensions = dict(
                tokens=x.shape[0] // runtime.up, layout=ctx.layout if ctx.indexed else None
            )
        else:
            prepare = prepare_headshard_pull_gemm_bshd
            dimensions = dict(batch=x.shape[0], sequence=x.shape[1] // runtime.up)
        cached, key, stream = None, None, None
        if ctx.experiment == "output_prepared":
            cached = contract.get("output_pull")
            stream = torch.cuda.current_stream(x.device)
            key = (
                id(runtime),
                weight.data_ptr(),
                tuple(weight.shape),
                tuple(weight.stride()),
                weight.storage_offset(),
                weight.dtype,
                weight.device,
                tuple(x.shape),
                stream.cuda_stream,
            )
        if cached is not None and cached[0] is weight and cached[1] == key:
            pull = cached[2]
        else:
            pull = prepare(
                weight.detach(),
                runtime,
                heads=shard_heads * runtime.up,
                head_dim=head_dim,
                workspace_epochs=8 if ctx.experiment == "output_prepared" else 1,
                validate_group=validate_group,
                **dimensions,
            )
            if ctx.experiment == "output_prepared":
                # Keep baseline clones and serialize reuse on the same CUDA stream:
                # the arena lease finishes before these integration clones do.
                # A stream change binds fresh buffers rather than sharing a slot.
                contract["output_pull"] = (weight, key, pull, stream)
        output = pull(x.detach().contiguous(), validate_group=validate_group)
        materialized = pull.materialized_input
        if ctx.experiment != "clone_free":
            output, materialized = output.clone(), materialized.clone()
        else:
            # Remove view metadata without copying owned storage, preserving
            # downstream in-place residual updates across this autograd boundary.
            output = output.detach()
        # Clone-free mode uses a fresh single-use Pull: these are owned allocations,
        # not arena aliases. Output-prepared mode always retains both clones.
        ctx.save_for_backward(materialized, weight, cu_seqlens)
        ctx.runtime = runtime
        ctx.validate_group = validate_group
        ctx.contract = contract
        return output

    @staticmethod
    def backward(ctx, grad):
        from mega_up import peer_write_gemm_bshd, peer_write_gemm_thd

        materialized, weight, cu_seqlens = ctx.saved_tensors
        peer = peer_write_gemm_thd if cu_seqlens is not None else peer_write_gemm_bshd
        # Rebind the saved weight so optimizer updates cannot leave a stale transpose.
        dx = peer(
            grad.detach().contiguous(),
            weight.detach().t().contiguous(),
            ctx.runtime,
            validate_group=ctx.validate_group,
            **({"layout": ctx.layout if ctx.indexed else None} if cu_seqlens is not None else {}),
        )
        if cu_seqlens is not None and not ctx.indexed:
            dx = _thd_reorder(dx, cu_seqlens, ctx.runtime.up, ctx.layout, True)
        dw = grad.reshape(-1, weight.shape[0]).t() @ materialized
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
    experiment = getattr(mha, "_megaup_experiment", None)
    if experiment not in (
        None,
        "clone_free",
        "qkv_prepared",
        "output_prepared",
        "basic_separate",
        "basic_indexed",
    ):
        raise RuntimeError("Unknown private MegaUP experiment")
    basic = experiment in ("basic_separate", "basic_indexed")
    if experiment is not None and not trusted:
        raise RuntimeError("Private MegaUP experiments require trusted fixed-contract mode")
    if basic and mha.qkv_format != "thd":
        raise RuntimeError("Basic MegaUP ownership requires THD inputs")
    contract = getattr(mha, "_megaup_validated_contract", None) if trusted else None
    group = mha.core_attention.cp_group
    if contract is not None and contract["group"] is not group:
        raise RuntimeError("Trusted MegaUP requires the initialized CP group")
    if not isinstance(group, dist.ProcessGroup) or dist.get_world_size(group) == 1:
        if basic:
            raise RuntimeError("Basic MegaUP ownership cannot fall back to native dual-chunk CP")
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
        experiment,
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
            if basic:
                raise RuntimeError(
                    "Basic MegaUP ownership cannot fall back to native dual-chunk CP"
                )
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
            layout = None
            if basic:
                from mega_up import prepare_thd_layout

                # Validated global boundaries describe equally split samples.
                # Build once from host values, not another CUDA-to-host read.
                layout = prepare_thd_layout(tuple(value // cp for value in bounds), runtimes[0])
            contract = {
                "experiment": experiment,
                "layout": layout,
                "signature": signature[:-1] + (tuple((r.hidden, r.max_rows) for r in runtimes),),
                "group": group,
                "ready": False,
            }
            object.__setattr__(mha, "_megaup_validated_contract", contract)
    qkv_runtime, output_runtime = runtimes

    # Each peer must receive complete Q/K/V triplets for its heads. Reordering
    # non-interleaved weights stays in autograd and preserves parameter ownership.
    packed_weight = weight
    if not mha.qkv_weight_interleaved:
        packed_weight = weight.view(3, heads, head_dim, width).permute(1, 0, 2, 3)
    packed_weight = packed_weight.reshape(3 * width, width).contiguous()
    if experiment == "qkv_prepared":
        # Cache only a binding to the original live parameter storage. In-place
        # optimizer updates remain visible; derived packed copies are never cached.
        cacheable = (
            mha.qkv_weight_interleaved
            and weight.is_contiguous()
            and packed_weight.data_ptr() == weight.data_ptr()
        )
        cached = contract.get("qkv_peer")
        key = (
            id(qkv_runtime),
            weight.data_ptr(),
            tuple(weight.shape),
            tuple(weight.stride()),
            weight.storage_offset(),
            weight.dtype,
            weight.device,
            batch,
            sequence,
            heads,
            head_dim,
        )
        if not cacheable:
            contract.pop("qkv_peer", None)
        elif cached is None or cached[0] is not weight or cached[1] != key:
            from mega_up import prepare_peer_write_gemm_bshd, prepare_peer_write_gemm_thd

            prepare = prepare_peer_write_gemm_thd if thd else prepare_peer_write_gemm_bshd
            dimensions = dict(tokens=sequence) if thd else dict(batch=batch, sequence=sequence)
            prepared = prepare(
                packed_weight.detach(),
                qkv_runtime,
                heads=heads,
                head_dim=head_dim,
                validate_group=validate_group,
                **dimensions,
            )
            contract["qkv_peer"] = (weight, key, prepared)
    if thd:
        raw = _PeerProjection.apply(
            x.contiguous().view(sequence, heads, head_dim),
            packed_weight,
            qkv_runtime,
            cu_seqlens,
            validate_group,
            contract,
        )
        routed = raw.view(cp * sequence, heads // cp, 3, head_dim)
    else:
        x_bshd = (x if mha.qkv_format == "bshd" else x.transpose(0, 1)).contiguous()
        x_bshd = x_bshd.view(batch, sequence, heads, head_dim)
        before = get_seq_chunk_ids_for_reordering_before_attn(cp, x.device)
        after = get_seq_chunk_ids_for_reordering_after_attn(cp, x.device)
        raw = _PeerProjection.apply(
            x_bshd, packed_weight, qkv_runtime, None, validate_group, contract
        )
        chunks = raw.view(batch, 2 * cp, sequence // 2, heads // cp, 3, head_dim)
        routed = chunks.index_select(1, before).reshape(
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
        output = _PullProjection.apply(
            attention.view(cp * sequence, heads // cp, head_dim),
            out_weight,
            output_runtime,
            cu_seqlens,
            validate_group,
            contract,
        ).reshape(sequence, width)
    else:
        chunks = attention.view(batch, 2 * cp, sequence // 2, heads // cp, head_dim)
        rank_order = chunks.index_select(1, after).reshape(
            batch, cp * sequence, heads // cp, head_dim
        )
        output = _PullProjection.apply(
            rank_order, out_weight, output_runtime, None, validate_group, contract
        ).reshape(batch, sequence, width)
    if mha.qkv_format == "sbhd":
        output = output.transpose(0, 1).contiguous()
    # Bias-free TE Linear still returns its empty bias tensor when requested.
    return (output, mha.proj.bias.to(dtype=x.dtype)) if mha.return_bias else output
