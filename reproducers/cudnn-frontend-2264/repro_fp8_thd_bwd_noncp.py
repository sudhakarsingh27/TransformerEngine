#!/usr/bin/env python3
"""Measure non-CP Transformer Engine FP8/BF16 backward memory for THD or BSHD."""

import argparse
import json
import os


Q_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128


def emit(kind, **fields):
    print(json.dumps({"type": kind, **fields}, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-sequences", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--mode", choices=("fp8", "bf16"), default="fp8")
    parser.add_argument("--qkv-format", choices=("thd", "bshd"), default="thd")
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def memory(torch, device, phase):
    free, total = torch.cuda.mem_get_info(device)
    gib = 1024**3
    emit(
        "memory",
        phase=phase,
        allocated_gib=torch.cuda.memory_allocated(device) / gib,
        reserved_gib=torch.cuda.memory_reserved(device) / gib,
        peak_allocated_gib=torch.cuda.max_memory_allocated(device) / gib,
        peak_reserved_gib=torch.cuda.max_memory_reserved(device) / gib,
        free_gib=free / gib,
        total_gib=total / gib,
    )


def main():
    args = parse_args()
    if args.num_sequences <= 0 or args.sequence_length <= 0 or args.device < 0:
        raise ValueError("sequence count/length must be positive and device non-negative")

    # These are read while Transformer Engine attention modules are imported.
    os.environ["NVTE_DPA_FP8CS_O_in_F16"] = "1"
    use_fp8 = args.mode == "fp8"
    os.environ["NVTE_FP8_DPA_BWD"] = "1" if use_fp8 else "0"
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "1"
    os.environ["NVTE_UNFUSED_ATTN"] = "0"

    import torch
    from transformer_engine.common import recipe
    from transformer_engine.pytorch import DotProductAttention, autocast

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    dtype = torch.bfloat16
    tokens = args.num_sequences * args.sequence_length

    # Dense BSHD provides a same-storage-volume format control.
    if args.qkv_format == "thd":
        q_shape = (tokens, Q_HEADS, HEAD_DIM)
        kv_shape = (tokens, KV_HEADS, HEAD_DIM)
        out_shape = (tokens, Q_HEADS * HEAD_DIM)
        offsets = torch.arange(args.num_sequences + 1, dtype=torch.int32, device=device)
        cu_seqlens = offsets * args.sequence_length
    else:
        q_shape = (args.num_sequences, args.sequence_length, Q_HEADS, HEAD_DIM)
        kv_shape = (args.num_sequences, args.sequence_length, KV_HEADS, HEAD_DIM)
        out_shape = (args.num_sequences, args.sequence_length, Q_HEADS * HEAD_DIM)
        cu_seqlens = None
    q = torch.zeros(q_shape, dtype=dtype, device=device, requires_grad=True)
    k = torch.zeros(kv_shape, dtype=dtype, device=device, requires_grad=True)
    v = torch.zeros_like(k, requires_grad=True)
    dout = torch.zeros(out_shape, dtype=dtype, device=device)

    dpa = DotProductAttention(
        Q_HEADS,
        HEAD_DIM,
        num_gqa_groups=KV_HEADS,
        attention_dropout=0.0,
        sequence_parallel=False,
        tp_size=1,
        get_rng_state_tracker=None,
        tp_group=None,
        layer_number=1,
        attention_type="self",
        qkv_format=args.qkv_format,
    ).to(dtype=dtype, device=device)
    fp8_recipe = (
        recipe.Float8CurrentScaling(fp8_format=recipe.Format.HYBRID, fp8_dpa=True)
        if use_fp8
        else None
    )

    emit(
        "config",
        context_parallel=False,
        mode=args.mode,
        qkv_format=args.qkv_format,
        num_sequences=args.num_sequences,
        sequence_length=args.sequence_length,
        tokens=tokens,
        q_shape=list(q.shape),
        kv_shape=list(k.shape),
        q_heads=Q_HEADS,
        kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        gpu=torch.cuda.get_device_name(device),
    )
    torch.cuda.reset_peak_memory_stats(device)
    memory(torch, device, "inputs_ready")

    emit("forward", status="start")
    try:
        with autocast(enabled=use_fp8, recipe=fp8_recipe):
            out = dpa(
                q,
                k,
                v,
                qkv_format=args.qkv_format,
                window_size=(-1, 0),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=args.sequence_length,
                max_seqlen_kv=args.sequence_length,
                attn_mask_type="padding_causal" if cu_seqlens is not None else "causal",
                checkpoint_core_attention=False,
                core_attention_bias_type="no_bias",
                fp8_output=use_fp8,
            )
        torch.cuda.synchronize(device)
    except torch.OutOfMemoryError as error:
        emit("forward", status="oom", error=str(error))
        raise
    memory(torch, device, "after_forward")
    emit("forward", status="ok", output_shape=list(out.shape))

    torch.cuda.reset_peak_memory_stats(device)
    emit("backward", status="start")
    try:
        out.backward(dout)
        torch.cuda.synchronize(device)
    except torch.OutOfMemoryError as error:
        emit("backward", status="oom", error=str(error))
        raise
    memory(torch, device, "after_backward")
    emit("backward", status="ok", gradient_shapes=[list(x.grad.shape) for x in (q, k, v)])


if __name__ == "__main__":
    main()
