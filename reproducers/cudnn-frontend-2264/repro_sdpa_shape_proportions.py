#!/usr/bin/env python3
"""Time CP-free THD SDPA calls at the Q/KV shapes used by all-gather splits.

Examples:
  python repro_sdpa_shape_proportions.py --backend fused
  python repro_sdpa_shape_proportions.py --backend fa3 --batch 1 \
      --q-seqlen 131072 --kv-seqlens 131072 262144 393216 524288

CUDA-event results cover each complete backend call. The printed causal-pair
ratio is the full-causal reference, including when --window-left selects SWA.
To isolate primary kernels:
  nsys profile --trace=cuda,nvtx -o sdpa_shapes python repro_sdpa_shape_proportions.py
  nsys stats --report nvtx_kern_sum sdpa_shapes.nsys-rep
"""

import argparse
import gc

import torch


DTYPE = torch.bfloat16


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("fused", "fa3", "fa4"), default="fused")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--q-seqlen", type=int, default=8192)
    parser.add_argument(
        "--kv-seqlens", type=int, nargs="+", default=(8192, 16384, 24576, 32768)
    )
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--window-left", type=int, default=-1, help="-1 for full causal; nonnegative for sliding window")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cp-geometry", type=int, choices=(2, 8))
    args = parser.parse_args()
    dimensions = (
        args.batch,
        args.q_seqlen,
        *args.kv_seqlens,
        args.q_heads,
        args.kv_heads,
        args.head_dim,
        args.iters,
    )
    if any(value <= 0 for value in dimensions) or args.warmup < 0:
        parser.error("tensor dimensions and iterations must be positive")
    if args.window_left < -1:
        parser.error("--window-left must be -1 or nonnegative")
    if args.q_heads % args.kv_heads:
        parser.error("--q-heads must be divisible by --kv-heads")
    return args


def measure(name, fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.nvtx.range_push(name)
    start.record()
    for iteration in range(1, iters + 1):
        # Per-call attribution allows primary-kernel final-five extraction
        # without counting warmups or whole-call helper/allocator overhead.
        torch.cuda.nvtx.range_push(f"{name}.iteration_{iteration}")
        fn()
        torch.cuda.nvtx.range_pop()
    end.record()
    torch.cuda.nvtx.range_pop()
    end.synchronize()
    return start.elapsed_time(end) / iters


def make_inputs(args, kv_seqlen, requires_grad):
    device = torch.device("cuda", args.device)
    # THD all-gather geometry uses both rank-local query chunks and full
    # gathered K/V. Padded offsets and seqused select one logical split.
    q_storage = args.q_seqlen * 2 if args.cp_geometry else args.q_seqlen
    kv_storage = args.q_seqlen * 2 * args.cp_geometry if args.cp_geometry else kv_seqlen
    q = torch.randn(
        args.batch * q_storage,
        args.q_heads,
        args.head_dim,
        device=device,
        dtype=DTYPE,
        requires_grad=requires_grad,
    )
    k = torch.randn(
        args.batch * kv_storage,
        args.kv_heads,
        args.head_dim,
        device=device,
        dtype=DTYPE,
        requires_grad=requires_grad,
    )
    v = torch.randn_like(k, requires_grad=requires_grad)
    cu_q = (
        torch.arange(args.batch + 1, dtype=torch.int32, device=device) * args.q_seqlen
    )
    cu_kv = torch.arange(args.batch + 1, dtype=torch.int32, device=device) * kv_seqlen
    return q, k, v, cu_q, cu_kv


def padded_metadata(args, kv_seqlen, cu_q, cu_kv):
    if not args.cp_geometry:
        return cu_q, cu_kv
    split = kv_seqlen // args.q_seqlen - 1
    if kv_seqlen % args.q_seqlen or not 0 <= split < 2 * args.cp_geometry:
        raise ValueError("KV extent must select a DCS split")
    q_padded = torch.arange(args.batch + 1, device=cu_q.device, dtype=torch.int32) * (2 * args.q_seqlen)
    if split >= args.cp_geometry:
        q_padded[:-1] += args.q_seqlen
    kv_padded = torch.arange(args.batch + 1, device=cu_q.device, dtype=torch.int32) * (2 * args.cp_geometry * args.q_seqlen)
    return q_padded, kv_padded


def bench_fused(args, kv_seqlen, marker):
    import transformer_engine.pytorch  # Loads TE's common library.
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.cpp_extensions.fused_attn import (
        fused_attn_bwd,
        fused_attn_fwd,
    )

    q, k, v, cu_q, cu_kv = make_inputs(args, kv_seqlen, requires_grad=False)
    q_padded, kv_padded = padded_metadata(args, kv_seqlen, cu_q, cu_kv)
    backend = tex.NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen
    common = dict(
        cu_seqlens_q_padded=q_padded,
        cu_seqlens_kv_padded=kv_padded,
        attn_scale=args.head_dim ** -0.5,
        qkv_layout="thd_thd_thd",
        o_format="thd",
        attn_mask_type="padding_causal_bottom_right",
        window_size=(args.window_left, 0),
    )

    def forward():
        return fused_attn_fwd(
            True,
            args.q_seqlen,
            kv_seqlen,
            cu_q,
            cu_kv,
            q,
            k,
            v,
            DTYPE,
            backend,
            **common,
        )

    fwd_ms = measure(f"{marker}.fwd", forward, args.warmup, args.iters)
    out, aux, *_ = forward()
    dout = torch.randn_like(out)

    def backward():
        return fused_attn_bwd(
            args.q_seqlen,
            kv_seqlen,
            cu_q,
            cu_kv,
            q,
            k,
            v,
            out,
            dout,
            DTYPE,
            aux,
            backend,
            do_format="thd",
            dqkv_layout="thd_thd_thd",
            **common,
        )

    bwd_ms = measure(f"{marker}.bwd", backward, args.warmup, args.iters)
    return fwd_ms, bwd_ms


def bench_flash(args, kv_seqlen, marker):
    if args.cp_geometry:
        return bench_flash_physical(args, kv_seqlen, marker)
    if args.backend == "fa4":
        from flash_attn.cute.interface import flash_attn_varlen_func
    else:
        try:
            from flash_attn_interface import flash_attn_varlen_func
        except ImportError:
            from flash_attn_3.flash_attn_interface import flash_attn_varlen_func

    q, k, v, cu_q, cu_kv = make_inputs(args, kv_seqlen, requires_grad=True)

    def forward():
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_kv,
            args.q_seqlen,
            kv_seqlen,
            softmax_scale=args.head_dim ** -0.5,
            causal=True,
            # FA4 uses None for an unlimited window bound; FA3 uses -1.
            window_size=(None if args.window_left == -1 else args.window_left, 0)
            if args.backend == "fa4" else (args.window_left, 0),
        )

    fwd_ms = measure(f"{marker}.fwd", forward, args.warmup, args.iters)
    out = forward()
    dout = torch.randn_like(out)

    def backward():
        return torch.autograd.grad(out, (q, k, v), dout, retain_graph=True)

    bwd_ms = measure(f"{marker}.bwd", backward, args.warmup, args.iters)
    return fwd_ms, bwd_ms


def bench_flash_physical(args, kv_seqlen, marker):
    """Use TE raw Flash APIs with THD padded storage; private APIs are version-sensitive."""
    from transformer_engine.pytorch.attention.dot_product_attention import backends
    from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import get_fa_args

    use_fa4 = args.backend == "fa4"
    forward_api = getattr(backends, "_flash_attn_fwd_v4" if use_fa4 else "_flash_attn_fwd_v3")
    backward_api = getattr(backends, "_flash_attn_bwd_v4" if use_fa4 else "_flash_attn_bwd_v3")
    assert callable(forward_api) and callable(backward_api), "requested Flash API unavailable"
    q, k, v, cu_q, cu_kv = make_inputs(args, kv_seqlen, requires_grad=False)
    q_padded, kv_padded = padded_metadata(args, kv_seqlen, cu_q, cu_kv)
    metadata = dict(cu_seqlens_q=q_padded, cu_seqlens_kv=kv_padded,
                    max_seqlen_q=args.q_seqlen, max_seqlen_kv=kv_seqlen,
                    seqused_q=cu_q[1:] - cu_q[:-1], seqused_k=cu_kv[1:] - cu_kv[:-1])
    kwargs = dict(softmax_scale=args.head_dim ** -0.5, window_size_left=args.window_left, window_size_right=0)

    def forward():
        fa_args = get_fa_args(True, not use_fa4, "thd", **metadata, use_flash_attn_4=use_fa4)
        if use_fa4:
            return forward_api(q, k, v, **fa_args, causal=True, return_lse=True, **kwargs)
        return forward_api(q, k, v, *fa_args, causal=True, **kwargs)

    fwd_ms = measure(f"{marker}.fwd", forward, args.warmup, args.iters)
    out, lse, *_ = forward()
    dout = torch.randn_like(out)

    def backward():
        dq, dk, dv = [torch.zeros_like(x) for x in (q, k, v)]
        fa_args = get_fa_args(False, not use_fa4, "thd", **metadata,
                              dq=dq, dk=dk, dv=dv, use_flash_attn_4=use_fa4)
        if use_fa4:
            return backward_api(q, k, v, out, dout, lse, **fa_args,
                                causal=True, deterministic=False, **kwargs)
        return backward_api(dout, q, k, v, out, lse, *fa_args,
                            is_causal=True, deterministic=False, **kwargs)

    bwd_ms = measure(f"{marker}.bwd", backward, args.warmup, args.iters)
    return fwd_ms, bwd_ms


def valid_pairs(q_seqlen, kv_seqlen):
    diagonal = min(q_seqlen, kv_seqlen)
    return diagonal * (diagonal + 1) // 2 + q_seqlen * max(kv_seqlen - q_seqlen, 0)


def main():
    args = parse_args()
    torch.cuda.set_device(args.device)
    torch.manual_seed(1234)
    bench = bench_fused if args.backend == "fused" else bench_flash
    base_kv = args.kv_seqlens[0]
    base_rect = args.q_seqlen * base_kv
    base_valid = valid_pairs(args.q_seqlen, base_kv)
    base_fwd = base_bwd = None

    print(
        f"backend={args.backend} dtype=bf16 device={torch.cuda.get_device_name(args.device)} "
        f"B={args.batch} Hq={args.q_heads} Hkv={args.kv_heads} D={args.head_dim} "
        f"window_left={args.window_left}",
        flush=True,
    )
    header = (
        f"{'Sq':>8} {'Sk':>8} {'rect':>7} {'causal':>7} "
        f"{'fwd_ms':>10} {'fwd':>7} {'bwd_ms':>10} {'bwd':>7}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for kv_seqlen in args.kv_seqlens:
        marker = (
            f"sdpa_shape.{args.backend}.B{args.batch}.Sq{args.q_seqlen}.Sk{kv_seqlen}."
            f"Hq{args.q_heads}.Hkv{args.kv_heads}.D{args.head_dim}"
        )
        fwd_ms, bwd_ms = bench(args, kv_seqlen, marker)
        if base_fwd is None:
            base_fwd, base_bwd = fwd_ms, bwd_ms
        print(
            f"{args.q_seqlen:8d} {kv_seqlen:8d} "
            f"{args.q_seqlen * kv_seqlen / base_rect:7.3f} "
            f"{valid_pairs(args.q_seqlen, kv_seqlen) / base_valid:7.3f} "
            f"{fwd_ms:10.3f} {fwd_ms / base_fwd:7.3f} "
            f"{bwd_ms:10.3f} {bwd_ms / base_bwd:7.3f}",
            flush=True,
        )
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
