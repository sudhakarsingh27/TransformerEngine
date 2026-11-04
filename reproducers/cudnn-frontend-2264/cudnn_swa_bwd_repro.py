"""Compare full-causal and sliding-window call times for padding/non-padding masks.

Single-GPU BF16 BSHD discriminator, with a required FA3 reference dependency.
Reported CUDA-event times cover complete calls, not primary kernels alone.
Usage: python cudnn_swa_bwd_repro.py
"""

import torch
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.cpp_extensions.fused_attn import fused_attn_fwd, fused_attn_bwd
from flash_attn_3.flash_attn_interface import flash_attn_func

B, S, H, D = 4, 16384, 8, 128
H_KV = 8
ITERS, WARMUP = 5, 3
DEVICE = "cuda:0"
DTYPE = torch.bfloat16
BACKEND = tex.NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen

q = torch.randn(B, S, H, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
k = torch.randn(B, S, H_KV, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
v = torch.randn(B, S, H_KV, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
cu = torch.arange(0, (B + 1) * S, S, dtype=torch.int32, device=DEVICE)


def time_it(fn):
    torch.cuda.synchronize()
    t = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    t[0].record()
    for _ in range(ITERS):
        fn()
    t[1].record()
    torch.cuda.synchronize()
    return t[0].elapsed_time(t[1]) / ITERS


# ── cuDNN (FusedAttention) ───────────────────────────────────────────────────

def bench_cudnn(mask, window):
    kw = dict(
        qkv_layout="bshd_bshd_bshd",
        attn_mask_type=mask,
        window_size=window,
        cu_seqlens_q_padded=cu,
        cu_seqlens_kv_padded=cu,
    )
    for _ in range(WARMUP):
        out, aux, *_ = fused_attn_fwd(
            True, S, S, cu, cu, q, k, v, DTYPE, BACKEND, **kw
        )
        dout = torch.randn_like(out)
        fused_attn_bwd(S, S, cu, cu, q, k, v, out, dout, DTYPE, aux, BACKEND, **kw)

    fwd = time_it(
        lambda: fused_attn_fwd(True, S, S, cu, cu, q, k, v, DTYPE, BACKEND, **kw)
    )
    out, aux, *_ = fused_attn_fwd(True, S, S, cu, cu, q, k, v, DTYPE, BACKEND, **kw)
    dout = torch.randn_like(out)
    bwd = time_it(
        lambda: fused_attn_bwd(
            S, S, cu, cu, q, k, v, out, dout, DTYPE, aux, BACKEND, **kw
        )
    )
    return fwd, bwd


# ── FlashAttention v3 ────────────────────────────────────────────────────────

def bench_fa3(causal, window):
    for _ in range(WARMUP):
        out = flash_attn_func(q, k, v, causal=causal, window_size=window)
        out.backward(torch.randn_like(out))

    fwd = time_it(lambda: flash_attn_func(q, k, v, causal=causal, window_size=window))

    out = flash_attn_func(q, k, v, causal=causal, window_size=window)
    dout = torch.randn_like(out)
    bwd = time_it(lambda: out.backward(dout, retain_graph=True))
    return fwd, bwd


# ── Run ──────────────────────────────────────────────────────────────────────

print(f"B={B}  S={S}  H={H}  D={D}  cudnn={tex.get_cudnn_version()}")
print()

hdr = f"{'backend':<14} {'mask':<36} {'window':>10} {'fwd':>8} {'bwd':>8} {'bwd/fwd':>8}"
print(hdr)
print("-" * len(hdr))

for mask in ["causal_bottom_right", "padding_causal_bottom_right"]:
    for window in [(-1, 0), (512, 0)]:
        fwd, bwd = bench_cudnn(mask, window)
        print(
            f"{'cuDNN':<14} {mask:<36} {str(window):>10} "
            f"{fwd:>7.2f}ms {bwd:>7.2f}ms {bwd/fwd:>7.1f}x"
        )
    print()

for causal, label in [(True, "causal"), (False, "no_mask")]:
    for window in [(-1, -1), (512, 0)]:
        fwd, bwd = bench_fa3(causal, window)
        mask_str = f"{'causal' if causal else 'no_mask'}"
        print(
            f"{'FA3':<14} {mask_str:<36} {str(window):>10} "
            f"{fwd:>7.2f}ms {bwd:>7.2f}ms {bwd/fwd:>7.1f}x"
        )
    print()
