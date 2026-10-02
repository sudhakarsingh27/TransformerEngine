# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Reuse MHA/GQA's fused RoPE kernel with DSv4 and other angle schedules."""

import math

import pytest
import torch

from transformer_engine.pytorch.attention.sparse_attention._dsv4_rope import (
    _DSv4RotaryEmbedding,
    apply_rotary,
)


def _angles(variant, seq, width, device):
    """Example angle policies; the test verifies the rotation, not model policy."""
    base = 40000.0
    pair = torch.arange(0, width, 2, device=device, dtype=torch.float32) / width
    inv = base**-pair
    if variant == "linear":
        inv /= 4
    elif variant == "dynamic_ntk":
        inv = (base * (4 * seq / 128 - 3) ** (width / (width - 2))) ** -pair
    elif variant == "longrope":
        inv /= torch.linspace(1, 4, width // 2, device=device)
    elif variant == "llama3":
        wavelength = 2 * math.pi / inv
        slow, fast = 256.0, 64.0
        scaled = inv / 4
        mix = ((256 / wavelength - 1) / 3).clamp(0, 1)
        inv = torch.where(
            wavelength < fast,
            inv,
            torch.where(wavelength > slow, scaled, mix * inv + (1 - mix) * scaled),
        )
    elif variant == "yarn":
        # Megatron's DSv4 YaRN blend, with unit amplitude (mscale=1).
        def correction(rotations):
            return width * math.log(4096 / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(correction(32)), 0)
        high = min(math.ceil(correction(1)), width - 1)
        ramp = ((torch.arange(width // 2, device=device) - low) / (high - low)).clamp(0, 1)
        inv = inv * (1 - ramp) + inv / 40 * ramp
    return (
        torch.outer(torch.arange(seq, device=device, dtype=torch.float32), inv)
        .repeat_interleave(2, dim=-1)[:, None, None, :]
        .contiguous()
    )


@pytest.mark.parametrize(
    "variant", ["plain", "linear", "dynamic_ntk", "longrope", "llama3", "yarn"]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_rope_accepts_variant_angles(variant, dtype):
    """The CUDA rotation and gradient match PyTorch for varied angle schedules."""
    if not torch.cuda.is_available():
        pytest.skip("TE fused RoPE requires CUDA")
    seq, width = 256, 64
    angles = _angles(variant, seq, width, "cuda")
    cos, sin = angles.cos().transpose(0, 1), angles.sin().transpose(0, 1)
    original = torch.randn(2, seq, 3, 128, device="cuda", dtype=dtype)
    fused_input = original.detach().requires_grad_()
    reference_input = original.detach().requires_grad_()
    fused = apply_rotary(fused_input, angles, fused=True)
    reference = apply_rotary(reference_input, (cos, sin))
    grad = torch.randn_like(fused)
    fused_grad = torch.autograd.grad(fused, fused_input, grad)[0]
    reference_grad = torch.autograd.grad(reference, reference_input, grad)[0]
    tolerance = 1e-2 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(fused, reference, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(fused_grad, reference_grad, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("ratio", [4, 128])
def test_dsv4_fused_rope_uses_window_starts_and_inverse(ratio):
    """Compressed rows rotate at window starts and inverse rotation recovers input."""
    if not torch.cuda.is_available():
        pytest.skip("TE fused RoPE requires CUDA")
    rope = _DSv4RotaryEmbedding(ratio, 64, 160000.0, "cuda", max_seqlen=256).to(
        dtype=torch.bfloat16
    )
    token, compressed = rope.angles(256, "cuda")
    _, compressed_pair = rope(256, "cuda")
    assert token.dtype == compressed.dtype == torch.float32
    assert rope.cos.dtype == compressed_pair[0].dtype == torch.float32
    assert compressed.is_contiguous()
    torch.testing.assert_close(compressed, token[::ratio].contiguous())
    x = torch.randn(1, 256 // ratio, 1, 128, device="cuda", dtype=torch.bfloat16)
    rotated = apply_rotary(x, compressed, fused=True)
    torch.testing.assert_close(rotated, apply_rotary(x, compressed_pair), atol=0.01, rtol=0.01)
    restored = apply_rotary(rotated, -compressed, fused=True)
    torch.testing.assert_close(restored, x, atol=0.02, rtol=0.02)
