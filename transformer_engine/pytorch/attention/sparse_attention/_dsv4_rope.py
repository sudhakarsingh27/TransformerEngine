# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""DSv4's trailing-channel, interleaved RoPE for full-sequence attention."""

import torch

from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb


def rotary_angles(seq, width, theta, device):
    """Return FP32 interleaved angles accepted by TE's fused RoPE kernel."""
    inv_freq = theta ** (-torch.arange(0, width, 2, device=device, dtype=torch.float32) / width)
    return torch.outer(
        torch.arange(seq, device=device, dtype=torch.float32), inv_freq
    ).repeat_interleave(2, dim=-1)[:, None, None, :]


def rotary_embeddings(seq, ratio, width, theta, device):
    """Return token and compressed-window (cos, sin) pairs in FP32."""
    angles = rotary_angles(seq, width, theta, device)
    cos = angles.cos().transpose(0, 1)
    sin = angles.sin().transpose(0, 1)
    window_positions = torch.arange(seq // ratio, device=device) * ratio
    return (cos, sin), (cos[:, window_positions], sin[:, window_positions])


class _DSv4RotaryEmbedding(torch.nn.Module):
    """Cache shared FP32 token frequencies; compressed positions are window starts."""

    def __init__(self, ratio, width, theta, device, max_seqlen=None):
        super().__init__()
        self.ratio, self.width, self.theta = ratio, width, theta
        cos, sin = (None, None)
        angles = None
        if max_seqlen is not None:
            (cos, sin), _ = rotary_embeddings(max_seqlen, ratio, width, theta, device)
            angles = rotary_angles(max_seqlen, width, theta, device)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("angles_cache", angles, persistent=False)

    def _apply(self, fn):
        cached = {name: getattr(self, name) for name in ("cos", "sin", "angles_cache")}
        super()._apply(fn)
        # Module.to(bfloat16) must not round reusable frequencies or angles.
        for name, value in cached.items():
            if value is not None:
                setattr(self, name, value.to(device=getattr(self, name).device))
        return self

    def forward(self, seq, device):
        """Return cached token and compressed-window frequency pairs."""
        cached_cos = getattr(self, "cos")
        if cached_cos is None or cached_cos.shape[1] < seq:
            (self.cos, self.sin), _ = rotary_embeddings(
                seq, self.ratio, self.width, self.theta, device
            )
        n_comp = seq // self.ratio
        return (self.cos[:, :seq], self.sin[:, :seq]), (
            self.cos[:, : n_comp * self.ratio : self.ratio],
            self.sin[:, : n_comp * self.ratio : self.ratio],
        )

    def angles(self, seq, device):
        """Return token and window-start angles for the MHA/GQA fused RoPE kernel."""
        cached_angles = getattr(self, "angles_cache")
        if cached_angles is None or cached_angles.shape[0] < seq:
            self.angles_cache = rotary_angles(seq, self.width, self.theta, device)
        n_comp = seq // self.ratio
        return (
            self.angles_cache[:seq],
            self.angles_cache[: n_comp * self.ratio : self.ratio].contiguous(),
        )


def apply_rotary(x, frequencies, *, fused=False):
    """Rotate DSv4's trailing interleaved channels using angles or cos/sin."""
    width = frequencies.shape[-1] if fused else frequencies[0].shape[-1]
    tail = x[..., -width:]
    if fused:
        rotated = apply_rotary_pos_emb(
            tail, frequencies, tensor_format="bshd", interleaved=True, fused=True
        )
    else:
        cos, sin = frequencies
        pair = torch.stack((-tail[..., 1::2], tail[..., 0::2]), -1).flatten(-2)
        rotated = (tail.float() * cos + pair.float() * sin).to(x.dtype)
    return torch.cat((x[..., :-width], rotated), -1)
