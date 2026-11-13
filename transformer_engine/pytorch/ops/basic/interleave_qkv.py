# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Explicit feature ordering for packed equal-width Q/K/V projections."""

from __future__ import annotations

from ..op import BasicOperation


class InterleaveQKV(BasicOperation):
    """Map 2-D [tokens, Q/K/V-major features] to head-major Q/K/V triplets.

    Keep this feature permutation separate from A2A's rank-major exchange so
    both the unfused sequence and its fused replacement have explicit semantics.
    """

    def __init__(self, num_heads: int, head_dim: int) -> None:
        super().__init__()
        if num_heads <= 0 or head_dim <= 0:
            raise ValueError("QKV head count and head dimension must be positive")
        self.num_heads, self.head_dim = num_heads, head_dim

    def permute(self, x, *, inverse=False):
        """Permute feature rows of a projection weight or its gradient."""
        if x.ndim < 1 or x.shape[0] != 3 * self.num_heads * self.head_dim:
            raise ValueError("QKV feature rows must match three complete head groups")
        tail = x.shape[1:]
        first = (self.num_heads, 3) if inverse else (3, self.num_heads)
        return x.reshape(*first, self.head_dim, *tail).transpose(0, 1).reshape_as(x)

    def op_forward(self, ctx, input_, prev_op_grad_output_quantizer, next_op_input_quantizer):
        del ctx, prev_op_grad_output_quantizer, next_op_input_quantizer
        if input_.ndim != 2 or input_.shape[1] != 3 * self.num_heads * self.head_dim:
            raise ValueError("InterleaveQKV expects [tokens, 3 * num_heads * head_dim]")
        return (
            input_.reshape(-1, 3, self.num_heads, self.head_dim).transpose(1, 2).reshape_as(input_)
        )

    def op_backward(self, ctx, grad_output):
        del ctx
        dx = grad_output.reshape(-1, self.num_heads, 3, self.head_dim).transpose(1, 2)
        return dx.reshape_as(grad_output), ()
