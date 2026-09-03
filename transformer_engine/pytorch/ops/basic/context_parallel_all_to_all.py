# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for context-parallel all-to-all."""

from __future__ import annotations
from typing import Literal, Optional

import torch

from .._common import maybe_dequantize
from ..op import BasicOperation, OperationContext
from ...tensor import Quantizer


class ContextParallelAllToAll(BasicOperation):
    """Exchange sequence and feature shards for Ulysses context parallelism.

    This is the unfused semantic reference for the communication adjacent to
    projection GEMMs. Inputs use a deliberately small 2-D contract so that a
    future fused operation can replace either ``BasicLinear ->
    ContextParallelAllToAll`` or ``ContextParallelAllToAll -> BasicLinear``
    without inheriting attention-layout policy.

    ``"sequence_to_head"`` maps ``[local_tokens, full_features]`` to
    ``[global_tokens, local_features]``. ``"head_to_sequence"`` applies the
    inverse mapping. Rows received from peers are ordered by source rank.
    ``local_tokens`` must be equal on every rank because this reference uses
    equal-size all-to-all splits.

    This operation does not implement BSHD/SBHD adapters, dual-chunk sequence
    reordering, THD padding, or ``cu_seqlens`` handling. Those transformations
    belong outside this basic operation unless a future kernel's documented
    output contract includes them.

    Parameters
    ----------
    process_group : torch.distributed.ProcessGroup, default = world group
        Context-parallel process group.
    direction : {"sequence_to_head", "head_to_sequence"}
        Direction of the context-parallel layout exchange.

    """

    def __init__(
        self,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        *,
        direction: Literal["sequence_to_head", "head_to_sequence"] = "sequence_to_head",
    ) -> None:
        super().__init__()
        if direction not in ("sequence_to_head", "head_to_sequence"):
            raise ValueError(
                "ContextParallelAllToAll direction must be "
                f"'sequence_to_head' or 'head_to_sequence' (got {direction!r})"
            )
        self.process_group: Optional[torch.distributed.ProcessGroup] = process_group
        self.direction = direction

    def _sequence_to_head(self, input_: torch.Tensor) -> torch.Tensor:
        """Gather token rows while sharding the last dimension."""
        group_size = torch.distributed.get_world_size(self.process_group)
        if input_.dim() != 2:
            raise ValueError(
                "ContextParallelAllToAll expects a 2-D tensor, "
                f"but got shape={list(input_.shape)}"
            )
        local_tokens, full_features = input_.shape
        if full_features % group_size != 0:
            raise ValueError(
                "The feature dimension must be divisible by the context-parallel "
                f"group size ({full_features=}, {group_size=})"
            )
        if group_size == 1:
            return input_.detach()

        local_features = full_features // group_size
        send = input_.reshape(local_tokens, group_size, local_features)
        send = send.movedim(1, 0).contiguous()
        recv = torch.empty_like(send)
        torch.distributed.all_to_all_single(recv, send, group=self.process_group)
        return recv.reshape(group_size * local_tokens, local_features)

    def _head_to_sequence(self, input_: torch.Tensor) -> torch.Tensor:
        """Shard token rows while gathering the last dimension."""
        group_size = torch.distributed.get_world_size(self.process_group)
        if input_.dim() != 2:
            raise ValueError(
                "ContextParallelAllToAll expects a 2-D tensor, "
                f"but got shape={list(input_.shape)}"
            )
        global_tokens, local_features = input_.shape
        if global_tokens % group_size != 0:
            raise ValueError(
                "The token dimension must be divisible by the context-parallel "
                f"group size ({global_tokens=}, {group_size=})"
            )
        if group_size == 1:
            return input_.detach()

        local_tokens = global_tokens // group_size
        send = input_.reshape(group_size, local_tokens, local_features).contiguous()
        recv = torch.empty_like(send)
        torch.distributed.all_to_all_single(recv, send, group=self.process_group)
        return recv.movedim(0, 1).reshape(local_tokens, group_size * local_features)

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
    ) -> torch.Tensor:
        del ctx, prev_op_grad_output_quantizer, next_op_input_quantizer
        input_ = maybe_dequantize(input_)
        if self.direction == "sequence_to_head":
            return self._sequence_to_head(input_)
        return self._head_to_sequence(input_)

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:
        del ctx
        grad_output = maybe_dequantize(grad_output)
        if self.direction == "sequence_to_head":
            grad_input = self._head_to_sequence(grad_output)
        else:
            grad_input = self._sequence_to_head(grad_output)
        return grad_input, ()
