# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Joint GEMM/A2A projection fusion using the optional MegaUP package."""

from __future__ import annotations

import torch
import torch.distributed as dist

from ...cpu_offload import is_cpu_offload_enabled
from ..basic import BasicLinear, ContextParallelAllToAll, InterleaveQKV
from ..op import FusedOperation


class MegaUPLinear(FusedOperation):
    """Replace Linear→A2A or A2A→Linear, including matching dgrad and wgrad.

    Intermediate rows use source-rank order. Attention-specific sequence
    restoration stays outside this operation. Each invocation owns its saved
    input; prepared Pull storage is never shared across outstanding graphs.
    """

    def fuser_forward(
        self,
        basic_op_ctxs,
        input_,
        *,
        basic_op_extra_inputs,
        prev_op_grad_output_quantizer,
        next_op_input_quantizer,
        basic_op_kwargs,
    ):
        from mega_up import peer_write_gemm_thd, prepare_headshard_pull_gemm_thd

        peer = isinstance(self.basic_ops[0], BasicLinear)
        linear_idx, exchange_idx = (0, len(self.basic_ops) - 1) if peer else (1, 0)
        linear, exchange = self.basic_ops[linear_idx], self.basic_ops[exchange_idx]
        ctx = basic_op_ctxs[linear_idx]
        runtime, d = exchange.megaup_runtime, exchange.head_dim
        validate = basic_op_kwargs[exchange_idx].get("validate_group", True)
        if any(kwargs for i, kwargs in enumerate(basic_op_kwargs) if i != exchange_idx):
            raise ValueError("MegaUP projection only accepts A2A validate_group metadata")
        if set(basic_op_kwargs[exchange_idx]) - {"validate_group", "validation_state"}:
            raise ValueError("Unknown MegaUP A2A metadata")
        if input_.ndim != 2 or input_.dtype != torch.bfloat16:
            raise ValueError("MegaUP projections require a 2-D BF16 input")
        weight = linear.weight
        n, k = weight.shape
        if input_.shape[1] != (k if peer else k // runtime.up):
            raise ValueError("MegaUP projection input width does not match the linear weight")
        if not peer and input_.shape[0] % runtime.up:
            raise ValueError("MegaUP head-to-sequence input rows must be divisible by CP size")
        interleave = self.basic_ops[1] if peer and len(self.basic_ops) == 3 else None
        packed = (interleave.permute(weight) if interleave is not None else weight).contiguous()
        if peer:
            x = input_.detach().contiguous().view(-1, k // d, d)
            output = peer_write_gemm_thd(x, packed.detach(), runtime, validate_group=validate)
            saved_input = input_
        else:
            x = input_.detach().contiguous().view(-1, k // (runtime.up * d), d)
            pull = prepare_headshard_pull_gemm_thd(
                packed.detach(),
                runtime,
                tokens=x.shape[0] // runtime.up,
                heads=k // d,
                head_dim=d,
                workspace_epochs=1,
                validate_group=validate,
            )
            output = pull(x, validate_group=validate)
            # Wgrad consumes the exchanged full-width input, not the incoming head shard.
            saved_input = pull.materialized_input if weight.requires_grad else None
        if ctx.requires_grad:
            # Save original weight too: packing must not hide optimizer version changes.
            ctx.save_for_backward(saved_input, weight, packed)
            ctx.runtime, ctx.head_dim, ctx.validate_group = runtime, d, validate
            ctx.peer, ctx.interleave = peer, interleave
            ctx.weight_requires_grad = weight.requires_grad
            # Private MHA metadata; each graph captures its own completed-validation state.
            ctx.validation_state = basic_op_kwargs[exchange_idx].get("validation_state")
        # Fresh Pull preparations and Peer copies own their storage. Detach only view
        # metadata after the final reshape, so downstream in-place residuals remain valid.
        return output.reshape(-1, n // runtime.up if peer else n).detach(), [()] * len(
            self.basic_ops
        )

    def fuser_backward(self, basic_op_ctxs, grad_output, *, basic_op_grad_extra_outputs):
        from mega_up import peer_write_gemm_thd, prepare_headshard_pull_gemm_thd

        peer = isinstance(self.basic_ops[0], BasicLinear)
        linear_idx = 0 if peer else 1
        ctx = basic_op_ctxs[linear_idx]
        x, original, weight = ctx.saved_tensors
        runtime, d, validate = ctx.runtime, ctx.head_dim, ctx.validate_group
        n, k = weight.shape
        # Refresh from this graph's live saved weight; never retain a stale transpose binding.
        transposed = weight.detach().t().contiguous()
        if peer:
            dy = grad_output.detach().contiguous().view(-1, n // (runtime.up * d), d)
            pull = prepare_headshard_pull_gemm_thd(
                transposed,
                runtime,
                tokens=dy.shape[0] // runtime.up,
                heads=n // d,
                head_dim=d,
                workspace_epochs=1,
                validate_group=validate,
            )
            dx = pull(dy, validate_group=validate)
            dw = pull.materialized_input.t() @ x if ctx.weight_requires_grad else None
            if dw is not None and ctx.interleave is not None:
                dw = ctx.interleave.permute(dw, inverse=True)
        else:
            dy = grad_output.detach().contiguous().view(-1, n // d, d)
            dx = peer_write_gemm_thd(dy, transposed, runtime, validate_group=validate)
            dw = grad_output.t() @ x if ctx.weight_requires_grad else None
        if peer and validate and ctx.validation_state is not None:
            ctx.validation_state["ready"] = True
        del original  # Saved only to preserve the original Parameter's version check.
        grads = [()] * len(self.basic_ops)
        grads[linear_idx] = (dw,)
        return dx.reshape(-1, k if peer else k // runtime.up), grads, [()] * len(self.basic_ops)

    @staticmethod
    def fuse_ops(ops, *, recipe):
        """Fuse only explicit runtime opt-ins; all other sequences remain unchanged."""
        if recipe is not None or is_cpu_offload_enabled():
            return ops
        result, i = [], 0
        while i < len(ops):
            candidate = None
            if isinstance(ops[i], BasicLinear):
                end = i + 1
                if end < len(ops) and isinstance(ops[end], InterleaveQKV):
                    end += 1
                if end < len(ops) and isinstance(ops[end], ContextParallelAllToAll):
                    if ops[end].direction == "sequence_to_head":
                        candidate = ops[i : end + 1]
            elif isinstance(ops[i], ContextParallelAllToAll) and i + 1 < len(ops):
                if ops[i].direction == "head_to_sequence" and isinstance(ops[i + 1], BasicLinear):
                    candidate = ops[i : i + 2]
            if candidate is not None:
                linear = candidate[0] if isinstance(candidate[0], BasicLinear) else candidate[1]
                exchange = candidate[-1] if isinstance(candidate[0], BasicLinear) else candidate[0]
                runtime = exchange.megaup_runtime
                group = (
                    exchange.process_group
                    if exchange.process_group is not None
                    else dist.group.WORLD
                )
                if (
                    runtime is not None
                    and runtime.group is group
                    and linear.weight.dtype == torch.bfloat16
                    and linear.weight.is_cuda
                    and not linear._with_quantized_weight
                    and linear.tensor_parallel_size == 1
                    and not linear.sequence_parallel
                    and not linear._accumulate_into_main_grad
                    and linear._userbuffers_options is None
                ):
                    result.append(MegaUPLinear(candidate))
                    i += len(candidate)
                    continue
            result.append(ops[i])
            i += 1
        return result
