# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

import argparse
from collections.abc import Iterable
import functools
import itertools
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import Optional

import pytest
import torch

import transformer_engine
import transformer_engine.common.recipe
import transformer_engine.pytorch as te
from transformer_engine.pytorch import (
    QuantizedTensor,
    Float8Quantizer,
    Float8CurrentScalingQuantizer,
    MXFP8Quantizer,
    NVFP4Quantizer,
    is_bf16_available,
)
import transformer_engine.pytorch.ops as te_ops

# Import utility functions
_current_file = pathlib.Path(__file__).resolve()
# Prepend so installed packages with a top-level utils module cannot shadow the test helpers.
sys.path = [str(_current_file.parent.parent)] + sys.path
from utils import assert_close, dtype_tols, make_recipe, quantization_tols


# Check what quantization schemes are supported
fp8_available, reason_for_no_fp8 = te.is_fp8_available(return_reason=True)
mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
nvfp4_available, reason_for_no_nvfp4 = te.is_mxfp8_available(return_reason=True)
quantization_list: list[Optional[str]] = [None]
if fp8_available:
    quantization_list.extend(("fp8_delayed_scaling", "fp8_current_scaling"))
if mxfp8_available:
    quantization_list.append("mxfp8")
if nvfp4_available:
    quantization_list.append("nvfp4")


@functools.cache
def world_group() -> torch.distributed.ProcessGroup:
    """Get NCCL process group, initializing if needed"""
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    group = torch.distributed.init_process_group(
        "nccl",
        # Each parallel job must use a fresh FileStore shared by only its ranks.
        init_method=f"file://{os.environ['NVTE_TEST_RDZV_PATH']}",
        world_size=world_size,
        rank=rank,
    )
    return group


def reset_rng(seed: int = 1234) -> None:
    """Reset random number generators"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


@torch.no_grad()
def make_reference_and_test_tensors(
    shape: int | Iterable[int],
    quantization: Optional[str] = None,
    ref_dtype: torch.dtype = torch.float64,
    ref_device: torch.device = "cpu",
    test_dtype: torch.dtype = torch.float32,
    test_device: torch.device = "cuda",
    test_is_quantized: bool = False,
    requires_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct tensors with the same values

    The reference tensor is intended for use in plain PyTorch
    operations in high precision. The test tensor is intended for use
    in Transformer Engine operations.

    If a quantization scheme is provided, the tensor values are
    quantized so that they are representable.

    """

    # Random reference tensor
    ref = torch.rand(shape, dtype=ref_dtype, device=ref_device)

    # Construct test tensor from reference tensor
    test = ref.to(device=test_device, dtype=test_dtype)
    if quantization is None:
        if test_is_quantized:
            raise ValueError("Quantization scheme not provided")
        if test.data_ptr() == ref.data_ptr():
            test = test.clone()
    elif quantization in ("fp8", "fp8_delayed_scaling"):
        quantizer = Float8Quantizer(
            scale=torch.ones(1, dtype=torch.float32, device=test_device).squeeze(),
            amax=torch.zeros(1, dtype=torch.float32, device=test_device),
            fp8_dtype=te.DType.kFloat8E4M3,
        )
        test = quantizer(test)
    elif quantization == "fp8_current_scaling":
        quantizer = Float8CurrentScalingQuantizer(
            fp8_dtype=te.DType.kFloat8E4M3,
            device=test_device,
        )
        test = quantizer(test)
    elif quantization == "mxfp8":
        test = MXFP8Quantizer(fp8_dtype=te.DType.kFloat8E4M3)(test)
    elif quantization == "nvfp4":
        test = NVFP4Quantizer(
            with_rht=False,
            with_post_rht_amax=False,
            with_2d_quantization=False,
            stochastic_rounding=False,
            with_random_sign_mask=False,
        )(test)
    else:
        raise ValueError(f"Unsupported quantization scheme ({quantization})")
    if isinstance(test, QuantizedTensor) and not test_is_quantized:
        test = test.dequantize()

    # Make sure reference and test tensors match each other
    ref.copy_(test)

    ref.requires_grad_(requires_grad)
    test.requires_grad_(requires_grad)
    return ref, test


def _test_all_reduce(
    *,
    local_size: int = 32,
    dtype: torch.dtype = torch.float32,
    device: torch.device = "cuda",
    quantization: Optional[str] = None,
) -> None:

    # Distributed process group
    process_group = world_group()
    rank = torch.distributed.get_rank(process_group)
    world_size = torch.distributed.get_world_size(process_group)

    # Tensor dimensions
    in_shape = [world_size, local_size, local_size]
    out_shape = [local_size, local_size]

    # Random data
    reset_rng()
    with_quantization = quantization is not None
    x_ref, x_test = make_reference_and_test_tensors(
        in_shape,
        quantization=quantization,
        test_dtype=dtype,
        test_device=device,
        test_is_quantized=with_quantization,
    )
    dy_ref, dy_test = make_reference_and_test_tensors(
        out_shape,
        quantization=quantization,
        test_dtype=dtype,
        test_device=device,
        test_is_quantized=with_quantization,
    )

    # Plain PyTorch implementation
    y_ref = x_ref.sum(0)
    y_ref.backward(dy_ref)

    # Convert to distributed tensors
    with torch.no_grad():
        dx_ref = x_ref.grad[rank]
        x_ref = x_ref[rank]
        x_test = x_test[rank].clone()
    x_test.requires_grad_()

    # Implementation with fusible operation
    op = te_ops.AllReduce(process_group=process_group)
    y_test = op(x_test)
    y_test.backward(dy_test)

    # Check results
    y_test = y_test.to(dtype=torch.float64, device="cpu")
    dx_test = x_test.grad.to(dtype=torch.float64, device="cpu")
    torch.testing.assert_close(y_test, y_ref, **dtype_tols(dtype))
    torch.testing.assert_close(dx_test, dx_ref, rtol=0, atol=0)


def _test_all_gather(
    *,
    local_size: int = 32,
    dtype: torch.dtype = torch.float32,
    device: torch.device = "cuda",
    quantization: Optional[str] = None,
) -> None:

    # Distributed process group
    process_group = world_group()
    rank = torch.distributed.get_rank(process_group)
    world_size = torch.distributed.get_world_size(process_group)

    # Tensor dimensions
    in_shape = [world_size, local_size, local_size]
    out_shape = [world_size, world_size * local_size, local_size]

    # Random data
    reset_rng()
    with_quantization = quantization is not None
    x_ref, x_test = make_reference_and_test_tensors(
        in_shape,
        quantization=quantization,
        test_dtype=dtype,
        test_device=device,
        test_is_quantized=with_quantization,
    )
    dy_ref, dy_test = make_reference_and_test_tensors(
        out_shape,
        quantization=quantization,
        test_dtype=dtype,
        test_device=device,
        test_is_quantized=with_quantization,
    )

    # Plain PyTorch implementation
    y_ref = x_ref.tile((world_size, 1, 1)).reshape(out_shape)
    y_ref.backward(dy_ref)

    # Convert to distributed tensors
    with torch.no_grad():
        dx_ref = x_ref.grad[rank]
        x_ref = x_ref[rank]
        x_test = x_test[rank].clone()
        y_ref = y_ref[rank]
        dy_ref = dy_ref[rank]
        dy_test = dy_test[rank].clone()
    x_test.requires_grad_()

    # Implementation with fusible operation
    op = te_ops.AllGather(process_group=process_group)
    y_test = op(x_test)
    y_test.backward(dy_test)

    # Check results
    y_test = y_test.to(dtype=torch.float64, device="cpu")
    dx_test = x_test.grad.to(dtype=torch.float64, device="cpu")
    torch.testing.assert_close(y_test, y_ref, rtol=0, atol=0)
    torch.testing.assert_close(dx_test, dx_ref, **dtype_tols(dtype))


def _test_reduce_scatter(
    *,
    local_size: int = 32,
    dtype: torch.dtype = torch.float32,
    device: torch.device = "cuda",
    quantization: Optional[str] = None,
) -> None:

    # Distributed process group
    process_group = world_group()
    rank = torch.distributed.get_rank(process_group)
    world_size = torch.distributed.get_world_size(process_group)

    # Tensor dimensions
    in_shape = [world_size, world_size * local_size, local_size]
    out_shape = [world_size, local_size, local_size]

    # Random data
    reset_rng()
    with_quantization = quantization is not None
    x_ref, x_test = make_reference_and_test_tensors(
        in_shape,
        quantization=quantization,
        test_dtype=dtype,
        test_device=device,
        test_is_quantized=with_quantization,
    )
    dy_ref, dy_test = make_reference_and_test_tensors(
        out_shape,
        quantization=quantization,
        test_dtype=dtype,
        test_device=device,
        test_is_quantized=with_quantization,
    )

    # Plain PyTorch implementation
    y_ref = x_ref.sum(0).reshape(out_shape)
    y_ref.backward(dy_ref)

    # Convert to distributed tensors
    with torch.no_grad():
        dx_ref = x_ref.grad[rank]
        x_ref = x_ref[rank]
        x_test = x_test[rank].clone()
        y_ref = y_ref[rank]
        dy_ref = dy_ref[rank]
        dy_test = dy_test[rank].clone()
    x_test.requires_grad_()

    # Implementation with fusible operation
    op = te_ops.ReduceScatter(process_group=process_group)
    y_test = op(x_test)
    y_test.backward(dy_test)

    # Check results
    y_test = y_test.to(dtype=torch.float64, device="cpu")
    dx_test = x_test.grad.to(dtype=torch.float64, device="cpu")
    torch.testing.assert_close(y_test, y_ref, **dtype_tols(dtype))
    torch.testing.assert_close(dx_test, dx_ref, rtol=0, atol=0)


def _test_context_parallel_all_to_all(
    *,
    local_tokens: int = 5,
    local_features: int = 7,
    dtype: torch.dtype = torch.float32,
    device: torch.device = "cuda",
) -> None:
    """Check both CP-A2A directions and their inverse gradients."""
    process_group = world_group()
    rank = torch.distributed.get_rank(process_group)
    world_size = torch.distributed.get_world_size(process_group)
    full_features = world_size * local_features

    # Replicate the global reference on each rank so expected peer slices can
    # be computed without using the implementation under test.
    x_global = torch.arange(
        world_size * local_tokens * full_features,
        dtype=dtype,
        device=device,
    ).reshape(world_size, local_tokens, full_features)
    grad_head_global = torch.arange(
        world_size * world_size * local_tokens * local_features,
        dtype=dtype,
        device=device,
    ).reshape(world_size, world_size * local_tokens, local_features)

    x = x_global[rank].clone().requires_grad_()
    expected_head = torch.cat(
        [
            x_global[src, :, rank * local_features : (rank + 1) * local_features]
            for src in range(world_size)
        ],
        dim=0,
    )
    expected_dx = torch.cat(
        [
            grad_head_global[dst].reshape(world_size, local_tokens, local_features)[rank]
            for dst in range(world_size)
        ],
        dim=-1,
    )

    to_head = te_ops.ContextParallelAllToAll(
        process_group=process_group,
        direction="sequence_to_head",
    )
    head = to_head(x)
    head.backward(grad_head_global[rank])
    torch.testing.assert_close(head, expected_head, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, expected_dx, rtol=0, atol=0)

    grad_sequence_global = torch.arange(
        world_size * local_tokens * full_features,
        dtype=dtype,
        device=device,
    ).reshape(world_size, local_tokens, full_features)
    expected_dhead = torch.cat(
        [
            grad_sequence_global[src][:, rank * local_features : (rank + 1) * local_features]
            for src in range(world_size)
        ],
        dim=0,
    )

    head_input = expected_head.clone().requires_grad_()
    to_sequence = te_ops.ContextParallelAllToAll(
        process_group=process_group,
        direction="head_to_sequence",
    )
    sequence = to_sequence(head_input)
    sequence.backward(grad_sequence_global[rank])
    torch.testing.assert_close(sequence, x_global[rank], rtol=0, atol=0)
    torch.testing.assert_close(head_input.grad, expected_dhead, rtol=0, atol=0)


def _test_cp_projection_pairs(process_group=None, *, use_megaup=False) -> None:
    """Independent forward/dX/dW oracles for the two complete CP projection pairs."""
    if process_group is None:
        world_group()
        process_group = torch.distributed.group.WORLD
    cp, rank = (
        torch.distributed.get_world_size(process_group),
        torch.distributed.get_rank(process_group),
    )
    m, k = 128, 1024

    def compare(actual, expected):
        passed = torch.isclose(actual, expected, atol=0.03, rtol=0.03).all().int()
        torch.distributed.all_reduce(passed, op=torch.distributed.ReduceOp.MIN, group=process_group)
        assert passed.item(), "CP projection output or gradient differs from independent matmul"

    for direction, interleave in (
        ("sequence_to_head", False),
        ("sequence_to_head", True),
        ("head_to_sequence", False),
    ):
        n = 3072 if direction == "sequence_to_head" else 1024
        reset_rng()
        owner = torch.nn.Module()
        owner.weight = torch.nn.Parameter(
            torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
        )
        linear = te_ops.BasicLinear(k, n, device="meta", dtype=torch.bfloat16)
        linear.weight = owner.weight  # Bind the original Parameter without initializing/copying it.
        runtime = None
        if use_megaup:
            from mega_up import initialize_runtime

            runtime = initialize_runtime(
                process_group,
                hidden=n if direction == "sequence_to_head" else k,
                max_rows=m,
                device=owner.weight.device,
            )
        exchange = te_ops.ContextParallelAllToAll(
            process_group, direction=direction, megaup_runtime=runtime
        )
        modules = [linear]
        if interleave:
            modules.append(te_ops.InterleaveQKV(k // 128, 128))
        modules = modules + [exchange] if direction == "sequence_to_head" else [exchange, linear]
        sequence = te_ops.Sequential(*modules)
        object.__setattr__(owner, "_projection", sequence)
        if use_megaup:
            local_width = k if direction == "sequence_to_head" else k // cp
            with pytest.raises(ValueError, match="input width"):
                sequence(torch.empty(m, local_width * 2, device="cuda", dtype=torch.bfloat16))
            if direction == "head_to_sequence":
                with pytest.raises(ValueError, match="input rows"):
                    sequence(
                        torch.empty(m * cp + 1, local_width, device="cuda", dtype=torch.bfloat16)
                    )
        assert set(owner.state_dict()) == {"weight"}
        assert next(owner.parameters()) is linear.weight

        for step in range(2):
            graphs = []
            for microbatch in range(2):
                reset_rng(1700 + 10 * step + microbatch)
                if direction == "sequence_to_head":
                    all_x = torch.randn(cp, m, k, device="cuda", dtype=torch.bfloat16) * 0.1
                    all_dy = (
                        torch.randn(cp, cp * m, n // cp, device="cuda", dtype=torch.bfloat16) * 0.1
                    )
                    x = all_x[rank].clone().requires_grad_()
                    # Source and destination slices are derived independently of the A2A op.
                    all_y = all_x @ owner.weight.detach().t()
                    if interleave:
                        all_y = (
                            all_y.view(cp, m, 3, k // 128, 128).transpose(2, 3).reshape(cp, m, n)
                        )
                    expected_y = all_y[:, :, rank * (n // cp) : (rank + 1) * (n // cp)].reshape(
                        cp * m, n // cp
                    )
                    dense_dy = all_dy[:, rank * m : (rank + 1) * m].movedim(0, 1).reshape(m, n)
                    if interleave:
                        dense_dy = dense_dy.view(m, k // 128, 3, 128).transpose(1, 2).reshape(m, n)
                    expected_dx = dense_dy @ owner.weight.detach()
                    expected_dw = dense_dy.t() @ all_x[rank]
                else:
                    all_x = (
                        torch.randn(cp, cp * m, k // cp, device="cuda", dtype=torch.bfloat16) * 0.1
                    )
                    all_dy = torch.randn(cp, m, n, device="cuda", dtype=torch.bfloat16) * 0.1
                    x = all_x[rank].clone().requires_grad_()
                    materialized = all_x[:, rank * m : (rank + 1) * m].movedim(0, 1).reshape(m, k)
                    expected_y = materialized @ owner.weight.detach().t()
                    all_dx = all_dy @ owner.weight.detach()
                    expected_dx = all_dx[:, :, rank * (k // cp) : (rank + 1) * (k // cp)].reshape(
                        cp * m, k // cp
                    )
                    expected_dw = all_dy[rank].t() @ materialized
                y = sequence(x)
                if microbatch == 0:
                    y.add_(0.125)
                    expected_y = expected_y + 0.125
                from transformer_engine.pytorch.ops.fused.megaup_linear import MegaUPLinear

                forward_ops = sequence._module_groups[0]._forward_ops
                assert any(isinstance(op, MegaUPLinear) for op, _ in forward_ops) == use_megaup
                graphs.append((x, y, all_dy[rank], expected_y, expected_dx, expected_dw))
            # Both outputs and saved materializations must survive another forward.
            for x, y, dy, expected_y, expected_dx, expected_dw in reversed(graphs):
                owner.weight.grad = None
                compare(y, expected_y)
                repeat = step == 0 and y is graphs[-1][1]
                saved_activation = None
                if repeat and direction == "head_to_sequence":
                    import weakref

                    # Keep the output/grad_fn alive while checking saved activation release.
                    saved_activation = weakref.ref(y.grad_fn.saved_tensors[0])
                y.backward(dy, retain_graph=repeat)
                compare(x.grad, expected_dx)
                compare(owner.weight.grad, expected_dw)
                if repeat:
                    if saved_activation is not None:
                        assert saved_activation() is not None
                    x.grad = owner.weight.grad = None
                    y.backward(dy)
                    compare(x.grad, expected_dx)
                    compare(owner.weight.grad, expected_dw)
                    if saved_activation is not None:
                        assert saved_activation() is None
            with torch.no_grad():
                # Identical update on each CP replica; bindings must see live storage.
                owner.weight.add_(0.001)
            assert linear.weight is owner.weight and set(owner.state_dict()) == {"weight"}

        # Frozen weights still require the inverse communication for input gradients.
        owner.weight.requires_grad_(False)
        owner.weight.grad = None
        x = x.detach().clone().requires_grad_()
        y = sequence(x)
        y.backward(torch.ones_like(y))
        if direction == "sequence_to_head":
            expected_dx = torch.ones(m, n, device="cuda", dtype=x.dtype) @ owner.weight
        else:
            dense_dx = torch.ones(cp * m, n, device="cuda", dtype=x.dtype) @ owner.weight
            expected_dx = dense_dx[:, rank * (k // cp) : (rank + 1) * (k // cp)]
        compare(x.grad, expected_dx)
        assert owner.weight.grad is None
    if rank == 0:
        print(
            f"CP projection pairs passed: CP={cp}, MegaUP={use_megaup}, retained graphs, dX/dW,"
            " live weights",
            flush=True,
        )


def run_cp_a2a_tests(*, use_megaup=False) -> None:
    """Run focused CP semantics/pairs on the world group and disjoint subgroups."""
    group = world_group()
    with pytest.raises(ValueError):
        te_ops.InterleaveQKV(0, 128)
    packing = te_ops.InterleaveQKV(8, 128)
    with pytest.raises(ValueError):
        packing.op_forward(None, torch.empty(2, 1536, device="cuda"), None, None)
    _test_context_parallel_all_to_all()
    cp = torch.distributed.get_world_size(group)
    from contextlib import ExitStack
    from unittest.mock import patch

    with ExitStack() as stack:
        if use_megaup:
            import mega_up

            peer = stack.enter_context(
                patch.object(mega_up, "peer_write_gemm_thd", wraps=mega_up.peer_write_gemm_thd)
            )
            pull = stack.enter_context(
                patch.object(
                    mega_up,
                    "prepare_headshard_pull_gemm_thd",
                    wraps=mega_up.prepare_headshard_pull_gemm_thd,
                )
            )
        _test_cp_projection_pairs(group, use_megaup=use_megaup)
        if cp >= 4:
            # Every world rank creates groups in the same order and participates in one.
            for members in (list(range(0, cp, 2)), list(range(1, cp, 2))):
                subgroup = torch.distributed.new_group(members)
                if torch.distributed.get_rank(group) in members:
                    _test_cp_projection_pairs(subgroup, use_megaup=use_megaup)
        if use_megaup:
            # Five forward/backwards per layout plus a repeated backward: the two
            # Peer layouts add two Pull calls; the Pull layout adds one Peer call.
            groups = 2 if cp >= 4 else 1
            assert (peer.call_count, pull.call_count) == (16 * groups, 17 * groups)
    torch.distributed.barrier(group)


def _test_basic_linear(
    *,
    local_weight_shape: tuple[int, int] = (32, 32),
    local_batch_size: int = 32,
    dtype: torch.dtype = torch.float32,
    device: torch.device = "cuda",
    quantization: Optional[str] = None,
    quantized_weight: bool = False,
    tensor_parallel_mode: str = "column",
    sequence_parallel: bool = False,
    compile_model: bool = False,
) -> None:

    # Skip invalid configurations
    quantized_compute = quantization is not None
    if not quantized_compute and quantized_weight:
        return

    # Distributed process group
    process_group = world_group()
    rank = torch.distributed.get_rank(process_group)
    world_size = torch.distributed.get_world_size(process_group)

    # Tensor dimensions
    local_out_features, local_in_features = local_weight_shape
    out_features, in_features = local_out_features, local_in_features
    batch_size = local_batch_size
    if tensor_parallel_mode == "column":
        out_features *= world_size
    elif tensor_parallel_mode == "row":
        in_features *= world_size
    if sequence_parallel:
        batch_size *= world_size
    in_shape = [batch_size, in_features]
    out_shape = [batch_size, out_features]

    # Random data
    reset_rng()
    x_ref, x_test = make_reference_and_test_tensors(
        in_shape,
        quantization=quantization,
        test_dtype=dtype,
        test_device=device,
    )
    w_ref, w_test = make_reference_and_test_tensors(
        (out_features, in_features),
        quantization=quantization,
        test_dtype=dtype,
        test_device=device,
    )
    dy_ref, dy_test = make_reference_and_test_tensors(
        out_shape,
        quantization=quantization,
        test_dtype=dtype,
        test_device=device,
        requires_grad=False,
    )

    # Plain PyTorch implementation
    y_ref = torch.nn.functional.linear(x_ref, w_ref)
    y_ref.backward(dy_ref)

    # Convert to distributed tensors
    with torch.no_grad():
        dw_ref = w_ref.grad
        dx_ref = x_ref.grad
        if tensor_parallel_mode == "column":
            local_out_features = out_features // world_size
            local_slice = slice(
                rank * local_out_features,
                (rank + 1) * local_out_features,
            )
            w_ref = w_ref[local_slice, :]
            dw_ref = dw_ref[local_slice, :]
            w_test = w_test[local_slice, :]
            y_ref = y_ref[..., local_slice]
            dy_ref = dy_ref[..., local_slice]
            dy_test = dy_test[..., local_slice].clone()
        elif tensor_parallel_mode == "row":
            local_in_features = in_features // world_size
            local_slice = slice(
                rank * local_in_features,
                (rank + 1) * local_in_features,
            )
            w_ref = w_ref[:, local_slice]
            dw_ref = dw_ref[:, local_slice]
            w_test = w_test[:, local_slice]
            x_ref = x_ref[..., local_slice]
            dx_ref = dx_ref[..., local_slice]
            x_test = x_test[..., local_slice].clone()
        if sequence_parallel:
            local_batch_size = batch_size // world_size
            local_slice = slice(
                rank * local_batch_size,
                (rank + 1) * local_batch_size,
            )
            if tensor_parallel_mode == "column":
                x_ref = x_ref[local_slice, ...]
                dx_ref = dx_ref[local_slice, ...]
                x_test = x_test[local_slice, ...].clone()
            elif tensor_parallel_mode == "row":
                y_ref = y_ref[local_slice, ...]
                dy_ref = dy_ref[local_slice, ...]
                dy_test = dy_test[local_slice, ...].clone()
    x_test.requires_grad_()

    # Implementation with fusible operation
    recipe = make_recipe(quantization)
    if compile_model and recipe is not None:
        recipe.backward_override = None
    with te.quantized_model_init(enabled=quantized_weight, recipe=recipe):
        op = te_ops.BasicLinear(
            in_features,
            out_features,
            device=device,
            dtype=dtype,
            tensor_parallel_mode=tensor_parallel_mode,
            tensor_parallel_group=process_group,
            sequence_parallel=sequence_parallel,
        )
    with torch.no_grad():
        op.weight.copy_(w_test)
        del w_test
    model = te_ops.Sequential(op)

    def forward(x):
        if not quantized_compute:
            return model(x)
        with te.autocast(enabled=quantized_compute, recipe=recipe):
            return model(x)

    if compile_model:
        torch._dynamo.reset()
        forward = torch.compile(forward, fullgraph=True)

    # Expected numerical error
    tols = dtype_tols(dtype)
    if dtype == torch.float32:
        tols = dtype_tols(torch.float16)  # TF32 GEMM
    if quantized_compute:
        tols = quantization_tols(quantization)

    # Repeated calls also exercise the compiled backward and cached graph.
    for _ in range(3 if compile_model else 1):
        model.zero_grad(set_to_none=True)
        x_test.grad = None
        y_test = forward(x_test)
        y_test.backward(dy_test)
        assert_close(y_test, y_ref, **tols)
        assert_close(x_test.grad, dx_ref, **tols)
        assert_close(op.weight.grad, dw_ref, **tols)


def _test_linear(
    *,
    bias: bool = True,
    local_weight_shape: tuple[int, int] = (32, 32),
    local_batch_size: int = 32,
    dtype: torch.dtype = torch.float32,
    device: torch.device = "cuda",
    quantization: Optional[str] = None,
    quantized_weight: bool = False,
    tensor_parallel_mode: str = "column",
    sequence_parallel: bool = False,
    compile_model: bool = False,
) -> None:

    # Skip invalid configurations
    quantized_compute = quantization is not None
    if not quantized_compute and quantized_weight:
        return

    # Distributed process group
    process_group = world_group()
    rank = torch.distributed.get_rank(process_group)
    world_size = torch.distributed.get_world_size(process_group)

    # Tensor dimensions
    local_out_features, local_in_features = local_weight_shape
    out_features, in_features = local_out_features, local_in_features
    batch_size = local_batch_size
    if tensor_parallel_mode == "column":
        out_features *= world_size
    elif tensor_parallel_mode == "row":
        in_features *= world_size
    if sequence_parallel:
        batch_size *= world_size
    in_shape = [batch_size, in_features]
    out_shape = [batch_size, out_features]

    # Random data
    reset_rng()
    x_ref, x_test = make_reference_and_test_tensors(
        in_shape,
        quantization=quantization,
        test_dtype=dtype,
        test_device=device,
    )
    w_ref, w_test = make_reference_and_test_tensors(
        (out_features, in_features),
        quantization=quantization,
        test_dtype=dtype,
        test_device=device,
    )
    b_ref, b_test = None, None
    if bias:
        if tensor_parallel_mode == "row":
            bias_shape = [world_size, out_features]
        else:
            bias_shape = [out_features]
        b_ref, b_test = make_reference_and_test_tensors(
            bias_shape,
            test_dtype=dtype,
            test_device=device,
        )
    dy_ref, dy_test = make_reference_and_test_tensors(
        out_shape,
        quantization=quantization,
        test_dtype=dtype,
        test_device=device,
        requires_grad=False,
    )

    # Plain PyTorch implementation
    y_ref = torch.nn.functional.linear(x_ref, w_ref)
    if bias:
        if tensor_parallel_mode == "row":
            y_ref += b_ref.sum(dim=0)
        else:
            y_ref += b_ref
    y_ref.backward(dy_ref)

    # Convert to distributed tensors
    with torch.no_grad():
        dw_ref = w_ref.grad
        db_ref = b_ref.grad if bias else None
        dx_ref = x_ref.grad
        if tensor_parallel_mode == "column":
            local_out_features = out_features // world_size
            local_slice = slice(
                rank * local_out_features,
                (rank + 1) * local_out_features,
            )
            w_ref = w_ref[local_slice, :]
            dw_ref = dw_ref[local_slice, :]
            w_test = w_test[local_slice, :]
            if bias:
                b_ref = b_ref[local_slice]
                db_ref = db_ref[local_slice]
                b_test = b_test[local_slice]
            y_ref = y_ref[..., local_slice]
            dy_ref = dy_ref[..., local_slice]
            dy_test = dy_test[..., local_slice].clone()
        elif tensor_parallel_mode == "row":
            local_in_features = in_features // world_size
            local_slice = slice(
                rank * local_in_features,
                (rank + 1) * local_in_features,
            )
            w_ref = w_ref[:, local_slice]
            dw_ref = dw_ref[:, local_slice]
            w_test = w_test[:, local_slice]
            if bias:
                b_ref = b_ref[rank, :]
                db_ref = db_ref[rank, :]
                b_test = b_test[rank, :]
            x_ref = x_ref[..., local_slice]
            dx_ref = dx_ref[..., local_slice]
            x_test = x_test[..., local_slice].clone()
        if sequence_parallel:
            local_batch_size = batch_size // world_size
            local_slice = slice(
                rank * local_batch_size,
                (rank + 1) * local_batch_size,
            )
            if tensor_parallel_mode == "column":
                x_ref = x_ref[local_slice, ...]
                dx_ref = dx_ref[local_slice, ...]
                x_test = x_test[local_slice, ...].clone()
            elif tensor_parallel_mode == "row":
                y_ref = y_ref[local_slice, ...]
                dy_ref = dy_ref[local_slice, ...]
                dy_test = dy_test[local_slice, ...].clone()
    x_test.requires_grad_()

    # Implementation with fusible operation
    recipe = make_recipe(quantization)
    if compile_model and recipe is not None:
        recipe.backward_override = None
    with te.quantized_model_init(enabled=quantized_weight, recipe=recipe):
        model = te_ops.Sequential(
            te_ops.Linear(
                in_features,
                out_features,
                bias=bias,
                device=device,
                dtype=dtype,
                tensor_parallel_mode=tensor_parallel_mode,
                tensor_parallel_group=process_group,
                sequence_parallel=sequence_parallel,
            ),
        )
    with torch.no_grad():
        model[0].weight.copy_(w_test)
        if bias:
            model[0].bias.copy_(b_test)
        del w_test
        del b_test

    def forward(x):
        if not quantized_compute:
            return model(x)
        with te.autocast(enabled=quantized_compute, recipe=recipe):
            return model(x)

    if compile_model:
        torch._dynamo.reset()
        forward = torch.compile(forward, fullgraph=True)

    # Expected numerical error
    tols = dtype_tols(dtype)
    if dtype == torch.float32:
        tols = dtype_tols(torch.float16)  # TF32 GEMM
    if quantized_compute:
        tols = quantization_tols(quantization)

    for _ in range(3 if compile_model else 1):
        model.zero_grad(set_to_none=True)
        x_test.grad = None
        y_test = forward(x_test)
        y_test.backward(dy_test)
        assert_close(y_test, y_ref, **tols)
        assert_close(x_test.grad, dx_ref, **tols)
        assert_close(model[0].weight.grad, dw_ref, **tols)
        if bias:
            assert_close(model[0].bias.grad, db_ref, **tols)


def _test_mlp(
    *,
    bias: bool = True,
    hidden_size: int = 32,
    local_batch_size: int = 32,
    dtype: torch.dtype = torch.float32,
    device: torch.device = "cuda",
    quantization: Optional[str] = None,
    quantized_weight: bool = False,
    sequence_parallel: bool = False,
) -> None:
    """2-layer MLP

    MLP includes GELU activation in order to test op fusions. Model
    performs warmup steps in order to test inter-step logic.

    """

    # Skip invalid configurations
    quantized_compute = quantization is not None
    if not quantized_compute and quantized_weight:
        return

    # Distributed process group
    process_group = world_group()
    rank = torch.distributed.get_rank(process_group)
    world_size = torch.distributed.get_world_size(process_group)

    # Tensor dimensions
    mlp_size = hidden_size * world_size
    batch_size = local_batch_size
    if sequence_parallel:
        batch_size *= world_size
    in_shape = (batch_size, hidden_size)

    # Random data
    reset_rng()
    x_ref, x_test = make_reference_and_test_tensors(
        in_shape,
        quantization=quantization,
        test_dtype=dtype,
        test_device=device,
    )
    w1_ref, w1_test = make_reference_and_test_tensors(
        (mlp_size, hidden_size),
        quantization=quantization,
        test_dtype=dtype,
        test_device=device,
    )
    b1_ref, b1_test = None, None
    w2_ref, w2_test = make_reference_and_test_tensors(
        (hidden_size, mlp_size),
        quantization=quantization,
        test_dtype=dtype,
        test_device=device,
    )
    b2_ref, b2_test = None, None
    if bias:
        b1_ref, b1_test = make_reference_and_test_tensors(
            (mlp_size,),
            test_dtype=dtype,
            test_device=device,
        )
        b2_ref, b2_test = make_reference_and_test_tensors(
            (world_size, hidden_size),
            test_dtype=dtype,
            test_device=device,
        )
    dy_ref, dy_test = make_reference_and_test_tensors(
        in_shape,
        quantization=quantization,
        test_dtype=dtype,
        test_device=device,
        requires_grad=False,
    )

    # Plain PyTorch implementation
    y_ref = torch.nn.functional.gelu(x_ref, approximate="tanh")
    y_ref = torch.nn.functional.linear(y_ref, w1_ref)
    if bias:
        y_ref += b1_ref
    y_ref = torch.nn.functional.gelu(y_ref, approximate="tanh")
    y_ref = torch.nn.functional.linear(y_ref, w2_ref)
    if bias:
        y_ref += b2_ref.sum(dim=0)
    y_ref = torch.nn.functional.gelu(y_ref, approximate="tanh")
    y_ref.backward(dy_ref)

    # Convert to distributed tensors
    with torch.no_grad():
        local_mlp_size = mlp_size // world_size
        local_mlp_slice = slice(rank * local_mlp_size, (rank + 1) * local_mlp_size)
        dx_ref = x_ref.grad
        dw1_ref = w1_ref.grad[local_mlp_slice, :]
        w1_ref = w1_ref[local_mlp_slice, :]
        w1_test = w1_test[local_mlp_slice, :]
        dw2_ref = w2_ref.grad[:, local_mlp_slice]
        w2_ref = w2_ref[:, local_mlp_slice]
        w2_test = w2_test[:, local_mlp_slice]
        if bias:
            db1_ref = b1_ref.grad[local_mlp_slice]
            b1_ref = b1_ref[local_mlp_slice]
            b1_test = b1_test[local_mlp_slice]
            db2_ref = b2_ref.grad[rank, :]
            b2_ref = b2_ref[rank, :]
            b2_test = b2_test[rank, :]
        else:
            db1_ref = None
            db2_ref = None
        if sequence_parallel:
            local_batch_slice = slice(
                rank * local_batch_size,
                (rank + 1) * local_batch_size,
            )
            x_ref = x_ref[local_batch_slice, ...]
            dx_ref = dx_ref[local_batch_slice, ...]
            x_test = x_test[local_batch_slice, ...].clone()
            y_ref = y_ref[local_batch_slice, ...]
            dy_ref = dy_ref[local_batch_slice, ...]
            dy_test = dy_test[local_batch_slice, ...].clone()
    x_test.requires_grad_()

    # Implementation with fusible operation
    recipe = make_recipe(quantization)
    with te.quantized_model_init(enabled=quantized_weight, recipe=recipe):
        model = te_ops.Sequential(
            te_ops.GELU(),
            te_ops.Linear(
                hidden_size,
                mlp_size,
                bias=bias,
                device=device,
                dtype=dtype,
                tensor_parallel_mode="column",
                tensor_parallel_group=process_group,
                sequence_parallel=sequence_parallel,
            ),
            te_ops.GELU(),
            te_ops.Linear(
                mlp_size,
                hidden_size,
                bias=bias,
                device=device,
                dtype=dtype,
                tensor_parallel_mode="row",
                tensor_parallel_group=process_group,
                sequence_parallel=sequence_parallel,
            ),
            te_ops.GELU(),
        )
    with torch.no_grad():
        model[1].weight.copy_(w1_test)
        model[3].weight.copy_(w2_test)
        if bias:
            model[1].bias.copy_(b1_test)
            model[3].bias.copy_(b2_test)
        del w1_test, w2_test, b1_test, b2_test

    # Warmup steps
    for _ in range(3):
        with te.autocast(enabled=quantized_compute, recipe=recipe):
            y_test = model(x_test)
        y_test.backward(dy_test)
    x_test.grad = None
    model[1].weight.grad = None
    model[3].weight.grad = None
    if bias:
        model[1].bias.grad = None
        model[3].bias.grad = None

    # Forward and backward step
    with te.autocast(enabled=quantized_compute, recipe=recipe):
        y_test = model(x_test)
    y_test.backward(dy_test)

    # Expected numerical error
    tols = dtype_tols(dtype)
    if dtype == torch.float32:
        tols = dtype_tols(torch.float16)  # TF32 GEMM
    if quantized_compute:
        tols = quantization_tols(quantization)

    # Check results
    y_test = y_test.to(dtype=torch.float64, device="cpu")
    dx_test = x_test.grad.to(dtype=torch.float64, device="cpu")
    dw1_test = model[1].weight.grad.to(dtype=torch.float64, device="cpu")
    dw2_test = model[3].weight.grad.to(dtype=torch.float64, device="cpu")
    torch.testing.assert_close(y_test, y_ref, **tols)
    torch.testing.assert_close(dx_test, dx_ref, **tols)
    torch.testing.assert_close(dw1_test, dw1_ref, **tols)
    torch.testing.assert_close(dw2_test, dw2_ref, **tols)
    if bias:
        db1_test = model[1].bias.grad.to(dtype=torch.float64, device="cpu")
        db2_test = model[3].bias.grad.to(dtype=torch.float64, device="cpu")
        torch.testing.assert_close(db1_test, db1_ref, **tols)
        torch.testing.assert_close(db2_test, db2_ref, **tols)


def _test_fp8_scale_update(
    *,
    amax_history_len: int = 31,
    amax_compute_algo: str = "max",
    margin: float = 2,
    local_weight_shape: tuple[int, int] = (32, 32),
    batch_size: int = 32,
    dtype: torch.dtype = torch.float32,
    device: torch.device = "cuda",
    tensor_parallel_mode: str = "column",
) -> None:

    # Distributed process group
    process_group = world_group()
    rank = torch.distributed.get_rank(process_group)
    world_size = torch.distributed.get_world_size(process_group)

    # Tensor dimensions
    local_out_features, local_in_features = local_weight_shape
    out_features, in_features = local_out_features, local_in_features
    if tensor_parallel_mode == "column":
        out_features *= world_size
    elif tensor_parallel_mode == "row":
        in_features *= world_size
    in_shape = [batch_size, in_features]
    out_shape = [batch_size, out_features]

    # Random data
    reset_rng()
    x_ref, x_test = make_reference_and_test_tensors(
        in_shape,
        test_dtype=dtype,
        test_device=device,
    )
    w_ref, w_test = make_reference_and_test_tensors(
        (out_features, in_features),
        test_dtype=dtype,
        test_device=device,
    )
    dy_ref, dy_test = make_reference_and_test_tensors(
        out_shape,
        test_dtype=dtype,
        test_device=device,
        requires_grad=False,
    )

    def ref_amax_and_scale(
        ref: torch.Tensor,
        stage: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expected absmax and FP8 scale"""
        amax = ref.abs().amax()
        max_val = {
            "forward": 448.0,
            "backward": 57344.0,
        }[stage]
        scale = (max_val / amax) / (2**margin)
        amax = amax.to(dtype=torch.float32, device="cpu")
        scale = scale.to(dtype=torch.float32, device="cpu")
        return amax, scale

    # Compute expected amaxes and FP8 scales
    x_amax_ref, x_scale_ref = ref_amax_and_scale(x_ref, "forward")
    w_amax_ref, w_scale_ref = ref_amax_and_scale(w_ref, "forward")
    dy_amax_ref, dy_scale_ref = ref_amax_and_scale(dy_ref, "backward")

    # Convert to distributed tensors
    with torch.no_grad():
        if tensor_parallel_mode == "column":
            local_out_features = out_features // world_size
            local_slice = slice(
                rank * local_out_features,
                (rank + 1) * local_out_features,
            )
            w_ref = w_ref[local_slice, :]
            w_test = w_test[local_slice, :]
            dy_ref = dy_ref[..., local_slice]
            dy_test = dy_test[..., local_slice].clone()
        elif tensor_parallel_mode == "row":
            local_in_features = in_features // world_size
            local_slice = slice(
                rank * local_in_features,
                (rank + 1) * local_in_features,
            )
            w_ref = w_ref[:, local_slice]
            w_test = w_test[:, local_slice]
            x_ref = x_ref[..., local_slice]
            x_test = x_test[..., local_slice].clone()
    x_test.requires_grad_()

    # Initialize fusible operation
    op = te_ops.BasicLinear(
        in_features,
        out_features,
        device=device,
        dtype=dtype,
        tensor_parallel_mode=tensor_parallel_mode,
        tensor_parallel_group=process_group,
    )
    with torch.no_grad():
        op.weight.copy_(w_test)
        del w_test

    # Forward and backward pass
    fp8_format = transformer_engine.common.recipe.Format.HYBRID
    recipe = transformer_engine.common.recipe.DelayedScaling(
        margin=margin,
        fp8_format=fp8_format,
        amax_history_len=amax_history_len,
        amax_compute_algo=amax_compute_algo,
    )
    with te.autocast(recipe=recipe):
        y_test = op(x_test)
    y_test.backward(dy_test)

    # Check results
    x_quantizer = op.get_quantizer("forward", 0)
    w_quantizer = op.get_quantizer("forward", 1)
    dy_quantizer = op.get_quantizer("backward", 0)
    x_scale_test = x_quantizer.scale.to(dtype=torch.float32, device="cpu").reshape([])
    w_scale_test = w_quantizer.scale.to(dtype=torch.float32, device="cpu").reshape([])
    dy_scale_test = dy_quantizer.scale.to(dtype=torch.float32, device="cpu").reshape([])
    torch.testing.assert_close(x_scale_test, x_scale_ref)
    torch.testing.assert_close(w_scale_test, w_scale_ref)
    torch.testing.assert_close(dy_scale_test, dy_scale_ref)


def run_parallel_tests() -> None:
    """Run parallel tests"""

    run_cp_a2a_tests()

    # Distributed process group
    process_group = world_group()
    rank = torch.distributed.get_rank(process_group)
    world_size = torch.distributed.get_world_size(process_group)

    # Collective communication ops
    if rank == 0:
        print(f"Running _test_all_reduce")
    _test_all_reduce()
    for quantization in quantization_list:
        if rank == 0:
            print(f"Running _test_all_gather with quantization={quantization}")
        _test_all_gather(quantization=quantization)
    if rank == 0:
        print(f"Running _test_reduce_scatter")
    _test_reduce_scatter()

    # Basic linear op
    for config in itertools.product(
        quantization_list,
        ("column", "row"),
        (False, True),
    ):
        if rank == 0:
            print(f"Running _test_basic_linear with {config=}")
        quantization, tensor_parallel_mode, sequence_parallel = config
        _test_basic_linear(
            quantization=quantization,
            tensor_parallel_mode=tensor_parallel_mode,
            sequence_parallel=sequence_parallel,
        )

    # Linear op
    for config in itertools.product(
        quantization_list,
        ("column", "row"),
        (False, True),
    ):
        if rank == 0:
            print(f"Running _test_linear with {config=}")
        quantization, tensor_parallel_mode, sequence_parallel = config
        dtype = torch.bfloat16 if is_bf16_available() else torch.float32
        _test_linear(
            bias=True,  # bias=False is tested in _test_basic_linear
            dtype=dtype,
            quantization=quantization,
            tensor_parallel_mode=tensor_parallel_mode,
            sequence_parallel=sequence_parallel,
        )

    # MLP
    for config in itertools.product(quantization_list, (False, True)):
        if rank == 0:
            print(f"Running _test_mlp with {config=}")
        quantization, sequence_parallel = config
        dtype = torch.bfloat16 if is_bf16_available() else torch.float32
        _test_mlp(
            bias=True,  # bias=False is tested in _test_basic_linear
            dtype=dtype,
            quantization=quantization,
            sequence_parallel=sequence_parallel,
        )

    # FP8 scale update
    if fp8_available:
        if rank == 0:
            print(f"Running _test_fp8_scale_update")
        _test_fp8_scale_update()


def run_compile_parallel_tests() -> None:
    """Compile the existing TP numerical test, including the Linear+Bias fusion."""
    quantizations = [None, "fp8_current_scaling"] if fp8_available else [None]
    for quantization, mode, sp in itertools.product(
        quantizations, ("column", "row"), (False, True)
    ):
        if torch.distributed.get_rank(world_group()) == 0:
            print(f"Compile BasicLinear: {quantization=}, {mode=}, {sp=}", flush=True)
        _test_basic_linear(
            dtype=torch.bfloat16,
            quantization=quantization,
            tensor_parallel_mode=mode,
            sequence_parallel=sp,
            compile_model=True,
        )
        # Row-parallel Linear uses standalone collectives without compile support.
        if mode == "column":
            if torch.distributed.get_rank(world_group()) == 0:
                print(f"Compile Linear: {quantization=}, {mode=}, {sp=}", flush=True)
            _test_linear(
                bias=True,
                dtype=torch.bfloat16,
                quantization=quantization,
                tensor_parallel_mode=mode,
                sequence_parallel=sp,
                compile_model=True,
            )


# Parallel job sizes
_world_sizes = [torch.cuda.device_count()]
if 1 not in _world_sizes:
    _world_sizes.append(1)
if torch.cuda.device_count() >= 2 and 2 not in _world_sizes:
    _world_sizes.append(2)


@pytest.mark.parametrize("world_size", _world_sizes)
@pytest.mark.parametrize("compile_model", [False, True], ids=["eager", "compile"])
def test_distributed_fuser_ops(world_size: int, compile_model: bool) -> None:
    """Launch parallel job that runs parallel tests"""
    python_exe = pathlib.Path(sys.executable)
    current_file = pathlib.Path(__file__).resolve()
    command = [
        python_exe,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={world_size}",
        current_file,
        "--parallel",
    ]
    if compile_model:
        command.append("--compile")
    with tempfile.TemporaryDirectory(prefix="te-test-fusible-ops-") as temp_dir:
        env = dict(os.environ)
        env["NVTE_TEST_RDZV_PATH"] = str(pathlib.Path(temp_dir) / "rdzv")
        subprocess.run(command, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", action="store_true", help="Run parallel tests")
    parser.add_argument("--compile", action="store_true", help="Use the compiled TP test suite")
    parser.add_argument(
        "--cp-a2a", action="store_true", help="Run only CP A2A and projection pairs"
    )
    parser.add_argument(
        "--megaup", action="store_true", help="Enable MegaUP joint CP projection fusion"
    )
    args = parser.parse_args()
    if args.compile and not args.parallel:
        parser.error("--compile requires --parallel")
    if args.parallel:
        try:
            if args.cp_a2a:
                run_cp_a2a_tests(use_megaup=args.megaup)
            elif args.compile:
                run_compile_parallel_tests()
            else:
                run_parallel_tests()
            torch.distributed.barrier(world_group())
            torch.cuda.synchronize()
        finally:
            # Stop NCCL background work before the CUDA runtime shuts down.
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
