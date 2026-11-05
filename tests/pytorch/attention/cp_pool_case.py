# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Run a narrowly supported attention Case inside the existing CP worker pool."""

import gc
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import weakref

import torch
import torch.distributed as dist
from transformer_engine.common.testing import Case
from transformer_engine.common.testing.artifacts import write_run_artifacts
from transformer_engine.common.testing.distributed import LAUNCH_ENV
from transformer_engine.common.testing.runner import run_benchmark_point, run_correctness

from cp_case_helpers import (
    build_point,
    project_reference,
    reset_leaves,
    run_fwd_bwd,
    verify_results,
)


def make_pool_case(output, benchmark_only):
    rank, world, group = dist.get_rank(), dist.get_world_size(), dist.group.WORLD
    point, refs = [], []
    evidence = dict(
        rank=rank,
        world_size=world,
        pid=os.getpid(),
        world_group_id=id(group),
        auxiliary_group_id=id(sys.modules["run_attention_with_cp"]._pool_cp_comm_group),
        pool_completed_cases_before=sys.modules["__main__"]._case_counter,
        evaluations=0,
        references=0,
        verifications=0,
        initial_seed=torch.initial_seed(),
    )

    def track(state):
        values = (
            *state["bases"],
            *state["leaves"],
            state["dout"],
            state["attn"],
            state["idx"],
            *state["lengths"].values(),
            *state.get("full_bases", ()),
            state.get("full_dout"),
            state.get("reference_attn"),
        )
        refs.extend(weakref.ref(value) for value in values if value is not None)

    def borrow(rank_arg, world_arg, address, port):
        assert (rank_arg, world_arg) == (rank, world)
        assert dist.is_initialized() and dist.group.WORLD is group
        return {"rank": rank, "world": world, "evidence": evidence}

    def setup(state):
        state = build_point(state, benchmark_only)
        point[:] = [state]
        track(state)
        return state

    def reset(state):
        reset_leaves(state)
        track(state)

    def evaluate(state):
        evidence["evaluations"] += 1
        return run_fwd_bwd(
            state["attn"],
            state["leaves"],
            state["dout"],
            state["lengths"],
            retain_result=not benchmark_only,
        )

    def reference(state):
        evidence["references"] += 1
        return project_reference(state)

    def verify(actual, expected):
        evidence["max_abs_errors"] = verify_results(actual, expected, point[0]["cfg"])
        evidence["verifications"] += 1

    def barrier(state):
        dist.barrier()
        torch.cuda.synchronize()

    def clean(state):
        # The pool owns its groups; this callback releases only point-local objects.
        state.clear()
        point.clear()
        gc.collect()
        evidence["point_objects_released"] = all(ref() is None for ref in refs)
        evidence["pool_group_alive"] = dist.is_initialized() and dist.group.WORLD is group
        assert evidence["point_objects_released"] and evidence["pool_group_alive"]
        (output / f"rank{rank}.json").write_text(json.dumps(evidence, indent=2) + "\n")

    return Case(
        setup=setup,
        evaluate=evaluate,
        reference=None if benchmark_only else reference,
        verify=None if benchmark_only else verify,
        reset=reset,
        dist_init=borrow,
        dist_clean=clean,
        barrier=barrier,
        num_gpus=2,
        batchable=False,
        time_reference=False,
    )


def run_pool_case(output_dir, mode, kwargs):
    required = dict(
        dtype="bf16",
        model="cp_2_0",
        qkv_format="thd",
        kernel_backend="FusedAttention",
        cp_comm_type="all_gather",
        fp8_bwd="False",
        fp8_dpa="False",
        fp8_mha="False",
        scaling_mode="None",
        f16_O="True",
        is_training="True",
        fa_pad_between_seqs="False",
        deterministic="False",
        load_balancing_strategy="DUAL_CHUNK_SWAP",
        softcap="0.0",
    )
    assert all(
        kwargs[key] == value for key, value in required.items()
    ), "unsupported pool Case controls"
    assert mode in ("correctness", "benchmark") and output_dir is not None
    assert os.environ.get("NVTE_CP_POOL_PG") == "1" and dist.is_initialized()
    assert dist.get_world_size() == 2
    assert [
        os.environ.get(name) for name in ("NVTE_FUSED_ATTN", "NVTE_FLASH_ATTN", "NVTE_UNFUSED_ATTN")
    ] == ["1", "0", "0"]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # A pool worker inherits pytest's marker but has no pytest plugin. The runner
    # below consumes the Case directly, so remove only that stale marker temporarily.
    old_test = os.environ.pop("PYTEST_CURRENT_TEST", None)
    old_launch = os.environ.get(LAUNCH_ENV)
    node_id = f"{(old_test or 'test_cp_pool_case').removesuffix(' (call)')}/{output.name}"
    os.environ[LAUNCH_ENV] = node_id
    try:
        case = make_pool_case(output, mode == "benchmark")
        if mode == "correctness":
            run_correctness(case)
        else:
            # Integration TODO: replace this private launch marker and record proxy
            # with an explicit external-rank execution context when the runner exposes one.
            item = SimpleNamespace(
                path=__file__,
                originalname="test_cp_pool_case",
                nodeid=node_id,
                module=SimpleNamespace(__name__="test_benchmarkable_cp_pool"),
                callspec=SimpleNamespace(params=required),
            )
            settings = dict(
                warmup=2, iterations=3, inner_iterations=1, min_run_time=0.0, no_reference=True
            )
            records = run_benchmark_point(case, settings, item)
            write_run_artifacts(
                output / "framework" / f"rank{dist.get_rank()}",
                records,
                sys.argv,
                {"execution": "existing CP pool, direct Case runner"},
            )
    finally:
        if old_launch is None:
            os.environ.pop(LAUNCH_ENV, None)
        else:
            os.environ[LAUNCH_ENV] = old_launch
        if old_test is not None:
            os.environ["PYTEST_CURRENT_TEST"] = old_test
