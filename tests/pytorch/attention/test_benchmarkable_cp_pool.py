# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Run with python -m pytest tests/pytorch/attention/test_benchmarkable_cp_pool.py.

One point exercises correctness -> timing -> correctness; heterogeneous requests
and failure recovery remain integration follow-ups. No --nvte-benchmark flag is needed.
"""

import json
import math
from pathlib import Path
import os

import pytest
import torch
from transformer_engine.pytorch import get_device_compute_capability


@pytest.mark.skipif(
    torch.cuda.device_count() < 2 or get_device_compute_capability() < (9, 0),
    reason="THD context parallelism needs two SM90-or-newer GPUs",
)
def test_cp_pool_case(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.setenv("TE_PATH", str(root))
    monkeypatch.setenv("PYTHONPATH", str(root) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    for name, value in dict(
        NVTE_FUSED_ATTN="1",
        NVTE_FLASH_ATTN="0",
        NVTE_UNFUSED_ATTN="0",
        NVTE_ALLOW_NONDETERMINISTIC_ALGO="1",
        NVTE_FP8_DPA_BWD="0",
        NVTE_CP_POOL_TIMING="1",
    ).items():
        monkeypatch.setenv(name, value)
    from test_attention_with_cp import PoolWorker, _submit

    pool = PoolWorker(2)
    pool._MAX_RETRIES = 0  # A replacement worker would invalidate the persistence check.
    requests = ("correctness_A", "benchmark_B", "correctness_C")
    evidence, launcher_pids = [], []
    try:
        for name in requests:
            output = tmp_path / name
            benchmark = name == "benchmark_B"
            _submit(
                pool,
                dtype="bf16",
                model="cp_2_0",
                qkv_format="thd",
                kernel_backend="FusedAttention",
                cp_comm_type="all_gather",
                fp8_bwd=False,
                fp8_dpa=False,
                fp8_mha=False,
                scaling_mode=None,
                f16_O=True,
                is_training=True,
                fa_pad_between_seqs=False,
                deterministic=False,
                load_balancing_strategy="DUAL_CHUNK_SWAP",
                softcap="0.0",
                case_mode="benchmark" if benchmark else "correctness",
                case_output_dir=str(output),
            )
            launcher_pids.append(pool.proc.pid)
            rows = [json.loads((output / f"rank{rank}.json").read_text()) for rank in range(2)]
            for rank, row in enumerate(rows):
                assert row["rank"] == rank and row["world_size"] == 2
                assert row["evaluations"] == (5 if benchmark else 1)
                assert row["references"] == row["verifications"] == (0 if benchmark else 1)
                assert row["full_reference_allocated"] is (not benchmark)
                assert row["global_shapes"][0][0] == 8192 and row["local_shapes"][0][0] == 4096
                assert row["point_objects_released"] and row["pool_group_alive"]
                assert row["initial_seed"] == 1234
                if not benchmark:
                    assert set(row["max_abs_errors"]) == {"out", "dq", "dk", "dv"}
                    assert all(math.isfinite(value) for value in row["max_abs_errors"].values())
            evidence.append(rows)
            if benchmark:
                records = [
                    json.loads((output / f"framework/rank{r}/benchmark_report.json").read_text())[
                        "records"
                    ]
                    for r in range(2)
                ]
                assert all(len(rank_records) == 1 for rank_records in records)
                records = [rank_records[0] for rank_records in records]
                assert records[0]["case_id"] == records[1]["case_id"]
                for rank, record in enumerate(records):
                    assert record["rank"] == rank and record["world_size"] == 2
                    assert record["status"] == "completed" and record["variant"] == "evaluation"
                    assert (
                        record["precondition_verified"] is False and record["inner_iterations"] == 1
                    )
                    assert record["warmup_iterations"] == 2 and record["iterations"] == 3
                    assert len(record["samples_ms"]) == 3
                    assert all(math.isfinite(value) and value > 0 for value in record["samples_ms"])
                maxima = [
                    max(values) for values in zip(*(record["samples_ms"] for record in records))
                ]
                (output / "aggregate.json").write_text(
                    json.dumps(
                        {
                            "statistic": "mean_of_per_iteration_rank_max_ms",
                            "samples_ms": maxima,
                            "latency_ms": sum(maxima) / len(maxima),
                        },
                        indent=2,
                    )
                    + "\n"
                )
        assert len(set(launcher_pids)) == 1
        for rank in range(2):
            rows = [point[rank] for point in evidence]
            assert [row["pool_completed_cases_before"] for row in rows] == (
                [0, 1, 2] if rank == 0 else [0, 0, 0]
            )
            for key in ("pid", "world_group_id", "auxiliary_group_id"):
                assert len({row[key] for row in rows}) == 1, key
        proc = pool.proc
    finally:
        pool.shutdown()
    assert proc.returncode == 0
