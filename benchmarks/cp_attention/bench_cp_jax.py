# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Context Parallel Attention Benchmark - JAX (Striped/RING or DualChunkSwap/RING)

Supports both THD (Striped) and BSHD (DualChunkSwap) layouts. Optionally runs
a no-CP single-device baseline for comparison (--baseline).

Launch:
  CUDA_VISIBLE_DEVICES=0,1 python bench_cp_jax.py --config llama3_8b --seq_len 32768 --layout thd
  CUDA_VISIBLE_DEVICES=0,1 python bench_cp_jax.py --config cp_1_0 --seq_len 4096 --layout thd --baseline
  CUDA_VISIBLE_DEVICES=0,1 python bench_cp_jax.py --config llama3_8b --seq_len 32768 --layout bshd
"""

import os
import argparse
import time
import statistics
from math import sqrt
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import value_and_grad, jit
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from transformer_engine.jax import autocast
from transformer_engine.jax.sharding import MeshResource
from transformer_engine.jax.attention import (
    AttnBiasType,
    AttnMaskType,
    AttnSoftmaxType,
    QKVLayout,
    reorder_causal_load_balancing,
    fused_attn,
    SequenceDescriptor,
    CPStrategy,
    ReorderStrategy,
)

# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    # name: (num_heads, num_gqa_groups, head_dim)
    "llama2_7b": (32, 32, 128),   # MHA
    "llama2_70b": (64, 8, 128),   # GQA
    "llama3_8b": (32, 8, 128),    # GQA
    # Test-size configs
    "cp_1_0": (12, 12, 128),      # MHA
    "cp_2_0": (12, 2, 128),       # GQA
}


def parse_args():
    parser = argparse.ArgumentParser(description="CP Attention Benchmark (JAX)")
    parser.add_argument("--config", type=str, default="llama3_8b",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--seq_len", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--layout", type=str, default="thd",
                        choices=["thd", "bshd"],
                        help="THD (Striped) or BSHD (DualChunkSwap)")
    parser.add_argument("--stripe_size", type=int, default=1,
                        help="Stripe size for Striped reordering (THD only)")
    parser.add_argument("--cp_strategy", type=str, default="ring",
                        choices=["ring", "all_gather"],
                        help="CP communication strategy")
    parser.add_argument("--window_size", type=int, default=None,
                        help="Sliding window size (symmetric). None = full causal attention")
    parser.add_argument("--baseline", action="store_true",
                        help="Also run a no-CP single-device baseline for comparison")
    parser.add_argument("--profile", action="store_true",
                        help="Reduce iters for nsys profiling (nsys wraps externally)")
    return parser.parse_args()


def setup_mesh(cp_size=None):
    """Create JAX mesh. If cp_size is None, use all devices."""
    devices = jax.devices()
    if cp_size is None:
        cp_size = len(devices)
    mesh_shape = (1, cp_size, 1)
    devices_array = np.asarray(devices[:cp_size]).reshape(mesh_shape)
    mesh = Mesh(devices_array, ("dp", "cp", "tpsp"))
    mesh_resource = MeshResource(dp_resource="dp", cp_resource="cp", tpsp_resource="tpsp")
    return mesh, mesh_resource, cp_size


def setup_mesh_no_cp():
    """Create JAX mesh with a single device (no CP)."""
    devices = jax.devices()
    mesh_shape = (1, 1, 1)
    devices_array = np.asarray(devices[:1]).reshape(mesh_shape)
    mesh = Mesh(devices_array, ("dp", "cp", "tpsp"))
    mesh_resource = MeshResource(dp_resource="dp", cp_resource="cp", tpsp_resource="tpsp")
    return mesh, mesh_resource


def make_fused_attn_fn(layout, head_dim, cp_load_balanced, stripe_size, cp_strategy,
                       window_size=None):
    """Build a wrapper around fused_attn with the right static kwargs."""
    scaling_factor = 1.0 / sqrt(head_dim)

    if layout == "thd":
        qkv_layout = QKVLayout.THD_THD_THD
        attn_mask_type = AttnMaskType.PADDING_CAUSAL_MASK
    else:
        qkv_layout = QKVLayout.BSHD_BSHD_BSHD
        attn_mask_type = AttnMaskType.CAUSAL_MASK

    def attn_fn(q, k, v, seq_desc, dropout_rng):
        return fused_attn(
            (q, k, v),
            None,  # bias
            seq_desc,
            dropout_rng,
            attn_bias_type=AttnBiasType.NO_BIAS,
            attn_mask_type=attn_mask_type,
            qkv_layout=qkv_layout,
            softmax_type=AttnSoftmaxType.VANILLA_SOFTMAX,
            scaling_factor=scaling_factor,
            dropout_probability=0.0,
            is_training=True,
            max_segments_per_seq=1,
            window_size=window_size,
            context_parallel_strategy=cp_strategy,
            context_parallel_causal_load_balanced=cp_load_balanced,
            stripe_size=stripe_size if layout == "thd" else None,
        ).astype(q.dtype)

    return attn_fn


def make_fused_attn_fn_no_cp(layout, head_dim, window_size=None):
    """Build fused_attn wrapper with no CP (single device baseline)."""
    scaling_factor = 1.0 / sqrt(head_dim)

    if layout == "thd":
        qkv_layout = QKVLayout.THD_THD_THD
        attn_mask_type = AttnMaskType.PADDING_CAUSAL_MASK
    else:
        qkv_layout = QKVLayout.BSHD_BSHD_BSHD
        attn_mask_type = AttnMaskType.CAUSAL_MASK

    def attn_fn(q, k, v, seq_desc, dropout_rng):
        return fused_attn(
            (q, k, v),
            None,
            seq_desc,
            dropout_rng,
            attn_bias_type=AttnBiasType.NO_BIAS,
            attn_mask_type=attn_mask_type,
            qkv_layout=qkv_layout,
            softmax_type=AttnSoftmaxType.VANILLA_SOFTMAX,
            scaling_factor=scaling_factor,
            dropout_probability=0.0,
            is_training=True,
            max_segments_per_seq=1,
            window_size=window_size,
            context_parallel_strategy=CPStrategy.DEFAULT,
            context_parallel_causal_load_balanced=False,
        ).astype(q.dtype)

    return attn_fn


def make_inputs_thd(batch_size, seq_len, num_heads_q, num_heads_kv, head_dim, cp_size,
                    stripe_size, dtype):
    """Generate Q/K/V and SequenceDescriptor for THD + Striped layout."""
    key = jax.random.PRNGKey(0)
    q_key, k_key, v_key = jax.random.split(key, 3)

    q = jax.random.uniform(q_key, (batch_size, seq_len, num_heads_q, head_dim), dtype, -1.0)
    k = jax.random.uniform(k_key, (batch_size, seq_len, num_heads_kv, head_dim), dtype, -1.0)
    v = jax.random.uniform(v_key, (batch_size, seq_len, num_heads_kv, head_dim), dtype, -1.0)

    segment_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    segment_pos = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch_size, seq_len))

    reorder_fn = partial(
        reorder_causal_load_balancing,
        strategy=ReorderStrategy.Striped,
        cp_size=cp_size,
        seq_dim=1,
        stripe_size=stripe_size,
    )
    q = reorder_fn(q)
    k = reorder_fn(k)
    v = reorder_fn(v)
    segment_ids_r = reorder_fn(segment_ids)
    segment_pos_r = reorder_fn(segment_pos)

    seq_desc = SequenceDescriptor.from_segment_ids_and_pos(
        (segment_ids_r, segment_ids_r),
        (segment_pos_r, segment_pos_r),
        is_thd=True,
        is_segment_ids_reordered=True,
    )

    return q, k, v, seq_desc


def make_inputs_bshd(batch_size, seq_len, num_heads_q, num_heads_kv, head_dim, cp_size, dtype):
    """Generate Q/K/V and SequenceDescriptor for BSHD + DualChunkSwap layout."""
    key = jax.random.PRNGKey(0)
    q_key, k_key, v_key = jax.random.split(key, 3)

    q = jax.random.uniform(q_key, (batch_size, seq_len, num_heads_q, head_dim), dtype, -1.0)
    k = jax.random.uniform(k_key, (batch_size, seq_len, num_heads_kv, head_dim), dtype, -1.0)
    v = jax.random.uniform(v_key, (batch_size, seq_len, num_heads_kv, head_dim), dtype, -1.0)

    reorder_fn = partial(
        reorder_causal_load_balancing,
        strategy=ReorderStrategy.DualChunkSwap,
        cp_size=cp_size,
        seq_dim=1,
    )
    q = reorder_fn(q)
    k = reorder_fn(k)
    v = reorder_fn(v)

    seqlens = jnp.full((batch_size,), seq_len, dtype=jnp.int32)
    seq_desc = SequenceDescriptor.from_seqlens((seqlens, seqlens))

    return q, k, v, seq_desc


def make_inputs_no_cp_thd(batch_size, seq_len, num_heads_q, num_heads_kv, head_dim, dtype):
    """Generate Q/K/V and SequenceDescriptor for THD, no CP."""
    key = jax.random.PRNGKey(0)
    q_key, k_key, v_key = jax.random.split(key, 3)

    q = jax.random.uniform(q_key, (batch_size, seq_len, num_heads_q, head_dim), dtype, -1.0)
    k = jax.random.uniform(k_key, (batch_size, seq_len, num_heads_kv, head_dim), dtype, -1.0)
    v = jax.random.uniform(v_key, (batch_size, seq_len, num_heads_kv, head_dim), dtype, -1.0)

    segment_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    segment_pos = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch_size, seq_len))

    seq_desc = SequenceDescriptor.from_segment_ids_and_pos(
        (segment_ids, segment_ids),
        (segment_pos, segment_pos),
        is_thd=True,
        is_segment_ids_reordered=False,
    )

    return q, k, v, seq_desc


def make_inputs_no_cp_bshd(batch_size, seq_len, num_heads_q, num_heads_kv, head_dim, dtype):
    """Generate Q/K/V and SequenceDescriptor for BSHD, no CP."""
    key = jax.random.PRNGKey(0)
    q_key, k_key, v_key = jax.random.split(key, 3)

    q = jax.random.uniform(q_key, (batch_size, seq_len, num_heads_q, head_dim), dtype, -1.0)
    k = jax.random.uniform(k_key, (batch_size, seq_len, num_heads_kv, head_dim), dtype, -1.0)
    v = jax.random.uniform(v_key, (batch_size, seq_len, num_heads_kv, head_dim), dtype, -1.0)

    seqlens = jnp.full((batch_size,), seq_len, dtype=jnp.int32)
    seq_desc = SequenceDescriptor.from_seqlens((seqlens, seqlens))

    return q, k, v, seq_desc


def get_seq_desc_sharding(seq_desc, mesh, mesh_resource):
    """Compute sharding for SequenceDescriptor leaves."""
    def to_dp_shardings(x):
        if x.ndim == 1:
            return NamedSharding(mesh, PartitionSpec(mesh_resource.dp_resource))
        return NamedSharding(mesh, PartitionSpec(
            mesh_resource.dp_resource, mesh_resource.cp_resource
        ))
    return jax.tree.map(to_dp_shardings, seq_desc)


def benchmark_jax(grad_fn_jit, q, k, v, seq_desc, mesh, mesh_resource, warmup, iters):
    """Run warmup + timed iterations, return list of times in ms."""
    with mesh, autocast(mesh_resource=mesh_resource):
        for i in range(warmup):
            loss, grads = grad_fn_jit(q, k, v, seq_desc)
            jax.block_until_ready((loss, grads))
            if i == 0:
                print(f"  First warmup done. Loss: {float(loss):.4f}")

    times_ms = []
    with mesh, autocast(mesh_resource=mesh_resource):
        for _ in range(iters):
            start = time.perf_counter()
            loss, grads = grad_fn_jit(q, k, v, seq_desc)
            jax.block_until_ready((loss, grads))
            elapsed = (time.perf_counter() - start) * 1000.0
            times_ms.append(elapsed)

    return times_ms


def compute_flops(batch_size, seq_len, num_heads, head_dim):
    """Compute causal fwd+bwd FLOPs."""
    fwd_flops = 4 * batch_size * num_heads * seq_len * seq_len * head_dim
    fwd_bwd_flops = 3.5 * fwd_flops
    causal_flops = 0.5 * fwd_bwd_flops
    return causal_flops


def report_results(label, times_ms, batch_size, seq_len, num_heads, head_dim):
    """Compute and return a dict of metrics."""
    median_ms = statistics.median(times_ms)
    mean_ms = statistics.mean(times_ms)
    std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    flops = compute_flops(batch_size, seq_len, num_heads, head_dim)
    tflops = flops / (median_ms / 1000.0) / 1e12
    tokens_per_sec = batch_size * seq_len / (median_ms / 1000.0)
    return {
        "label": label,
        "median_ms": median_ms,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "tflops": tflops,
        "tokens_per_sec": tokens_per_sec,
    }


def print_results(results_list):
    """Print a comparison table from a list of result dicts."""
    print(f"\n{'='*72}")
    print(f"{'Mode':<20} {'Median(ms)':>10} {'Mean(ms)':>10} {'Std(ms)':>10} "
          f"{'TFLOPS':>8} {'Tok/s':>12}")
    print(f"{'-'*72}")
    for r in results_list:
        print(f"{r['label']:<20} {r['median_ms']:>10.2f} {r['mean_ms']:>10.2f} "
              f"{r['std_ms']:>10.2f} {r['tflops']:>8.1f} {r['tokens_per_sec']:>12.0f}")

    if len(results_list) == 2:
        baseline = results_list[0]["median_ms"]
        cp = results_list[1]["median_ms"]
        speedup = baseline / cp
        print(f"{'-'*72}")
        print(f"  Speedup (CP vs no-CP): {speedup:.2f}x")
    print(f"{'='*72}")


def main():
    args = parse_args()
    num_heads, num_gqa_groups, head_dim = MODEL_CONFIGS[args.config]
    dtype = jnp.bfloat16

    strategy_map = {"ring": CPStrategy.RING, "all_gather": CPStrategy.ALL_GATHER}
    cp_strategy = strategy_map[args.cp_strategy]
    strategy_label = args.cp_strategy.upper()
    window_size = (args.window_size, args.window_size) if args.window_size else None
    window_label = f"W={args.window_size}" if args.window_size else "full"

    if args.profile:
        args.iters = min(args.iters, 3)
        args.warmup = min(args.warmup, 2)

    print(f"Config: {args.config} | Layout: {args.layout} | "
          f"B={args.batch_size}, S={args.seq_len}, H={num_heads}, "
          f"KV_groups={num_gqa_groups}, D={head_dim}")
    print(f"Devices: {[str(d) for d in jax.devices()]} | "
          f"CP strategy: {strategy_label} | Window: {window_label} | Baseline: {args.baseline}")
    print(f"Warmup: {args.warmup} | Timed iters: {args.iters}"
          f"{' | Profile: ON' if args.profile else ''}")

    results = []

    # ---- No-CP baseline (single device) ----
    if args.baseline:
        print(f"\n--- No-CP baseline (1 device, full S={args.seq_len}) ---")
        mesh_nocp, mr_nocp = setup_mesh_no_cp()

        if args.layout == "thd":
            bq, bk, bv, bseq_desc = make_inputs_no_cp_thd(
                args.batch_size, args.seq_len, num_heads, num_gqa_groups, head_dim, dtype,
            )
        else:
            bq, bk, bv, bseq_desc = make_inputs_no_cp_bshd(
                args.batch_size, args.seq_len, num_heads, num_gqa_groups, head_dim, dtype,
            )

        qkvo_sharding_nocp = NamedSharding(mesh_nocp, PartitionSpec(
            mr_nocp.dp_resource, mr_nocp.cp_resource, mr_nocp.tpsp_resource, None,
        ))
        seq_desc_sharding_nocp = get_seq_desc_sharding(bseq_desc, mesh_nocp, mr_nocp)

        bq = jax.device_put(bq, qkvo_sharding_nocp)
        bk = jax.device_put(bk, qkvo_sharding_nocp)
        bv = jax.device_put(bv, qkvo_sharding_nocp)
        bseq_desc = jax.device_put(bseq_desc, seq_desc_sharding_nocp)

        attn_fn_nocp = make_fused_attn_fn_no_cp(args.layout, head_dim, window_size=window_size)

        def loss_fn_nocp(q, k, v, seq_desc):
            return attn_fn_nocp(q, k, v, seq_desc, None).sum()

        grad_fn_nocp = value_and_grad(loss_fn_nocp, argnums=(0, 1, 2))
        grad_fn_nocp_jit = jit(
            grad_fn_nocp,
            in_shardings=[
                qkvo_sharding_nocp, qkvo_sharding_nocp, qkvo_sharding_nocp,
                seq_desc_sharding_nocp,
            ],
        )

        baseline_times = benchmark_jax(
            grad_fn_nocp_jit, bq, bk, bv, bseq_desc,
            mesh_nocp, mr_nocp, args.warmup, args.iters,
        )
        results.append(report_results(
            f"no-CP ({args.layout})", baseline_times,
            args.batch_size, args.seq_len, num_heads, head_dim,
        ))

        del bq, bk, bv, bseq_desc, grad_fn_nocp_jit

    # ---- CP run ----
    mesh, mesh_resource, cp_size = setup_mesh()
    print(f"\n--- CP ({strategy_label}, {args.layout}, {cp_size} devices, {window_label}) ---")

    if args.layout == "thd":
        q, k, v, seq_desc = make_inputs_thd(
            args.batch_size, args.seq_len, num_heads, num_gqa_groups, head_dim,
            cp_size, args.stripe_size, dtype,
        )
    else:
        q, k, v, seq_desc = make_inputs_bshd(
            args.batch_size, args.seq_len, num_heads, num_gqa_groups, head_dim,
            cp_size, dtype,
        )

    qkvo_pspec = PartitionSpec(
        mesh_resource.dp_resource,
        mesh_resource.cp_resource,
        mesh_resource.tpsp_resource,
        None,
    )
    qkvo_sharding = NamedSharding(mesh, qkvo_pspec)
    seq_desc_sharding = get_seq_desc_sharding(seq_desc, mesh, mesh_resource)

    q = jax.device_put(q, qkvo_sharding)
    k = jax.device_put(k, qkvo_sharding)
    v = jax.device_put(v, qkvo_sharding)
    seq_desc = jax.device_put(seq_desc, seq_desc_sharding)

    attn_fn = make_fused_attn_fn(
        args.layout, head_dim,
        cp_load_balanced=True,
        stripe_size=args.stripe_size,
        cp_strategy=cp_strategy,
        window_size=window_size,
    )

    def loss_fn(q, k, v, seq_desc):
        return attn_fn(q, k, v, seq_desc, None).sum()

    grad_fn = value_and_grad(loss_fn, argnums=(0, 1, 2))
    grad_fn_jit = jit(
        grad_fn,
        in_shardings=[qkvo_sharding, qkvo_sharding, qkvo_sharding, seq_desc_sharding],
    )

    cp_times = benchmark_jax(
        grad_fn_jit, q, k, v, seq_desc,
        mesh, mesh_resource, args.warmup, args.iters,
    )
    results.append(report_results(
        f"CP-{strategy_label} ({args.layout})", cp_times,
        args.batch_size, args.seq_len, num_heads, head_dim,
    ))

    # ---- Print comparison ----
    print_results(results)


if __name__ == "__main__":
    main()
