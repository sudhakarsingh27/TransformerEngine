# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import copy
import os
import sys
import time
import logging
from contextlib import nullcontext
import torch
import torch.distributed as dist
from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import (
    get_cu_seqlens_on_cp_rank,
)
from transformer_engine.pytorch.attention.dot_product_attention.utils import combine_and_quantize
import transformer_engine_torch as tex
from transformer_engine.pytorch import DType
from test_attention_with_cp import model_configs_flash_attn, model_configs_fused_attn
from transformer_engine.pytorch import (
    autocast,
    DotProductAttention,
    Float8Quantizer,
    Float8CurrentScalingQuantizer,
    MXFP8Quantizer,
    make_graphed_callables,
)
from transformer_engine.common.recipe import (
    DelayedScaling,
    Float8CurrentScaling,
    MXFP8BlockScaling,
    Format,
)
from utils import ModelConfig, compare_and_assert

# Merge benchmark/stress configs into both backend maps so worker runs can use
# aliases like rl16k, bucket32k, and mixed32k_swa512 with either backend.
try:
    from benchmark_cp import model_configs_fused_attn as _bench_cfgs_fused_attn

    for _k, _v in _bench_cfgs_fused_attn.items():
        model_configs_fused_attn.setdefault(_k, _v)
        model_configs_flash_attn.setdefault(_k, _v)
except ImportError:
    pass

# Pool mode (NVTE_CP_POOL_PG=1) only: shared CP collective groups, created once
# per pool by run_attention_with_cp_pool.main() and reused across every case in
# that pool. world_size and the rank set don't change per case, so re-creating
# these per call would be wasted NCCL setup (~50-100 ms each). Single-shot
# subprocess mode leaves these None / [] and run_dpa_with_cp creates/destroys
# its own groups inline.
_pool_cp_comm_group = None
_pool_cp_comm_sub_groups: list = []

dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8": torch.bfloat16}


class _CUDAGraphDotProductAttentionAdapter(torch.nn.Module):
    """Expose a tensor-only Q/K/V surface for one fixed attention configuration."""

    def __init__(
        self,
        core_attn,
        *,
        core_attention_bias_type,
        core_attention_bias,
        cu_seqlens_q,
        cu_seqlens_kv,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        max_seqlen_q,
        max_seqlen_kv,
        pad_between_seqs,
    ):
        super().__init__()
        if core_attention_bias is not None:
            raise ValueError("CUDA graph prototype does not support an explicit attention bias")
        # Register the attention module so make_graphed_callables discovers its
        # TE state, while binding Python metadata here keeps strings and booleans
        # out of the helper's tensor-only sample input surface.
        self.core_attn = core_attn
        self.core_attention_bias_type = core_attention_bias_type
        self.register_buffer("cu_seqlens_q", cu_seqlens_q, persistent=False)
        self.register_buffer("cu_seqlens_kv", cu_seqlens_kv, persistent=False)
        self.register_buffer("cu_seqlens_q_padded", cu_seqlens_q_padded, persistent=False)
        self.register_buffer("cu_seqlens_kv_padded", cu_seqlens_kv_padded, persistent=False)
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_kv = max_seqlen_kv
        self.pad_between_seqs = pad_between_seqs

    def forward(self, q, k, v):
        return self.core_attn(
            q,
            k,
            v,
            core_attention_bias_type=self.core_attention_bias_type,
            core_attention_bias=None,
            cu_seqlens_q=self.cu_seqlens_q,
            cu_seqlens_kv=self.cu_seqlens_kv,
            cu_seqlens_q_padded=self.cu_seqlens_q_padded,
            cu_seqlens_kv_padded=self.cu_seqlens_kv_padded,
            max_seqlen_q=self.max_seqlen_q,
            max_seqlen_kv=self.max_seqlen_kv,
            # Backend choice, THD padding policy, and output dtype are
            # deliberately fixed for the lifetime of this per-config graph.
            pad_between_seqs=self.pad_between_seqs,
            fp8_output=False,
        )


def generate_input_shapes(
    qkv_format: str,
    config: ModelConfig,
    world_size: int,
    kernel_backend: str,
    fa_pad_between_seqs: str = "False",
    thd_seqlen_pattern: str = "random",
):
    if qkv_format == "bshd":
        q_input_shape = (
            config.batch_size,
            config.max_seqlen_q,
            config.num_heads,
            config.head_dim_qk,
        )
        k_input_shape = (
            config.batch_size,
            config.max_seqlen_kv,
            config.num_gqa_groups,
            config.head_dim_qk,
        )
        v_input_shape = (
            config.batch_size,
            config.max_seqlen_kv,
            config.num_gqa_groups,
            config.head_dim_v,
        )
        attn_output_shape = (
            config.batch_size,
            config.max_seqlen_q,
            config.num_heads * config.head_dim_v,
        )
        cu_seqlens_q = None
        cu_seqlens_kv = None
        cu_seqlens_q_padded = None
        cu_seqlens_kv_padded = None
    elif qkv_format == "sbhd":
        q_input_shape = (
            config.max_seqlen_q,
            config.batch_size,
            config.num_heads,
            config.head_dim_qk,
        )
        k_input_shape = (
            config.max_seqlen_kv,
            config.batch_size,
            config.num_gqa_groups,
            config.head_dim_qk,
        )
        v_input_shape = (
            config.max_seqlen_kv,
            config.batch_size,
            config.num_gqa_groups,
            config.head_dim_v,
        )
        attn_output_shape = (
            config.max_seqlen_q,
            config.batch_size,
            config.num_heads * config.head_dim_v,
        )
        cu_seqlens_q = None
        cu_seqlens_kv = None
        cu_seqlens_q_padded = None
        cu_seqlens_kv_padded = None
    elif qkv_format == "thd":
        b, s = config.batch_size, config.max_seqlen_q
        # Custom list: "24576,28672,30720,32768" -> explicit per-seq lengths
        if "," in thd_seqlen_pattern:
            seqlens_q = torch.tensor(
                [int(x) for x in thd_seqlen_pattern.split(",")], dtype=torch.int32
            )
            b = len(seqlens_q)
            s = int(seqlens_q.max())
            config.batch_size = b
            config.max_seqlen_q = s
            config.max_seqlen_kv = s
        elif thd_seqlen_pattern == "max":
            seqlens_q = torch.full([b], s, dtype=torch.int32)
        elif thd_seqlen_pattern == "half":
            seqlens_q = torch.full([b], s // 2, dtype=torch.int32)
        elif thd_seqlen_pattern == "linear":
            seqlens_q = torch.linspace(1, s, b).to(torch.int32)
        elif thd_seqlen_pattern == "alternating":
            seqlens_q = torch.tensor(
                [s if i % 2 == 0 else s // 4 for i in range(b)], dtype=torch.int32
            )
        else:  # "random"
            seqlens_q = torch.randint(0, s + 1, [b]).to(torch.int32)
        seqlens_q_padded = (seqlens_q + 2 * world_size - 1) // (world_size * 2) * (world_size * 2)
        cu_seqlens_q_padded = torch.cat(
            [
                torch.zeros([1], dtype=torch.int32),
                seqlens_q_padded.cumsum(0, dtype=torch.int32),
            ]
        ).cuda()
        cu_seqlens_q = torch.clone(cu_seqlens_q_padded)

        # Generate padded data (cu_seqlens_q reflects non-padded lengths, so it
        # differs from cu_seqlens_q_padded) for FusedAttention always, and for
        # FlashAttention only when its test param requests it. DPA auto-detects
        # pad_between_seqs downstream from the cu_seqlens_q vs cu_seqlens_q_padded
        # mismatch.
        if kernel_backend == "FusedAttention" or fa_pad_between_seqs == "True":
            cu_seqlens_q[1:] = seqlens_q.cumsum(0, dtype=torch.int32).cuda()

        # NOTE: In case of Cross-Attention, `cu_seqlens_kv` and `cu_seqlens_kv_padded`
        # will not be the same as `cu_seqlens_q` and `cu_seqlens_q_padded` respectively.
        cu_seqlens_kv = cu_seqlens_q
        cu_seqlens_kv_padded = cu_seqlens_q_padded

        total_tokens = cu_seqlens_q_padded[-1]

        q_input_shape = (
            total_tokens,
            config.num_heads,
            config.head_dim_qk,
        )
        k_input_shape = (
            total_tokens,
            config.num_gqa_groups,
            config.head_dim_qk,
        )
        v_input_shape = (
            total_tokens,
            config.num_gqa_groups,
            config.head_dim_v,
        )
        attn_output_shape = (
            total_tokens,
            config.num_heads * config.head_dim_v,
        )
    else:
        assert False, f"{qkv_format=} is not supported!"

    return (
        q_input_shape,
        k_input_shape,
        v_input_shape,
        attn_output_shape,
        cu_seqlens_q,
        cu_seqlens_kv,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
    )


def get_tols(config, dtype):
    if dtype == "bf16":
        if config.num_heads == config.num_gqa_groups:
            atol = 2.5e-2
            rtol = 2.5e-2
        else:
            atol = 3.5e-2
            rtol = 3.5e-2
        rmse_tol = 0.01
    elif dtype == "fp16":
        atol = 5e-3
        rtol = 5e-3
        rmse_tol = 0.01
    elif dtype == "fp8":
        atol = 5e-1
        rtol = 5e-1
        rmse_tol = 0.15
    else:
        assert False, f"{dtype=} is not supported!"

    return atol, rtol, rmse_tol


def run_dpa_with_cp(
    dtype="bf16",
    model=None,
    qkv_format="bshd",
    kernel_backend="FlashAttention",
    cp_comm_type="p2p",
    fp8_bwd="True",
    fp8_dpa="False",
    fp8_mha="False",
    scaling_mode="delayed",
    f16_O="False",
    is_training="True",
    fa_pad_between_seqs="False",
    deterministic="False",
    log_level=logging.WARNING,
    benchmark="0",
    thd_seqlen_pattern="random",
    cuda_graph="False",
    cuda_graph_warmup="3",
):
    """Test DotProductAttention module with context parallelism"""
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    logging.root.setLevel(log_level)
    # When is_training is False, gradient outputs are None.
    is_training = is_training == "True"
    benchmark_iters = int(benchmark)
    use_cuda_graph = cuda_graph == "True"
    cuda_graph_warmup_iters = int(cuda_graph_warmup)
    cp_bench_only = os.getenv("NVTE_CP_BENCH_ONLY", "0") == "1"
    if cp_bench_only and benchmark_iters <= 0:
        raise ValueError("NVTE_CP_BENCH_ONLY requires benchmark > 0")
    if use_cuda_graph:
        # Keep the prototype narrow enough that capture failures remain
        # attributable to CP/NCCL. FlashAttention uses the same static
        # BF16/THD/training graph adapter as FusedAttention, so allow both while
        # retaining the restrictions on stateful FP8, layout, and inference.
        if dtype != "bf16":
            raise ValueError("cuda_graph=True prototype only supports dtype=bf16")
        if qkv_format != "thd":
            raise ValueError("cuda_graph=True prototype only supports qkv_format=thd")
        if kernel_backend not in ("FusedAttention", "FlashAttention"):
            raise ValueError(
                "cuda_graph=True prototype only supports FusedAttention or FlashAttention"
            )
        if not is_training:
            raise ValueError("cuda_graph=True prototype only supports is_training=True")
        if benchmark_iters <= 0:
            raise ValueError("cuda_graph=True requires benchmark > 0")
        if cuda_graph_warmup_iters <= 0:
            raise ValueError("cuda_graph_warmup must be positive")
        # Benchmark-only mode skips the full non-CP reference, but the graph
        # path below still runs one eager CP pass as its correctness oracle.
        # This keeps large fixed-token workloads on the rank-local input path
        # used by benchmark_cp without giving up eager-versus-graph validation.
    if cp_bench_only and int(os.getenv("RANK", "0")) == 0:
        correctness_paths = "eager_cp_vs_graph" if use_cuda_graph else "skipped"
        print(
            f"CP_BENCH_ONLY correctness_paths={correctness_paths}"
            " non_cp_reference=skipped inputs=rank_local",
            flush=True,
        )

    # set up environment variables and config
    if deterministic == "True":
        os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
    else:
        os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "1"
    fp8_bwd = fp8_bwd == "True" and dtype == "fp8"
    os.environ["NVTE_FP8_DPA_BWD"] = "1" if fp8_bwd else "0"
    fp8_dpa = fp8_dpa == "True" and dtype == "fp8"
    fp8_mha = fp8_mha == "True" and dtype == "fp8" and scaling_mode != "mxfp8"
    f16_O = dtype == "fp8" and scaling_mode in ["current", "mxfp8"] and f16_O == "True"
    os.environ["NVTE_DPA_FP8CS_O_in_F16"] = "1" if f16_O else "0"
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    if kernel_backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
        # Deep-copy: the module-level dict is shared across pool cases; the
        # THD branch below rewrites attn_mask_type in place, which would
        # otherwise leak into subsequent cases reusing the same model key.
        config = copy.deepcopy(model_configs_flash_attn[model])
    if kernel_backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"
        config = copy.deepcopy(model_configs_fused_attn[model])
    assert config.attn_mask_type in [
        "causal",
        "no_mask",
    ], f"{config.attn_mask_type=} is not supported!"
    if qkv_format == "thd":
        if "causal" in config.attn_mask_type:
            config.attn_mask_type = "padding_causal"
        else:
            config.attn_mask_type = "padding"

    # set up distributed group
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    # When NVTE_CP_POOL_PG=1, the pool runner owns the lifecycle of the main
    # process group across many cases; here we only reuse it.
    _pool_managed_pg = os.getenv("NVTE_CP_POOL_PG", "0") == "1"
    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        device_count = torch.cuda.device_count()
        device = rank % device_count
        torch.cuda.set_device(device)
    logging.info(f"[Rank {rank}] Setup: world_size {world_size}")
    if not _pool_managed_pg:
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    if use_cuda_graph and world_size != 2:
        raise ValueError("cuda_graph=True prototype requires exactly 2 ranks (CP=2)")

    # Set up communication group for CP. In pool mode, the pool worker has
    # already pre-created world-scoped and a2a+p2p sub-groups once and stashed
    # them in module-level pointers; we reuse those and the pool destroys them
    # at shutdown. In single-shot mode we create them per call and destroy in
    # the finally below.
    cp_comm_ranks = range(world_size)
    assert rank in cp_comm_ranks
    _reusing_pool_groups = _pool_managed_pg and _pool_cp_comm_group is not None
    cp_comm_group = None
    cp_comm_sub_groups: list = []
    if _reusing_pool_groups:
        cp_comm_group = _pool_cp_comm_group
        cp_comm_sub_groups = _pool_cp_comm_sub_groups if cp_comm_type == "a2a+p2p" else []
    else:
        cp_comm_group = dist.new_group(cp_comm_ranks, backend="nccl")
        if cp_comm_type == "a2a+p2p":
            assert world_size % 2 == 0, (
                "{cp_comm_type=} requires world_size % 2 = 0 as it assumes the a2a level has"
                " cp_size = 2."
            )
            cp_comm_sub_ranks = [range(i * 2, (i + 1) * 2) for i in range(world_size // 2)]
            cp_comm_sub_ranks += [range(i, world_size, 2) for i in range(2)]
            for sub_ranks in cp_comm_sub_ranks:
                sub_group = dist.new_group(sub_ranks, backend="nccl")
                if rank in sub_ranks:
                    cp_comm_sub_groups.append(sub_group)
    if dtype == "fp8":
        if scaling_mode == "delayed":
            fp8_recipe = DelayedScaling(fp8_dpa=fp8_dpa, fp8_mha=fp8_mha)
        if scaling_mode == "current":
            fp8_recipe = Float8CurrentScaling(fp8_dpa=fp8_dpa, fp8_mha=fp8_mha)
        if scaling_mode == "mxfp8":
            fp8_recipe = MXFP8BlockScaling(fp8_format=Format.E4M3, fp8_dpa=fp8_dpa, fp8_mha=fp8_mha)

    # instantiate attention module
    core_attn = DotProductAttention(
        config.num_heads,
        (config.head_dim_qk, config.head_dim_v),
        num_gqa_groups=config.num_gqa_groups,
        attention_dropout=config.dropout_p,
        qkv_format=qkv_format,
        attn_mask_type=config.attn_mask_type,
        window_size=config.window_size,
        softmax_type=config.softmax_type,
        return_max_logit=config.return_max_logit,
    ).cuda()
    if not is_training:
        core_attn.eval()
    if is_training and config.softmax_type != "vanilla":
        core_attn.softmax_offset.requires_grad = True

    # generate attention inputs
    (
        q_input_shape,
        k_input_shape,
        v_input_shape,
        attn_output_shape,
        cu_seqlens_q,
        cu_seqlens_kv,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
    ) = generate_input_shapes(
        qkv_format, config, world_size, kernel_backend, fa_pad_between_seqs, thd_seqlen_pattern
    )
    input_shapes = [q_input_shape, k_input_shape, v_input_shape, attn_output_shape]
    if cp_bench_only:
        # Performance mode needs only rank-local CP leaves; full inputs exist solely
        # to make correctness comparisons deterministic across the two paths.
        if qkv_format in ("bshd", "sbhd"):
            seq_dim = qkv_format.index("s")
            input_shapes = [list(shape) for shape in input_shapes]
            for shape in input_shapes:
                shape[seq_dim] //= world_size
            seq_idx = torch.tensor([rank, 2 * world_size - rank - 1], device="cuda")
        elif qkv_format == "thd":
            seq_idx_q = tex.thd_get_partitioned_indices(
                cu_seqlens_q_padded, int(q_input_shape[0]), world_size, rank
            )
            seq_idx_kv = tex.thd_get_partitioned_indices(
                cu_seqlens_kv_padded, int(k_input_shape[0]), world_size, rank
            )
            input_shapes = [
                (seq_idx_q.numel(), *q_input_shape[1:]),
                (seq_idx_kv.numel(), *k_input_shape[1:]),
                (seq_idx_kv.numel(), *v_input_shape[1:]),
                (seq_idx_q.numel(), *attn_output_shape[1:]),
            ]
        q_, k_, v_, dout_ = [
            torch.clamp(torch.randn(shape, dtype=dtypes[dtype]), min=-1, max=1).cuda()
            for shape in input_shapes
        ]
    else:
        q_orig, k_orig, v_orig, dout_orig = [
            torch.clamp(torch.randn(shape, dtype=dtypes[dtype]), min=-1, max=1).cuda()
            for shape in input_shapes
        ]
    # Save inputs for cross-backend comparison
    _save_path = os.environ.get("CP_CROSS_BACKEND_SAVE_DIR")
    if _save_path and not cp_bench_only:
        os.makedirs(_save_path, exist_ok=True)
        torch.save(
            {
                "q": q_orig,
                "k": k_orig,
                "v": v_orig,
                "dout": dout_orig,
                "cu_seqlens_q": cu_seqlens_q,
                "cu_seqlens_q_padded": cu_seqlens_q_padded,
            },
            os.path.join(_save_path, f"inputs_rank{rank}.pt"),
        )
    if scaling_mode == "delayed":
        qkv_quantizer = Float8Quantizer(
            fp8_dtype=DType.kFloat8E4M3,
            scale=torch.tensor([1], dtype=torch.float32).cuda(),
            amax=torch.tensor([0], dtype=torch.float32).cuda(),
        )
        dout_quantizer = Float8Quantizer(
            fp8_dtype=DType.kFloat8E5M2,
            scale=torch.tensor([1], dtype=torch.float32).cuda(),
            amax=torch.tensor([0], dtype=torch.float32).cuda(),
        )
    if scaling_mode == "current":
        qkv_quantizer = Float8CurrentScalingQuantizer(
            fp8_dtype=DType.kFloat8E4M3,
            device="cuda",
        )
        dout_quantizer = Float8CurrentScalingQuantizer(
            fp8_dtype=DType.kFloat8E5M2,
            device="cuda",
        )
    if scaling_mode == "mxfp8":
        qkv_quantizer = MXFP8Quantizer(
            fp8_dtype=DType.kFloat8E4M3,
            rowwise=True,
            columnwise=True,
        )
        qkv_quantizer.optimize_for_gemm = True
        qkv_quantizer.internal = False
        dout_quantizer = MXFP8Quantizer(
            fp8_dtype=DType.kFloat8E5M2,
            rowwise=True,
            columnwise=True,
        )
        dout_quantizer.optimize_for_gemm = True
        dout_quantizer.internal = False
    qkv_layout = "_".join([qkv_format] * 3)
    if not cp_bench_only:
        # Correctness mode shares full storage until independent CP shards exist.
        q, k, v, dout = [x.detach() for x in [q_orig, k_orig, v_orig, dout_orig]]
        if fp8_mha:
            q, k, v, qkv_layout, _ = combine_and_quantize(qkv_layout, q, k, v, qkv_quantizer)
        for x in [q, k, v]:
            x.requires_grad = True

    if config.attn_bias_type not in ["no_bias", "alibi"]:
        bias_shape_map = {
            "1hss": (1, config.num_heads, config.max_seqlen_q, config.max_seqlen_kv),
            "11ss": (1, 1, config.max_seqlen_q, config.max_seqlen_kv),
            "b1ss": (config.batch_size, 1, config.max_seqlen_q, config.max_seqlen_kv),
            "bhss": (
                config.batch_size,
                config.num_heads,
                config.max_seqlen_q,
                config.max_seqlen_kv,
            ),
            "111s": (1, 1, 1, config.max_seqlen_kv),
        }
        attn_bias_shape = bias_shape_map.get(config.bias_shape)
        if attn_bias_shape is None:
            assert False, f"cuDNN does not support {config.bias_shape=}"
        bias = torch.randn(*attn_bias_shape, dtype=dtypes[dtype]).cuda()
        # cuDNN does not support dbias calculation for 111s as of cuDNN 9.18
        # TODO(KshitijLakhani): Set requires_grad to True for all shapes once 111s is supported
        bias.requires_grad = True if config.bias_shape != "111s" else False
    else:
        bias = None

    if dtype == "fp8":
        fp8_context = autocast(enabled=True, recipe=fp8_recipe, amax_reduction_group=cp_comm_group)
    else:
        fp8_context = nullcontext()
    if not cp_bench_only:
        ############ run without CP ############
        logging.info(f"[Rank {rank}] Run without context parallelism")
        max_logit = None
        with fp8_context:
            # q, k, v, out in FP8; dout in F16
            out = core_attn(
                q,
                k,
                v,
                core_attention_bias_type=config.attn_bias_type,
                core_attention_bias=bias,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                cu_seqlens_q_padded=cu_seqlens_q_padded,
                cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                # Test runner sets cu_seqlens_q == cu_seqlens_q_padded for the
                # FlashAttention path, i.e. no inter-sequence padding. Declare this
                # explicitly so the sync-free auto-detect (which conservatively
                # picks True when padded cu_seqlens are present) does not disable FA.
                pad_between_seqs=(
                    (kernel_backend != "FlashAttention") if qkv_format == "thd" else None
                ),
                fp8_output=fp8_mha,
            )
            if config.return_max_logit:
                out, max_logit = out
            if is_training:
                if fp8_bwd and fp8_mha:
                    dout_fp8 = dout_quantizer(dout)
                    out.backward(dout_fp8)
                else:
                    out.backward(dout)
        if is_training:
            dq, dk, dv, dbias = q.grad, k.grad, v.grad, bias.grad if bias is not None else None
            d_softmax_offset = (
                core_attn.softmax_offset.grad if config.softmax_type != "vanilla" else None
            )
        else:
            dq, dk, dv, dbias = None, None, None, None
            d_softmax_offset = None

    ############ run with CP ############
    logging.info(f"[Rank {rank}] Run with context parallelism")

    # set up inputs
    bias_ = bias.clone().detach() if bias is not None else None
    if not cp_bench_only:
        q_, k_, v_, dout_ = [x.detach() for x in [q_orig, k_orig, v_orig, dout_orig]]
        if qkv_format == "bshd" or qkv_format == "sbhd":
            seq_dim = qkv_format.index("s")
            q_, k_, v_, dout_ = [
                x.view(
                    *x.shape[:seq_dim],
                    2 * world_size,
                    x.shape[seq_dim] // (2 * world_size),
                    *x.shape[(seq_dim + 1) :],
                )
                for x in [q_, k_, v_, dout_]
            ]
            seq_idx = torch.tensor([rank, 2 * world_size - rank - 1], device=q_.device)
            q_, k_, v_, dout_ = [
                x.index_select(seq_dim, seq_idx) for x in [q_, k_, v_, dout_]
            ]
            q_, k_, v_, dout_ = [
                x.view(*x.shape[:seq_dim], -1, *x.shape[(seq_dim + 2) :])
                for x in [q_, k_, v_, dout_]
            ]
        elif qkv_format == "thd":
            seq_idx_q = tex.thd_get_partitioned_indices(
                cu_seqlens_q_padded, q_.shape[0], world_size, rank
            )
            seq_idx_kv = tex.thd_get_partitioned_indices(
                cu_seqlens_kv_padded, k_.shape[0], world_size, rank
            )
            q_, dout_ = [x.index_select(0, seq_idx_q) for x in [q_, dout_]]
            k_, v_ = [x.index_select(0, seq_idx_kv) for x in [k_, v_]]
        else:
            assert False, f"{qkv_format} is an unsupported qkv_format!"
    q_, k_, v_, dout_ = [x.contiguous() for x in [q_, k_, v_, dout_]]
    if not cp_bench_only:
        # index_select owns shard storage, so full generated inputs are no longer needed.
        out = out.detach()
        if max_logit is not None:
            max_logit = max_logit.detach()
        del q, k, v, dout, q_orig, k_orig, v_orig, dout_orig
    if scaling_mode == "delayed":
        qkv_quantizer.scale.fill_(1.0)
        qkv_quantizer.amax.fill_(0.0)
        dout_quantizer.scale.fill_(1.0)
        dout_quantizer.amax.fill_(0.0)
    if fp8_mha:
        q_, k_, v_, qkv_layout, _ = combine_and_quantize(qkv_layout, q_, k_, v_, qkv_quantizer)
    if is_training:
        q_, k_, v_ = [x.requires_grad_() for x in [q_, k_, v_]]
    if bias_ is not None:
        ndim = bias_.ndim
        seq_q_dim = ndim - 2
        if qkv_format == "thd":
            bias_seq_idx = seq_idx_q
        else:
            bias_seq_idx = seq_idx
        shape_before_seq = bias_.shape[:seq_q_dim]
        seq_q_size = bias_.shape[seq_q_dim]
        seq_kv_size = bias_.shape[-1]
        if seq_q_size == 1:
            # TODO(KshitijLakhani): Set to True always once cuDNN supports dbias for 111s
            bias_.requires_grad = False
            # Bias is broadcast, no need to partition along sequence dimension
            pass
        else:
            bias_ = bias_.view(
                *shape_before_seq, 2 * world_size, seq_q_size // (2 * world_size), seq_kv_size
            )
            bias_ = bias_.index_select(seq_q_dim, bias_seq_idx)
            bias_ = bias_.view(*shape_before_seq, -1, seq_kv_size)
            bias_.requires_grad = True
    # set up environment
    core_attn.set_context_parallel_group(
        cp_comm_sub_groups if cp_comm_type == "a2a+p2p" else cp_comm_group,
        cp_comm_ranks,
        torch.cuda.Stream(),
        cp_comm_type,
    )
    if config.softmax_type != "vanilla" and core_attn.softmax_offset.grad is not None:
        core_attn.softmax_offset.grad.zero_()
    if dtype == "fp8":
        core_attn.fp8_initialized = False
        core_attn.fp8_meta_tensors_initialized = False
        fp8_context = autocast(enabled=True, recipe=fp8_recipe, amax_reduction_group=cp_comm_group)
    else:
        fp8_context = nullcontext()
    if use_cuda_graph:
        if config.dropout_p != 0.0:
            raise ValueError("cuda_graph=True prototype requires attention_dropout=0")
        if config.attn_bias_type not in ("no_bias", "alibi"):
            raise ValueError("cuda_graph=True prototype does not capture an explicit attention bias")
        if config.softmax_type != "vanilla" or config.return_max_logit:
            raise ValueError(
                "cuda_graph=True prototype requires vanilla softmax without return_max_logit"
            )

    if not cp_bench_only or use_cuda_graph:
        # CUDA Graph benchmark-only runs still need one eager CP result as the
        # graph correctness oracle; ordinary benchmark-only runs enter timing
        # directly and continue to skip all correctness work.
        max_logit_ = None
        with fp8_context:
            # q, k, v, out in FP8; dout in F16
            out_ = core_attn(
                q_,
                k_,
                v_,
                core_attention_bias_type=config.attn_bias_type,
                core_attention_bias=bias_,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                cu_seqlens_q_padded=cu_seqlens_q_padded,
                cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                # See note above (non-CP branch): same explicit declaration so
                # FlashAttention isn't disabled by the conservative sync-free
                # auto-detect when this test path constructs no inter-seq padding.
                pad_between_seqs=(
                    (kernel_backend != "FlashAttention") if qkv_format == "thd" else None
                ),
                fp8_output=fp8_mha,
            )
            if config.return_max_logit:
                out_, max_logit_ = out_
            if is_training:
                if fp8_bwd and fp8_mha:
                    dout_fp8_ = dout_quantizer(dout_)
                    out_.backward(dout_fp8_)
                else:
                    out_.backward(dout_)
        if is_training:
            dq_, dk_, dv_, dbias_ = (
                q_.grad,
                k_.grad,
                v_.grad,
                bias_.grad if bias_ is not None else None,
            )
            if use_cuda_graph:
                # The graph checks need the eager values, not their autograd
                # history. Drop that history before allocating the capture
                # pool for the largest fixed-token configurations.
                out_ = out_.detach()
            d_softmax_offset_ = (
                core_attn.softmax_offset.grad.clone() if config.softmax_type != "vanilla" else None
            )
        else:
            dq_, dk_, dv_, dbias_ = None, None, None, None
            d_softmax_offset_ = None

    if _save_path and not cp_bench_only:
        torch.save(
            {
                "out": out_.detach(),
                "dq": dq_.detach() if dq_ is not None else None,
                "dk": dk_.detach() if dk_ is not None else None,
                "dv": dv_.detach() if dv_ is not None else None,
            },
            os.path.join(_save_path, f"outputs_{cp_comm_type}_rank{rank}.pt"),
        )

    # Benchmark: re-run forward+backward with timing
    if benchmark_iters > 0:
        warmup = 10
        t0 = None
        for it in range(warmup + benchmark_iters):
            q_b, k_b, v_b = [x.clone().detach().requires_grad_() for x in [q_, k_, v_]]
            torch.cuda.synchronize()
            if it == warmup:
                torch.cuda.cudart().cudaProfilerStart()
                t0 = time.perf_counter()
            with fp8_context:
                out_b = core_attn(
                    q_b,
                    k_b,
                    v_b,
                    core_attention_bias_type=config.attn_bias_type,
                    core_attention_bias=bias_,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_kv,
                    cu_seqlens_q_padded=cu_seqlens_q_padded,
                    cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                    # Match the correctness and graph paths. FlashAttention
                    # receives compact uniform THD input, while FusedAttention
                    # uses its padded-sequence path.
                    pad_between_seqs=(
                        (kernel_backend != "FlashAttention") if qkv_format == "thd" else None
                    ),
                    fp8_output=fp8_mha,
                )
                if isinstance(out_b, tuple):
                    out_b = out_b[0]
                if is_training:
                    out_b.backward(dout_)
            torch.cuda.synchronize()
            # Do not carry a completed graph and its leaf gradients into the next iteration.
            del out_b, q_b, k_b, v_b
        elapsed = (time.perf_counter() - t0) / benchmark_iters * 1000
        torch.cuda.cudart().cudaProfilerStop()
        print(
            f"[Rank {rank}] {cp_comm_type} {qkv_format} {dtype}: {elapsed:.2f} ms/iter"
            f" ({benchmark_iters} iters)",
            flush=True,
        )
        # Distributed latency is bounded by the slower CP rank. Emit a
        # machine-readable MAX just like the CUDA Graph result so comparisons
        # do not depend on parsing interleaved per-rank stdout.
        eager_max_ms = torch.tensor(elapsed, dtype=torch.float64, device="cuda")
        dist.all_reduce(eager_max_ms, op=dist.ReduceOp.MAX, group=cp_comm_group)
        if rank == 0:
            print(
                "EAGER_RESULT"
                f" model={model} backend={kernel_backend} comm={cp_comm_type}"
                f" cp={world_size} qkv={qkv_format} dtype={dtype}"
                f" wall_ms={eager_max_ms.item():.3f}"
                f" iters={benchmark_iters}",
                flush=True,
            )

    if use_cuda_graph:
        def rounded_max_seqlen(cu_seqlens, cu_seqlens_padded):
            boundaries = cu_seqlens_padded if cu_seqlens_padded is not None else cu_seqlens
            seqlens = boundaries[1:] - boundaries[:-1]
            return int((seqlens.max().item() + 63) // 64 * 64)

        # DPA performs this same THD inference when max_seqlen is omitted, but
        # its device-to-host .item() is illegal once CUDA capture has started.
        # One graph is built per exact static sample, so bind the inferred Python
        # integers before capture without changing eager execution semantics.
        graph_max_seqlen_q = rounded_max_seqlen(cu_seqlens_q, cu_seqlens_q_padded)
        graph_max_seqlen_kv = rounded_max_seqlen(cu_seqlens_kv, cu_seqlens_kv_padded)
        graph_adapter = _CUDAGraphDotProductAttentionAdapter(
            core_attn,
            core_attention_bias_type=config.attn_bias_type,
            core_attention_bias=bias_,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            cu_seqlens_q_padded=cu_seqlens_q_padded,
            cu_seqlens_kv_padded=cu_seqlens_kv_padded,
            max_seqlen_q=graph_max_seqlen_q,
            max_seqlen_kv=graph_max_seqlen_kv,
            pad_between_seqs=(
                (kernel_backend != "FlashAttention") if qkv_format == "thd" else None
            ),
        )
        graph_inputs = tuple(x.detach().clone().requires_grad_() for x in (q_, k_, v_))

        def capture_barrier():
            # Every rank must issue CP collectives in the same capture order.
            # Synchronizing both sides of make_graphed_callables' eager warmup
            # prevents a faster rank from entering NCCL capture early.
            torch.cuda.synchronize()
            dist.barrier(group=cp_comm_group)
            torch.cuda.synchronize()

        capture_barrier()
        torch.cuda.reset_peak_memory_stats()
        capture_start = time.perf_counter()
        graphed_core_attn = make_graphed_callables(
            graph_adapter,
            graph_inputs,
            num_warmup_iters=cuda_graph_warmup_iters,
            enabled=False,
            pre_warmup_hook=capture_barrier,
            post_warmup_hook=capture_barrier,
        )
        capture_barrier()
        capture_ms = (time.perf_counter() - capture_start) * 1000.0
        capture_peak_mib = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

        # Validate one replay against the eager CP pass before collecting timing.
        # Graph buffers are reused, so compare before the next replay overwrites them.
        for tensor in graph_inputs:
            tensor.grad = None
        graph_out = graphed_core_attn(*graph_inputs)
        torch.cuda.synchronize()
        atol, rtol, rmse_tol = get_tols(config, dtype)
        # Check before backward so a failure identifies forward replay itself,
        # rather than a forward buffer overwritten by the captured backward.
        compare_and_assert(
            out_,
            graph_out,
            "out_cp_eager",
            "out_cp_cuda_graph_pre_backward",
            atol,
            rtol,
            rmse_tol,
            False,
        )
        if rank == 0:
            print(
                "CUDA_GRAPH_FORWARD_CORRECTNESS"
                f" model={model} backend={kernel_backend} comm={cp_comm_type}"
                " cp=2 qkv=thd dtype=bf16 eager_cp_vs_graph=passed",
                flush=True,
            )
        graph_out.backward(dout_)
        torch.cuda.synchronize()
        graph_tensors = [graph_out, *(tensor.grad for tensor in graph_inputs)]
        eager_tensors = [out_, dq_, dk_, dv_]
        graph_names = ["out", "dq", "dk", "dv"]
        for eager_tensor, graph_tensor, tensor_name in zip(
            eager_tensors, graph_tensors, graph_names
        ):
            compare_and_assert(
                eager_tensor,
                graph_tensor,
                f"{tensor_name}_cp_eager",
                f"{tensor_name}_cp_cuda_graph",
                atol,
                rtol,
                rmse_tol,
                False,
            )
        if rank == 0:
            print(
                "CUDA_GRAPH_CORRECTNESS"
                f" model={model} backend={kernel_backend} comm={cp_comm_type}"
                " cp=2 qkv=thd dtype=bf16 eager_cp_vs_graph=passed",
                flush=True,
            )
        del graph_out

        # Reuse the same leaves so graph replay avoids host allocation and input
        # copies. Clearing .grad matches the eager loop's fresh-leaf semantics.
        replay_warmup = 10
        for _ in range(replay_warmup):
            for tensor in graph_inputs:
                tensor.grad = None
            graph_out = graphed_core_attn(*graph_inputs)
            graph_out.backward(dout_)
            del graph_out
        capture_barrier()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        wall_start = time.perf_counter()
        for _ in range(benchmark_iters):
            for tensor in graph_inputs:
                tensor.grad = None
            graph_out = graphed_core_attn(*graph_inputs)
            graph_out.backward(dout_)
            del graph_out
        end_event.record()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - wall_start) * 1000.0 / benchmark_iters
        cuda_event_ms = start_event.elapsed_time(end_event) / benchmark_iters

        # Rank 0 reports the slowest rank, which is the distributed iteration
        # latency that constrains useful throughput.
        max_metrics = torch.tensor(
            [wall_ms, cuda_event_ms, capture_ms, capture_peak_mib],
            dtype=torch.float64,
            device="cuda",
        )
        dist.all_reduce(max_metrics, op=dist.ReduceOp.MAX, group=cp_comm_group)
        if rank == 0:
            print(
                "CUDA_GRAPH_RESULT"
                f" model={model} backend={kernel_backend} comm={cp_comm_type}"
                " cp=2 qkv=thd dtype=bf16"
                f" wall_ms={max_metrics[0].item():.3f}"
                f" cuda_event_ms={max_metrics[1].item():.3f}"
                f" iters={benchmark_iters} replay_warmup={replay_warmup}"
                f" capture_warmup={cuda_graph_warmup_iters}"
                f" capture_ms={max_metrics[2].item():.1f}"
                f" capture_peak_mib={max_metrics[3].item():.1f}",
                flush=True,
            )
        for tensor in graph_inputs:
            tensor.grad = None
        torch.cuda.synchronize()
        graphed_core_attn.reset()

    if cp_bench_only:
        # Benchmark-only mode has no correctness result to compare.
        if not _reusing_pool_groups:
            if cp_comm_group is not None:
                dist.destroy_process_group(cp_comm_group)
            for group in cp_comm_sub_groups:
                dist.destroy_process_group(group)
        if not _pool_managed_pg:
            dist.destroy_process_group()
        return

    # get outputs
    tensors = [out, dq, dk, dv, dbias, out_, dq_, dk_, dv_, dbias_]
    names = ["out", "dq", "dk", "dv", "dbias", "out_cp", "dq_cp", "dk_cp", "dv_cp", "dbias_cp"]
    if fp8_mha:
        tensors_to_deq = [out, out_] if not fp8_bwd else tensors
        for i, tensor in enumerate(tensors_to_deq):
            # dbias/dbias_ could be None, so skip check for it
            if tensor is not None:
                tensors_to_deq[i] = tensor.dequantize()
        if not fp8_bwd:
            tensors[0], tensors[5] = tensors_to_deq
    for tensor, name in zip(tensors, names):
        # dbias/dbias_ could be None, so skip check for it
        if tensor is not None:
            assert torch.all(~torch.isnan(tensor)), f"{name} has nan values"
            assert torch.all(~torch.isinf(tensor)), f"{name} has inf values"
    out, dq, dk, dv, dbias, out_, dq_, dk_, dv_, dbias_ = tensors

    ############  compare results between CP and no-CP ############
    if qkv_format == "bshd" or qkv_format == "sbhd":
        if is_training:
            dq, dk, dv, out = [
                x.view(
                    *x.shape[:seq_dim],
                    2 * world_size,
                    x.shape[seq_dim] // (2 * world_size),
                    *x.shape[(seq_dim + 1) :],
                )
                for x in [dq, dk, dv, out]
            ]
            dq, dk, dv, out = [x.index_select(seq_dim, seq_idx) for x in [dq, dk, dv, out]]
            dq_, dk_, dv_, out_ = [
                x.view(*x.shape[:seq_dim], 2, x.shape[seq_dim] // 2, *x.shape[(seq_dim + 1) :])
                for x in [dq_, dk_, dv_, out_]
            ]
            if dbias is not None and dbias_ is not None:
                ndim = dbias.ndim
                # Query seq is at dim -2
                seq_q_dim = ndim - 2
                shape_before_seq = dbias.shape[:seq_q_dim]
                seq_q_size = dbias.shape[seq_q_dim]
                seq_kv_size = dbias.shape[-1]
                # Reshape to split seq_q dimension
                dbias = dbias.view(
                    *shape_before_seq,
                    2 * world_size,
                    seq_q_size // (2 * world_size),
                    seq_kv_size,
                )
                # Index select on the newly created dimension (now at position seq_q_dim)
                dbias = dbias.index_select(seq_q_dim, seq_idx)
                dbias_ = dbias_.view(
                    *shape_before_seq, 2, dbias_.shape[seq_q_dim] // 2, seq_kv_size
                )
        else:
            # Forward-only: reshape only out/out_ for comparison
            out = out.view(
                *out.shape[:seq_dim],
                2 * world_size,
                out.shape[seq_dim] // (2 * world_size),
                *out.shape[(seq_dim + 1) :],
            )
            out = out.index_select(seq_dim, seq_idx)
            out_ = out_.view(
                *out_.shape[:seq_dim], 2, out_.shape[seq_dim] // 2, *out_.shape[(seq_dim + 1) :]
            )

    elif qkv_format == "thd":
        if is_training:
            dq, out = [x.index_select(0, seq_idx_q).contiguous() for x in [dq, out]]
            dk, dv = [x.index_select(0, seq_idx_kv).contiguous() for x in [dk, dv]]
            cu_seqlens_q_padded = cu_seqlens_q_padded // world_size
            cu_seqlens_q = get_cu_seqlens_on_cp_rank(
                cu_seqlens_q, cu_seqlens_q_padded, world_size, rank, True, True
            )
            cu_pads_q = cu_seqlens_q_padded - cu_seqlens_q
            num_pads_q = cu_pads_q[1:] - cu_pads_q[:-1]
            cu_seqlens_kv_padded = cu_seqlens_kv_padded // world_size
            cu_seqlens_kv = get_cu_seqlens_on_cp_rank(
                cu_seqlens_kv, cu_seqlens_kv_padded, world_size, rank, True, True
            )
            num_pads_kv = (cu_seqlens_kv_padded - cu_seqlens_kv)[1:] - (
                cu_seqlens_kv_padded - cu_seqlens_kv
            )[:-1]
            # FA3 leaves garbage at padding positions despite seqused_q/k (tile spillover).
            # Forward out_ can't be pre-zeroed because FA3's custom op returns out_ as an
            # output rather than mutating it in-place, triggering PyTorch's aliasing constraint.
            # Backward dq/dk/dv CAN be pre-zeroed because FA3 marks them as mutated inputs.
            if fa_pad_between_seqs == "True":
                # out_ is a view inside the CP custom autograd Function, so in-place
                # zeroing is blocked by PyTorch. Clone to break the view relationship.
                out_ = out_.clone()
                for x in [out, out_, dq]:
                    for b in range(config.batch_size):
                        x[
                            cu_seqlens_q_padded[b + 1] - num_pads_q[b] : cu_seqlens_q_padded[b + 1]
                        ] = 0.0
                    x[cu_seqlens_q_padded[-1] :] = 0.0
                for x in [dk, dv]:
                    for b in range(config.batch_size):
                        x[
                            cu_seqlens_kv_padded[b + 1]
                            - num_pads_kv[b] : cu_seqlens_kv_padded[b + 1]
                        ] = 0.0
                    x[cu_seqlens_kv_padded[-1] :] = 0.0
                # Verify CP backward tensors have clean padding (pre-zeroed in context_parallel.py).
                for xname, x, cu, np_ in [
                    ("dq_", dq_, cu_seqlens_q_padded, num_pads_q),
                    ("dk_", dk_, cu_seqlens_kv_padded, num_pads_kv),
                    ("dv_", dv_, cu_seqlens_kv_padded, num_pads_kv),
                ]:
                    nnz = torch.count_nonzero(x[cu[-1] :]).item()
                    assert nnz == 0, (
                        f"{xname} has {nnz} nonzero values in tail padding — "
                        "context_parallel.py should zero padding positions"
                    )
                    for b in range(config.batch_size):
                        if np_[b] > 0:
                            nnz = torch.count_nonzero(x[cu[b + 1] - np_[b] : cu[b + 1]]).item()
                            assert nnz == 0, (
                                f"{xname} has {nnz} nonzero values in batch {b} padding — "
                                "context_parallel.py should zero padding positions"
                            )
        else:
            out = out.index_select(0, seq_idx_q).contiguous()
            out_ = out_

    atol, rtol, rmse_tol = get_tols(config, dtype)
    tensors_cp = [out_, dq_, dk_, dv_, dbias_, d_softmax_offset_, max_logit_]
    tensors_no_cp = [out, dq, dk, dv, dbias, d_softmax_offset, max_logit]
    names = ["out", "dq", "dk", "dv", "dbias", "d_softmax_offset", "max_logit"]
    names_cp = [x + "_cp" for x in names]
    names_no_cp = [x + "_no_cp" for x in names]
    is_fp8 = dtype == "fp8"
    for i, t in enumerate(tensors_no_cp):
        if t is not None:
            if "softmax_offset" not in names[i] and "max_logit" not in names[i]:
                if qkv_format == "bshd":
                    # Compare the two sequence chunks separately
                    # Compare dbias
                    if names[i] == "dbias":
                        # Compare the two chunks along dimension 2 (the split sequence dimension)
                        seq_q_dim_bias = 2
                        ndim_bias = t.ndim
                        slice_0 = [slice(None)] * ndim_bias
                        slice_0[seq_q_dim_bias] = 0
                        slice_1 = [slice(None)] * ndim_bias
                        slice_1[seq_q_dim_bias] = 1
                        compare_and_assert(
                            t[tuple(slice_0)],
                            tensors_cp[i][tuple(slice_0)],
                            names_no_cp[i],
                            names_cp[i],
                            atol,
                            rtol,
                            rmse_tol,
                            is_fp8,
                        )
                        compare_and_assert(
                            t[tuple(slice_1)],
                            tensors_cp[i][tuple(slice_1)],
                            names_no_cp[i],
                            names_cp[i],
                            atol,
                            rtol,
                            rmse_tol,
                            is_fp8,
                        )
                    # Compare Q/K/V/out
                    else:
                        #  Compare the two chunks along dimension 1 (the split sequence dimension)
                        compare_and_assert(
                            t[:, 0],
                            tensors_cp[i][:, 0],
                            names_no_cp[i],
                            names_cp[i],
                            atol,
                            rtol,
                            rmse_tol,
                            is_fp8,
                        )
                        compare_and_assert(
                            t[:, 1],
                            tensors_cp[i][:, 1],
                            names_no_cp[i],
                            names_cp[i],
                            atol,
                            rtol,
                            rmse_tol,
                            is_fp8,
                        )
                elif qkv_format == "sbhd":
                    # Compare the two sequence chunks separately
                    # Compare dbias (same as BSHD)
                    if names[i] == "dbias":
                        # Same as bshd: Compare the two chunks along dimension 2 (the split sequence dimension)
                        seq_q_dim_bias = 2
                        ndim_bias = t.ndim
                        slice_0 = [slice(None)] * ndim_bias
                        slice_0[seq_q_dim_bias] = 0
                        slice_1 = [slice(None)] * ndim_bias
                        slice_1[seq_q_dim_bias] = 1
                        compare_and_assert(
                            t[tuple(slice_0)],
                            tensors_cp[i][tuple(slice_0)],
                            names_no_cp[i],
                            names_cp[i],
                            atol,
                            rtol,
                            rmse_tol,
                            is_fp8,
                        )
                        compare_and_assert(
                            t[tuple(slice_1)],
                            tensors_cp[i][tuple(slice_1)],
                            names_no_cp[i],
                            names_cp[i],
                            atol,
                            rtol,
                            rmse_tol,
                            is_fp8,
                        )
                    # Compare Q/K/V/out
                    else:
                        #  Compare the two chunks along dimension 0 (the split sequence dimension)
                        compare_and_assert(
                            t[0],
                            tensors_cp[i][0],
                            names_no_cp[i],
                            names_cp[i],
                            atol,
                            rtol,
                            rmse_tol,
                            is_fp8,
                        )
                        compare_and_assert(
                            t[1],
                            tensors_cp[i][1],
                            names_no_cp[i],
                            names_cp[i],
                            atol,
                            rtol,
                            rmse_tol,
                            is_fp8,
                        )
                elif qkv_format == "thd":
                    compare_and_assert(
                        t,
                        tensors_cp[i],
                        names_no_cp[i],
                        names_cp[i],
                        atol,
                        rtol,
                        rmse_tol,
                        is_fp8,
                    )
            else:
                compare_and_assert(
                    t, tensors_cp[i], names_no_cp[i], names_cp[i], atol, rtol, rmse_tol, is_fp8
                )
            logging.info(f"[Rank {rank}] CP vs no-CP: {names[i]} matches")

    # Teardown on the success path. Pool mode: cp_comm_group / cp_comm_sub_groups
    # point at pool-shared groups owned by the pool runner (which destroys them
    # at pool shutdown), and the main PG is also pool-owned — both branches
    # below are no-ops. Single-shot mode: destroy what we created here. If the
    # body above raises, we skip this — the subprocess dies at function return
    # and NCCL releases the communicators with the process.
    if not _reusing_pool_groups:
        if cp_comm_group is not None:
            try:
                dist.destroy_process_group(cp_comm_group)
            except Exception:
                pass
        for g in cp_comm_sub_groups:
            try:
                dist.destroy_process_group(g)
            except Exception:
                pass
    if not _pool_managed_pg:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def main(**kwargs):
    run_dpa_with_cp(**kwargs)


if __name__ == "__main__":
    kwargs = dict(arg.split("=") for arg in sys.argv[2:])
    main(**kwargs)
