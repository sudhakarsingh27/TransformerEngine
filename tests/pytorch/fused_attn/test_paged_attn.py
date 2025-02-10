# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from collections import OrderedDict
import os
import logging

import pytest
import torch

from torch.distributions import Exponential
from transformer_engine.pytorch.attention import (
    DotProductAttention,
    InferenceParams,
)
from transformer_engine.pytorch.utils import is_bf16_compatible
from test_fused_attn import (
    ModelConfig,
    reset_rng_states,
    _get_attention_backends,
)
from tests.pytorch.test_numerics import assert_allclose

# Initialize RNG state
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
_cpu_rng_state = torch.get_rng_state()
_cuda_rng_state = torch.cuda.get_rng_state()


param_types = [torch.float16]
if is_bf16_compatible():
    param_types.append(torch.bfloat16)

model_configs_infer = {
    #    test:             b,  h, hg,  d,  sq, skv,   p,      mask,      bias
    "infer_0": ModelConfig(4, 16, 16, 64, 66, 66, 0.0, "no_mask", "no_bias", total_requests=8),
    #"infer_1": ModelConfig(2, 16, 4, 64, 66, 66, 0.0, "no_mask", "no_bias", total_requests=6),
}

qkv_formats = ["bshd", "sbhd", "thd"]


def to_pretty_string(x: torch.Tensor):
    return "[" + ",".join(["{:>3s}".format(str(i)) for i in x.tolist()]) + "]"

class Simulation:
    def __init__(
        self,
        total_requests: int = 10,
        max_seqlen_kv: int = 1024,
        context_ratio: float = 0.25,
        max_batch_size: int = 5,
        poisson_rate: float = None,
    ):
        self.total_requests = total_requests
        self.max_seqlen_kv = max_seqlen_kv
        self.context_ratio = context_ratio
        self.max_batch_size = max_batch_size
        self.poisson_rate = poisson_rate

        # calculate maximum context/generation length
        self.max_context_len = int(max_seqlen_kv * context_ratio)
        self.max_gen_len = max_seqlen_kv - self.max_context_len

        # simulate sequence ids in monotonically increasing fashion
        self.seq_ids = torch.range(0, total_requests-1, dtype=torch.int32, device="cpu")

        # simulate context lengths in Uniform distribution
        self.context_lens = torch.randint(
            1, self.max_context_len, [total_requests], dtype=torch.int32, device="cpu"
        )

        # simulate gen lengths in Exponential distribution
        gen_dist = Exponential(1 / self.max_gen_len)
        gen_lens = gen_dist.sample((total_requests,))
        gen_lens = torch.where(gen_lens > self.max_gen_len, self.max_gen_len, gen_lens).to(
            dtype=torch.int32, device="cpu"
        )
        self.gen_lens = torch.where(gen_lens == 0, 1, gen_lens).to(
            dtype=torch.int32, device="cpu"
        )

        # simulate arrival times in Poisson distribution
        if poisson_rate is None:
            self.poisson_rate = torch.randint(1, max_batch_size, [1]).item()
        interval_dist = Exponential(self.poisson_rate)
        arrival_intervals = interval_dist.sample((total_requests,))
        self.arrival_times = torch.cumsum(arrival_intervals, dim=0).to(dtype=torch.int32, device="cpu")
        self.last_arrival = self.arrival_times.max().item()

        # initialize tensors
        self.reset()

    def reset(self):
        self.t = 0
        self.request_delays = torch.zeros([self.total_requests], dtype=torch.int32, device="cpu")
        self.delayed_seq_ids = torch.Tensor().to(dtype=torch.int32, device="cpu")
        self.serving_times = self.arrival_times
        self.complete_times = self.arrival_times

        # time-stepping workflow
        # t-1: ...
        #      compute for seq_ids = [0, 1, 2], ctx_lens = [5, 2, 3], gen_lens = [2, 9, 4],
        #              batch_size = 3, step_lens = [1, 1, 1]
        #      increase counter for gen_lens = [3, 10, 5]
        # t:   detect seq 1 is finished since expected_gen_lens = [12, 10, 15]
        #      add two new seqs 3 and 4, with ctx lens 10 and 11
        #      compute for seq_ids = [0, 2, 3, 4], ctx_lens = [5, 3, 10, 11], gen_lens = [3, 5, 0, 0],
        #              batch_size = 4, step_lens = [1, 1, 10, 11]
        #      increase counter for gen_lens = [3, 5, 1, 1]

        # batch info at step t
        self.t_seq_ids = torch.Tensor([]).to(dtype=torch.bool, device="cpu")
        self.t_ctx_lens = torch.Tensor([]).to(dtype=torch.bool, device="cpu")
        self.t_gen_lens = torch.Tensor([]).to(dtype=torch.bool, device="cpu")
        self.t_total_lens = self.t_ctx_lens + self.t_gen_lens #+ self.step_lens
        self.t_batch_size = 0

        # step info from step t-1 to t
        self.step_lens = torch.Tensor([]).to(dtype=torch.int32, device="cpu")

    def print(self, logger, label="setup"):
        if label == "setup":
            logger.info("Simulation:")
            logger.info("  {:<33s}: {}".format("total number of requests", self.total_requests))
            logger.info("  {:<33s}: {}".format("max sequence length per request", self.max_seqlen_kv))
            logger.info("  {:<33s}: {}".format("max context lengh", self.max_context_len))
            logger.info("  {:<33s}: {}".format("max generation lengh", self.max_gen_len))
            logger.info("  {:<33s}: {}".format("max batch size per iteration", self.max_batch_size))
            logger.info("  {:<33s}: {}".format("Poisson rate", self.poisson_rate))
            logger.info("  {:<18s}: {}".format("sequence ids", to_pretty_string(self.seq_ids)))
            logger.info("  {:<18s}: {}".format("arrival times", to_pretty_string(self.arrival_times)))
            logger.info("  {:<18s}: {}".format("context lenghs", to_pretty_string(self.context_lens)))
            logger.info("  {:<18s}: {}".format("generation lenghs", to_pretty_string(self.gen_lens)))
        if label == "step":
            logger.info(f"Step t = {self.t}:")
            logger.info("  {:<15s}: {}".format("t_batch_size", self.t_batch_size))
            logger.info("  {:<15s}: {}".format("t_seq_ids", self.t_seq_ids.tolist()))
            logger.info("  {:<15s}: {}".format("t_ctx_lens", self.t_ctx_lens.tolist()))
            logger.info("  {:<15s}: {}".format("t_gen_lens", self.t_gen_lens.tolist()))
            logger.info("  {:<15s}: {}".format("t_total_lens", self.t_total_lens.tolist()))
            logger.info("  {:<15s}: {}".format("step_lens", self.step_lens.tolist()))
        if label == "summary":
            logger.info("Summary:")
            logger.info("  {:<18s}: {}".format("total steps taken", self.t))
            logger.info("  {:<18s}: {}".format("arrival_times",     to_pretty_string(self.arrival_times)))
            logger.info("  {:<18s}: {}".format("serving_times",     to_pretty_string(self.serving_times)))
            logger.info("  {:<18s}: {}".format("total_gen_lens",    to_pretty_string(self.gen_lens)))
            logger.info("  {:<18s}: {}".format("complete_times",    to_pretty_string(self.complete_times)))

    def add_new_seqs(self, new_seq_ids):
        # get ctx_lens for new seqs
        self.t_seq_ids = torch.cat([self.t_seq_ids, new_seq_ids], dim=0)
        self.t_ctx_lens = torch.cat([self.t_ctx_lens, self.context_lens[new_seq_ids]], dim=0)
        gen_lens = torch.Tensor([0] * len(new_seq_ids)).to(dtype=torch.int32, device="cpu")
        self.t_gen_lens = torch.cat([self.t_gen_lens, gen_lens], dim=0)
        # append new seqs' ctx_lens to step_lens
        self.step_lens = torch.cat([self.step_lens, self.context_lens[new_seq_ids]], dim=0)

    def remove_finished(self):
        # figure out which seqs have finished
        finished = torch.where(self.t_gen_lens - self.gen_lens[self.t_seq_ids] < 0, False, True).to(
            dtype=torch.bool, device="cpu"
        )
        self.t_seq_ids = self.t_seq_ids[~finished]
        self.t_ctx_lens = self.t_ctx_lens[~finished]
        self.t_gen_lens = self.t_gen_lens[~finished]
        # add ones for unfinished seqs to step_lens
        self.step_lens = torch.ones([len(self.t_seq_ids)], dtype=torch.int32, device="cpu")

    def step(self, dynamic_fill: bool = True):
        # remove finished seqs
        if self.t != 0:
            self.remove_finished()
        # get allowed new seqs
        arrived_seq_ids = torch.where(self.arrival_times == self.t, True, False).nonzero().view(-1)
        queuing_seq_ids = torch.cat([self.delayed_seq_ids, arrived_seq_ids], dim=0)
        if dynamic_fill:
            allowed_num_new_seqs = self.max_batch_size - len(self.t_seq_ids)
        else:
            allowed_num_new_seqs = 0 if len(self.t_seq_ids) else self.max_batch_size
        if len(queuing_seq_ids) > allowed_num_new_seqs:
            new_seq_ids = queuing_seq_ids[:allowed_num_new_seqs]
            self.delayed_seq_ids = queuing_seq_ids[allowed_num_new_seqs:]
            self.request_delays[self.delayed_seq_ids.tolist()] += 1
        else:
            new_seq_ids = queuing_seq_ids
            self.delayed_seq_ids = torch.Tensor().to(dtype=torch.int32)
        # add new seqs to batch
        self.add_new_seqs(new_seq_ids)
        # update batch variables
        self.t_batch_size = len(self.t_seq_ids)
        self.t_total_lens = self.t_ctx_lens + self.t_gen_lens


@pytest.mark.parametrize("dtype", [torch.float16])#param_types)
@pytest.mark.parametrize("model", model_configs_infer.keys())
@pytest.mark.parametrize("qkv_format", ['bshd'])#qkv_formats)
@pytest.mark.parametrize("is_paged", [False])#, True])
@pytest.mark.parametrize("backend", ["FusedAttention"])#, "FlashAttention", "UnfusedAttention"])
@pytest.mark.parametrize("is_cuda_graph", [False])#, True])
def test_paged_attn(dtype, model, qkv_format, is_paged, backend, is_cuda_graph):
    reset_rng_states()
    logger = logging.getLogger("test_paged_attn")

    config = model_configs_infer[model]
    layer_number = 1

    # figure out supported backends
    inference_params_qkv_format = "bshd"
    if is_paged:
        qkv_layout = "paged_kv_" + inference_params_qkv_format + "_2" + inference_params_qkv_format
    else:
        qkv_layout = "_".join([inference_params_qkv_format] * 3)
    available_backends, fused_attn_backends = _get_attention_backends(
        config,
        qkv_dtype=dtype,
        qkv_layout=qkv_layout,
        window_size=config.window_size,
        pad_between_seqs=False,
    )
    flash_attn_supported, fused_attn_supported, unfused_attn_supported = available_backends
    if backend == "FlashAttention" and not flash_attn_supported:
        pytest.skip("FlashAttention backend is not supported")
    if backend == "FusedAttention" and not fused_attn_supported:
        pytest.skip("FusedAttention backend is not supported")
    if backend == "UnfusedAttention" and not unfused_attn_supported:
        pytest.skip("UnfusedAttention backend is not supported")
    os.environ["NVTE_FLASH_ATTN"] = str(int(backend == "FlashAttention"))
    os.environ["NVTE_FUSED_ATTN"] = str(int(backend == "FusedAttention"))
    os.environ["NVTE_UNFUSED_ATTN"] = str(int(backend == "UnfusedAttention"))

    # set up various parameters
    total_requests = config.total_requests
    max_batch_size = config.batch_size
    max_seqlen_kv = config.max_seqlen_kv
    attn_mask_type = "padding"
    page_size = 256 if backend == "FlashAttention" else 16
    max_seqlen_kv_roundup = max_seqlen_kv
    if is_paged:
        # round up max_seqlen_kv to nearest page size
        max_seqlen_kv_roundup = int((max_seqlen_kv + page_size - 1) // page_size * page_size)
    else:
        # round up max_seqlen_kv to nearest multiple of 64
        max_seqlen_kv_roundup = int((max_seqlen_kv + 63) // 64 * 64)
    cache_size = max_batch_size * max_seqlen_kv_roundup
    total_num_pages = int(cache_size / page_size)

    # set up simulation
    sim = Simulation(
        total_requests=total_requests,
        max_seqlen_kv=max_seqlen_kv,
        context_ratio=0.25,
        max_batch_size=max_batch_size,
        poisson_rate=2,
        )
    sim.print(logger, label="setup")

    # create model and data
    model = (
        DotProductAttention(
            kv_channels=config.head_dim_qk,
            num_attention_heads=config.num_heads,
            num_gqa_groups=config.num_gqa_groups,
            layer_number=layer_number,
            attention_dropout=0.0,
            attn_mask_type="causal",
            qkv_format="bshd",
        )
        .cuda()
        .eval()
    )
    q = 0.1 * torch.randn(
        (total_requests, max_seqlen_kv, config.num_heads, config.head_dim_qk),
        dtype=dtype,
        device="cuda",
    )
    k = 0.1 * torch.randn(
        (total_requests, max_seqlen_kv, config.num_gqa_groups, config.head_dim_qk),
        dtype=dtype,
        device="cuda",
    )
    v = 0.1 * torch.randn(
        (total_requests, max_seqlen_kv, config.num_gqa_groups, config.head_dim_v),
        dtype=dtype,
        device="cuda",
    )

    # generate all tokens at once
    logger.info("=== Generating all tokens at once ===")
    full_output = model(
        query_layer=q,
        key_layer=k,
        value_layer=v,
        qkv_format="bshd",
        attn_mask_type="causal",
    )

    # generate tokens one at a time
    logger.info("=== Generating one token at a time ===")
    inference_params = InferenceParams(
        max_batch_size=max_batch_size,
        max_seqlen_kv=max_seqlen_kv_roundup,
        num_heads_kv=config.num_gqa_groups,
        head_dim_k=config.head_dim_qk,
        head_dim_v=config.head_dim_v,
        dtype=dtype,
        is_paged=is_paged,
        page_size=page_size,
        total_num_pages=total_num_pages,
        is_cuda_graph=is_cuda_graph,
        num_heads_q=config.num_heads,
        head_dim_q=config.head_dim_qk,
    )
    inference_params.allocate_memory(layer_number)
    inference_params.print()

#    def generate_data(
#        model_config: ModelConfig,
#        dtype: torch.dtype,
#        warmup: bool = False,
#    ) -> List[torch.Tensor]:
#        """Generate synthetic data for dot product attention."""
#        gen_func = torch.ones if warmup else torch.randn
#        aa=[
#            gen_func(
#                model_config.batch_size,
#                model_config.max_seqlen_q,
#                model_config.num_heads,
#                model_config.head_dim_qk,
#                device="cuda",
#                #requires_grad=True,
#                dtype=dtype,
#            )
#            for _ in range(3)
#        ]
#        #aa.extend([model_config.sequence_length, model_config.sequence_length])
#        return aa
#
#    def gen_cu(
#        model_config: ModelConfig,
#        dtype: torch.dtype,
#        ):
#        cu_dict = {}
#        cu_dict["cu_seqlens_q"] = torch.linspace( 0,
#            model_config.batch_size * model_config.max_seqlen_q,
#            steps=model_config.batch_size+1,
#            device="cuda",
#            dtype=torch.int32,
#        )
#        cu_dict["cu_seqlens_kv"] = torch.linspace( 0,
#            model_config.batch_size * model_config.max_seqlen_kv,
#            steps=model_config.batch_size+1,
#            device="cuda",
#            dtype=torch.int32,
#        )
#        cu_dict["max_seqlen_q"] = model_config.max_seqlen_q
#        cu_dict["max_seqlen_kv"] = model_config.max_seqlen_kv
#        return cu_dict
#
#    model = make_graphed_callables(
#        model,
#        generate_data_for_dot_product_attention(model_config, dtype, warmup=True),
#        num_warmup_iters=10,
#        fp8_enabled=False,
#        #sample_kwargs={"qkv_format":"thd"},
#        sample_kwargs=gen_cu(model_config, dtype),
#    )

    # similate step by step
    sim.reset()
    while True:
        if inference_params.is_paged:
            inference_params.cache_manager.print_cache()

        dynamic_fill = True #inference_params.is_paged
        sim.step(dynamic_fill=dynamic_fill)
        sim.print(logger, label="step")

        if sim.t_batch_size == 0:
            # all sequences are finished
            if sim.t > sim.last_arrival:
                sim.serving_times = sim.arrival_times + sim.request_delays
                sim.complete_times = sim.serving_times + sim.gen_lens
                break
            # not finished; run next iteration
            else:
                sim.t += 1
                continue

        if not is_cuda_graph:
            max_seqlen_q_infer = int((sim.max_context_len + 63)// 64 * 64)
        else:
            max_seqlen_q_infer = max_seqlen_kv_roundup

        # create incremental input
        if qkv_format == "thd":
            incremental_q = torch.Tensor().to(dtype=dtype, device="cuda")
            incremental_k = torch.Tensor().to(dtype=dtype, device="cuda")
            incremental_v = torch.Tensor().to(dtype=dtype, device="cuda")
            for i, seq in enumerate(sim.t_seq_ids):
                start = (sim.t_total_lens[i] - sim.step_lens[i]).item()
                end = sim.t_total_lens[i].item()
                incremental_q = torch.cat([incremental_q, q[seq, start:end, :, :]], dim=0)
                incremental_k = torch.cat(
                    [
                        incremental_k,
                        k[seq, start:end, :, :].view(-1, config.num_gqa_groups, config.head_dim_qk),
                    ],
                    dim=0,
                )
                incremental_v = torch.cat(
                    [
                        incremental_v,
                        v[seq, start:end, :, :].view(-1, config.num_gqa_groups, config.head_dim_v),
                    ],
                    dim=0,
                )
        else:
            incremental_q = torch.zeros(
                sim.t_batch_size,
                max_seqlen_q_infer,
                config.num_heads,
                config.head_dim_qk,
                dtype=dtype,
                device="cuda",
            )
            incremental_k = torch.zeros(
                sim.t_batch_size,
                max_seqlen_q_infer,
                config.num_gqa_groups,
                config.head_dim_qk,
                dtype=dtype,
                device="cuda",
            )
            incremental_v = torch.zeros(
                sim.t_batch_size,
                max_seqlen_q_infer,
                config.num_gqa_groups,
                config.head_dim_v,
                dtype=dtype,
                device="cuda",
            )
            for i, seq in enumerate(sim.t_seq_ids):
                start = (sim.t_total_lens[i] - sim.step_lens[i]).item()
                end = sim.t_total_lens[i].item()
                incremental_q[i, : sim.step_lens[i], :, :] = q[seq, start:end, :, :]
                incremental_k[i, : sim.step_lens[i], :, :] = k[seq, start:end, :, :]
                incremental_v[i, : sim.step_lens[i], :, :] = v[seq, start:end, :, :]
            if qkv_format == "sbhd":
                incremental_q, incremental_k, incremental_v = [
                    x.transpose(0, 1) for x in [incremental_q, incremental_k, incremental_v]
                ]

        cu_seqlens_q = torch.zeros(sim.t_batch_size + 1, dtype=torch.int32, device="cuda")
        cu_seqlens_q[1 : sim.t_batch_size + 1] = torch.cumsum(sim.step_lens, dim=0)
        cu_seqlens_kv = torch.zeros(sim.t_batch_size + 1, dtype=torch.int32, device="cuda")
        cu_seqlens_kv[1 : sim.t_batch_size + 1] = torch.cumsum(sim.t_total_lens, dim=0)

        inference_params.step_dict = OrderedDict(
            zip(sim.t_seq_ids.tolist(), sim.step_lens.tolist())
        )

        line_output = model(
            query_layer=incremental_q,
            key_layer=incremental_k,
            value_layer=incremental_v,
            inference_params=inference_params,
            attn_mask_type=attn_mask_type,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q_infer,
            max_seqlen_kv=max_seqlen_kv_roundup,
            qkv_format=qkv_format,
        )

        if backend != "FlashAttention":
            tols = {
                torch.float32: 1e-3,
                torch.half: 1e-3,
                torch.bfloat16: 1e-2,
            }
        else:
            tols = {
                torch.float32: 1e-3,
                torch.half: 4e-3,
                torch.bfloat16: 1e-2,
            }
        for i, seq in enumerate(sim.t_seq_ids):
            if qkv_format == "bshd":
                torch.testing.assert_close(
                    full_output[seq, sim.t_total_lens[i] - 1, :],
                    line_output[i, sim.step_lens[i] - 1, :],
                    atol=tols[dtype],
                    rtol=tols[dtype],
                )
            if qkv_format == "sbhd":
                torch.testing.assert_close(
                    full_output[seq, sim.t_total_lens[i] - 1, :],
                    line_output[sim.step_lens[i] - 1, i, :],
                    atol=tols[dtype],
                    rtol=tols[dtype],
                )
            if qkv_format == "thd":
                torch.testing.assert_close(
                    full_output[seq, sim.t_total_lens[i] - 1, :],
                    line_output[cu_seqlens_q[i + 1] - 1, :],
                    atol=tols[dtype],
                    rtol=tols[dtype],
                )
        sim.t += 1
        sim.t_gen_lens = sim.t_gen_lens + 1

    sim.serving_times = sim.arrival_times + sim.request_delays
    sim.complete_times = sim.serving_times + sim.gen_lens
    sim.print(logger, label="summary")
