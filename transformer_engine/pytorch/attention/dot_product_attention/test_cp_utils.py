import torch
import transformer_engine.pytorch as te
from utils import get_sequences_on_this_cp_rank
import os
import datetime
import torch.distributed as dist


def dist_print(msg, src=None, end="\n"):
    """Print message from a specific rank (default: rank 0)."""
    world_rank = int(os.getenv("RANK", "0"))
    if world_rank == (0 if src is None else src):
        print(f"[rank{world_rank}] {msg}", end=end)

def setup_distributed():
    """Initialize distributed training."""
    world_rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    local_size = int(os.getenv("LOCAL_WORLD_SIZE", "1"))

    assert world_size == local_size  # this example supports only 1 node
    assert local_size <= torch.cuda.device_count()

    dist_init_kwargs = {
        "backend": "nccl",
        "rank": world_rank,
        "world_size": world_size,
        "timeout": datetime.timedelta(seconds=30),
    }
    dist_init_kwargs["init_method"] = "env://"
    dist_init_kwargs["device_id"] = torch.device(f"cuda:{local_rank}")

    assert dist.is_nccl_available()
    torch.cuda.set_device(local_rank)
    dist.init_process_group(**dist_init_kwargs)

    nccl_world = dist.new_group(backend="nccl")

    dist_print(f"Running the example with {world_size} GPUs with {local_size} process per GPU!")
    return world_rank, world_size, local_rank, nccl_world

if __name__ == "__main__":

    # 1. setup distributed
    world_rank, world_size, local_rank, nccl_world = setup_distributed()

    # 2. create random data (make sure the seed is set after the distributed setup)
    torch.manual_seed(0)
    seqlens = [4,4,4]
    cu_seqlens = torch.cumsum(torch.tensor([0] + seqlens), dim=0).cuda()
    x = torch.randn(1, sum(seqlens)).cuda()

    seqids = torch.arange(cu_seqlens[-1]).reshape(1, -1).cuda()

    # 3. Create batch similar to how it's accepted in the get_sequences_on_this_cp_rank function
    batch = {k:None for k in ['input_ids_padded', 'labels_padded', 'position_ids_padded']}
    batch['input_ids_padded'] = x
    batch['position_ids_padded'] = seqids
    batch_on_rank = get_sequences_on_this_cp_rank(batch, qkv_format="thd", cu_seqlens_padded=cu_seqlens, cp_group=nccl_world)

    # 4. Prepare the buffer for all_gather
    all_gather_batch = {}
    all_gather_batch['input_ids_padded'] = [torch.empty_like(batch_on_rank['input_ids_padded']) for _ in range(world_size)]
    all_gather_batch['position_ids_padded'] = [torch.empty_like(batch_on_rank['position_ids_padded']) for _ in range(world_size)]

    # 5. Perform all_gather
    dist.all_gather(all_gather_batch['input_ids_padded'], batch_on_rank['input_ids_padded'], group=nccl_world)
    dist.all_gather(all_gather_batch['position_ids_padded'], batch_on_rank['position_ids_padded'], group=nccl_world)

    # 6. Check the results
    if local_rank == 0:

        pos0 = all_gather_batch['position_ids_padded'][0]
        val0 = all_gather_batch['input_ids_padded'][0]
        pos1 = all_gather_batch['position_ids_padded'][1]
        val1 = all_gather_batch['input_ids_padded'][1]

        # Determine the size of the final tensor (max index + 1)
        max_index = torch.cat([pos0, pos1], dim=1).max().item()
        final_tensor = torch.empty((1, max_index + 1)).cuda()

        final_tensor[0, pos0] = val0
        final_tensor[0, pos1] = val1

        # Combine positions and values
        all_pos = torch.cat([pos0, pos1], dim=1).flatten()
        all_vals = torch.cat([val0, val1], dim=1).flatten()

        # Determine the size of the reconciled tensor (max position index + 1)
        max_pos = all_pos.max().item() + 1
        reconciled = torch.empty(max_pos, device='cuda:0')

        # Fill based on positions
        reconciled[all_pos] = all_vals

        assert (reconciled == x).all()