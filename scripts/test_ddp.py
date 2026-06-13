import os
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")

rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ["LOCAL_RANK"])

torch.cuda.set_device(local_rank)

x = torch.ones(1, device=f"cuda:{local_rank}") * (rank + 1)
dist.all_reduce(x)

print(f"rank={rank}, local_rank={local_rank}, world_size={world_size}, result={x.item()}")

dist.destroy_process_group()
