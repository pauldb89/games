import datetime
import functools
import os

import torch.distributed
import tqdm


def distributed_setup() -> int:
    """Initializes the process group for distributed training"""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.distributed.init_process_group("nccl", timeout=datetime.timedelta(minutes=10))
    torch.cuda.set_device(local_rank)
    return local_rank


def distributed_cleanup() -> None:
    """Cleans up the distributed process group"""
    torch.distributed.destroy_process_group()


def is_root_process() -> bool:
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def barrier() -> None:
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


@functools.cache
def world_size() -> int:
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


@functools.cache
def get_rank() -> int:
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def print_once(message: str) -> None:
    if is_root_process():
        print(message)


def tqdm_once(*args, **kwargs) -> tqdm.tqdm:
    return tqdm.tqdm(*args, **kwargs, disable=not is_root_process())