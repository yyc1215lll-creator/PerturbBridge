"""
Helpers for distributed training.
"""

import os

import torch
import torch.distributed as dist

# torchrun provides these variables. Defaults keep model construction and CPU
# preflight imports usable in a normal single-process Python invocation.
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
WORLD_RANK = int(os.environ.get("RANK", 0))


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    backend = "gloo" if not torch.cuda.is_available() else "nccl"
    if torch.cuda.is_available():
        torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group(backend)


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
