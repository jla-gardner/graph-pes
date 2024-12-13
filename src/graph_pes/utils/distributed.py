from __future__ import annotations

from dataclasses import dataclass

import torch.distributed as dist
from typing_extensions import Self


@dataclass
class DistributedInfo:
    rank: int
    world_size: int

    @classmethod
    def from_env(cls) -> Self:
        if not dist.is_available():
            return cls(rank=0, world_size=1)
        if not dist.is_initialized():
            return cls(rank=0, world_size=1)
        return cls(rank=dist.get_rank(), world_size=dist.get_world_size())
