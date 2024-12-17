from __future__ import annotations

import os
from dataclasses import dataclass

from pytorch_lightning import Trainer
from typing_extensions import Self


@dataclass
class DistributedCommunication:
    global_rank: int
    world_size: int

    @classmethod
    def from_env(cls) -> Self:
        # dirty hack: just get lightning to work this out
        trainer = Trainer(logger=False)
        return cls(
            global_rank=trainer.global_rank,
            world_size=trainer.world_size,
        )

    def send(self, key: str, value: str) -> None:
        os.environ[key] = value

    def receive(self, key: str) -> str:
        return os.environ[key]
