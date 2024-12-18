from __future__ import annotations

import logging
import os
from typing import Final

from pytorch_lightning import Trainer

# dirty hack: just get lightning to work this out,
# and ensure no annoying printing happens
logger = logging.getLogger("pytorch_lightning.utilities.rank_zero")
logger.setLevel(logging.ERROR)
_trainer = Trainer(logger=False)
logger.setLevel(logging.INFO)

GLOBAL_RANK: Final[int] = _trainer.global_rank
WORLD_SIZE: Final[int] = _trainer.world_size
IS_RANK_0: Final[bool] = GLOBAL_RANK == 0


def send_to_other_ranks(key: str, value: str) -> None:
    """Must be called by rank 0 and before `Trainer.fit` is called."""
    os.environ[key] = value


def receive_from_rank_0(key: str) -> str:
    """Must be called from a non-0 rank."""
    return os.environ[key]
