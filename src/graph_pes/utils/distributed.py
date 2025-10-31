from __future__ import annotations

import logging
from typing import Final

import pytorch_lightning as pl


def silently_create_trainer(**trainer_kwargs) -> pl.Trainer:
    logger = logging.getLogger("pytorch_lightning.utilities.rank_zero")
    logger.setLevel(logging.ERROR)
    _trainer = pl.Trainer(**trainer_kwargs)
    logger.setLevel(logging.INFO)
    return _trainer


# dirty hack: just get lightning to work this out,
# and ensure no annoying printing happens
_trainer = silently_create_trainer(logger=False, devices="auto")
GLOBAL_RANK: Final[int] = _trainer.global_rank
WORLD_SIZE: Final[int] = _trainer.world_size
IS_RANK_0: Final[bool] = GLOBAL_RANK == 0
