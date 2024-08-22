from __future__ import annotations

import time
from typing import Literal

import pytorch_lightning as pl
from graph_pes.logger import logger
from pytorch_lightning.callbacks import StochasticWeightAveraging
from typing_extensions import override


class VerboseSWACallback(StochasticWeightAveraging):
    @override
    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if (not self._initialized) and (
            self.swa_start <= trainer.current_epoch <= self.swa_end
        ):
            logger.info("SWA: starting SWA")

        return super().on_train_epoch_start(trainer, pl_module)


class ModelTimer(pl.Callback):
    def __init__(self):
        super().__init__()
        self.tick_ms: float | None = None

    def start(self):
        self.tick_ms = time.time_ns() // 1_000_000

    def stop(
        self, pl_module: pl.LightningModule, stage: Literal["train", "valid"]
    ):
        assert self.tick_ms is not None
        duration_ms = max((time.time_ns() // 1_000_000) - self.tick_ms, 1)
        self.tick_ms = None

        for name, x in (
            ("step_duration_ms", duration_ms),
            ("its_per_s", 1_000 / duration_ms),
        ):
            pl_module.log(
                f"timer/{name}/{stage}",
                x,
                batch_size=1,
                on_epoch=stage == "valid",
                on_step=stage == "train",
            )

    @override
    def on_train_batch_start(self, *args, **kwargs):
        self.start()

    @override
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *args,
        **kwargs,
    ):
        self.stop(pl_module, "train")

    @override
    def on_validation_batch_start(self, *args, **kwargs):
        self.start()

    @override
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *args,
        **kwargs,
    ):
        self.stop(pl_module, "valid")
