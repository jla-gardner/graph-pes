from __future__ import annotations

import time
from typing import Any, Literal

import pytorch_lightning as pl
from graph_pes.logger import logger
from pytorch_lightning.callbacks import ProgressBar, StochasticWeightAveraging
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
                prog_bar=name == "its_per_s",
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


class LoggedProgressBar(ProgressBar):
    """
    A progress bar that logs all metrics at the end of each validation epoch.
    """

    def __init__(self):
        super().__init__()
        self._enabled = True

    @override
    def disable(self):
        self._enabled = False

    @override
    def enable(self):
        self._enabled = True

    @override
    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ):
        if not self._enabled or trainer.sanity_checking:
            return

        metrics = self.get_metrics(trainer, pl_module)
        widths = {k: max(len(k), len(v)) + 3 for k, v in metrics.items()}

        if trainer.current_epoch == 0:
            print("".join(f"{k:>{widths[k]}}" for k in metrics))
        print("".join(f"{v:>{widths[k]}}" for k, v in metrics.items()))

    @override
    def get_metrics(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> dict[str, str]:
        def logged_value(v: float | int | Any):
            return f"{v:.5f}" if isinstance(v, float) else str(v)

        metrics = {"epoch": f"{trainer.current_epoch:>5}"}
        super_metrics = super().get_metrics(trainer, pl_module)
        super_metrics.pop("v_num", None)
        for k, v in super_metrics.items():
            metrics[k] = logged_value(v)

        # rearrange according to: epoch | valid/* | rest...
        sorted_metrics = {"epoch": metrics.pop("epoch")}
        for k in list(metrics):
            if k.startswith("valid/"):
                sorted_metrics[k] = metrics.pop(k)
        sorted_metrics.update(metrics)

        return sorted_metrics
