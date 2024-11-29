from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import cast

import torch
from ase.data import chemical_symbols
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.models.addition import AdditionModel
from graph_pes.models.offsets import LearnableOffset
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import Logger


class GraphPESCallback(Callback, ABC):
    def __init__(self):
        self.root: Path = None  # type: ignore

    def _register_root(self, root: Path):
        # called by us before any training starts
        self.root = root

    def get_model(self, pl_module: LightningModule) -> GraphPESModel:
        return cast(GraphPESModel, pl_module)


class DumpModel(GraphPESCallback):
    def __init__(self, every_n_val_checks: int = 10):
        self.every_n_val_checks = every_n_val_checks

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ):
        epoch = trainer.current_epoch
        if epoch % self.every_n_val_checks != 0:
            return

        model_path = self.root / "dumps" / f"model_{epoch}.pt"
        model_path.parent.mkdir(exist_ok=True)
        torch.save(self.get_model(pl_module).cpu(), model_path)


def log_offset(model: LearnableOffset, logger: Logger):
    Zs = model._offsets._accessed_Zs
    for Z in Zs:
        logger.log_metrics(
            {f"offset.{chemical_symbols[Z]}": model._offsets[Z].item()}
        )


class OffsetLogger(GraphPESCallback):
    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ):
        if not trainer.logger:
            return

        model = self.get_model(pl_module)
        if not isinstance(model, AdditionModel):
            return

        offsets = [c for c in model.models if isinstance(c, LearnableOffset)]
        if offsets:
            log_offset(offsets[0], trainer.logger)
