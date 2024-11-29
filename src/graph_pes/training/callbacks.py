from __future__ import annotations

from abc import ABC
from pathlib import Path

import torch
from graph_pes.training.task import PESLearningTask
from pytorch_lightning import Callback, Trainer


class GraphPESCallback(Callback, ABC):
    def __init__(self):
        self.root: Path = None  # type: ignore

    def _register_root(self, root: Path):
        # called by us before any training starts
        self.root = root


class DumpModel(GraphPESCallback):
    def __init__(self, every_n_val_checks: int = 10):
        self.every_n_val_checks = every_n_val_checks

    def on_validation_epoch_end(  # type: ignore
        self,
        trainer: Trainer,
        pl_module: PESLearningTask,
    ):
        epoch = trainer.current_epoch
        if epoch % self.every_n_val_checks != 0:
            return

        model_path = self.root / f"model_{epoch}.pt"
        torch.save(pl_module.model.cpu(), model_path)
