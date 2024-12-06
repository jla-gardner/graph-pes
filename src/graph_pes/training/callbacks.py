from __future__ import annotations

import copy
from abc import ABC
from pathlib import Path
from typing import cast

import torch
from ase.data import chemical_symbols
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.models.addition import AdditionModel
from graph_pes.models.offsets import LearnableOffset
from graph_pes.training.util import VALIDATION_LOSS_KEY
from graph_pes.utils.lammps import deploy_model
from graph_pes.utils.logger import logger
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import Logger


class GraphPESCallback(Callback, ABC):
    """
    A base class for all callbacks that require access to useful
    information generated by the ``graph-pes-train`` command.
    """

    def __init__(self):
        self.root: Path = None  # type: ignore

    def _register_root(self, root: Path):
        # called by us before any training starts
        self.root = root

    def get_model(self, pl_module: LightningModule) -> GraphPESModel:
        return cast(GraphPESModel, pl_module.model)

    def get_model_on_cpu(self, pl_module: LightningModule) -> GraphPESModel:
        model = self.get_model(pl_module)
        model = copy.deepcopy(model)
        model.to("cpu")
        return model

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(root={self.root})"


class DumpModel(GraphPESCallback):
    """
    Dump the model to ``<output_dir>/dumps/model_{epoch}.pt``
    at regular intervals.

    Parameters
    ----------
    every_n_val_checks: int
        The number of validation epochs between dumps.
    """

    def __init__(self, every_n_val_checks: int = 10):
        self.every_n_val_checks = every_n_val_checks

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ):
        if not trainer.is_global_zero:
            return

        epoch = trainer.current_epoch
        if epoch % self.every_n_val_checks != 0:
            return

        model_path = self.root / "dumps" / f"model_{epoch}.pt"
        model_path.parent.mkdir(exist_ok=True)
        torch.save(self.get_model_on_cpu(pl_module), model_path)


def log_offset(model: LearnableOffset, logger: Logger):
    Zs = model._offsets._accessed_Zs
    for Z in Zs:
        logger.log_metrics(
            {f"offset.{chemical_symbols[Z]}": model._offsets[Z].item()}
        )


class OffsetLogger(GraphPESCallback):
    """
    Log any learned, per-element offsets of the model at the
    end of each validation epoch.
    """

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ):
        if not trainer.is_global_zero:
            return

        if not trainer.logger:
            return

        model = self.get_model(pl_module)
        if not isinstance(model, AdditionModel):
            return

        offsets = [c for c in model.models if isinstance(c, LearnableOffset)]
        if offsets:
            log_offset(offsets[0], trainer.logger)


class SaveBestModel(GraphPESCallback):
    """
    Save the best model to ``<output_dir>/model.pt`` and deploy it to
    ``<output_dir>/lammps_model.pt``.
    """

    def __init__(self):
        super().__init__()
        self.best_val_loss = float("inf")
        self.try_to_deploy = True

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule):
        if not trainer.is_global_zero or not self.root:
            return

        # get current validation loss
        current_loss = trainer.callback_metrics.get(VALIDATION_LOSS_KEY, None)
        if current_loss is None:
            return

        # if the validation loss has improved, save the model
        if current_loss < self.best_val_loss:
            self.best_val_loss = current_loss
            cpu_model = self.get_model_on_cpu(pl_module)
            logger.debug(f"New best model: validation loss {current_loss}")

            # log the final path to the trainer.logger.summary
            model_path = self.root / "model.pt"
            lammps_model_path = self.root / "lammps_model.pt"

            assert trainer.logger is not None
            trainer.logger.log_hyperparams(
                {
                    "model_path": model_path,
                    "lammps_model_path": lammps_model_path,
                }
            )
            torch.save(cpu_model, model_path)
            logger.debug(f"Model saved to {model_path}")

            if self.try_to_deploy:
                try:
                    deploy_model(cpu_model, path=lammps_model_path)
                    logger.debug(
                        f"Deployed model for use with LAMMPS to "
                        f"{lammps_model_path}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to deploy model for use with LAMMPS: {e}"
                    )
                    self.try_to_deploy = False
