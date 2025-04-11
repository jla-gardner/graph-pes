from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, Literal

import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig

from graph_pes.atomic_graph import (
    AtomicGraph,
    PropertyKey,
    number_of_atoms,
    number_of_structures,
    to_batch,
)
from graph_pes.config.training import FittingOptions
from graph_pes.data.datasets import DatasetCollection, GraphDataset
from graph_pes.data.loader import GraphDataLoader
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.training.loss import (
    MAE,
    Loss,
    PerAtomEnergyLoss,
    PropertyLoss,
    TotalLoss,
)
from graph_pes.training.loss import (
    RMSE as RMSE_batchwise,
)
from graph_pes.training.opt import LRScheduler, Optimizer
from graph_pes.training.utils import (
    VALIDATION_LOSS_KEY,
    log_model_info,
    sanity_check,
)
from graph_pes.utils.logger import logger
from graph_pes.utils.nn import PerElementParameter, UniformModuleList
from graph_pes.utils.sampling import SequenceSampler
from graph_pes.utils.shift_and_scale import add_auto_offset


def train_with_lightning(
    trainer: pl.Trainer,
    model: GraphPESModel,
    data: DatasetCollection,
    loss: TotalLoss,
    fit_config: FittingOptions,
    optimizer: Optimizer,
    user_eval_metrics: list[Loss] | None = None,
    scheduler: LRScheduler | None = None,
) -> GraphPESModel:
    # - prepare the data
    if trainer.global_rank == 0:
        logger.info("Preparing data")
        data.train.prepare_data()
        data.valid.prepare_data()
    trainer.strategy.barrier("data prepare")

    logger.info("Setting up datasets")
    data.train.setup()
    data.valid.setup()

    loader_kwargs = {**fit_config.loader_kwargs}
    loader_kwargs["shuffle"] = True
    train_loader = GraphDataLoader(
        data.train,
        **loader_kwargs,
        three_body_cutoff=model.three_body_cutoff.item() or None,
    )
    loader_kwargs["shuffle"] = False
    valid_loader = GraphDataLoader(
        data.valid,
        **loader_kwargs,
        three_body_cutoff=model.three_body_cutoff.item() or None,
    )

    eval_metrics = get_all_eval_metrics(
        [data.train, data.valid], user_eval_metrics
    )

    # - do some pre-fitting
    pre_fit_graphs = SequenceSampler(data.train.graphs)
    if fit_config.max_n_pre_fit is not None:
        pre_fit_graphs = pre_fit_graphs.sample_at_most(fit_config.max_n_pre_fit)
    pre_fit_graphs = list(pre_fit_graphs)

    # optionally pre-fit the model
    if fit_config.pre_fit_model:
        logger.info(f"Pre-fitting the model on {len(pre_fit_graphs):,} samples")
        model.pre_fit_all_components(pre_fit_graphs)
    trainer.strategy.barrier("pre-fit")

    # optionally account for reference energies
    if fit_config.auto_fit_reference_energies:
        model = add_auto_offset(model, pre_fit_graphs)
    trainer.strategy.barrier("auto-fit reference energies")

    # always register the elements in the training set
    for param in model.parameters():
        if isinstance(param, PerElementParameter):
            param.register_elements(
                torch.unique(to_batch(pre_fit_graphs).Z).tolist()
            )

    # always pre-fit the losses
    for subloss in loss.losses:
        subloss.pre_fit(to_batch(pre_fit_graphs))

    # - log the model info
    if trainer.global_rank == 0:
        log_model_info(model, trainer.logger)

    # - sanity checks
    logger.info("Sanity checking the model...")
    sanity_check(model, next(iter(train_loader)))

    # - create the task (a pytorch lightning module)
    task = TrainingTask(model, loss, optimizer, scheduler, eval_metrics)

    # - train the model
    logger.info("Starting fit...")
    try:
        trainer.fit(task, train_loader, valid_loader)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise e
    finally:
        try:
            task.load_best_weights(model, trainer)
        except Exception as e:
            logger.error(f"Failed to load best weights: {e}")

    return model


class TrainingTask(pl.LightningModule):
    def __init__(
        self,
        model: GraphPESModel,
        loss: TotalLoss,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        eval_metrics: list[Loss],
    ):
        super().__init__()
        self.model = model
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler
        self.total_loss = loss
        self.train_properties: list[PropertyKey] = list(
            set.union(
                set(),
                *[
                    set(loss.required_properties)
                    for loss in self.total_loss.losses
                ],
            )
        )
        self.eval_metrics = eval_metrics
        self.validation_properties = list(
            set.union(
                set(), *[set(m.required_properties) for m in eval_metrics]
            )
        )
        self._torchmetrics = UniformModuleList(
            [
                m
                for m in self.eval_metrics
                if isinstance(m.metric, torchmetrics.Metric)
            ]
        )

    def forward(self, graphs: AtomicGraph) -> torch.Tensor:
        """Get the energy"""
        return self.model.predict_energy(graphs)

    def on_train_start(self):
        super().on_train_start()
        self.model.train()

    def on_validation_start(self):
        super().on_validation_start()
        self.model.eval()

    def on_validation_end(self) -> None:
        super().on_validation_end()
        self.model.train()

    def _step(
        self,
        graph: AtomicGraph,
        mode: Literal["train", "valid"],
    ) -> torch.Tensor:
        """
        Get (and log) the losses and metrics for a step.

        Parameters
        ----------
        graph: AtomicGraph
            The atomic graph to compute the losses and metrics for.
        prefix: Literal["train", "valid", "test"]
            The prefix of the step, either "train", "valid", or "test".

        Returns
        -------
        torch.Tensor
            The total loss for the step.
        """

        def log(
            name: str,
            value: torch.Tensor | float | torchmetrics.Metric,
            per_atom: bool = False,
        ):
            if isinstance(value, torch.Tensor):
                value = value.item()

            validating = mode == "valid"

            return self.log(
                f"{mode}/{name}",
                value,
                prog_bar=(
                    validating and "metric" in name and "batchwise" not in name
                ),
                on_step=not validating,
                on_epoch=validating,
                sync_dist=validating,
                batch_size=(
                    number_of_atoms(graph)
                    if per_atom
                    else number_of_structures(graph)
                ),
            )

        # generate prediction:
        if mode == "train":
            desired_properties = self.train_properties
        else:
            desired_properties = list(graph.properties.keys())

        predictions = self.model.predict(graph, properties=desired_properties)

        # compute the loss and its sub-components
        total_loss_result = self.total_loss(self.model, graph, predictions)

        # log
        log("loss/total", total_loss_result.loss_value)
        for name, loss_pair in total_loss_result.components.items():
            log(f"metrics/{name}", loss_pair.loss_value, loss_pair.is_per_atom)
            log(
                f"loss/{name}_weighted",
                loss_pair.weighted_loss_value,
                loss_pair.is_per_atom,
            )

        # log additional values during validation
        if mode == "valid":
            for eval in self.eval_metrics:
                if eval.name in total_loss_result.components:
                    # don't double log
                    continue

                if any(
                    p not in graph.properties for p in eval.required_properties
                ):
                    warnings.warn(
                        f"Metric {eval.name} requires properties "
                        f"{eval.required_properties} that are "
                        "not present in the graph: "
                        f"{list(graph.properties.keys())}. We won't log this "
                        "in this batch",
                        stacklevel=2,
                    )
                    continue
                value = eval(self.model, graph, predictions)
                log(f"metrics/{eval.name}", value, eval.is_per_atom)

        return total_loss_result.loss_value

    def training_step(self, structure: AtomicGraph, _):
        return self._step(structure, "train")

    def validation_step(self, structure: AtomicGraph, _):
        return self._step(structure, "valid")

    def configure_optimizers(
        self,
    ) -> torch.optim.Optimizer | OptimizerLRSchedulerConfig:
        opt = self.optimizer_factory(self.model)
        if self.scheduler_factory is None:
            return opt

        scheduler = self.scheduler_factory(opt)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            trainer = self.trainer
            assert trainer is not None

            # if the user is validating once every n training epochs,
            # set the scheduler to update every n training epochs
            if trainer.check_val_every_n_epoch:
                config = {
                    "interval": "epoch",  # really this means training epoch
                    "frequency": trainer.check_val_every_n_epoch,
                    "monitor": VALIDATION_LOSS_KEY,
                    "strict": True,
                }

            # otherwise, they are validating every n training steps:
            # set the scheduler to update every n training steps
            else:
                if isinstance(trainer.val_check_interval, float):
                    raise ValueError(
                        "`graph-pes` does not support specifying "
                        "`val_check_interval` as a fraction of the training "
                        "data size. Please specify it as an integer number of "
                        "training steps."
                    )
                config = {
                    "interval": "step",
                    "frequency": trainer.val_check_interval,
                    "monitor": VALIDATION_LOSS_KEY,
                    "strict": True,
                }

        else:
            # vanilla LR scheduler: update every training step
            config = {
                "interval": "step",
                "frequency": 1,
            }

        logger.debug(f"Using LR scheduler config:\n{config}")
        config["scheduler"] = scheduler
        return {"optimizer": opt, "lr_scheduler": config}  # type: ignore

    # we override the defaults to turn on gradient tracking for the
    # validation/test steps since we (might) compute the forces using autograd
    def on_validation_model_eval(self, *args, **kwargs):
        super().on_validation_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)

    @classmethod
    def load_best_weights(
        cls,
        model: GraphPESModel,
        trainer: pl.Trainer | None = None,
        checkpoint_path: Path | str | None = None,
    ):
        if checkpoint_path is None and trainer is None:
            raise ValueError(
                "Either trainer or checkpoint_path must be provided"
            )
        if checkpoint_path is None:
            path = trainer.checkpoint_callback.best_model_path  # type: ignore
        else:
            path = Path(checkpoint_path)
        logger.info(f'Loading best weights from "{path}"')
        checkpoint = torch.load(path, weights_only=True)
        state_dict = {
            k.replace("model.", "", 1): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("model.")
        }
        model.load_state_dict(state_dict)


def test_with_lightning(
    trainer: pl.Trainer,
    model: GraphPESModel,
    data: dict[str, GraphDataset],
    loader_kwargs: dict,
    logging_prefix: str,
    user_eval_metrics: list[Loss] | None = None,
):
    assert trainer.test_loop.inference_mode is False

    test_loaders = [
        GraphDataLoader(dataset, **{**loader_kwargs, "shuffle": False})
        for dataset in data.values()
    ]

    task = TestingTask(
        model,
        list(data.keys()),
        get_all_eval_metrics(data.values(), user_eval_metrics),
        logging_prefix,
    )
    trainer.test(task, test_loaders)


class TestingTask(pl.LightningModule):
    def __init__(
        self,
        model: GraphPESModel,
        test_names: list[str],
        eval_metrics: list[Loss],
        logging_prefix: str,
    ):
        super().__init__()
        self.model = model
        self.test_names = test_names
        self.eval_metrics = eval_metrics
        self.test_properties = list(
            set.union(
                set(), *[set(m.required_properties) for m in eval_metrics]
            )
        )
        self.logging_prefix = logging_prefix

        self._torchmetrics = UniformModuleList(
            [
                m
                for m in self.eval_metrics
                if isinstance(m.metric, torchmetrics.Metric)
            ]
        )

    def test_step(
        self, structure: AtomicGraph, batch_idx: int, dataloader_idx: int = 0
    ):
        predictions = self.model.predict(
            structure, properties=self.test_properties
        )
        test_name = self.test_names[dataloader_idx]
        if test_name == "test" and len(self.test_names) == 1:
            prefix = self.logging_prefix
        else:
            prefix = f"{self.logging_prefix}/{test_name}"

        for metric in self.eval_metrics:
            if not all(
                p in structure.properties for p in metric.required_properties
            ):
                continue

            value = metric(self.model, structure, predictions)
            self.log(
                f"{prefix}/{metric.name}",
                value,
                on_epoch=True,
                sync_dist=True,
                batch_size=number_of_structures(structure),
                add_dataloader_idx=False,
            )

        return torch.tensor(0.0, requires_grad=True)

    def configure_optimizers(self):
        # no need for optimizer or scheduler during testing
        return None

    def on_test_model_eval(self):
        self.model.eval()
        # we need grad enabled to compute forces using autograd
        torch.set_grad_enabled(True)


def get_eval_metrics_for(dataset: GraphDataset) -> list[Loss]:
    evals = []

    # we use the torchmetrics RMSE implementation to allow
    # for correct accumulation of the RMSE over multiple batches
    # but also keep the batchwise RMSE for historic reasons
    def tm_RMSE():
        return torchmetrics.MeanSquaredError(squared=False)

    if "energy" in dataset.properties:
        evals.append(PerAtomEnergyLoss(metric=tm_RMSE()))
        evals.append(PerAtomEnergyLoss(metric=MAE()))
        evals.append(PropertyLoss("energy", tm_RMSE()))
        evals.append(PropertyLoss("energy", MAE()))
    if "forces" in dataset.properties:
        evals.append(PropertyLoss("forces", tm_RMSE()))
        evals.append(PropertyLoss("forces", RMSE_batchwise()))
        evals.append(PropertyLoss("forces", MAE()))
    if "stress" in dataset.properties:
        evals.append(PropertyLoss("stress", tm_RMSE()))
        evals.append(PropertyLoss("stress", MAE()))
    if "virial" in dataset.properties:
        evals.append(PropertyLoss("virial", tm_RMSE()))
        evals.append(PropertyLoss("virial", MAE()))

    return evals


def get_all_eval_metrics(
    data_sources: Iterable[GraphDataset],
    user_specified_metrics: list[Loss] | None = None,
) -> list[Loss]:
    if user_specified_metrics is None:
        user_specified_metrics = []

    # de-duplicate based on the name of the metric
    de_duped_metrics = {}
    for data_source in data_sources:
        de_duped_metrics.update(
            {m.name: m for m in get_eval_metrics_for(data_source)}
        )
    de_duped_metrics.update({m.name: m for m in user_specified_metrics})

    return list(de_duped_metrics.values())
