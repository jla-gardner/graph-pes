from __future__ import annotations

import time
from typing import Any, Literal

import pytorch_lightning as pl
from graph_pes.atomic_graph import (
    AtomicGraph,
    number_of_atoms,
    number_of_structures,
)
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.models.addition import AdditionModel
from graph_pes.utils.logger import logger
from graph_pes.utils.nn import learnable_parameters
from pytorch_lightning.callbacks import ProgressBar, StochasticWeightAveraging
from pytorch_lightning.loggers import Logger as PTLLogger
from typing_extensions import override


def log_model_info(
    model: GraphPESModel,
    ptl_logger: PTLLogger | None = None,
) -> None:
    """Log the number of parameters in a model."""

    logger.info(f"Model:\n{model}")

    if isinstance(model, AdditionModel):
        model_names = [
            f"{given_name} ({component.__class__.__name__})"
            for given_name, component in model.models.items()
        ]
        params = [
            learnable_parameters(component)
            for component in model.models.values()
        ]
        width = max(len(name) for name in model_names)
        info_str = "Number of learnable params:"
        for name, param in zip(model_names, params):
            info_str += f"\n    {name:<{width}}: {param:,}"
        logger.info(info_str)

    else:
        params = sum(p.numel() for p in model.parameters())
        logger.info(f"Number of learnable params : {params:,}")

    if ptl_logger is not None:
        all_params = sum(p.numel() for p in model.parameters())
        learnable_params = learnable_parameters(model)
        ptl_logger.log_metrics(
            {
                "n_parameters": all_params,
                "n_learnable_parameters": learnable_params,
            }
        )


def sanity_check(model: GraphPESModel, batch: AtomicGraph) -> None:
    outputs = model.get_all_PES_predictions(batch)

    N = number_of_atoms(batch)
    S = number_of_structures(batch)
    expected_shapes = {
        "local_energies": (N,),
        "forces": (N, 3),
        "energy": (S,),
        "stress": (S, 3, 3),
    }

    incorrect = []
    for key, value in outputs.items():
        if value.shape != expected_shapes[key]:
            incorrect.append((key, value.shape, expected_shapes[key]))

    if len(incorrect) > 0:
        raise ValueError(
            "Sanity check failed for the following outputs:\n"
            + "\n".join(
                f"{key}: {value} != {expected}"
                for key, value, expected in incorrect
            )
        )


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
                sync_dist=stage == "valid",
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


class LoggedProgressBar(ProgressBar):
    """
    A progress bar that logs all metrics at the end of each validation epoch.
    """

    def __init__(self):
        super().__init__()
        self._enabled = True
        self._start_time = time.time()
        self._widths: dict[str, int] = {}

    @override
    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        self._start_time = time.time()

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

        if not self._widths:
            # first time we're logging things:
            # calculate the widths of the columns, and print the headers

            # Split headers with more than 1 slash
            # over two rows
            def split_header(h: str) -> list[str]:
                if "/" in h:
                    parts = h.split("/")
                    if len(parts) > 2:
                        return [f"{parts[0]}/{parts[1]}", "/".join(parts[2:])]
                return [h]

            headers = list(metrics.keys())
            split_headers = [split_header(h) for h in headers]

            header_widths = {
                header: max(len(line) for line in lines)
                for header, lines in zip(headers, split_headers)
            }
            content_widths = {
                header: len(metrics[header]) for header in headers
            }
            self._widths = {
                header: max(header_widths[header], content_widths[header])
                + (6 if header == "time" else 3)
                for header in headers
            }

            # print the headers
            first_row = [
                "" if len(lines) == 1 else lines[0] for lines in split_headers
            ]
            second_row = [lines[-1] for lines in split_headers]

            print(
                "".join(
                    f"{part:>{self._widths[header]}}"
                    for part, header in zip(first_row, headers)
                )
            )
            print(
                "".join(
                    f"{part:>{self._widths[header]}}"
                    for part, header in zip(second_row, headers)
                )
            )

        # print the values for this epoch
        print("".join(f"{v:>{self._widths[k]}}" for k, v in metrics.items()))

    @override
    def get_metrics(  # type: ignore
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> dict[str, str]:
        def logged_value(v: float | int | Any):
            return f"{v:.5f}" if isinstance(v, float) else str(v)

        metrics = {"epoch": f"{trainer.current_epoch + 1:>5}"}
        super_metrics = super().get_metrics(trainer, pl_module)
        super_metrics.pop("v_num", None)
        for k, v in super_metrics.items():
            metrics[k] = logged_value(v)

        # rearrange according to: epoch | time | valid/* | rest...
        sorted_metrics = {"epoch": metrics.pop("epoch")}
        sorted_metrics["time"] = f"{time.time() - self._start_time:.1f}"
        for k in list(metrics):
            if k.startswith("valid/"):
                sorted_metrics[k] = metrics.pop(k)
        sorted_metrics.update(metrics)

        return sorted_metrics


VALIDATION_LOSS_KEY = "valid/loss/total"
