from __future__ import annotations

import os
from typing import Final

import pytorch_lightning as pl

from graph_pes.config.shared import instantiate_config_from_dict
from graph_pes.config.testing import TestingConfig
from graph_pes.data.loader import GraphDataLoader
from graph_pes.models import load_model
from graph_pes.scripts.utils import extract_config_dict_from_command_line
from graph_pes.training.loss import PerAtomEnergyLoss, PropertyLoss
from graph_pes.training.tasks import test_with_lightning
from graph_pes.utils import distributed
from graph_pes.utils.logger import logger

DEFAULT_LOADER_KWARGS: Final[dict] = dict(batch_size=2, num_workers=0)


def test(config: TestingConfig) -> None:
    logger.info(f"Testing model at {config.model_path}...")

    model = load_model(config.model_path)
    logger.info("Loaded model.")
    logger.debug(f"Model: {model}")

    loader_kwargs = {
        **DEFAULT_LOADER_KWARGS,
        **config.loader_kwargs,
        "shuffle": False,
    }
    datasets = (
        config.data if isinstance(config.data, dict) else {"test": config.data}
    )

    for dataset in datasets.values():
        if distributed.IS_RANK_0:
            dataset.prepare_data()
        dataset.setup()

    dataloaders = {
        name: GraphDataLoader(dataset, **loader_kwargs)
        for name, dataset in datasets.items()
    }
    trainer = pl.Trainer(
        logger=config.get_logger(),
        accelerator=config.accelerator,
        inference_mode=False,
    )

    all_properties = set.union(
        *[set(dataset.properties) for dataset in datasets.values()]
    )
    eval_metrics = []
    if "energy" in all_properties:
        eval_metrics.append(PerAtomEnergyLoss("RMSE"))
        eval_metrics.append(PerAtomEnergyLoss("MAE"))
        eval_metrics.append(PropertyLoss("energy", "RMSE"))
        eval_metrics.append(PropertyLoss("energy", "MAE"))
    if "forces" in all_properties:
        eval_metrics.append(PropertyLoss("forces", "RMSE"))
        # Force MAE is not invariant wrt. rotations, so we don't log it
        # see "How to validate machine-learned interatomic potentials"
        #      -> https://doi.org/10.1063/5.0139611
    if "stress" in all_properties:
        eval_metrics.append(PropertyLoss("stress", "RMSE"))
    if "virial" in all_properties:
        eval_metrics.append(PropertyLoss("virial", "RMSE"))

    test_with_lightning(trainer, model, dataloaders, eval_metrics)


def main():
    # set the load-atoms verbosity to 1 by default to avoid
    # spamming logs with `rich` output
    os.environ["LOAD_ATOMS_VERBOSE"] = os.getenv("LOAD_ATOMS_VERBOSE", "1")

    config_dict = extract_config_dict_from_command_line(
        "Test a GraphPES model using PyTorch Lightning."
    )
    try:
        _, config = instantiate_config_from_dict(config_dict, TestingConfig)
    except Exception as e:
        raise ValueError(
            "Your configuration file could not be successfully parsed. "
            "Please check that it is formatted correctly. For examples, "
            "please see https://jla-gardner.github.io/graph-pes/cli/graph-pes-test.html"
        ) from e
    test(config)


if __name__ == "__main__":
    main()
