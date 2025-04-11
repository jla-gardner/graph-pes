from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from pytorch_lightning.loggers import CSVLogger, Logger

from graph_pes.config.shared import TorchConfig
from graph_pes.training.callbacks import WandbLogger
from graph_pes.utils import distributed
from graph_pes.utils.logger import logger
from graph_pes.utils.misc import (
    build_single_nested_dict,
    nested_merge,
    nested_merge_all,
)


def extract_config_dict_from_command_line(description: str) -> dict:
    parser = argparse.ArgumentParser(
        description=description,
        epilog="Copyright 2023-25, John Gardner",
    )
    parser.add_argument(
        "args",
        nargs="*",
        help=(
            "Config files and command line specifications. "
            "Config files should be YAML (.yaml/.yml) files. "
            "Command line specifications should be in the form "
            "my/nested/key=value. "
            "Final config is built up from these items in a left "
            "to right manner, with later items taking precedence "
            "over earlier ones in the case of conflicts. "
        ),
    )
    args = parser.parse_args()
    return nested_merge_all(*map(get_data_from_cli_arg, args.args))


def get_data_from_cli_arg(arg: str) -> dict:
    if arg.endswith(".yaml") or arg.endswith(".yml"):
        # it's a config file
        try:
            with open(arg) as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(
                f"You specified a config file ({arg}) " "that we couldn't load."
            )
            raise e

    elif "=" in arg:
        # it's an override
        key, value = arg.split("=", maxsplit=1)
        keys = key.split("/")

        # parse the value
        with contextlib.suppress(yaml.YAMLError):
            value = yaml.safe_load(value)

        return build_single_nested_dict(keys, value)

    raise ValueError(
        f"Invalid argument: {arg}. "
        "Expected a YAML file or an override in the form key=value"
    )


def configure_general_options(torch_config: TorchConfig, seed: int):
    prec = torch_config.float32_matmul_precision
    torch.set_float32_matmul_precision(prec)
    logger.debug(f"Using {prec} precision for float32 matrix multiplications.")

    ftype = torch_config.dtype
    logger.debug(f"Using {ftype} as default dtype.")
    torch.set_default_dtype(
        {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }[ftype]
    )
    # a nice setting for e3nn components that get scripted upon instantiation
    # - DYNAMIC refers to the fact that they will expect different input sizes
    #   at every iteration (graphs are not all the same size)
    # - 4 is the number of times we attempt to recompile before giving up
    torch.jit.set_fusion_strategy([("DYNAMIC", 4)])

    # a non-verbose version of pl.seed_everything
    logger.debug(f"Using seed {seed} for reproducibility.")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PL_SEED_WORKERS"] = "0"


def extract_summary(logs: Logger) -> dict:
    if isinstance(logs, WandbLogger):
        return dict(logs.experiment.summary)

    elif isinstance(logs, CSVLogger):
        try:
            experiment = logs.experiment
            with experiment._fs.open(
                experiment.metrics_file_path, "r", newline=""
            ) as file:
                metrics = list(csv.DictReader(file))
        except Exception as e:
            logger.warning(f"Failed to read metrics file: {e}")
            return {}

        summary = {}
        for metric_row in metrics:
            m = {
                k: ast.literal_eval(v) for k, v in metric_row.items() if v != ""
            }
            summary.update(m)
        return summary

    else:
        logger.warning(
            f"Unsupported logger type: {type(logs)}. "
            "We can't extract a summary."
        )
        return {}


def update_summary(logger: Logger | None, summary_file: Path):
    if logger is None:
        return
    if not distributed.IS_RANK_0:
        return

    if summary_file.exists():
        with open(summary_file) as f:
            existing = yaml.safe_load(f)
    else:
        existing = {}

    summary = extract_summary(logger)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w") as f:
        yaml.dump(nested_merge(existing, summary), f)
