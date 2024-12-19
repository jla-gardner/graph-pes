from __future__ import annotations

import argparse
import contextlib
import os
import random

import numpy as np
import torch
import yaml

from graph_pes.config.shared import TorchConfig
from graph_pes.utils.logger import logger
from graph_pes.utils.misc import build_single_nested_dict, nested_merge_all


def extract_config_dict_from_command_line(description: str) -> dict:
    parser = argparse.ArgumentParser(
        description=description,
        epilog="Copyright 2023-24, John Gardner",
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
            "The data2objects package is used to resolve references "
            "and create objects directly from the config dictionary."
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
