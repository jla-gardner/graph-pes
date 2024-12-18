from __future__ import annotations

import argparse
import contextlib

import yaml

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
