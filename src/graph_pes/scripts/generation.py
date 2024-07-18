from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from graph_pes.config import get_default_config_values
from graph_pes.config.spec import Config
from graph_pes.util import nested_merge


def get_automation_options() -> dict[str, Any]:
    with open(Path(__file__).parent / "automation.yaml") as f:
        return yaml.safe_load(f)


def recursive_search(in_dict: dict, target_dict: dict) -> None:
    for k, v in in_dict.items():
        print(k, v)
        if all(i in v for i in ["prompt", "type", "value"]):
            # Get input from user and update out_dict value key
            type_check = eval(v["type"])
            print(target_dict[k]["value"])
            if "prefix" in v:
                target_dict[k]["value"] = v["prefix"] + type_check(
                    input(v["prompt"]) or v["value"]
                )
            else:
                target_dict[k]["value"] = type_check(
                    input(v["prompt"]) or v["value"]
                )
        elif isinstance(v, dict) and v.keys() != {"prompt", "type", "value"}:
            # Recurse through nested dict, updating out_dict as needed
            recursive_search(v, target_dict[k])
        else:
            continue


def clean_dict(in_dict: dict) -> dict:
    out_dict = {}
    for k, v in in_dict.items():
        if all(i in v for i in ["prompt", "type", "value"]):
            out_dict[k] = v["value"]
        elif isinstance(v, dict) and v.keys() != {"prompt", "type", "value"}:
            out_dict[k] = clean_dict(v)
        else:
            continue
    return out_dict


def config_auto_generation() -> Config:
    # Get prompts, types, and default values from automation.yaml

    user_inputs = get_automation_options()
    # Recursively search through prompts and update config_dict
    recursive_search(user_inputs, user_inputs)
    defaults = get_default_config_values()
    user_inputs = clean_dict(user_inputs)
    print(user_inputs)
    config_dict = nested_merge(defaults, user_inputs)
    # Generate config
    return Config.from_dict(config_dict)
