from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .config import Config, FittingOptions

__all__ = [
    "Config",
    "get_default_config_values",
    "FittingOptions",
]


def get_default_config_values() -> dict[str, Any]:
    with open(Path(__file__).parent / "defaults.yaml") as f:
        return yaml.safe_load(f)
