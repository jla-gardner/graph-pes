from __future__ import annotations

import sys
from pathlib import Path

from graph_pes.scripts.train import extract_config_from_command_line, parse_args


def test_arg_parse():
    config_path = (
        Path(__file__).parent.parent / "src/graph_pes/configs/minimal.yaml"
    )
    command = f"""\
graph-pes-train --config {config_path} \
    fitting^loader_kwargs^batch_size=32 \
    data^graph_pes.data.load_atoms_dataset^n_train=10
"""
    sys.argv = command.split()

    args = parse_args()
    assert args.config == [str(config_path)]
    assert args.overrides == [
        "fitting^loader_kwargs^batch_size=32",
        "data^graph_pes.data.load_atoms_dataset^n_train=10",
    ]

    config = extract_config_from_command_line()
    assert config.fitting.loader_kwargs["batch_size"] == 32
    assert config.data["graph_pes.data.load_atoms_dataset"]["n_train"] == 10  # type: ignore
