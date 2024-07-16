from __future__ import annotations

import sys
from pathlib import Path

import yaml
from graph_pes.config import Config, get_default_config_values
from graph_pes.scripts.train import (
    extract_config_from_command_line,
    parse_args,
    train_from_config,
)
from graph_pes.util import nested_merge


def test_arg_parse():
    config_path = Path(__file__).parent.parent / "configs/minimal.yaml"
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


def test_train_script(tmp_path: Path):
    structure_path = Path(__file__).parent / "test.xyz"
    root = tmp_path / "root"
    config_str = f"""\
general:
    root_dir: {root}
wandb: null
loss: graph_pes.training.loss.PerAtomEnergyLoss()
model: graph_pes.models.LennardJones()    
data:
    graph_pes.data.load_atoms_datasets:
        id: {structure_path}
        cutoff: 3.0
        n_train: 8
        n_valid: 2
fitting:
    trainer_kwargs:
        max_epochs: 1
        accelerator: cpu
        callbacks: []

    loader_kwargs:
        batch_size: 2
"""
    config = Config.from_dict(
        nested_merge(
            get_default_config_values(),
            yaml.safe_load(config_str),
        )
    )

    train_from_config(config)

    assert root.exists()
    sub_dir = next(root.iterdir())
    assert (sub_dir / "model.pt").exists()
