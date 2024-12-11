from __future__ import annotations

import sys
from pathlib import Path

import yaml

from graph_pes.config import get_default_config_values
from graph_pes.config.config import SWAConfig
from graph_pes.scripts.train import (
    extract_config_from_command_line,
    parse_args,
    train_from_config,
)
from graph_pes.utils.misc import nested_merge

from .. import helpers


def test_arg_parse():
    config_path = helpers.CONFIGS_DIR / "minimal.yaml"
    command = f"""\
graph-pes-train {config_path} \
    fitting/loader_kwargs/batch_size=32 \
    data/+load_atoms_dataset/n_train=10
"""
    sys.argv = command.split()

    args = parse_args()
    assert args.args == [
        str(config_path),
        "fitting/loader_kwargs/batch_size=32",
        "data/+load_atoms_dataset/n_train=10",
    ]

    config_data = extract_config_from_command_line()
    assert config_data["fitting"]["loader_kwargs"]["batch_size"] == 32
    assert config_data["data"]["+load_atoms_dataset"]["n_train"] == 10


def test_train_script(tmp_path: Path):
    root = tmp_path / "root"
    config = _get_quick_train_config(root)

    train_from_config(config)

    assert root.exists()
    sub_dir = next(root.iterdir())
    assert (sub_dir / "model.pt").exists()


def test_run_id(tmp_path: Path):
    root = tmp_path / "root"

    # first round: train with no explicit run_id ...
    config = _get_quick_train_config(root)
    assert config["general"]["run_id"] is None

    train_from_config(config)

    # second round: train with an explicit run_id
    config = _get_quick_train_config(root)
    config["general"]["run_id"] = "explicit-id"
    train_from_config(config)
    assert (root / "explicit-id").exists()

    # third round: train with the same explicit run_id
    # and check that the collision is avoided
    config = _get_quick_train_config(root)
    config["general"]["run_id"] = "explicit-id"
    train_from_config(config)
    assert (root / "explicit-id-1").exists()


def test_swa(tmp_path: Path, caplog):
    root = tmp_path / "root"
    config = _get_quick_train_config(root)
    config["fitting"]["swa"] = SWAConfig(lr=0.1, start=1)
    config["fitting"]["trainer_kwargs"]["max_epochs"] = 3

    train_from_config(config)

    assert "SWA: starting SWA" in caplog.text


def _get_quick_train_config(root) -> dict:
    config_str = f"""\
general:
    root_dir: {root}
    run_id: null
wandb: null
loss: +graph_pes.training.loss.PerAtomEnergyLoss()
model: +graph_pes.models.LennardJones()    
data:
    +graph_pes.data.load_atoms_dataset:
        id: {helpers.CU_STRUCTURES_FILE}
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
    return nested_merge(
        get_default_config_values(),
        yaml.safe_load(config_str),
    )
