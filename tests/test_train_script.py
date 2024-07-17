from __future__ import annotations

import sys
from pathlib import Path

import helpers
import yaml
from graph_pes.config import Config, get_default_config_values
from graph_pes.scripts.train import (
    extract_config_from_command_line,
    parse_args,
    train_from_config,
)
from graph_pes.util import nested_merge


def test_arg_parse():
    config_path = helpers.CONFIGS_DIR / "minimal.yaml"
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


# def mimic_autogen_inputs(prompt):
#     responses = {
#         f"Enter the model type. Must be one of {STR_ALL_MODELS} "
#         "(Required, Case Sensitive): ": "LennardJones",
#         "Data type, 'ase_database' or 'load_atoms_datasets' "
#         "(Required): ": "load_atoms_datasets",
#         "Data Source (Required): ": "QM7",
#         "Neighbour List Cutoff Radius (Default: 4): ": 3,
#         "Number of training structures (Default: 500): ": "480",
#         "Number of validation structures (Default: 100): ": 100,
#         "Convert labels to energy, forces, stress by writing in dict form "
#         "e.g. {'energy': 'U0'}: ": "",
#         "Max Epochs (Default: 100): ": 1,
#         "Learning Rate (Default: 0.001): ": "0.005",
#         "Optimizer Name (Default: AdamW): ": "",
#         "Loss Function, check docs for options (Default: per atom): ": "",
#     }
#     return responses[prompt]


# def test_autogen(monkeypatch):
#     command = "graph-pes-train --autogen"
#     sys.argv = command.split()

#     args = parse_args()
#     assert args.autogen
#     # TODO: Mimic user input to test the auto-generation of config
#     monkeypatch.setattr("builtins.input", mimic_autogen_inputs)
#     config = extract_config_from_command_line()
#     assert config.model == {"graph_pes.models.LennardJones": {}}
#     assert config.data == {
#         "graph_pes.data.load_atoms_datasets": {
#             "id": "QM7",
#             "cutoff": 3.0,
#             "n_train": 480,
#             "n_valid": 100,
#         }
#     }


def test_train_script(tmp_path: Path):
    root = tmp_path / "root"
    config_str = f"""\
general:
    root_dir: {root}
wandb: null
loss: graph_pes.training.loss.PerAtomEnergyLoss()
model: graph_pes.models.LennardJones()    
data:
    graph_pes.data.load_atoms_datasets:
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
