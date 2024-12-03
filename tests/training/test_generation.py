from __future__ import annotations

import sys

from graph_pes.config.config import Config
from graph_pes.scripts.generation import get_automation_options
from graph_pes.scripts.train import extract_config_from_command_line, parse_args
from graph_pes.training.loss import PerAtomEnergyLoss


def mimic_autogen_inputs(prompt):
    responses = {
        "Select a model to train. Must be a member of [PaiNN(), LennardJones(),"
        " SchNet(), TensorNet(), Morse(), LennardJonesMixture(), NequIP(), "
        "ZEmbeddingNequIP(), MACE(), ZEmbeddingMACE()]. "
        "Default: SchNet()\n": "LennardJones()",
        "Select a loss function to use. Check documentation for options."
        " Default: PerAtomEnergyLoss()\n": "",
        "Select a dataset to train on.  This must be either a) a path to "
        "an ASE-readable file, or b) an id from load-atoms. Default: QM7\n": "",
        "Select a cutoff for the dataset. Default: 4.0": "3.5",
        "Select the number of training samples. Default: 500": "10",
        "Select the number of validation samples. Default: 100": "5",
        "Select an optimizer, check documentation for options. "
        "Default: AdamW\n": "",
        "Select learning rate for optimizer. Default: 0.001": "",
        "Select the maximum number of epochs to train for. Default: 100\n": "1",
        "Select the batch size for training. Default: 4 ": "",
    }
    return responses[prompt]


def test_autogen(monkeypatch):
    configs = get_automation_options()
    assert configs["model"]["value"] == "SchNet()"
    assert (
        configs["fitting"]["optimizer"]["+Optimizer"]["name"]["type"] == "str"
    )
    command = "graph-pes-train"
    sys.argv = command.split()

    args = parse_args()
    assert not args.args

    monkeypatch.setattr("builtins.input", mimic_autogen_inputs)
    config_data = extract_config_from_command_line()
    _, config = Config.from_raw_config_dicts(config_data)
    assert isinstance(config.loss, PerAtomEnergyLoss)
