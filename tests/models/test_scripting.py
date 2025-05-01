import pytest
import torch
from ase.build import molecule

from graph_pes.models import SchNet, ScriptedModel, load_model


def test_scripting():
    model = SchNet()
    water = molecule("H2O")
    pred = model.ase_calculator().get_potential_energy(water)

    _scripted = torch.jit.script(model)
    scripted = ScriptedModel(_scripted)
    pred_scripted = scripted.ase_calculator().get_potential_energy(water)

    assert pred == pytest.approx(pred_scripted)

    torch.jit.script(scripted).save("scripted.pt")
    scripted_from_file = load_model("scripted.pt")
    pred_scripted_from_file = (
        scripted_from_file.ase_calculator().get_potential_energy(water)
    )
    assert pred == pytest.approx(pred_scripted_from_file)
