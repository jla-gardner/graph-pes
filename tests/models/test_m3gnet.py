from __future__ import annotations

import pytest
import torch
from ase.build import molecule, bulk

from graph_pes import AtomicGraph
from graph_pes.atomic_graph import number_of_atoms, to_batch
from graph_pes.models import M3GNet

CUTOFF = 3.5

def test_m3gnet_init():
    """Test M3GNet initialization with different parameters"""
    model = M3GNet(cutoff=CUTOFF)
    assert model.cutoff == CUTOFF
    assert model.implemented_properties == ["local_energies"]
    
    model = M3GNet(cutoff=CUTOFF, channels=32, expansion_features=40, layers=2)
    assert len(model.interactions) == 2

@pytest.mark.filterwarnings("ignore:No energy data found in training data")
def test_m3gnet_predictions():
    """Test M3GNet predictions on a simple molecule"""
    model = M3GNet(cutoff=CUTOFF)
    
    copper = bulk("Cu", cubic=True)
    graph = AtomicGraph.from_ase(copper, cutoff=CUTOFF)
    model.pre_fit_all_components([graph])
    
    predictions = model.get_all_PES_predictions(graph)
    
    # Check shapes
    n_atoms = number_of_atoms(graph)
    assert "energy" in predictions
    assert "forces" in predictions
    assert "local_energies" in predictions
    assert predictions["energy"].shape == ()
    assert predictions["forces"].shape == (n_atoms, 3)
    assert predictions["local_energies"].shape == (n_atoms,)
    

@pytest.mark.filterwarnings("ignore:No energy data found in training data")
def test_m3gnet_batch_predictions():
    """Test M3GNet predictions on batched molecules"""
    model = M3GNet(cutoff=CUTOFF)
    
    molecules = [molecule("H2O"), molecule("NH3")]
    graphs = [AtomicGraph.from_ase(mol, cutoff=CUTOFF) for mol in molecules]
    batch = to_batch(graphs)
    model.pre_fit_all_components(graphs)
    predictions = model.get_all_PES_predictions(batch)
    
    # Check shapes
    assert predictions["energy"].shape == (2,)
    assert predictions["forces"].shape == (number_of_atoms(batch), 3)
    assert predictions["local_energies"].shape == (number_of_atoms(batch),)

@pytest.mark.filterwarnings("ignore:No energy data found in training data")
def test_m3gnet_isolated_atom():
    """Test M3GNet predictions on an isolated atom"""
    model = M3GNet(cutoff=CUTOFF)
    
    atom = molecule("H")
    graph = AtomicGraph.from_ase(atom, cutoff=CUTOFF)
    model.pre_fit_all_components([graph])
    
    predictions = model.get_all_PES_predictions(graph)
    
    assert torch.allclose(
        predictions["forces"], 
        torch.zeros_like(predictions["forces"])
    )

@pytest.mark.filterwarnings("ignore:No energy data found in training data")
def test_m3gnet_serialization(tmp_path):
    """Test saving and loading M3GNet model"""
    model = M3GNet(cutoff=CUTOFF)
    
    methane = molecule("CH4")
    graph = AtomicGraph.from_ase(methane, cutoff=CUTOFF)
    model.pre_fit_all_components([graph])
    
    original_predictions = model.get_all_PES_predictions(graph)
    
    save_path = tmp_path / "m3gnet.pt"
    torch.save(model.state_dict(), save_path)
    
    loaded_model = M3GNet(cutoff=CUTOFF)
    loaded_model.load_state_dict(torch.load(save_path))
    
    loaded_predictions = loaded_model.get_all_PES_predictions(graph)
    
    # Check predictions are the same
    assert torch.allclose(
        original_predictions["energy"],
        loaded_predictions["energy"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        original_predictions["forces"],
        loaded_predictions["forces"],
        atol=1e-6,
        rtol=1e-6,
    ) 