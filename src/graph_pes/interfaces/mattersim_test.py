from __future__ import annotations

from pathlib import Path

import ase.build
import numpy as np
import pytest
import torch
from mattersim.datasets.utils.convertor import GraphConvertor
from mattersim.forcefield.potential import (
    MatterSimCalculator,
    Potential,
    batch_to_dict,
)
from torch_geometric.loader import DataLoader as DataLoader_pyg

from graph_pes.atomic_graph import AtomicGraph, to_batch
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.interfaces._mattersim import (
    MatterSim_M3Gnet_Wrapper,
    count_number_of_triplets_per_leading_edge,
    mattersim,
)
from graph_pes.utils.lammps import deploy_model
from graph_pes.utils.threebody import (
    triplet_edge_pairs,
)

# Test molecules/crystals
CH4 = ase.build.molecule("CH4")
CH4.center(vacuum=10)
DIAMOND = ase.build.bulk("C", "diamond")
DIAMOND.rattle(0.1)

# models
MATTERSIM_MODEL = Potential.from_checkpoint(  # type: ignore
    "mattersim-v1.0.0-1m", load_training_state=False, device="cpu"
).model
GRAPH_PES_MODEL = MatterSim_M3Gnet_Wrapper(MATTERSIM_MODEL)


def test_output_shapes():
    graph = AtomicGraph.from_ase(DIAMOND)
    outputs = GRAPH_PES_MODEL.get_all_PES_predictions(graph)

    assert outputs["energy"].shape == ()
    assert outputs["forces"].shape == (2, 3)  # 2 atoms in unit cell
    assert outputs["stress"].shape == (3, 3)

    batch = to_batch([graph, graph])
    outputs = GRAPH_PES_MODEL.get_all_PES_predictions(batch)

    assert outputs["energy"].shape == (2,)
    assert outputs["forces"].shape == (4, 3)
    assert outputs["stress"].shape == (2, 3, 3)


@pytest.mark.parametrize("structure", [CH4, DIAMOND])
def test_raw_model_agreement(structure: ase.Atoms):
    graph = AtomicGraph.from_ase(structure)
    us = GRAPH_PES_MODEL.predict_energy(graph)

    c = GraphConvertor(
        twobody_cutoff=GRAPH_PES_MODEL.cutoff.item(),
        threebody_cutoff=GRAPH_PES_MODEL.model.model_args["threebody_cutoff"],
    )
    data = c.convert(structure)
    dl = DataLoader_pyg([data], batch_size=1)
    batch = next(iter(dl))
    batch_dict = batch_to_dict(batch)
    them = MATTERSIM_MODEL(batch_dict)

    assert us.item() == pytest.approx(them.item(), abs=1e-5)


def test_calculator_agreement():
    our_calc = GRAPH_PES_MODEL.ase_calculator(skin=0.0)
    our_calc.calculate(DIAMOND, properties=["energy", "forces", "stress"])
    our_results = our_calc.results

    # Compare with the raw model
    ms_calc = MatterSimCalculator(
        Potential.from_checkpoint(  # type: ignore
            "mattersim-v1.0.0-1m", load_training_state=False, device="cpu"
        )
    )
    ms_calc.calculate(DIAMOND, properties=["energy", "forces", "stress"])
    their_results = ms_calc.results
    np.testing.assert_allclose(
        our_results["energy"], their_results["energy"], atol=1e-5
    )
    np.testing.assert_allclose(
        our_results["forces"], their_results["forces"], atol=1e-5
    )
    np.testing.assert_allclose(
        our_results["stress"], their_results["stress"], atol=1e-5
    )


def test_batch_agreement():
    graph = AtomicGraph.from_ase(DIAMOND)
    our_batch = to_batch([graph, graph])
    us = GRAPH_PES_MODEL.predict_energy(our_batch)
    assert us.shape == (2,)

    c = GraphConvertor(
        twobody_cutoff=GRAPH_PES_MODEL.cutoff.item(),
        threebody_cutoff=GRAPH_PES_MODEL.model.model_args["threebody_cutoff"],
    )
    data = c.convert(DIAMOND)
    dl = DataLoader_pyg([data, data], batch_size=2)
    batch = next(iter(dl))
    assert batch.num_graphs == 2
    batch_dict = batch_to_dict(batch)
    them = MATTERSIM_MODEL(batch_dict)
    assert them.shape == (2,)
    torch.testing.assert_close(us, them)

    # test forces
    pot = Potential(MATTERSIM_MODEL)
    their_forces = pot(batch_dict)["forces"]
    assert their_forces.shape == (4, 3)

    our_forces = GRAPH_PES_MODEL.predict_forces(our_batch)
    torch.testing.assert_close(our_forces, their_forces)


def test_api():
    ms = mattersim()
    assert isinstance(ms, GraphPESModel)


def test_implementation():
    c = GraphConvertor(
        twobody_cutoff=GRAPH_PES_MODEL.cutoff.item(),
        threebody_cutoff=GRAPH_PES_MODEL.model.model_args["threebody_cutoff"],
    )
    data = c.convert(DIAMOND)
    dl = DataLoader_pyg([data], batch_size=1)
    batch = next(iter(dl))
    batch_dict = batch_to_dict(batch)

    graph = AtomicGraph.from_ase(DIAMOND)
    graph = graph._replace(
        neighbour_list=batch_dict["edge_index"],
        neighbour_cell_offsets=batch_dict["pbc_offsets"],
    )

    tep = triplet_edge_pairs(graph, GRAPH_PES_MODEL.three_body_cutoff.item())
    assert torch.all(batch_dict["three_body_indices"] == tep).item()

    count = count_number_of_triplets_per_leading_edge(tep, graph).unsqueeze(-1)
    assert torch.all(batch_dict["num_triple_ij"] == count).item()


def test_deploy(tmp_path: Path):
    pytest.skip("TODO: fix threebody torchscripting")

    deploy_model(GRAPH_PES_MODEL, tmp_path / "test_model.pt")
