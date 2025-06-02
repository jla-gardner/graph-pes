from __future__ import annotations

import ase.build
import numpy as np
import pytest
import torch
from e3nn import o3
from mace.calculators import MACECalculator
from mace.modules import ScaleShiftMACE, gate_dict, interaction_classes

from graph_pes.atomic_graph import AtomicGraph, to_batch
from graph_pes.interfaces._mace import (
    MACEWrapper,
    go_mace_23,
    mace_mp,
    mace_off,
)
from graph_pes.utils.calculator import GraphPESCalculator

ELEMENTS = [1, 6, 8]
CUTOFF = 5.0


def create_default_scaleshift_mace(
    atomic_numbers: list,
    r_max: float = 5.0,
    num_bessel: int = 8,
    num_polynomial_cutoff: int = 5,
    max_ell: int = 3,
    num_interactions: int = 3,
    hidden_irreps: str = "32x0e + 32x1o",
    mlp_irreps: str = "16x0e",
    atomic_energies: torch.Tensor | None = None,
    avg_num_neighbors: float = 1.0,
    atomic_inter_scale: float = 1.0,
    atomic_inter_shift: float = 0.0,
) -> ScaleShiftMACE:
    if atomic_energies is None:
        atomic_energies = torch.zeros(len(atomic_numbers))

    interaction_cls = interaction_classes[
        "RealAgnosticResidualInteractionBlock"
    ]
    interaction_cls_first = interaction_classes["RealAgnosticInteractionBlock"]

    # Default gate function
    gate = gate_dict["silu"]

    # Create model
    model = ScaleShiftMACE(
        r_max=r_max,
        num_bessel=num_bessel,
        num_polynomial_cutoff=num_polynomial_cutoff,
        max_ell=max_ell,
        num_interactions=num_interactions,
        num_elements=len(atomic_numbers),
        hidden_irreps=o3.Irreps(hidden_irreps),
        MLP_irreps=o3.Irreps(mlp_irreps),
        atomic_energies=atomic_energies,
        avg_num_neighbors=avg_num_neighbors,
        atomic_numbers=atomic_numbers,
        correlation=3,
        atomic_inter_scale=atomic_inter_scale,
        atomic_inter_shift=atomic_inter_shift,
        # Added required parameters
        interaction_cls=interaction_cls,
        interaction_cls_first=interaction_cls_first,
        gate=gate,
        # Optional parameters with defaults
        pair_repulsion=True,
        distance_transform="None",
        radial_MLP=[64, 64, 64],
        radial_type="bessel",
        heads=None,
        cueq_config=None,
    )

    return model


MACE_MODEL = create_default_scaleshift_mace(ELEMENTS, CUTOFF)

# Test molecules/crystals
CH4 = ase.build.molecule("CH4")
DIAMOND = ase.build.bulk("C", "diamond", a=3.5668)

# Pre-configured calculators
MACE_CALC = MACECalculator(models=MACE_MODEL)
GRAPH_PES_MODEL = MACEWrapper(MACE_MODEL)
GRAPH_PES_CALC = GRAPH_PES_MODEL.ase_calculator(skin=0.0)


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


def test_molecular():
    MACE_CALC.calculate(CH4, properties=["energy", "forces"])
    GRAPH_PES_CALC.calculate(CH4, properties=["energy", "forces"])

    assert MACE_CALC.results["energy"] == pytest.approx(
        GRAPH_PES_CALC.results["energy"], abs=1e-5
    )
    np.testing.assert_allclose(
        MACE_CALC.results["forces"], GRAPH_PES_CALC.results["forces"], atol=1e-5
    )


def test_periodic():
    MACE_CALC.calculate(DIAMOND, properties=["energy", "forces", "stress"])
    GRAPH_PES_CALC.calculate(DIAMOND, properties=["energy", "forces", "stress"])

    assert MACE_CALC.results["energy"] == pytest.approx(
        GRAPH_PES_CALC.results["energy"], abs=1e-5
    )
    np.testing.assert_allclose(
        MACE_CALC.results["forces"],
        GRAPH_PES_CALC.results["forces"],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        MACE_CALC.results["stress"].flatten(),
        GRAPH_PES_CALC.results["stress"].flatten(),
        atol=1e-5,
    )


def test_mace_mp():
    # check can download model
    base_model = mace_mp("small", "float32")
    calc = GraphPESCalculator(base_model)

    # check that forces on central atom are roughly zero
    calc.calculate(CH4, properties=["energy", "forces"])
    assert np.abs(calc.results["forces"][0]).max() < 1e-5


def test_mace_off():
    base_model = mace_off("small", "float32")
    calc = GraphPESCalculator(base_model)

    # check that forces on central atom are roughly zero
    calc.calculate(CH4, properties=["energy", "forces"])
    assert np.abs(calc.results["forces"][0]).max() < 1e-5


def test_go_mace_23():
    base_model = go_mace_23()
    calc = GraphPESCalculator(base_model)

    calc.calculate(CH4, properties=["energy", "forces"])
    assert np.abs(calc.results["forces"][0]).max() < 1e-5

def test_z_to_onehot_raises_error():
    with pytest.raises(ValueError, match="ZToOneHot received an atomic number"):
        MACEWrapper(MACE_MODEL).z_to_one_hot(torch.tensor([0, 1, 6, 8]))