from __future__ import annotations

import ase.build
import numpy as np
import pytest
import torch
from e3nn import o3
from mace.calculators import MACECalculator
from mace.modules import ScaleShiftMACE, gate_dict, interaction_classes

from graph_pes.interfaces.mace import MACEWrapper
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
        pair_repulsion=False,
        distance_transform="None",
        radial_MLP=[64, 64, 64],
        radial_type="bessel",
        heads=None,
        cueq_config=None,
    )

    return model


MACE_MODEL = create_default_scaleshift_mace(ELEMENTS, CUTOFF)


def test_molecular():
    ch4 = ase.build.molecule("CH4")

    mace_calc = MACECalculator(models=MACE_MODEL)
    mace_calc.calculate(ch4, properties=["energy", "forces"])

    graph_pes_model = MACEWrapper(MACE_MODEL)
    graph_pes_calc = GraphPESCalculator(graph_pes_model)
    graph_pes_calc.calculate(ch4, properties=["energy", "forces"])

    assert mace_calc.results["energy"] == pytest.approx(
        graph_pes_calc.results["energy"]
    )
    np.testing.assert_allclose(
        mace_calc.results["forces"],
        graph_pes_calc.results["forces"],
        atol=1e-4,
        rtol=100,
    )


def test_periodic():
    diamond = ase.build.bulk("C", "diamond", a=3.5668)

    mace_calc = MACECalculator(models=MACE_MODEL)
    mace_calc.calculate(diamond, properties=["energy", "forces", "stress"])

    graph_pes_model = MACEWrapper(MACE_MODEL)
    graph_pes_calc = GraphPESCalculator(graph_pes_model)
    graph_pes_calc.calculate(diamond, properties=["energy", "forces", "stress"])

    assert mace_calc.results["energy"] == pytest.approx(
        graph_pes_calc.results["energy"], abs=1e-4
    )
    np.testing.assert_allclose(
        mace_calc.results["forces"],
        graph_pes_calc.results["forces"],
        atol=1e-4,
        rtol=100,
    )
    np.testing.assert_allclose(
        mace_calc.results["stress"].flatten(),
        graph_pes_calc.results["stress"].flatten(),
        atol=1e-4,
        rtol=100,
    )
