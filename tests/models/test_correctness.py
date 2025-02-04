import numpy as np
import pytest
from ase import build
from ase.calculators.lj import LennardJones

from graph_pes.models import LennardJones as GraphPESLennardJones
from graph_pes.utils.calculator import GraphPESCalculator


def test_correctness():
    lj_ase = LennardJones(rc=3.0)
    lj_gp = GraphPESCalculator(GraphPESLennardJones.from_ase(rc=3.0))

    # test correct on molecular
    mol = build.molecule("H2O")
    mol.center(vacuum=10)

    assert lj_ase.get_potential_energy(mol) == pytest.approx(
        lj_gp.get_potential_energy(mol), abs=3e-5
    )
    np.testing.assert_allclose(  # type: ignore
        lj_ase.get_forces(mol),  # type: ignore
        lj_gp.get_forces(mol),  # type: ignore
        atol=3e-4,
    )
    np.testing.assert_allclose(  # type: ignore
        lj_ase.get_stress(mol),  # type: ignore
        lj_gp.get_stress(mol),  # type: ignore
        atol=3e-4,
    )

    # test correct on bulk
    bulk = build.bulk("C", "diamond", a=3.5668)
    assert lj_ase.get_potential_energy(bulk) == pytest.approx(
        lj_gp.get_potential_energy(bulk), abs=3e-5
    )
    np.testing.assert_allclose(  # type: ignore
        lj_ase.get_forces(bulk),  # type: ignore
        lj_gp.get_forces(bulk),  # type: ignore
        atol=3e-4,
    )
    np.testing.assert_allclose(  # type: ignore
        lj_ase.get_stress(bulk),  # type: ignore
        lj_gp.get_stress(bulk),  # type: ignore
        atol=3e-4,
    )
