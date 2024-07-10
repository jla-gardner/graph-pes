from ase.build import molecule
from graph_pes.calculator import GraphPESCalculator
from graph_pes.models import LennardJones


def test_calc():
    calc = GraphPESCalculator(LennardJones(), cutoff=5)
    ethanol = molecule("CH3CH2OH")
    ethanol.calc = calc

    assert ethanol.get_potential_energy().shape == ()
    assert ethanol.get_forces().shape == (9, 3)
