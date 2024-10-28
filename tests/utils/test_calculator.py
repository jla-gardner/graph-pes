from ase.build import bulk, molecule
from graph_pes.models import LennardJones
from graph_pes.utils.calculator import GraphPESCalculator


def test_calc():
    calc = GraphPESCalculator(LennardJones(), cutoff=5)
    ethanol = molecule("CH3CH2OH")
    ethanol.calc = calc

    assert isinstance(ethanol.get_potential_energy(), float)
    assert ethanol.get_forces().shape == (9, 3)

    copper = bulk("Cu")
    copper.calc = calc
    assert isinstance(copper.get_potential_energy(), float)
    assert copper.get_forces().shape == (1, 3)
    assert copper.get_stress().shape == (6,)
