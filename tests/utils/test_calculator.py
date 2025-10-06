import numpy
import pytest
from ase.build import bulk, molecule

from graph_pes.atomic_graph import AtomicGraph, PropertyKey
from graph_pes.models import LennardJones
from graph_pes.utils.calculator import merge_predictions


def test_calc():
    model = LennardJones()
    calc = model.ase_calculator(skin=0.0)
    ethanol = molecule("CH3CH2OH")
    ethanol.calc = calc

    # ensure right shapes
    assert isinstance(ethanol.get_potential_energy(), float)
    assert ethanol.get_forces().shape == (9, 3)

    # ensure correctness
    g = AtomicGraph.from_ase(ethanol, model.cutoff.item())
    assert ethanol.get_potential_energy() == pytest.approx(
        model.predict_energy(g).item()
    )
    numpy.testing.assert_allclose(
        ethanol.get_forces(), model.predict_forces(g).numpy()
    )

    copper = bulk("Cu")
    copper.calc = calc
    assert isinstance(copper.get_potential_energy(), float)
    assert copper.get_forces().shape == (1, 3)
    assert copper.get_stress().shape == (6,)

    g = AtomicGraph.from_ase(copper, model.cutoff.item())
    assert copper.get_potential_energy() == pytest.approx(
        model.predict_energy(g).item()
    )
    numpy.testing.assert_allclose(
        copper.get_forces(), model.predict_forces(g).numpy()
    )


def test_calc_all():
    calc = LennardJones().ase_calculator()
    molecules = [molecule(s) for s in ["CH4", "H2O", "CH3CH2OH", "C2H6"]]

    # add cell info so we can test stresses
    for m in molecules:
        m.center(vacuum=10)

    properties: list[PropertyKey] = ["energy", "forces", "stress"]
    one_by_one = []
    for m in molecules:
        calc.calculate(m, properties)
        one_by_one.append(calc.results)

    batched = calc.calculate_all(molecules, properties)

    assert len(one_by_one) == len(batched) == len(molecules)

    for single, parallel in zip(one_by_one, batched):
        for key in properties:
            numpy.testing.assert_allclose(
                single[key], parallel[key], rtol=10, atol=1e-8
            )

    merged = merge_predictions(batched)
    for i in range(len(molecules)):
        for key in "energy", "stress":
            numpy.testing.assert_allclose(merged[key][i], batched[i][key])

    n_atoms = sum(map(len, molecules))
    assert merged["forces"].shape == (n_atoms, 3)
