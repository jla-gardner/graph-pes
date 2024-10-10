# at least check a lack of failure of the plotting scripts
from ase.build import molecule
from graph_pes.analysis import dimer_curve, parity_plot
from graph_pes.data.io import to_atomic_graphs
from graph_pes.models import LennardJones


def test_parity_plot():
    structures = [molecule("H2O"), molecule("CO2")]
    energies = [-14.0, -14.0]
    for s, e in zip(structures, energies):
        s.info["energy"] = e

    graphs = to_atomic_graphs(structures, cutoff=3.0)
    model = LennardJones(cutoff=3.0)

    parity_plot(model, graphs)


def test_dimer_curve():
    dimer_curve(LennardJones(cutoff=3.0), system="SiO")
