from ase.build import molecule
from graph_pes.atomic_graph import AtomicGraph
from graph_pes.models import LennardJones
from graph_pes.utils.analysis import dimer_curve, parity_plot


def test_parity_plot():
    structures = [molecule("H2O"), molecule("CO2")]
    energies = [-14.0, -14.0]
    for s, e in zip(structures, energies):
        s.info["energy"] = e

    graphs = [AtomicGraph.from_ase(s, cutoff=3.0) for s in structures]
    model = LennardJones(cutoff=3.0)

    parity_plot(model, graphs)


def test_dimer_curve():
    dimer_curve(LennardJones(cutoff=3.0), system="SiO")
