import torch
from ase.build import molecule

from graph_pes import AtomicGraph
from graph_pes.models import LennardJones


def test_dtypes():
    torch.set_default_dtype(torch.float64)
    g = AtomicGraph.from_ase(molecule("C6H6"))
    assert g.R.dtype == torch.float64

    model = LennardJones()
    assert model._log_epsilon.dtype == torch.float64

    model(g)

    torch.set_default_dtype(torch.float32)
    g = AtomicGraph.from_ase(molecule("C6H6"))
    assert g.R.dtype == torch.float32

    model = LennardJones()
    assert model._log_epsilon.dtype == torch.float32

    model(g)
