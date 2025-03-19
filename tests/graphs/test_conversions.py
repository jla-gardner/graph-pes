import ase
import numpy as np
import pytest
import torch

from graph_pes import AtomicGraph


def test_there_and_back_again():
    for pbc in [True, False]:
        atoms = ase.Atoms(
            "H2",
            positions=[(0, 0, 0), (0, 0, 1)],
            pbc=pbc,
            cell=(4, 4, 4),
        )
        atoms.info["energy"] = 1.0
        atoms.arrays["forces"] = np.random.rand(2, 3)

        graph = AtomicGraph.from_ase(atoms)
        print(graph)

        atoms_back = graph.to_ase()
        print(atoms_back.info)

        assert atoms_back.info["energy"] == atoms.info["energy"]
        assert np.allclose(atoms_back.arrays["forces"], atoms.arrays["forces"])
        assert np.all(atoms_back.pbc == pbc)


def test_no_cell():
    atoms = ase.Atoms("H2", positions=[(0, 0, 0), (0, 0, 1)], pbc=True)
    with pytest.raises(ValueError, match="but cell is all zeros"):
        AtomicGraph.from_ase(atoms)

    atoms.cell = (4, 4, 4)
    graph = AtomicGraph.from_ase(atoms)
    assert (graph.cell == torch.eye(3) * 4).all()
