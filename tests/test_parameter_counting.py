from __future__ import annotations

import ase
import torch
from graph_pes.data.io import to_atomic_graph
from graph_pes.graphs.graph_typing import AtomicGraph
from graph_pes.models import (
    FixedOffset,
    LearnableOffset,
    LennardJones,
    LennardJonesMixture,
    SchNet,
)
from graph_pes.models.addition import AdditionModel
from graph_pes.nn import PerElementParameter
from helpers import DoesNothingModel

graphs: list[AtomicGraph] = []
for n in range(5):
    g = to_atomic_graph(ase.Atoms(f"CH{n}"), cutoff=5.0)
    g["energy"] = torch.tensor(n).float()
    graphs.append(g)


# Before a model has been pre_fit, all PerElementParameters should have 0
# relevant and countable values. After pre_fitting, the PerElementParameter
# values corresponding to elements seen in the pre-fit data should be counted.


def test_fixed():
    model = FixedOffset(H=1.0, C=2.0)

    # model should have a single parameter
    params = list(model.parameters())
    assert len(params) == 1

    # there should be 2 values, since we passed 2 offset energies
    assert params[0].numel() == 2


def test_scaling():
    model = DoesNothingModel(cutoff=0, auto_scale=True)
    # the model should have a single parameter: the per_element_scaling
    params = list(model.parameters())
    assert len(params) == 1
    assert params[0] is model.per_element_scaling

    # there should be no countable values in this parameter
    assert params[0].numel() == 0

    model.pre_fit(graphs)  # type: ignore
    # now the model has seen info about 2 elements:
    # there should be 2 countable elements on the model
    assert params[0].numel() == 2


def test_counting():
    _schnet_dim = 50
    model = AdditionModel(
        offset=LearnableOffset(),
        schnet=SchNet(node_features=_schnet_dim),
    )

    non_pre_fit_params = sum(p.numel() for p in model.parameters())

    # 3 per-element parameters:
    #   1. the offsets in LearnableOffset
    #   2. the per-element scaling in SchNet
    #   3. the chemical embedding of SchNet
    assert (
        sum(1 for p in model.parameters() if isinstance(p, PerElementParameter))
        == 3
    )

    model.pre_fit(graphs)  # type: ignore

    post_fit_params = sum(p.numel() for p in model.parameters())

    # seen 2 elements (C and H), leading to a total of:
    #   1. 2 countable elements in the offsets
    #   2. 2 countable elements in the per-element scaling
    #   3. 2*schnet emdedding dim countable elements in the chemical embedding
    assert post_fit_params == non_pre_fit_params + 2 + 2 + 2 * _schnet_dim


def test_lj():
    lj = LennardJones()
    # lj has two parameters: epsilon and sigma
    assert sum(p.numel() for p in lj.parameters()) == 2


def test_lj_mixture():
    lj_mixture = LennardJonesMixture()
    lj_mixture.pre_fit(graphs)  # type: ignore

    expected_params = 0
    # sigma and epsilon for each element
    expected_params += 2 * 2
    # nu and zeta term for each ordered pair of elements
    expected_params += 2 * 2 * 2

    assert sum(p.numel() for p in lj_mixture.parameters()) == expected_params
