from __future__ import annotations

import pytest
import torch
from ase import Atoms

from graph_pes.atomic_graph import (
    AtomicGraph,
    number_of_atoms,
    number_of_edges,
    number_of_neighbours,
    to_batch,
)
from graph_pes.models.components.aggregation import (
    MeanNeighbours,
    NeighbourAggregation,
    ScaledSumNeighbours,
    SumNeighbours,
    VariancePreservingSumNeighbours,
)

# Test structures
ISOLATED_ATOM = Atoms("H", positions=[(0, 0, 0)], pbc=False)
DIMER = Atoms("H2", positions=[(0, 0, 0), (0, 0, 1)], pbc=False)
TRIMER = Atoms("H3", positions=[(0, 0, 0), (0, 0, 1), (0, 1, 0)], pbc=False)


@pytest.fixture
def graphs():
    return [
        AtomicGraph.from_ase(ISOLATED_ATOM, cutoff=1.5),
        AtomicGraph.from_ase(DIMER, cutoff=1.5),
        AtomicGraph.from_ase(TRIMER, cutoff=1.5),
    ]


@pytest.mark.parametrize(
    "aggregation_class",
    [
        SumNeighbours,
        MeanNeighbours,
        ScaledSumNeighbours,
        VariancePreservingSumNeighbours,
    ],
)
def test_aggregation_shape(aggregation_class, graphs):
    aggregation = aggregation_class()

    for graph in graphs:
        N = number_of_atoms(graph)
        E = number_of_edges(graph)

        for shape in [(E,), (E, 2), (E, 2, 3)]:
            edge_property = torch.rand(shape)
            result = aggregation(edge_property, graph)
            assert result.shape == (N, *shape[1:])


def test_sum_neighbours(graphs):
    sum_agg = SumNeighbours()

    for graph in graphs:
        E = number_of_edges(graph)
        edge_property = torch.ones(E)
        result = sum_agg(edge_property, graph)
        assert torch.allclose(
            result,
            number_of_neighbours(graph, include_central_atom=False).float(),
        )


def test_mean_neighbours(graphs):
    mean_agg = MeanNeighbours()

    for graph in graphs:
        E = number_of_edges(graph)
        edge_property = torch.ones(E)
        result = mean_agg(edge_property, graph)
        expected = number_of_neighbours(
            graph, include_central_atom=False
        ) / number_of_neighbours(graph, include_central_atom=True)
        assert torch.allclose(result, expected)


@pytest.mark.parametrize("learnable", [True, False])
def test_scaled_sum_neighbours(graphs, learnable):
    scaled_sum_agg = ScaledSumNeighbours(learnable=learnable)

    # Test pre_fit
    batch = to_batch(graphs)
    scaled_sum_agg.pre_fit(batch)
    avg_neighbours = number_of_edges(batch) / number_of_atoms(batch)
    assert torch.isclose(scaled_sum_agg.scale, torch.tensor(avg_neighbours))

    # Test forward
    for graph in graphs:
        E = number_of_edges(graph)
        edge_property = torch.ones(E)
        result = scaled_sum_agg(edge_property, graph)
        expected = (
            number_of_neighbours(graph, include_central_atom=False).to(
                torch.float
            )
            / scaled_sum_agg.scale
        )
        assert torch.allclose(result, expected)


def test_variance_preserving_sum_neighbours(graphs):
    var_pres_sum_agg = VariancePreservingSumNeighbours()

    for graph in graphs:
        E = number_of_edges(graph)
        edge_property = torch.ones(E)
        result = var_pres_sum_agg(edge_property, graph)
        expected = number_of_neighbours(
            graph, include_central_atom=False
        ) / torch.sqrt(number_of_neighbours(graph, include_central_atom=True))
        assert torch.allclose(result, expected)


def test_parse_aggregation():
    assert isinstance(NeighbourAggregation.parse("sum"), SumNeighbours)
    assert isinstance(NeighbourAggregation.parse("mean"), MeanNeighbours)
    assert isinstance(
        NeighbourAggregation.parse("constant_fixed"), ScaledSumNeighbours
    )
    assert isinstance(
        NeighbourAggregation.parse("constant_learnable"), ScaledSumNeighbours
    )
    assert isinstance(
        NeighbourAggregation.parse("sqrt"), VariancePreservingSumNeighbours
    )

    with pytest.raises(ValueError):
        NeighbourAggregation.parse("invalid_mode")  # type: ignore


@pytest.mark.parametrize(
    "aggregation_class",
    [
        SumNeighbours,
        MeanNeighbours,
        ScaledSumNeighbours,
        VariancePreservingSumNeighbours,
    ],
)
def test_torchscript_compatibility(aggregation_class):
    aggregation = aggregation_class()
    scripted = torch.jit.script(aggregation)
    assert isinstance(scripted, torch.jit.ScriptModule)
