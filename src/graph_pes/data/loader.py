from __future__ import annotations

import warnings
from functools import partial
from typing import Iterator, Sequence

import torch.utils.data

from ..atomic_graph import AtomicGraph, to_batch
from .datasets import GraphDataset


class GraphDataLoader(torch.utils.data.DataLoader):
    r"""
    A helper class for merging :class:`~graph_pes.AtomicGraph` objects
    into a single batch, represented as another :class:`~graph_pes.AtomicGraph`
    containing disjoint subgraphs per structure (see
    :func:`~graph_pes.atomic_graph.to_batch`).

    Parameters
    ----------
    dataset: GraphDataset | Sequence[AtomicGraph]
        The dataset to load.
    batch_size
        The batch size.
    shuffle
        Whether to shuffle the dataset.
    **kwargs:
        Additional keyword arguments to pass to the underlying
        :class:`torch.utils.data.DataLoader`.
    """

    def __init__(
        self,
        dataset: GraphDataset | Sequence[AtomicGraph],
        batch_size: int = 1,
        shuffle: bool = False,
        three_body_cutoff: float | None = None,
        **kwargs,
    ):
        if not isinstance(dataset, GraphDataset):
            dataset = GraphDataset(dataset)

        if "collate_fn" in kwargs:
            warnings.warn(
                "graph-pes uses a custom collate_fn (`collate_atomic_graphs`), "
                "are you sure you want to override this?",
                stacklevel=2,
            )

        collate_fn = kwargs.pop(
            "collate_fn",
            partial(to_batch, three_body_cutoff=three_body_cutoff),
        )

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=collate_fn,
            **kwargs,
        )

    def __iter__(self) -> Iterator[AtomicGraph]:  # type: ignore
        return super().__iter__()
