from __future__ import annotations

from typing import Iterator, Sequence, TypeVar, overload

import numpy as np

T = TypeVar("T")


class SequenceSampler(Sequence[T]):
    """
    A class that wraps a :class:`Sequence` of ``T`` objects and
    provides methods for sampling from it without the need to manipulate or
    access the underlying data.

    This is useful for e.g. sub-sampling a ``collection`` where individual
    indexing operations are expensive, such as a database on disk, or when
    indexing involves some form of pre-processing.

    Parameters
    ----------
    collection
        The collection to wrap.
    indices
        The indices of the elements to include in the collection. If ``None``,
        all elements are included.
    """

    def __init__(
        self,
        collection: Sequence[T],
        indices: Sequence[int] | None = None,
    ):
        self.collection = collection
        self.indices = indices or range(len(collection))

    @overload
    def __getitem__(self, index: int) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> SequenceSampler[T]: ...
    def __getitem__(self, index: int | slice) -> T | SequenceSampler[T]:
        """
        Get the element/s at the given ``index``.

        Parameters
        ----------
        index
            The index/indices of the element/elements to get.
        """
        if isinstance(index, int):
            return self.collection[self.indices[index]]

        sampled_indices = self.indices[index]
        return SequenceSampler(self.collection, sampled_indices)

    def __len__(self) -> int:
        """The number of items in the collection."""
        return len(self.indices)

    def __iter__(self) -> Iterator[T]:
        """Iterate over the collection."""
        for i in range(len(self)):
            yield self[i]

    def shuffled(self, seed: int = 42) -> SequenceSampler[T]:
        """
        Return a shuffled version of this collection.

        Parameters
        ----------
        seed
            The random seed to use for shuffling.
        """
        # 1. make a copy of the indices
        indices = [*self.indices]
        # 2. shuffle them
        np.random.default_rng(seed).shuffle(indices)
        # 3. return a new OrderedCollection with the shuffled indices
        return SequenceSampler(self.collection, indices)

    def sample_at_most(self, n: int, seed: int = 42) -> SequenceSampler[T]:
        """
        Return a sampled collection with at most ``n`` elements.
        """
        assert n >= 0, "n must be non-negative"
        n = min(n, len(self))
        return self.shuffled(seed)[:n]
