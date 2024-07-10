from __future__ import annotations

from typing import Sequence, TypeVar

import numpy as np

E = TypeVar("E")


def random_split(
    sequence: Sequence[E],
    lengths: Sequence[int],
    seed: int | None = None,
) -> list[list[E]]:
    """
    Randomly split `sequence` into sub-sequences according to `lengths`.

    Parameters
    ----------
    sequence: Sequence[E]
        The sequence to split.
    lengths: Sequence[int]
        The lengths of the sub-sequences to create.
    seed: int | None
        The random seed to use. If `None`, the current random state is
        used (non-deterministic).

    Returns
    -------
    list[list[E]]
        A list of sub-sequences.

    Examples
    --------
    >>> random_split("abcde", [2, 3])
    [['b', 'c'], ['a', 'd', 'e']]
    """

    if sum(lengths) > len(sequence):
        raise ValueError("Not enough things to split")

    shuffle = np.random.RandomState(seed=seed).permutation(len(sequence))
    ptr = [0, *np.cumsum(lengths)]

    return [
        [sequence[i] for i in shuffle[ptr[n] : ptr[n + 1]]]
        for n in range(len(lengths))
    ]
