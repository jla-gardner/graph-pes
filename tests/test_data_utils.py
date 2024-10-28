from __future__ import annotations

import pytest
from graph_pes.utils.misc import random_split


def test_random_split():
    indices = list(range(10))
    split = random_split(indices, lengths=[2, 2], seed=0)
    assert split == [[2, 8], [4, 9]]

    with pytest.raises(ValueError, match="Not enough things to split"):
        random_split(indices, lengths=[20, 20], seed=0)
