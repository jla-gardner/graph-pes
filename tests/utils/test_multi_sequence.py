import pytest

from graph_pes.utils.misc import MultiSequence


def test_multi_sequence():
    a = [1, 2, 3, 4, 5]
    b = [6, 7, 8, 9, 10]

    ms = MultiSequence([a, b])
    assert len(ms) == 10
    assert ms[0] == 1
    assert ms[5] == 6

    # test slicing
    sliced = ms[2:5]
    assert isinstance(sliced, MultiSequence)
    assert len(sliced) == 3
    assert list(sliced) == [3, 4, 5]
    # nested slicing
    assert isinstance(sliced[:2], MultiSequence)
    assert list(sliced[:2]) == [3, 4]

    # test slicing across the boundary of two sequences
    sliced = ms[3:7]
    assert len(sliced) == 4
    assert list(sliced) == [4, 5, 6, 7]

    # test slicing with a step
    sliced = ms[::2]
    assert len(sliced) == 5
    assert list(sliced) == [1, 3, 5, 7, 9]

    # test negative slicing
    sliced = ms[-3:]
    assert len(sliced) == 3
    assert list(sliced) == [8, 9, 10]

    sliced = ms[::-1]
    assert len(sliced) == 10
    assert list(sliced) == [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

    with pytest.raises(IndexError):
        ms[100]
