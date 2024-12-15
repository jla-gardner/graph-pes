import pytest

from graph_pes.utils.sampling import SequenceSampler


@pytest.fixture
def sample_list():
    return list(range(10))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_sequence_sampler_init(sample_list):
    # Test initialization with default indices
    sampler = SequenceSampler(sample_list)
    assert len(sampler) == len(sample_list)

    # Test initialization with specific indices
    indices = [0, 2, 4]
    sampler = SequenceSampler(sample_list, indices)
    assert len(sampler) == len(indices)


def test_sequence_sampler_getitem(sample_list):
    sampler = SequenceSampler(sample_list)

    # Test integer indexing
    assert sampler[0] == 0
    assert sampler[-1] == 9

    # Test slice indexing
    sliced = sampler[2:5]
    assert isinstance(sliced, SequenceSampler)
    assert list(sliced) == [2, 3, 4]


def test_sequence_sampler_len(sample_list):
    sampler = SequenceSampler(sample_list)
    assert len(sampler) == len(sample_list)

    sampler = SequenceSampler(sample_list, [0, 1, 2])
    assert len(sampler) == 3


def test_sequence_sampler_iter(sample_list):
    sampler = SequenceSampler(sample_list)
    assert list(sampler) == sample_list


def test_sequence_sampler_shuffled(sample_list):
    sampler = SequenceSampler(sample_list)
    shuffled = sampler.shuffled(seed=42)

    # Test that shuffled returns a SequenceSampler
    assert isinstance(shuffled, SequenceSampler)

    # Test that shuffled has same length
    assert len(shuffled) == len(sampler)

    # Test that shuffled contains all original elements
    assert sorted(list(shuffled)) == sorted(sample_list)

    # Test that shuffling is deterministic with same seed
    shuffled2 = sampler.shuffled(seed=42)
    assert list(shuffled) == list(shuffled2)

    # Test that different seeds give different orders
    shuffled3 = sampler.shuffled(seed=43)
    assert list(shuffled) != list(shuffled3)


def test_sequence_sampler_sample_at_most(sample_list):
    sampler = SequenceSampler(sample_list)

    # Test sampling less than total length
    sample = sampler.sample_at_most(5, seed=42)
    assert len(sample) == 5
    assert isinstance(sample, SequenceSampler)

    # Test sampling more than total length
    sample = sampler.sample_at_most(15, seed=42)
    assert len(sample) == len(sample_list)


def test_sequence_sampler_with_custom_sequence():
    class CustomSequence:
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    custom_seq = CustomSequence(["a", "b", "c"])
    sampler = SequenceSampler(custom_seq)  # type: ignore

    assert len(sampler) == 3
    assert list(sampler) == ["a", "b", "c"]


def test_sequence_sampler_error_handling():
    with pytest.raises(TypeError):
        # Test with non-sequence
        SequenceSampler(42)  # type: ignore

    sampler = SequenceSampler([1, 2, 3])
    with pytest.raises(IndexError):
        # Test index out of range
        sampler[10]
