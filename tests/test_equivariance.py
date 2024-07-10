import pytest
from graph_pes.models import LennardJones, SchNet

models = [
    LennardJones(),
    SchNet(),
]


@pytest.mark.parametrize("model", models)
def test_equivariance(model):
    ...
    # TODO
