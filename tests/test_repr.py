from graph_pes.models.pairwise import LennardJones
from graph_pes.training.loss import Loss
from graph_pes.utils.nn import MLP


def test_reprs():
    assert str(LennardJones()) == "LennardJones(epsilon=0.1, sigma=1.0)"
    assert (
        str(LennardJones(epsilon=2.0)) == "LennardJones(epsilon=2.0, sigma=1.0)"
    )

    assert (
        str(MLP([10, 20, 1], "ReLU")) == "MLP(10 → 20 → 1, activation=ReLU())"
    )

    assert str(Loss("energy")) == 'Loss("energy", metric=MAE())'
