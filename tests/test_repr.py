from graph_pes.models.pairwise import LennardJones, Morse
from graph_pes.nn import MLP
from graph_pes.training.loss import Loss


def test_reprs():
    assert str(LennardJones()) == "LennardJones(epsilon=0.1, sigma=1.0)"
    assert (
        str(LennardJones(epsilon=2.0)) == "LennardJones(epsilon=2.0, sigma=1.0)"
    )

    assert (
        str(MLP([10, 20, 1], "ReLU")) == "MLP(10 → 20 → 1, activation=ReLU())"
    )

    assert str(Loss("energy")) == 'Loss("energy", metric=MAE())'

    addition = LennardJones() + Morse()
    assert (
        str(addition)
        == """\
AdditionModel(
  LennardJones(epsilon=0.1, sigma=1.0),
  Morse(D=0.1, a=5.0, r0=1.5)
)"""
    )
