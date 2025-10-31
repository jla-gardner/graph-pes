from graph_pes.models import (
    LennardJones,
    freeze,
    freeze_all_except,
    freeze_any_matching,
    freeze_matching,
)
from graph_pes.utils.nn import count_used_parameters


def test_freeze():
    model = LennardJones()
    model = freeze(model)

    assert count_used_parameters(model, only_learnable=True) == 0


def test_freeze_matching():
    model = LennardJones()
    model = freeze_matching(model, ".*")

    assert count_used_parameters(model, only_learnable=True) == 0

    model = LennardJones()
    model = freeze_matching(model, "_log_epsilon")
    assert count_used_parameters(model, only_learnable=True) == 1
    assert not model._log_epsilon.requires_grad


def test_freeze_all_except():
    model = LennardJones()
    model = freeze_all_except(model, "_log_epsilon")

    assert count_used_parameters(model, only_learnable=True) == 1
    assert model._log_epsilon.requires_grad


def test_freeze_any_matching():
    model = LennardJones()
    model = freeze_any_matching(model, ["_log_epsilon", "_log_sigma"])

    assert count_used_parameters(model, only_learnable=True) == 0
    assert not model._log_epsilon.requires_grad
    assert not model._log_sigma.requires_grad
