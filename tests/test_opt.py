import pytest
import torch
from graph_pes.models import LearnableOffset, LennardJones, SchNet
from graph_pes.models.addition import AdditionModel
from graph_pes.training.opt import LRScheduler, Optimizer


def test_non_decayable_params():
    opt_factory = Optimizer("Adam", weight_decay=1e-4)

    # learnable offsets have a single paramter, _offsets,
    # that is not decayable
    model = LearnableOffset()
    assert (
        set(model.non_decayable_parameters())
        == set([model._offsets])
        == set(model.parameters())
    )

    opt = opt_factory(model)
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["weight_decay"] == 0.0

    # LJ models have no non-decayable parameters
    model = LennardJones()
    assert len(model.non_decayable_parameters()) == 0

    opt = opt_factory(model)
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["weight_decay"] == 1e-4

    # schnet has many parameters, but only per_element_scaling is
    # not decayable
    model = SchNet()
    assert model.local_energies_scaler is not None
    assert set(model.non_decayable_parameters()) == set(
        [model.local_energies_scaler.per_element_scaling]
    )
    opt = opt_factory(model)
    assert len(opt.param_groups) == 2
    pg_by_name = {pg["name"]: pg for pg in opt.param_groups}
    assert pg_by_name["non-decayable"]["weight_decay"] == 0.0
    assert pg_by_name["normal"]["weight_decay"] == 1e-4

    # addition models should return the decayable parameters of their
    # components
    model = AdditionModel(energy_offset=LearnableOffset(), schnet=SchNet())
    assert model["schnet"].local_energies_scaler is not None
    assert set(model.non_decayable_parameters()) == set(
        [
            model["energy_offset"]._offsets,
            model["schnet"].local_energies_scaler.per_element_scaling,
        ]
    )


def test_opt():
    # test vanilla use
    model = SchNet()
    opt_factory = Optimizer("Adam", weight_decay=1e-4)
    opt = opt_factory(model)

    # check that opt is an Adam optimizer with two parameter groups
    assert isinstance(opt, torch.optim.Adam)

    # test error if optimizer class is not found
    with pytest.raises(ValueError, match="Could not find optimizer"):
        Optimizer("Unknown")

    # test error if optimizer class is not an optimizer
    with pytest.raises(
        ValueError,
        match="Expected the returned optimizer to be an instance of ",
    ):

        def fake_opt(*args, **kwargs):
            return None

        Optimizer(fake_opt)  # type: ignore

    # test custom optimizer class
    class CustomOptimizer(torch.optim.Adam):
        pass

    opt_factory = Optimizer(CustomOptimizer)
    opt = opt_factory(model)
    assert isinstance(opt, CustomOptimizer)


def test_lr_scheduler():
    params = [torch.nn.Parameter(torch.zeros(1))]
    sched_factory = LRScheduler("StepLR", step_size=10, gamma=0.1)
    sched = sched_factory(torch.optim.Adam(params))
    assert isinstance(sched, torch.optim.lr_scheduler.StepLR)

    # can't find scheduler
    with pytest.raises(ValueError, match="Could not find scheduler"):
        LRScheduler("Unknown")

    # custom scheduler
    class CustomScheduler(torch.optim.lr_scheduler.StepLR):
        pass

    sched_factory = LRScheduler(CustomScheduler, step_size=10, gamma=0.1)
    sched = sched_factory(torch.optim.Adam(params))
    assert isinstance(sched, CustomScheduler)
