import pytest
import torch
from graph_pes.models import LearnableOffset, SchNet
from graph_pes.models.addition import AdditionModel
from graph_pes.training.opt import SGD, Adam, LRScheduler, Optimizer


def test_opt():
    # test vanilla use
    model = SchNet()
    opt_factory = Optimizer("Adam", weight_decay=1e-4)
    opt = opt_factory(model)

    # check that opt is an Adam optimizer with a single parameter group
    assert isinstance(opt, torch.optim.Adam)
    assert len(opt.param_groups) == 1

    # test addition model with energy offset parameters
    model = AdditionModel(energy_offset=LearnableOffset(), schnet=SchNet())
    opt = opt_factory(model)
    assert len(opt.param_groups) == 2
    group1, group2 = opt.param_groups
    # first group should be the offset parameters
    assert group1["name"] == "offset" and group1["weight_decay"] == 0.0
    # second group should be the model parameters
    assert group2["name"] == "model" and group2["weight_decay"] == 1e-4

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

    # (edge case) when training just an offset model, the optimizer should
    # again have a single parameter group
    model = LearnableOffset()
    opt = opt_factory(model)
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["weight_decay"] == 0.0


def test_opt_conveniences():
    model = SchNet()
    opt_factory = Adam(weight_decay=1e-4)
    opt = opt_factory(model)
    assert isinstance(opt, torch.optim.Adam)

    opt_factory = SGD(lr=1e-3)
    opt = opt_factory(model)
    assert isinstance(opt, torch.optim.SGD)


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
