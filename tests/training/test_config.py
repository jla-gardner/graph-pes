import pytest
import torch
import yaml
from graph_pes.config import Config, get_default_config_values
from graph_pes.models import SchNet
from graph_pes.models.addition import AdditionModel
from graph_pes.utils.misc import nested_merge

from .. import helpers


def get_dummy_config_dict():
    return nested_merge(
        get_default_config_values(),
        yaml.safe_load((helpers.CONFIGS_DIR / "minimal.yaml").read_text()),
    )


def test_model_instantiation():
    # 1. test single model with no params:
    dummy_data = get_dummy_config_dict()
    dummy_data["model"] = "+SchNet()"
    dummy_data = Config.from_dict(dummy_data)
    model = dummy_data.get_model()
    assert isinstance(model, SchNet)

    # 2. test single model with params:
    dummy_data = get_dummy_config_dict()
    dummy_data["model"] = {"+SchNet": {"cutoff": 3.7}}
    dummy_data = Config.from_dict(dummy_data)
    model = dummy_data.get_model()
    assert isinstance(model, SchNet)
    assert model.cutoff == 3.7

    # 3. test AdditionModel creation
    dummy_data = get_dummy_config_dict()
    dummy_data["model"] = yaml.safe_load(
        r"""
offset: +LearnableOffset()
many-body:
   +SchNet: {cutoff: 3.7}
"""
    )
    dummy_data = Config.from_dict(dummy_data)
    model = dummy_data.get_model()
    assert isinstance(model, AdditionModel)
    assert len(model.models) == 2
    assert isinstance(model.models["many-body"], SchNet)

    # 4. test nice error messages:
    dummy_data = get_dummy_config_dict()
    # clearly incorrect
    dummy_data["model"] = 3
    with pytest.raises(ValueError, match="could not be successfully parsed."):
        Config.from_dict(dummy_data).get_model()

    # not a GraphPESModel
    dummy_data["model"] = "+torch.nn.ReLU()"
    with pytest.raises(ValueError):
        Config.from_dict(dummy_data).get_model()

    # incorrect AdditionModel spec
    dummy_data["model"] = {"a": "+torch.nn.ReLU()"}
    with pytest.raises(ValueError):
        Config.from_dict(dummy_data).get_model()


def test_optimizer():
    dummy_data = get_dummy_config_dict()
    user_data = yaml.safe_load("""
    fitting:
        optimizer:
            +Optimizer:
                name: Adam
                lr: 0.001
    """)
    actual_data = nested_merge(dummy_data, user_data)
    config = Config.from_dict(actual_data)

    dummy_model = SchNet()
    optimizer_instance = config.fitting.optimizer(dummy_model)
    assert optimizer_instance.param_groups[0]["lr"] == 0.001


def test_scheduler():
    # no default scheduler
    dummy_data = get_dummy_config_dict()
    config = Config.from_dict(dummy_data)
    scheduler = config.fitting.scheduler
    assert scheduler is None

    # with default scheduler
    user_data = yaml.safe_load("""
    fitting:
        scheduler:
            +LRScheduler:
                name: StepLR
                step_size: 10
                gamma: 0.1
    """)
    dummy_data = get_dummy_config_dict()
    actual_data = nested_merge(dummy_data, user_data)
    config = Config.from_dict(actual_data)

    scheduler = config.fitting.scheduler
    assert scheduler is not None
    dummy_optimizer = torch.optim.Adam(SchNet().parameters())
    scheduler_instance = scheduler(dummy_optimizer)
    assert isinstance(scheduler_instance, torch.optim.lr_scheduler.StepLR)
    assert scheduler_instance.step_size == 10
    assert scheduler_instance.gamma == 0.1
