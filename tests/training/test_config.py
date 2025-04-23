import pytest
import torch
import yaml

from graph_pes.config.shared import (
    instantiate_config_from_dict,
    parse_loss,
    parse_model,
)
from graph_pes.config.training import TrainingConfig
from graph_pes.models import SchNet
from graph_pes.models.addition import AdditionModel
from graph_pes.training.loss import ForceRMSE, PerAtomEnergyLoss, TotalLoss
from graph_pes.utils.misc import nested_merge

from .. import helpers


def get_dummy_config_dict():
    return nested_merge(
        TrainingConfig.defaults(),
        yaml.safe_load((helpers.CONFIGS_DIR / "minimal.yaml").read_text()),
    )


def test_model_instantiation():
    # 1. test single model with no params:
    dummy_data = get_dummy_config_dict()
    dummy_data["model"] = "+SchNet()"
    dummy_data, config = instantiate_config_from_dict(
        dummy_data, TrainingConfig
    )
    model = parse_model(config.model)
    assert isinstance(model, SchNet)

    # 2. test single model with params:
    dummy_data = get_dummy_config_dict()
    dummy_data["model"] = {"+SchNet": {"cutoff": 3.7}}
    dummy_data, config = instantiate_config_from_dict(
        dummy_data, TrainingConfig
    )
    model = parse_model(config.model)
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
    dummy_data, config = instantiate_config_from_dict(
        dummy_data, TrainingConfig
    )
    model = parse_model(config.model)
    assert isinstance(model, AdditionModel)
    assert len(model.models) == 2
    assert isinstance(model.models["many-body"], SchNet)

    # 4. test nice error messages:
    dummy_data = get_dummy_config_dict()
    # clearly incorrect
    dummy_data["model"] = 3
    with pytest.raises(ValueError, match="Failed to instantiate a config"):
        instantiate_config_from_dict(dummy_data, TrainingConfig)

    # not a GraphPESModel
    dummy_data["model"] = "+torch.nn.ReLU()"
    with pytest.raises(ValueError):
        instantiate_config_from_dict(dummy_data, TrainingConfig)

    # incorrect AdditionModel spec
    dummy_data["model"] = {"a": "+torch.nn.ReLU()"}
    with pytest.raises(ValueError):
        instantiate_config_from_dict(dummy_data, TrainingConfig)


def test_optimizer():
    dummy_data = get_dummy_config_dict()
    user_data = yaml.safe_load("""
    fitting:
        optimizer:
            name: Adam
            lr: 0.001
    """)
    actual_data = nested_merge(dummy_data, user_data)
    _, config = instantiate_config_from_dict(actual_data, TrainingConfig)

    dummy_model = SchNet()
    optimizer_instance = config.fitting.get_optimizer()(dummy_model)
    assert optimizer_instance.param_groups[0]["lr"] == 0.001


def test_scheduler():
    # no default scheduler
    dummy_data = get_dummy_config_dict()
    _, config = instantiate_config_from_dict(dummy_data, TrainingConfig)
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
    _, config = instantiate_config_from_dict(actual_data, TrainingConfig)

    scheduler = config.fitting.scheduler
    assert scheduler is not None
    dummy_optimizer = torch.optim.Adam(SchNet().parameters())
    scheduler_instance = scheduler(dummy_optimizer)
    assert isinstance(scheduler_instance, torch.optim.lr_scheduler.StepLR)
    assert scheduler_instance.step_size == 10
    assert scheduler_instance.gamma == 0.1


def test_extra_keys():
    dummy_data = get_dummy_config_dict()
    dummy_data["extra_key"] = "extra_value"
    final_data, config = instantiate_config_from_dict(
        dummy_data, TrainingConfig
    )
    assert "extra_key" in final_data


def test_parse_loss():
    # 1. a single loss should be wrapped in a TotalLoss
    loss = PerAtomEnergyLoss()
    total_loss = parse_loss(loss)
    assert isinstance(total_loss, TotalLoss)
    assert len(total_loss.losses) == 1
    assert total_loss.losses[0] is loss

    # 2. a list of losses should be wrapped in a TotalLoss
    loss = [PerAtomEnergyLoss(), ForceRMSE()]
    total_loss = parse_loss(loss)
    assert isinstance(total_loss, TotalLoss)
    assert len(total_loss.losses) == 2

    # 3. as should a dictionary of losses
    loss = {"energy": PerAtomEnergyLoss(), "forces": ForceRMSE()}
    total_loss = parse_loss(loss)
    assert isinstance(total_loss, TotalLoss)
    assert len(total_loss.losses) == 2
    assert total_loss.losses[0] is loss["energy"]
    assert total_loss.losses[1] is loss["forces"]

    # 4. a TotalLoss should be returned as is
    loss = TotalLoss([PerAtomEnergyLoss()])
    assert parse_loss(loss) is loss

    # 5. a non-Loss should raise an error
    with pytest.raises(ValueError):
        parse_loss(1)  # type: ignore


def test_parse_model():
    model = SchNet()
    parsed = parse_model(model)
    assert parsed is model

    parsed = parse_model({"schnet": model})
    assert isinstance(parsed, AdditionModel)
    assert len(parsed.models) == 1
    assert parsed.models["schnet"] is model

    with pytest.raises(ValueError):
        parse_model(1)  # type: ignore

    with pytest.raises(ValueError):
        parse_model({"schnet": 1})  # type: ignore
