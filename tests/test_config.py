import torch
import yaml
from graph_pes.config import Config, get_default_config_values
from graph_pes.config.utils import _import, create_from_dict, create_from_string
from graph_pes.core import AdditionModel
from graph_pes.models import SchNet
from graph_pes.training.loss import RMSE, Loss
from graph_pes.util import nested_merge


def test_import():
    # test that we can import:
    obj = _import("graph_pes.models.SchNet")
    assert obj is SchNet


def test_spec_instantiating():
    # 1. test that we get the object:
    schnet_class = create_from_string("graph_pes.models.SchNet")
    assert schnet_class is SchNet

    # 2. test that we get the object returned from a function:
    schnet_obj = create_from_string("graph_pes.models.SchNet()")
    assert isinstance(schnet_obj, SchNet)

    # 3. test that we get the returned object from a function with params:
    spec = """
    graph_pes.models.SchNet:
        cutoff: 3.7
    """
    schnet_obj = create_from_dict(yaml.safe_load(spec))
    assert isinstance(schnet_obj, SchNet)
    assert schnet_obj.cutoff == 3.7

    # 4. test nested instantiation:
    spec = """
    graph_pes.training.loss.Loss:
        property_key: energy
        metric: graph_pes.training.loss.RMSE()
    """
    loss_obj = create_from_dict(yaml.safe_load(spec))
    print(loss_obj)
    assert isinstance(loss_obj, Loss)
    assert loss_obj.property_key == "energy"
    assert isinstance(loss_obj.metric, RMSE)


def get_dummy_config_dict():
    config = get_default_config_values()
    config["data"] = "dummy"
    config["model"] = "dummy"
    config["loss"] = "dummy"
    return config


def test_from_dict():
    data = get_dummy_config_dict()
    config = Config.from_dict(data)
    assert config.data == "dummy"


def test_model_instantiation():
    # 1. test single model with no params:
    dummy_data = get_dummy_config_dict()
    dummy_data["model"] = "graph_pes.models.SchNet()"
    dummy_data = Config.from_dict(dummy_data)
    model = dummy_data.instantiate_model()
    assert isinstance(model, SchNet)

    # 2. test single model with params:
    dummy_data = get_dummy_config_dict()
    dummy_data["model"] = {"graph_pes.models.SchNet": {"cutoff": 3.7}}
    dummy_data = Config.from_dict(dummy_data)
    model = dummy_data.instantiate_model()
    assert isinstance(model, SchNet)
    assert model.cutoff == 3.7

    # 3. test a list of models:
    dummy_data = get_dummy_config_dict()
    dummy_data["model"] = yaml.safe_load(
        """
offset: graph_pes.models.LearnableOffset()
many-body: 
   graph_pes.models.SchNet:
       cutoff: 3.7
"""
    )
    dummy_data = Config.from_dict(dummy_data)
    model = dummy_data.instantiate_model()
    assert isinstance(model, AdditionModel)
    assert len(model.models) == 2
    assert isinstance(model.models["many-body"], SchNet)


def test_optimizer():
    dummy_data = get_dummy_config_dict()
    user_data = yaml.safe_load("""
    fitting:
        optimizer:
            graph_pes.training.opt.Optimizer:
                name: Adam
                lr: 0.001
    """)
    actual_data = nested_merge(dummy_data, user_data)
    config = Config.from_dict(actual_data)

    optimizer = config.fitting.instantiate_optimizer()
    dummy_model = SchNet()
    optimizer_instance = optimizer(dummy_model)
    assert optimizer_instance.param_groups[0]["lr"] == 0.001


def test_scheduler():
    # no default scheduler
    dummy_data = get_dummy_config_dict()
    config = Config.from_dict(dummy_data)
    scheduler = config.fitting.instantiate_scheduler()
    assert scheduler is None

    # with default scheduler
    user_data = yaml.safe_load("""
    fitting:
        scheduler:
            graph_pes.training.opt.LRScheduler:
                name: StepLR
                step_size: 10
                gamma: 0.1
    """)
    dummy_data = get_dummy_config_dict()
    print(user_data, dummy_data)
    actual_data = nested_merge(dummy_data, user_data)
    config = Config.from_dict(actual_data)

    scheduler = config.fitting.instantiate_scheduler()
    assert scheduler is not None
    dummy_optimizer = torch.optim.Adam(SchNet().parameters())
    scheduler_instance = scheduler(dummy_optimizer)
    assert isinstance(scheduler_instance, torch.optim.lr_scheduler.StepLR)
    assert scheduler_instance.step_size == 10
    assert scheduler_instance.gamma == 0.1
