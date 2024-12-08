from graph_pes.models.addition import AdditionModel
from graph_pes.models.offsets import LearnableOffset
from graph_pes.models.schnet import SchNet
from graph_pes.training.callbacks import log_offset, log_scales


class FakeLogger:
    def __init__(self, results: dict):
        self.results = results

    def log_metrics(self, metrics: dict):
        self.results.update(metrics)


def test_offset_logger():
    offset = LearnableOffset(H=1, O=2)
    offset._offsets.register_elements([1, 6, 8])  # H, C, O
    model = AdditionModel(offset=offset)
    logger = FakeLogger({})
    log_offset(model, logger)  # type: ignore
    assert logger.results == {"offset/H": 1.0, "offset/O": 2.0, "offset/C": 0.0}

    # clear the logger
    logger.results = {}
    log_offset(SchNet(), logger)  # type: ignore
    assert logger.results == {}


def test_scales_logger():
    model = SchNet()
    model.scaler.per_element_scaling.register_elements([1, 6])  # H, C, O
    model.scaler.per_element_scaling.data[1, 0] = 2.0
    logger = FakeLogger({})
    log_scales(model, logger)  # type: ignore
    assert logger.results == {"scale/H": 2.0, "scale/C": 1.0}

    addition_model = AdditionModel(schnet=model)
    logger = FakeLogger({})
    log_scales(addition_model, logger)  # type: ignore
    assert logger.results == {"scale/schnet/H": 2.0, "scale/schnet/C": 1.0}
