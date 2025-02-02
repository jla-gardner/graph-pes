from __future__ import annotations

from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.models import AdditionModel
from graph_pes.training.loss import Loss, TotalLoss, WeightedLoss


def parse_model(
    model: GraphPESModel | dict[str, GraphPESModel],
) -> GraphPESModel:
    if isinstance(model, GraphPESModel):
        return model
    elif isinstance(model, dict):
        if not all(isinstance(m, GraphPESModel) for m in model.values()):
            _types = {k: type(v) for k, v in model.items()}

            raise ValueError(
                "Expected all values in the model dictionary to be "
                "GraphPESModel instances, but got something else: "
                f"types: {_types}\n"
                f"values: {model}\n"
            )
        return AdditionModel(**model)
    raise ValueError(
        "Expected to be able to parse a GraphPESModel or a "
        "dictionary of named GraphPESModels from the model config, "
        f"but got something else: {model}"
    )


def parse_loss(
    loss: Loss | WeightedLoss | TotalLoss | dict[str, WeightedLoss | Loss],
) -> TotalLoss:
    if isinstance(loss, (Loss, WeightedLoss)):
        return TotalLoss([loss])
    elif isinstance(loss, TotalLoss):
        return loss
    elif isinstance(loss, dict):
        return TotalLoss(list(loss.values()))
    raise ValueError(
        "Expected to be able to parse a Loss, TotalLoss, or a dictionary "
        "of WeightedLoss instances from the loss config, but got something "
        f"else: {loss}"
    )
