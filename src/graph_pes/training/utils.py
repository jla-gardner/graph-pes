from __future__ import annotations

from pytorch_lightning.loggers import Logger as PTLLogger

from graph_pes.atomic_graph import (
    AtomicGraph,
    number_of_atoms,
    number_of_structures,
)
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.models.addition import AdditionModel
from graph_pes.utils.logger import logger
from graph_pes.utils.nn import learnable_parameters


def log_model_info(
    model: GraphPESModel,
    ptl_logger: PTLLogger | None = None,
) -> None:
    """Log the number of parameters in a model."""

    logger.debug(f"Model:\n{model}")

    if isinstance(model, AdditionModel):
        model_names = [
            f"{given_name} ({component.__class__.__name__})"
            for given_name, component in model.models.items()
        ]
        params = [
            learnable_parameters(component)
            for component in model.models.values()
        ]
        width = max(len(name) for name in model_names)
        info_str = "Number of learnable params:"
        for name, param in zip(model_names, params):
            info_str += f"\n    {name:<{width}}: {param:,}"
        logger.info(info_str)

    else:
        logger.info(
            f"Number of learnable params : {learnable_parameters(model):,}"
        )

    if ptl_logger is not None:
        all_params = sum(p.numel() for p in model.parameters())
        learnable_params = learnable_parameters(model)
        ptl_logger.log_metrics(
            {
                "n_parameters": all_params,
                "n_learnable_parameters": learnable_params,
            }
        )


def sanity_check(model: GraphPESModel, batch: AtomicGraph) -> None:
    outputs = model.get_all_PES_predictions(batch)

    N = number_of_atoms(batch)
    S = number_of_structures(batch)
    expected_shapes = {
        "local_energies": (N,),
        "forces": (N, 3),
        "energy": (S,),
        "stress": (S, 3, 3),
        "virial": (S, 3, 3),
    }

    incorrect = []
    for key, value in outputs.items():
        if value.shape != expected_shapes[key]:
            incorrect.append((key, value.shape, expected_shapes[key]))

    if len(incorrect) > 0:
        raise ValueError(
            "Sanity check failed for the following outputs:\n"
            + "\n".join(
                f"{key}: {value} != {expected}"
                for key, value, expected in incorrect
            )
        )

    if batch.cutoff < model.cutoff:
        logger.error(
            "Sanity check failed: you appear to be training on data "
            f"composed of graphs with a cutoff ({batch.cutoff}) that is "
            f"smaller than the cutoff used in the model ({model.cutoff}). "
            "This is almost certainly not what you want to do?",
        )


VALIDATION_LOSS_KEY = "valid/loss/total"
