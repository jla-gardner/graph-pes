from __future__ import annotations

from pytorch_lightning.loggers import Logger as PTLLogger

from graph_pes.atomic_graph import (
    AtomicGraph,
    number_of_atoms,
    number_of_structures,
)
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.graph_property_model import GraphPropertyModel, GraphTensorModel
from graph_pes.models.addition import AdditionModel, TensorAdditionModel
from graph_pes.utils.logger import logger
from graph_pes.utils.nn import count_used_parameters


def log_model_info(
    model: GraphPropertyModel,
    ptl_logger: PTLLogger | None = None,
) -> None:
    """Log the number of parameters in a model."""

    logger.debug(f"Model:\n{model}")

    if isinstance(model, (AdditionModel, TensorAdditionModel)):
        model_names = [
            f"{given_name} ({component.__class__.__name__})"
            for given_name, component in model.models.items()
        ]
        params = [
            count_used_parameters(component, only_learnable=True)
            for component in model.models.values()
        ]
        width = max(len(name) for name in model_names)
        info_str = "Number of learnable params:"
        for name, param in zip(model_names, params):
            info_str += f"\n    {name:<{width}}: {param:,}"
        logger.info(info_str)

    else:
        n = count_used_parameters(model, only_learnable=True)
        logger.info(f"Number of learnable params : {n:,}")

    if ptl_logger is not None:
        all_params = count_used_parameters(model, only_learnable=False)
        learnable_params = count_used_parameters(model, only_learnable=True)
        ptl_logger.log_metrics(
            {
                "n_parameters": all_params,
                "n_learnable_parameters": learnable_params,
            }
        )


def sanity_check(model: GraphPropertyModel, batch: AtomicGraph) -> None:
    if isinstance(model, GraphPESModel):
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

    elif isinstance(model, GraphTensorModel):
        outputs = model.predict(batch, ["tensor"])

        # for now, only a single output, "tensor", is supported
        if "tensor" not in outputs:
            raise ValueError(
                "Sanity check failed: the model did not predict the "
                "`tensor` property."
            )

        tensor = outputs["tensor"]
        N = number_of_atoms(batch)
        if tensor.shape[0] != N:
            raise ValueError(
                "Sanity check failed: the model predicted a tensor with "
                f"shape {tensor.shape} but the number of atoms in the graph "
                f"is {N}."
            )


VALIDATION_LOSS_KEY = "valid/loss/total"
