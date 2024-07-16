from graph_pes.core import GraphPESModel
from graph_pes.logger import logger
from graph_pes.models.addition import AdditionModel
from graph_pes.nn import learnable_parameters
from pytorch_lightning.loggers import Logger as PTLLogger


def log_model_info(
    model: GraphPESModel,
    ptl_logger: PTLLogger | None = None,
) -> None:
    """Log the number of parameters in a model."""

    logger.info(f"Model:\n{model}")

    if not isinstance(model, AdditionModel):
        params = sum(p.numel() for p in model.parameters())
        logger.info(f"Number of learnable params : {params:,}")

    else:
        model_names = [
            component.__class__.__name__ for component in model.models
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

    if ptl_logger is not None:
        all_params = sum(p.numel() for p in model.parameters())
        learnable_params = learnable_parameters(model)
        ptl_logger.log_hyperparams(
            {
                "n_parameters": all_params,
                "n_learnable_parameters": learnable_params,
            }
        )
