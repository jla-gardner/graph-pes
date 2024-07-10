from graph_pes.core import AdditionModel, GraphPESModel
from graph_pes.logger import logger


def log_model_info(model: GraphPESModel):
    """Log the number of parameters in a model."""

    logger.debug(f"Model: {model}")
    if not isinstance(model, AdditionModel):
        params = sum(p.numel() for p in model.parameters())
        logger.info(f"Number of learnable params : {params:,}")
    else:
        model_names = [
            component.__class__.__name__ for component in model.models
        ]
        params = [
            sum(p.numel() for p in component.parameters() if p.requires_grad)
            for component in model.models
        ]
        width = max(len(name) for name in model_names)
        info_str = "Number of learnable params:"
        for name, param in zip(model_names, params):
            info_str += f"\n    {name:<{width}}: {param:,}"
        logger.info(info_str)
