from __future__ import annotations

from typing import TypeVar

import dacite
import data2objects

T = TypeVar("T")


def instantiate_config_from_dict(
    config_dict: dict, config_class: type[T]
) -> tuple[dict, T]:
    """Instantiate a config object from a dictionary."""

    final_dict: dict = data2objects.fill_referenced_parts(config_dict)  # type: ignore

    import graph_pes
    import graph_pes.data
    import graph_pes.models
    import graph_pes.training
    import graph_pes.training.callbacks
    import graph_pes.training.loss
    import graph_pes.training.opt

    object_dict = data2objects.from_dict(
        final_dict,
        modules=[
            graph_pes,
            graph_pes.models,
            graph_pes.training,
            graph_pes.training.opt,
            graph_pes.training.loss,
            graph_pes.data,
            graph_pes.training.callbacks,
        ],
    )

    try:
        return (
            final_dict,
            dacite.from_dict(
                data_class=config_class,
                data=object_dict,
                config=dacite.Config(strict=True),
            ),
        )
    except Exception as e:
        raise ValueError(
            f"Failed to instantiate a config object of type {config_class} "
            "from the following object-replaced dictionary:\n{final_dict}"
        ) from e
