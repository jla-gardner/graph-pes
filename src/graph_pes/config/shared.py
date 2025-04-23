from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Literal, Protocol, TypeVar

import dacite
import data2objects

from graph_pes.data.datasets import (
    DatasetCollection,
    GraphDataset,
    file_dataset,
)
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.models import AdditionModel
from graph_pes.training.loss import Loss, TotalLoss
from graph_pes.utils.misc import nested_merge

T = TypeVar("T")


def _nice_dict_repr(d: dict) -> str:
    def _print_dict(d: dict, indent: int = 0) -> str:
        nice = {
            k: v
            if not isinstance(v, dict)
            else "\n" + _print_dict(v, indent + 1)
            for k, v in d.items()
        }
        return "\n".join([f"{'  ' * indent}{k}: {v}" for k, v in nice.items()])

    return _print_dict(d, 0)


class HasDefaults(Protocol):
    @classmethod
    def defaults(cls) -> dict: ...


HD = TypeVar("HD", bound=HasDefaults)


def instantiate_config_from_dict(
    config_dict: dict, config_class: type[HD]
) -> tuple[dict, HD]:
    """Instantiate a config object from a dictionary."""

    config_dict = nested_merge(config_class.defaults(), config_dict)
    final_dict: dict = data2objects.fill_referenced_parts(config_dict)  # type: ignore

    import graph_pes
    import graph_pes.data
    import graph_pes.interfaces
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
            graph_pes.interfaces,
        ],
    )
    field_names = {f.name for f in fields(config_class)}  # type: ignore
    object_dict = {
        k: v
        for k, v in object_dict.items()
        if k in field_names  # type: ignore
    }

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
            f"from the following dictionary:\n{_nice_dict_repr(final_dict)}"
        ) from e


@dataclass
class TorchConfig:
    """Configuration for PyTorch."""

    dtype: Literal["float16", "float32", "float64"]
    """
    The dtype to use for all model parameters and graph properties.
    Defaults is ``"float32"``.
    """

    float32_matmul_precision: Literal["highest", "high", "medium"]
    """
    The precision to use internally for float32 matrix multiplications. Refer to the
    `PyTorch documentation <https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html>`__
    for details.

    Defaults to ``"high"`` to favour accelerated learning over numerical
    exactness for matmuls.
    """  # noqa: E501


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
    loss: Loss | TotalLoss | dict[str, Loss] | list[Loss],
) -> TotalLoss:
    if isinstance(loss, Loss):
        return TotalLoss([loss])
    elif isinstance(loss, TotalLoss):
        return loss
    elif isinstance(loss, dict):
        return TotalLoss(list(loss.values()))
    elif isinstance(loss, list):
        return TotalLoss(loss)
    raise ValueError(
        "Expected to be able to parse a Loss, TotalLoss, a list of "
        "Loss instances, or a dictionary mapping keys to Loss instances from "
        "the loss config, but got something else:\n"
        f"{loss}"
    )


def parse_single_dataset(value: Any, model: GraphPESModel) -> GraphDataset:
    if isinstance(value, GraphDataset):
        return value
    elif isinstance(value, (str, dict)):
        kwargs: dict[str, Any] = {"cutoff": model.cutoff.item()}
        if isinstance(value, str):
            kwargs["path"] = value
        else:
            kwargs.update(value)
        return file_dataset(**kwargs)
    else:
        raise ValueError(
            "Expected a GraphDataset or a dict to pass to file_dataset, "
            f"but got {value}"
        )


def parse_dataset_collection(
    raw_data: DatasetCollection | dict[str, Any],
    model: GraphPESModel,
) -> DatasetCollection:
    if isinstance(raw_data, DatasetCollection):
        return raw_data

    if "train" not in raw_data or "valid" not in raw_data:
        raise ValueError(
            "Expected to parse a DatasetCollection instance or a "
            "dictionary mapping 'train' and 'valid' keys to GraphDataset "
            "instances from the data config, but got something else: "
            f"{raw_data}"
        )

    collection = {}
    for key in ["train", "valid"]:
        value = raw_data[key]
        collection[key] = parse_single_dataset(value, model)

    if "test" in raw_data:
        if isinstance(raw_data["test"], (str, GraphDataset)):
            collection["test"] = parse_single_dataset(raw_data["test"], model)
        elif isinstance(raw_data["test"], dict):
            # could either be a simple dict specifying a single test set:
            exception = None
            try:
                collection["test"] = parse_single_dataset(
                    raw_data["test"], model
                )
            except FileNotFoundError:
                raise
            except Exception as e:
                exception = e

            # or a dict of test sets:
            try:
                if "test" not in collection:
                    collection["test"] = {
                        _key: parse_single_dataset(_value, model)
                        for _key, _value in raw_data["test"].items()
                    }
            except Exception as e:
                raise ValueError(
                    "Failed to parse test sets. We encountered the following "
                    f"errors:\n{exception}\nand\n{e}\n"
                    "Please check your test set configuration and try again."
                ) from e

    return DatasetCollection(**collection)
