# ruff: noqa: UP006, UP007

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, TypeVar, Union

import dacite
import yaml

from graph_pes.core import AdditionModel, GraphPESModel
from graph_pes.data.dataset import FittingData
from graph_pes.training.loss import Loss, TotalLoss
from graph_pes.training.opt import LRScheduler, Optimizer

T = TypeVar("T")


def _import(thing: str) -> Any:
    """
    Import a module or object from a fully qualified name.

    Parameters
    ----------
    thing
        The fully qualified name.

    Returns
    -------
    Any
        The imported object.
    """

    module_name, obj_name = thing.rsplit(".", 1)
    module = __import__(module_name, fromlist=[obj_name])
    return getattr(module, obj_name)


def _import_and_maybe_call(thing: str) -> Any:
    """
    Import a module or object from a fully qualified name.

    If the name ends with ``()``, the object is called and the result
    is returned.

    Parameters
    ----------
    thing
        The fully qualified name.

    Returns
    -------
    Any
        The imported object.
    """

    if thing.endswith("()"):
        return _import(thing[:-2])()
    return _import(thing)


def _instantiate(thing: str | dict[str, Any]) -> Any:
    """
    Instantiate an object from a user-defined specification.

    If the ``thing`` is a string, it is assumed to be a fully qualified
    name of a module or object. The object is imported, optionally called
    if the name ends with ``()`` and returned.

    If the ``thing`` is a dictionary, it is assumed to be a specification
    for an object. The dictionary must have exactly one key, which is the
    fully qualified name of the object to import. The value is a dictionary
    of keyword arguments to pass to the object's constructor. We recursively
    instantiate these arguments, before ultimately instantiating the object
    and returning it.
    """

    def _from_string(s: str) -> tuple[Any, bool]:
        if "." not in s:
            return s, False

        # try to import the thing: return it and a success flag
        try:
            return _import_and_maybe_call(s), True
        # if we can't import it, return the string and a failure flag
        # if it looks like we should have been able to import it, warn
        except ImportError:
            warnings.warn(
                f"Encountered a string ({s}) that looks like it "
                "could be meant to be imported - we couldn't do "
                "this. This may cause issues later.",
                stacklevel=2,
            )
            return s, False

    def _from_dict(d: dict[str, Any]) -> tuple[Any, bool]:
        if len(d) == 1:
            # this could be a fully qualified name mapping to some arguments
            og_key, og_value = next(iter(d.items()))
            if (
                isinstance(og_key, str)
                and "." in og_key
                and isinstance(og_value, dict)
            ):
                # try to import the thing
                new_key, import_success = _from_string(og_key)

                # maybe instantiate the arguments recursively
                new_values = {}
                for kk, vv in og_value.items():
                    if isinstance(vv, str):
                        new_values[kk], _ = _from_string(vv)
                    elif isinstance(vv, dict):
                        new_values[kk], _ = _from_dict(vv)
                    else:
                        new_values[kk] = vv

                if import_success:
                    return new_key(**new_values), True

                return {og_key: new_values}, False

        # not a mapping from a fully qualified name to arguments, so
        # keep the keys as they are, but still try to instantiate the values
        new_dict = {}
        for key, value in d.items():
            if isinstance(value, str):
                new_dict[key], _ = _from_string(value)
            elif isinstance(value, dict):
                new_dict[key], _ = _from_dict(value)
            else:
                new_dict[key] = value
        return new_dict, False

    if isinstance(thing, str):
        result, success = _from_string(thing)
        if not success:
            raise ValueError(f"Could not import {thing}")
        return result

    elif isinstance(thing, dict):
        if len(thing) != 1:
            raise ValueError("Expected exactly one key in the dictionary.")

        result, success = _from_dict(thing)
        if not success:
            raise ValueError(f"Was not able to import {next(iter(thing))}")
        return result

    raise ValueError("Expected either a string or a dictionary.")


@dataclass
class LossSpec:
    component: Union[str, Dict[str, Any]]
    weight: Union[int, float] = 1.0


@dataclass
class FittingOptions:
    pre_fit_model: bool
    """Whether to pre-fit the model before training."""

    max_n_pre_fit: Union[int, None]
    """
    The maximum number of graphs to use for pre-fitting.
    Set to ``None`` to use all available data.
    """

    early_stopping_patience: Union[int, None]
    """
    The number of epochs to wait for improvement in the total validation loss
    before stopping training. Set to ``None`` to disable early stopping.
    """

    trainer_kwargs: Dict[str, Any]
    """
    Key-word arguments to pass to the PTL trainer.
    
    See their docs. # TODO
    
    Example
    -------
    .. code-block:: yaml
    
        trainer:
            max_epochs: 100
            gpus: 1
            check_val_every_n_epoch: 5
    """

    loader_kwargs: Dict[str, Any]
    """
    Key-word arguments to pass to the underlying 
    :class:`torch.utils.data.DataLoader`.

    See their docs. # TODO
    """


@dataclass
class FittingConfig(FittingOptions):
    optimizer: Union[str, Dict[str, Any]]
    """
    Specification for the optimizer.

    ``graph-pes`` provides a few common optimisers, but you can also 
    roll your own.
    
    Point to something that instantiates a graph_pes optimiser:

    Examples
    --------
    The default (see :func:`~graph_pes.training.opt.Optimizer` for details):
    .. code-block:: yaml
    
        optimizer:
            graph_pes.training.opt.Optimizer:
                name: Adam
                lr: 3e-3
                weight_decay: 0.0
                amsgrad: false

    Or a custom one:
    .. code-block:: yaml
    
        optimizer: my.module.MagicOptimizer()
    """

    scheduler: Union[str, Dict[str, Any], None]
    """
    Specification for the learning rate scheduler. Optional.

    # TODO: more schedules/flexibility
    Examples
    --------
    .. code-block:: yaml
    
        scheduler:
            graph_pes.training.opt.LRScheduler:
                name: ReduceLROnPlateau
                factor: 0.5
                patience: 10
    """

    ### Methods ###

    def instantiate_optimizer(self) -> Optimizer:
        # check correct input
        if not isinstance(self.optimizer, (str, dict)):
            raise ValueError("# TODO")

        # and then instantiate
        return _instantiate(self.optimizer)

    def instantiate_scheduler(self) -> LRScheduler | None:
        if self.scheduler is None:
            return None

        return _instantiate(self.scheduler)


# NB: dacite parsing requires the old type hint syntax in
# order to be compatible with all versions of Python that
# we are targeting (3.8+)
@dataclass
class Config:
    """
    A schema for a configuration file to train a
    :class:`~graph_pes.core.GraphPESModel`.

    While parsing your configuration file, we will attempt to import
    any class, object or function that you specify via a fully qualified
    name. This allows you to point both to classes and functions that
    ``graph-pes`` provides, as well as your own custom code.

    Notes
    -----
    To point to an object, simplify specify the **fully qualified name**,
    e.g. ``my_module.my_object``.

    If you want to use the return value of a
    function with no arguments, append ``()`` to the name, e.g.
    ``my_module.my_function()``.

    To point to a class or function with arguments, use a nested dictionary
    structure like so:
    .. code-block:: yaml

        graph_pes.models.SchNet:
            cutoff: 5.0
            n_layers: 3
    """

    model: Union[
        str,
        Dict[str, Any],
        List[Union[str, Dict[str, Any]]],
    ]
    """
    Specification for the model.

    Examples
    --------
    To specify a single model with parameters:
    .. code-block:: yaml
    
        model:
            graph_pes.models.LennardJones:
                sigma: 0.1
                epsilon: 1.0
    
    or, if no parameters are needed:
    .. code-block:: yaml
    
        model: my_model.SpecialModel()
    
    To specify multiple components of an :class:`~graph_pes.core.AdditionModel`,
    create a list of specications as above:
    .. code-block:: yaml
    
        model:
            - graph_pes.models.FixedOffset:
                  H: -123.4
                  C: -456.7
            - graph_pes.models.SchNet()
    """

    data: Union[str, Dict[str, Any]]
    """
    Specification for the data. 
    
    Point to one of the following:
    - a callable that returns a :class:`~graph_pes.data.dataset.FittingData` 
      instance
    - a dictionary mapping ``"train"`` and ``"valid"`` keys to callables that
      return :class:`~graph_pes.data.dataset.LabelledGraphDataset` instances

    Examples
    --------
    Load custom data from a function with no arguments:
    .. code-block:: yaml
        
        data: my_module.my_fitting_data()

    Point to :func:`graph_pes.data.load_atoms_datasets` with arguments:
    .. code-block:: yaml

        data:
            graph_pes.data.load_atoms_datasets:
                id: QM9
                cutoff: 5.0
                n_train: 10000
                n_val: 1000
                property_map:
                    energy: U0
    """

    loss: Union[str, Dict[str, Any], List[LossSpec]]
    """
    Specification for the loss function. This can be a single loss function
    or a list of loss functions with weights.

    Examples
    --------
    To specify a single loss function:
    .. code-block:: yaml
    
        loss: graph_pes.training.loss.PerAtomEnergyLoss()

    or with parameters:

    .. code-block:: yaml
        
            loss:
                graph_pes.training.loss.Loss:
                    property_key: energy
                    metric: graph_pes.training.loss.RMSE()

    To specify multiple loss functions with weights:
    .. code-block:: yaml
    
        loss:
            - component: graph_pes.training.loss.Loss:
                property_key: energy
                metric: graph_pes.training.loss.RMSE()
              weight: 1.0
            - component: graph_pes.training.loss.Loss:
                property_key: forces
                metric: graph_pes.training.loss.MAE()
              weight: 10.0
    """

    fitting: FittingConfig

    ### Methods ###

    def to_nested_dict(self) -> Dict[str, Any]:
        def _maybe_as_dict(obj):
            if isinstance(obj, list):
                return [_maybe_as_dict(v) for v in obj]
            elif not hasattr(obj, "__dict__"):
                return obj
            return {k: _maybe_as_dict(v) for k, v in obj.__dict__.items()}

        return _maybe_as_dict(self)  # type: ignore

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        try:
            return dacite.from_dict(
                data_class=cls,
                data=data,
                config=dacite.Config(strict=True),
            )
        except Exception as e:
            raise ValueError(
                "Your configuration file could not be successfully parsed. "
                "Please check that it is formatted correctly. For examples, "
                "please see ..."  # TODO
            ) from e

    def __repr__(self):
        # yaml dump the nested config, complete with defaults,
        # as it would appear in a config.yaml file
        return yaml.dump(self.to_nested_dict(), indent=3, sort_keys=False)

    def instantiate_model(self) -> GraphPESModel:
        if isinstance(self.model, (str, dict)):
            model = _instantiate(self.model)
            if not isinstance(model, GraphPESModel):
                raise ValueError(
                    f"Expected a GraphPESModel, got {type(model)}: {model}"
                )
            return model

        elif isinstance(self.model, list):
            models = [_instantiate(spec) for spec in self.model]
            if not all(isinstance(m, GraphPESModel) for m in models):
                raise ValueError("# TODO")
            return AdditionModel(models)

        raise ValueError("# TODO")

    def instantiate_data(self) -> FittingData:
        if isinstance(self.data, str):
            return _instantiate(self.data)

        if isinstance(self.data, dict):
            if len(self.data) == 1:
                return _instantiate(self.data)
            elif len(self.data) == 2:
                assert self.data.keys() == {"train", "valid"}
                return FittingData(
                    train=_instantiate(self.data["train"]),
                    valid=_instantiate(self.data["valid"]),
                )

        raise ValueError(
            "Unexpected data specification. "
            "Please provide a callable or a dictionary containing "
            "a single key (the fully qualified name of some callable) "
            "or two keys ('train' and 'valid') mapping to callables."
        )

    def instantiate_loss(self) -> TotalLoss:
        if isinstance(self.loss, (str, dict)):
            loss = _instantiate(self.loss)
            if isinstance(loss, Loss):
                return TotalLoss([loss])
            elif isinstance(loss, TotalLoss):
                return loss
            else:
                raise ValueError("# TODO")

        else:
            if not all(isinstance(l, LossSpec) for l in self.loss):
                raise ValueError("# TODO")

            weights = [l.weight for l in self.loss]
            losses = [_instantiate(l.component) for l in self.loss]

            if not all(isinstance(w, (int, float)) for w in weights):
                raise ValueError("# TODO")

            if not all(isinstance(l, Loss) for l in losses):
                raise ValueError("# TODO")

            return TotalLoss(losses, weights)
