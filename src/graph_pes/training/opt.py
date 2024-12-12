from __future__ import annotations

import torch

from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.models.offsets import LearnableOffset
from graph_pes.utils.misc import contains_tensor, uniform_repr


class Optimizer:
    """
    A factory class for delayed instantiation of :class:`torch.optim.Optimizer`
    objects.

    The generated optimizer splits the parameters of the model into two groups:

    - "non-decayable" parameters, which are all parameters returned by the
      :meth:`~graph_pes.GraphPESModel.non_decayable_parameters` method of
      the model.
    - "normal" parameters, corresponding to the remaining model parameters.

    Unsurprisingly, any specified weight decay is applied only to the normal
    model parameters.

    As an example, per-element energy offsets parameters of
    :class:`~graph_pes.models.offsets.LearnableOffset` models
    represent the arbitrary zero points of energies for different elements:
    it doesn't make sense to push these towards zero during training.

    .. note::

        We use delayed instantiation of optimizers when configuring our training
        runs to allow for arbitrary changes to the model and its parameters
        during the :class:`~graph_pes.GraphPESModel.pre_fit_all_components`
        method.


    Parameters
    ----------
    name
        The name of the :class:`torch.optim.Optimizer` class to use, e.g.
        ``"Adam"`` or ``"SGD"``. Alternatively, provide the type of any
        subclass of :class:`torch.optim.Optimizer`.
    **kwargs
        Additional keyword arguments to pass to the specified optimizer's
        constructor.

    Examples
    --------
    Pass a named optimiser:

    >>> from graph_pes.training.opt import Optimizer
    >>> optimizer_factory = Optimizer("AdamW", lr=1e-3)
    >>> optimizer_instance = optimizer_factory(model)

    Or pass the optimiser class directly:

    >>> from torch.optim import SGD
    >>> optimizer_factory = Optimizer(SGD, lr=1e-3)
    >>> optimizer_instance = optimizer_factory(model)

    Psuedo-code excerpt from ``graph-pes-train`` logic:

    >>> from graph_pes.training.opt import Optimizer
    >>> from graph_pes.models import LennardJones
    >>> ...
    >>> optimizer_factory = Optimizer("AdamW", lr=1e-3)
    >>> model = LennardJones()
    >>> model.pre_fit(train_loader)
    >>> optimizer_instance = optimizer_factory(model)
    """

    def __init__(
        self,
        name: str | type[torch.optim.Optimizer],
        **kwargs,
    ):
        self.kwargs = kwargs
        if isinstance(name, str):
            optimizer_class: type[torch.optim.Optimizer] = getattr(
                torch.optim, name, None
            )  # type: ignore
            if optimizer_class is not None:
                name = optimizer_class
            else:
                raise ValueError(
                    f"Could not find optimizer {name}. "
                    "Please provide the common PyTorch name, or a fully "
                    "qualified class name."
                )
        else:
            optimizer_class = name

        dummy_model = LearnableOffset()
        dummy_opt = optimizer_class(dummy_model.parameters(), **kwargs)
        if not isinstance(dummy_opt, torch.optim.Optimizer):
            raise ValueError(
                "Expected the returned optimizer to be an instance of "
                "torch.optim.Optimizer, but got "
                f"{type(dummy_opt)}"
            )

        self.optimizer_class = optimizer_class

    def __call__(self, model: GraphPESModel) -> torch.optim.Optimizer:
        """
        Create an instance of the specified optimizer class, with the correct
        parameter groups for the model.
        """

        all_params = list(set(model.parameters()))
        non_decayable_params = list(set(model.non_decayable_parameters()))

        param_groups = [
            {
                "name": "non-decayable",
                "params": non_decayable_params,
                "weight_decay": 0.0,
            },
            {
                "name": "normal",
                "params": [
                    p
                    for p in all_params
                    if not contains_tensor(non_decayable_params, p)
                ],
            },
        ]

        # remove empty groups
        param_groups = [pg for pg in param_groups if pg["params"]]

        return self.optimizer_class(param_groups, **self.kwargs)

    def __repr__(self):
        return uniform_repr(
            self.__class__.__name__,
            name=self.optimizer_class.__name__,
            **self.kwargs,
        )


class LRScheduler:
    """
    A factory class for delayed instantiation of
    :class:`torch.optim.lr_scheduler.LRScheduler` objects.

    Parameters
    ----------
    name
        The name of the :class:`torch.optim.lr_scheduler.LRScheduler` class to
        use, e.g. ``"ReduceLROnPlateau"``. Alternatively, provide any subclass
        of :class:`torch.optim.lr_scheduler.LRScheduler`.
    **kwargs
        Additional keyword arguments to pass to the specified scheduler's
        constructor.

    Examples
    --------
    >>> from graph_pes.training.opt import LRScheduler
    >>> ...
    >>> scheduler_factory = LRScheduler(
    ...     "LambdaLR", lr_lambda=lambda epoch: 0.95 ** epoch
    ... )
    >>> scheduler_instance = scheduler_factory(optimizer)
    """

    def __init__(
        self,
        name: str | type[torch.optim.lr_scheduler.LRScheduler],
        **kwargs,
    ):
        self.kwargs = kwargs
        if isinstance(name, str):
            scheduler_class: type[torch.optim.lr_scheduler.LRScheduler] = (
                getattr(torch.optim.lr_scheduler, name, None)
            )  # type: ignore
            if scheduler_class is not None:
                name = scheduler_class
            else:
                raise ValueError(
                    f"Could not find scheduler {name}. "
                    "Please provide the common PyTorch name, or a fully "
                    "qualified class name."
                )
        else:
            scheduler_class = name

        dummy_opt = torch.optim.Adam(LearnableOffset().parameters())
        dummy_scheduler = scheduler_class(dummy_opt, **kwargs)
        if not isinstance(
            dummy_scheduler,
            (
                torch.optim.lr_scheduler.LRScheduler,
                torch.optim.lr_scheduler.ReduceLROnPlateau,
            ),
        ):
            raise ValueError(
                "Expected the returned scheduler to be an instance of "
                "torch.optim.lr_scheduler.LRScheduler, but got "
                f"{type(dummy_scheduler)}"
            )

        self.scheduler_class = scheduler_class

    def __call__(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:
        return self.scheduler_class(optimizer, **self.kwargs)

    def __repr__(self):
        return uniform_repr(
            self.__class__.__name__,
            name=self.scheduler_class.__name__,
            **self.kwargs,
        )
