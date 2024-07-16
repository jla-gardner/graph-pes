from __future__ import annotations

import torch
from graph_pes.core import GraphPESModel
from graph_pes.models.addition import AdditionModel
from graph_pes.models.offsets import LearnableOffset
from graph_pes.util import uniform_repr


# TODO: document
class Optimizer:
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
        offset_params = []
        if isinstance(model, LearnableOffset):
            offset_params += list(model.parameters())
        elif isinstance(model, AdditionModel):
            for component in model.models:
                if isinstance(component, LearnableOffset):
                    offset_params += list(component.parameters())

        if not offset_params:
            return self.optimizer_class(
                [{"name": "model", "params": model.parameters()}],
                **self.kwargs,
            )

        param_groups = [
            {
                "name": "offset",
                "params": offset_params,
                "lr": min(self.kwargs["lr"] * 10, 1e-1),
                "weight_decay": 0.0,
            },
            {
                "name": "model",
                "params": [
                    p
                    for p in model.parameters()
                    # can't use simple "is in" due to pytorch's automated
                    # casting to strange tensor types
                    if not any(p is t for t in offset_params)
                ],
            },
        ]
        return self.optimizer_class(param_groups, **self.kwargs)

    def __repr__(self):
        return uniform_repr(
            self.__class__.__name__,
            name=self.optimizer_class.__name__,
            **self.kwargs,
        )


class LRScheduler:
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


## Util classes for simplicity ##


class Adam(Optimizer):
    def __init__(
        self,
        lr: float = 3e-4,
        weight_decay: float = 0.0,
    ):
        super().__init__(torch.optim.Adam, lr=lr, weight_decay=weight_decay)


class SGD(Optimizer):
    def __init__(
        self,
        lr: float = 3e-4,
        weight_decay: float = 0.0,
    ):
        super().__init__(torch.optim.SGD, lr=lr, weight_decay=weight_decay)


class ReduceLROnPlateau(LRScheduler):
    def __init__(
        self,
        factor: float = 0.5,
        patience: int = 10,
        threshold: float = 1e-4,
        min_lr: float = 1e-6,
    ):
        super().__init__(
            torch.optim.lr_scheduler.ReduceLROnPlateau,
            factor=factor,
            patience=patience,
            threshold=threshold,
            min_lr=min_lr,
        )
