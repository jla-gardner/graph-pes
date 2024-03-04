from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Sequence, TypeVar, overload

import torch
from torch import Tensor

T = TypeVar("T")


MAX_Z = 118
"""The maximum atomic number in the periodic table."""


@overload
def pairs(a: Sequence[T]) -> Iterator[tuple[T, T]]: ...


@overload
def pairs(a: Tensor) -> Iterator[tuple[Tensor, Tensor]]: ...


def pairs(a) -> Iterator[tuple[T, T] | tuple[Tensor, Tensor]]:
    """
    Iterate over pairs of elements in `a`

    Parameters
    ----------
    a
        The sequence or tensor to iterate over.

    Example
    -------
    >>> list(pairs([1, 2, 3]))
    [(1, 2), (2, 3)]

    >>> list(pairs(Tensor([1, 2, 3])))
    [(1, 2), (2, 3)]
    """
    return zip(a, a[1:])


def as_possible_tensor(value: object) -> Tensor | None:
    """
    Convert a value to a tensor if possible.

    Parameters
    ----------
    value
        The value to convert.
    """

    try:
        tensor = torch.as_tensor(value)
        if tensor.dtype == torch.float64:
            tensor = tensor.float()
        return tensor

    except Exception:
        return None


def differentiate(y: torch.Tensor, x: torch.Tensor):
    if y.grad_fn is None:
        raise ValueError(
            "The tensor `y` must be the result of a computation "
            "that requires gradients. Currently, there is no "
            "gradient function associated with this tensor."
        )
    with require_grad(x):
        grad = torch.autograd.grad(
            y.sum(),
            x,
            create_graph=True,
            allow_unused=True,
        )[0]

    default = torch.zeros_like(x, requires_grad=True)
    return grad if grad is not None else default


@contextmanager
def require_grad(tensor: torch.Tensor):
    # check if in a torch.no_grad() context: if so,
    # raise an error
    if not torch.is_grad_enabled():
        raise RuntimeError(
            "Autograd is disabled, but you are trying to "
            "calculate gradients. Please wrap your code in "
            "a torch.enable_grad() context."
        )

    req_grad = tensor.requires_grad
    tensor.requires_grad_(True)
    yield
    tensor.requires_grad_(req_grad)
