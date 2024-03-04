from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Sequence, TypeVar, overload

import torch
from ase.data import chemical_symbols
from torch import Tensor

T = TypeVar("T")


MAX_Z = 118
"""The maximum atomic number in the periodic table."""


@overload
def pairs(a: Sequence[T]) -> Iterator[tuple[T, T]]:
    ...


@overload
def pairs(a: Tensor) -> Iterator[tuple[Tensor, Tensor]]:
    ...


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


def shape_repr(dict_of_tensors: dict[str, Tensor], sep: str = ", ") -> str:
    """
    Generate a string representation of the shapes of the tensors in
    `dict_of_tensors`.

    Parameters
    ----------
    dict_of_tensors: dict[str, Tensor]
        The dictionary of tensors to represent.
    sep: str
        The separator to use between each tensor.

    Returns
    -------
    str
        The string representation of the shapes of the tensors.

    Example
    -------
    >>> shape_repr({'a': torch.zeros(3), 'b': torch.zeros(4)})
    'a=[3], b=[4]'
    """

    def _get_shape(tensor: Tensor) -> str:
        if len(tensor.shape) == 0:
            return f"{tensor.item():.2f}"
        return "[" + ",".join(str(i) for i in tensor.shape) + "]"

    return sep.join(f"{k}={_get_shape(v)}" for k, v in dict_of_tensors.items())


def to_chem_symbol(z: int):
    """
    Convert an atomic number to a chemical symbol.

    Parameters
    ----------
    z: int
        The atomic number.

    Returns
    -------
    str
        The chemical symbol.

    Example
    -------
    >>> to_chem_symbol(1)
    'H'
    >>> to_chem_symbol(118)
    'Og'
    """
    return chemical_symbols[z]


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
