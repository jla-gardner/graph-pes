from __future__ import annotations

import copy
import random
import string
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence, TypeVar, overload

import torch
import torch.distributed
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
        warnings.warn(
            "Expected the tensor `y` to be the result of a computation "
            "that requires gradients. Currently, there is no "
            "grad function associated with this tensor: "
            f"{y}.",
            stacklevel=2,
        )
        return torch.zeros_like(x, requires_grad=True)

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
def require_grad(*tensors: torch.Tensor):
    # check if in a torch.no_grad() context: if so,
    # raise an error
    if not torch.is_grad_enabled():
        raise RuntimeError(
            "Autograd is disabled, but you are trying to "
            "calculate gradients. Please wrap your code in "
            "a torch.enable_grad() context."
        )

    original = [tensor.requires_grad for tensor in tensors]
    for tensor in tensors:
        tensor.requires_grad_(True)
    yield
    for tensor, req_grad in zip(tensors, original):
        tensor.requires_grad_(req_grad)


def to_significant_figures(x: float | int, sf: int = 3) -> float:
    """
    Get a string representation of a float, rounded to
    `sf` significant figures.
    """

    # do the actual rounding
    possibly_scientific = f"{x:.{sf}g}"

    # this might be in e.g. 1.23e+02 format,
    # so convert to float and back to string
    return float(possibly_scientific)


def _is_being_documented():
    return "sphinx" in sys.modules


def uniform_repr(
    thing_name: str,
    *anonymous_things: Any,
    max_width: int = 60,
    stringify: bool = True,
    **named_things: Any,
) -> str:
    def _to_str(thing: Any) -> str:
        if isinstance(thing, str) and stringify:
            return f'"{thing}"'
        return str(thing)

    info = list(map(_to_str, anonymous_things))
    info += [f"{name}={_to_str(thing)}" for name, thing in named_things.items()]

    single_liner = f"{thing_name}({', '.join(info)})"
    if len(single_liner) < max_width and "\n" not in single_liner:
        return single_liner

    def indent(s: str, n=2) -> str:
        _indent = " " * n
        return "\n".join(f"{_indent}{line}" for line in s.split("\n"))

    # if we're here, we need to do a multi-line repr
    rep = f"{thing_name}("
    for thing in info:
        rep += "\n" + indent(thing) + ","

    # remove trailing comma, add final newline and close bracket
    return rep[:-1] + "\n)"


def force_to_single_line(s: str) -> str:
    # TODO: yuck: replace leading and trailing white space with single?
    # or just use tabulate
    lines = [line.strip() for line in s.split("\n")]
    return " ".join(lines)


def nested_merge(a: dict, b: dict):
    """
    Merge two nested dictionaries, with `b` taking precedence
    over `a`.
    """

    new_dict = copy.deepcopy(a)
    for key, value in b.items():
        if (
            key in new_dict
            and isinstance(value, dict)
            and isinstance(new_dict[key], dict)
        ):
            new_dict[key] = nested_merge(new_dict[key], value)
        else:
            new_dict[key] = value

    return new_dict


def random_id(
    lengths: list[int] | None = None,
    use_existing: bool = False,
) -> str:
    """
    Generate a random ID of the form ``abdc123-efgh456-...``.

    Parameters
    ----------
    lengths
        The lengths of the individual parts of the ID.
    use_existing
        Whether to use the existing random number generator in ``random``,
        or to create a new one.

    Example
    -------
    >>> random_id(lengths=[4, 4])
    "abcd-1234"
    """

    if lengths is None:
        lengths = [8]

    if use_existing:
        rng = random
    else:
        # seed with the current time
        rng = random.Random()
        rng.seed()

    return "-".join(
        "".join(rng.choices(string.ascii_lowercase + string.digits, k=k))
        for k in lengths
    )


def random_dir(root: Path) -> Path:
    """Find a random directory that doesn't exist in `root`."""

    while True:
        new_dir = root / random_id(lengths=[8, 8, 8])
        if not new_dir.exists():
            return new_dir
