from __future__ import annotations

import copy
import random
import string
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence, TypeVar, overload

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


def differentiate_all(
    y: torch.Tensor,
    xs: list[torch.Tensor],
    training: bool = False,
):
    """
    A ``Torchscript``-compatible way to differentiate `y` with respect
    to all of the tensors in `xs`, with a slightly nicer API than
    ``torch.autograd.grad``, and handling the (odd) cases where either
    or both of `y` or `xs` do not have a gradient function.
    """

    if not torch.is_grad_enabled():
        raise RuntimeError(
            "Autograd is disabled, but you are trying to "
            "calculate gradients. Please wrap your code in "
            "a torch.enable_grad() context."
        )

    defaults = [torch.zeros_like(x) for x in xs]

    x_did_require_grad = [x.requires_grad for x in xs]
    for x in xs:
        x.requires_grad_(True)

    y_total = y.sum()
    # ensure y has a grad_fn
    y_total = y_total + torch.tensor(0.0, requires_grad=True)

    grads = torch.autograd.grad(
        [y_total],
        xs,
        create_graph=training,
        allow_unused=True,
    )

    for x, did_require_grad in zip(xs, x_did_require_grad):
        x.requires_grad_(did_require_grad)

    return [
        grad if grad is not None else default
        for grad, default in zip(grads, defaults)
    ]


def differentiate(y: torch.Tensor, x: torch.Tensor, training: bool = False):
    """
    A torchscript-compatible way to differentiate `y` with respect
    to `x`, handling the (odd) cases where either or both of
    `y` or `x` do not have a gradient function: in these cases,
    we return a tensor of zeros with the correct shape and
    requires_grad set to True.
    """

    return differentiate_all(y, [x], training)[0]


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


def nested_merge_all(*dicts: dict) -> dict:
    """
    Merge multiple nested dictionaries, with later dictionaries
    taking precedence over earlier ones.
    """

    result = {}
    for d in dicts:
        result = nested_merge(result, d)
    return result


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


def build_single_nested_dict(keys: list[str], value: Any) -> dict:
    """
    Build a single nested dictionary from a list of keys and a value.
    """

    result = value
    for key in reversed(keys):
        result = {key: result}
    return result


def random_id(
    lengths: list[int] | None = None,
    use_existing: bool = False,
) -> str:
    """
    Generate a random ID of the form ``abdc123_efgh456_...``.

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
    "abcd_1234"
    """

    if lengths is None:
        lengths = [8]

    if use_existing:
        rng = random
    else:
        # seed with the current time
        rng = random.Random()
        rng.seed()

    return "_".join(
        "".join(rng.choices(string.ascii_lowercase + string.digits, k=k))
        for k in lengths
    )


def random_dir(root: Path) -> Path:
    """Find a random directory that doesn't exist in `root`."""

    while True:
        new_dir = root / random_id(lengths=[8, 8, 8])
        if not new_dir.exists():
            return new_dir


def contains_tensor(l: Iterable[torch.Tensor], tensor: torch.Tensor) -> bool:
    """
    A convenient way to check if a list contains a particular tensor,
    since ``tensor in l`` is broadcasted by torch to return a boolean
    tensor with the same shape as ``l``.
    """
    return any(tensor is t for t in l)


def left_aligned_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r"""
    Calculate :math:`z = x \odot y` such that:

    .. math::

            z_{i, j, \dots} = x_{i, j, \dots} \cdot y_i

    That is, broadcast :math:`y` to the far left of :math:`x` (the opposite
    sense of normal broadcasting in torch), and multiply the two tensors
    elementwise.

    Parameters
    ----------
    x
        of shape (n, ...)
    y
        of shape (n, )

    Returns
    -------
    torch.Tensor
        of same shape as x
    """
    if x.dim() == 1 or x.dim() == 0:
        return x * y

    # x of shape (n, ..., a)
    x = x.transpose(0, -1)  # shape: (a, ..., n)
    result = x * y  # shape: (a, ..., n)
    return result.transpose(0, -1)  # shape: (n, ..., a)


def left_aligned_div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r"""
    Calculate :math:`z = x \oslash y` such that:

    .. math::

            z_{i, j, \dots} = x_{i, j, \dots} / y_i

    That is, broadcast :math:`y` to the far left of :math:`x` (the opposite
    sense of normal broadcasting in torch), and divide the two tensors
    elementwise.

    Parameters
    ----------
    x
        of shape (n, ...)
    y
        of shape (n, )

    Returns
    -------
    torch.Tensor
        of same shape as x
    """
    if x.dim() == 1 or x.dim() == 0:
        return x / y

    # x of shape (n, ..., a)
    x = x.transpose(0, -1)  # shape: (a, ..., n)
    result = x / y  # shape: (a, ..., n)
    return result.transpose(0, -1)  # shape: (n, ..., a)
