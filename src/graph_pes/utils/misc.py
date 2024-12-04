from __future__ import annotations

import copy
import random
import re
import string
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence, TypeVar, overload

import numpy
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
    keep_graph: bool = False,
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
        create_graph=keep_graph,
        allow_unused=True,
    )

    for x, did_require_grad in zip(xs, x_did_require_grad):
        x.requires_grad_(did_require_grad)

    return [
        grad if grad is not None else default
        for grad, default in zip(grads, defaults)
    ]


def differentiate(y: torch.Tensor, x: torch.Tensor, keep_graph: bool = False):
    """
    A torchscript-compatible way to differentiate `y` with respect
    to `x`, handling the (odd) cases where either or both of
    `y` or `x` do not have a gradient function: in these cases,
    we return a tensor of zeros with the correct shape and
    requires_grad set to True.
    """

    return differentiate_all(y, [x], keep_graph)[0]


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


def is_being_documented():
    return "sphinx" in sys.modules


def uniform_repr(
    thing_name: str,
    *anonymous_things: Any,
    max_width: int = 60,
    stringify: bool = True,
    indent_width: int = 2,
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

    def indent(s: str) -> str:
        _indent = " " * indent_width
        return "\n".join(f"{_indent}{line}" for line in s.split("\n"))

    # if we're here, we need to do a multi-line repr
    rep = f"{thing_name}("
    for thing in info:
        rep += "\n" + indent(thing) + ","

    # remove trailing comma, add final newline and close bracket
    return rep[:-1] + "\n)"


def force_to_single_line(s: str) -> str:
    """
    Convert a multi-line string to a single line by replacing all whitespace
    sequences (including newlines) with a single space.
    """
    return re.sub(r"\s+", " ", s.strip())


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
        lengths = [6, 6]

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
        new_dir = root / random_id()
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


def random_split(
    sequence: Sequence[T],
    lengths: Sequence[int],
    seed: int | None = None,
) -> list[list[T]]:
    """
    Randomly split `sequence` into sub-sequences according to `lengths`.

    Parameters
    ----------
    sequence
        The sequence to split.
    lengths
        The lengths of the sub-sequences to create.
    seed
        The random seed to use. If `None`, the current random state is
        used (non-deterministic).

    Returns
    -------
    list[list[T]]
        A list of sub-sequences.

    Examples
    --------
    >>> random_split("abcde", [2, 3])
    [['b', 'c'], ['a', 'd', 'e']]
    """

    if sum(lengths) > len(sequence):
        raise ValueError("Not enough things to split")

    shuffle = numpy.random.RandomState(seed=seed).permutation(len(sequence))
    ptr = [0, *numpy.cumsum(lengths)]

    return [
        [sequence[i] for i in shuffle[ptr[n] : ptr[n + 1]]]
        for n in range(len(lengths))
    ]


def all_equal(iterable: Iterable[T]) -> bool:
    """
    Check if all elements in an iterable are the same. If the
    iterable is empty, return `False`.

    Returns
    -------
    bool
        Whether all elements in `iterable` are the same.
    """
    iterator = iter(iterable)
    try:
        first = next(iterator)
    except StopIteration:
        return False
    return all(first == x for x in iterator)


def groups_of(
    size: int,
    things: Iterable[T],
    drop_last: bool = False,
) -> Iterator[list[T]]:
    """
    Split an iterable into groups of a specified size.

    Parameters
    ----------
    size : int
        The size of each group.
    things : Iterable[T]
        The iterable to split into groups.
    drop_last : bool, default=False
        If True, drop the last group if it's smaller than the specified size.
        If False, yield the last group even if incomplete.

    Returns
    -------
    Iterator[list[T]]
        An iterator yielding lists of items, where each list has length `size`
        (except possibly the last one if `drop_last=False`).

    Examples
    --------
    >>> for group in groups_of(5, range(12)):
    ...     print(group)
    [0, 1, 2, 3, 4]
    [5, 6, 7, 8, 9]
    [10, 11]

    >>> list(groups_of(2, range(5), drop_last=True))
    [[0, 1], [2, 3]]
    """
    batch = []
    for thing in things:
        batch.append(thing)
        if len(batch) == size:
            yield batch
            batch = []

    if batch and not drop_last:
        yield batch


def angle_spanned_by(v1: torch.Tensor, v2: torch.Tensor):
    """
    Calculate angles between corresponding vectors in two batches.

    Parameters
    ----------
    v1
        First batch of vectors, shape (N, 3)
    v2
        Second batch of vectors, shape (N, 3)

    Returns
    -------
    torch.Tensor
        Angles in radians, shape (N,)
    """
    # Compute dot product
    dot_product = torch.sum(v1 * v2, dim=1)

    # Compute magnitudes
    v1_mag = torch.linalg.vector_norm(v1, dim=1)
    v2_mag = torch.linalg.vector_norm(v2, dim=1)

    # Compute cosine of angle, add small epsilon to prevent division by zero
    cos_angle = dot_product / (v1_mag * v2_mag + 1e-8)

    # Clamp cosine values to handle numerical instabilities
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)

    # Compute angle using arccos
    return torch.arccos(cos_angle)
