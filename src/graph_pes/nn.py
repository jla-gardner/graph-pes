from __future__ import annotations

from functools import reduce
from typing import Any, Generic, Iterable, TypeVar

import torch
import torch.nn as nn
from ase.data import atomic_numbers, chemical_symbols, covalent_radii
from torch import Tensor

from .util import MAX_Z, pairs, to_significant_figures, uniform_repr

V = TypeVar("V", bound=nn.Module)


class UniformModuleDict(nn.ModuleDict, Generic[V]):
    """
    A :class:`torch.nn.ModuleDict` sub-class for cases where
    the values are all of the same type.

    Examples
    --------
    >>> from graph_pes.nn import UniformModuleDict
    >>> from torch.nn import Linear
    >>> linear_dict = UniformModuleDict(a=Linear(10, 5), b=Linear(5, 1))
    """

    def __init__(self, **modules: V):
        super().__init__(modules)

    def values(self) -> Iterable[V]:
        return super().values()  # type: ignore

    def items(self) -> Iterable[tuple[str, V]]:
        return super().items()  # type: ignore

    def __getitem__(self, key: str) -> V:
        return super().__getitem__(key)  # type: ignore

    def __setitem__(self, key: str, value: V) -> None:
        super().__setitem__(key, value)

    def pop(self, key: str) -> V:
        return super().pop(key)  # type: ignore


class MLP(nn.Module):
    """
    A multi-layer perceptron model, alternating linear layers and activations.

    Parameters
    ----------
    layers
        The number of nodes in each layer.
    activation
        The activation function to use: either a named activation function
        from `torch.nn`, or a `torch.nn.Module` instance.
    activate_last
        Whether to apply the activation function after the last linear layer.
    bias
        Whether to include bias terms in the linear layers.

    Examples
    --------
    >>> import torch
    >>> from graph_pes.nn import MLP
    >>> model = MLP([10, 5, 1])
    >>> model
    MLP(10 → 5 → 1, activation=CELU())
    >>> MLP([10, 5, 1], activation=torch.nn.ReLU())
    MLP(10 → 5 → 1, activation=ReLU())
    >>> MLP([10, 5, 1], activation="Tanh")
    MLP(10 → 5 → 1, activation=Tanh())
    """

    def __init__(
        self,
        layers: list[int],
        activation: str | nn.Module = "CELU",
        activate_last: bool = False,
        bias: bool = True,
    ):
        super().__init__()

        self.activation = (
            parse_activation(activation)
            if isinstance(activation, str)
            else activation
        )
        self.activate_last = activate_last

        self.linear_layers = nn.ModuleList(
            [nn.Linear(_in, _out, bias=bias) for _in, _out in pairs(layers)]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform a forward pass through the network.

        Parameters
        ----------
        x
            The input to the network.
        """
        for i, linear in enumerate(self.linear_layers):
            x = linear(x)
            last_layer = i == len(self.linear_layers) - 1
            if not last_layer or self.activate_last:
                x = self.activation(x)

        return x

    @property
    def input_size(self):
        """The size of the input to the network."""
        return self.linear_layers[0].in_features

    @property
    def output_size(self):
        """The size of the output of the network."""
        return self.linear_layers[-1].out_features

    @property
    def layer_widths(self):
        """The widths of the layers in the network."""
        inputs = [layer.in_features for layer in self.linear_layers]
        return inputs + [self.output_size]

    def __repr__(self):
        layers = " → ".join(map(str, self.layer_widths))
        return uniform_repr(
            self.__class__.__name__,
            layers,
            activation=self.activation,
            stringify=False,
        )


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(x) - self.shift

    def __repr__(self):
        return uniform_repr(self.__class__.__name__)


def parse_activation(act: str) -> torch.nn.Module:
    """
    Parse a string into a PyTorch activation function.

    Parameters
    ----------
    act
        The activation function to parse.

    Returns
    -------
    torch.nn.Module
        The parsed activation function.
    """
    activation = getattr(torch.nn, act, None)
    if activation is None:
        raise ValueError(f"Activation function {act} not found in `torch.nn`.")
    return activation()


def prod(iterable):
    return reduce(lambda x, y: x * y, iterable, 1)


class PerElementParameter(torch.nn.Parameter):
    def __new__(
        cls, data: Tensor, requires_grad: bool = True
    ) -> PerElementParameter:
        pep = super().__new__(cls, data, requires_grad=requires_grad)
        pep._is_per_element_param = True  # type: ignore[assignment]
        return pep  # type: ignore[return-value]

    def __init__(self, data: Tensor, requires_grad: bool = True):
        super().__init__()
        # set extra state
        self._accessed_Zs = set()
        # set this to an arbitrary value: this gets updated post-init
        self._index_dims: int = 1

    def register_elements(self, Zs: Iterable[int]) -> None:
        self._accessed_Zs.update(sorted(Zs))

    @classmethod
    def of_shape(
        cls,
        shape: tuple[int, ...] = (),
        index_dims: int = 1,
        default_value: float | None = None,
        requires_grad: bool = True,
    ) -> PerElementParameter:
        actual_shape = tuple([MAX_Z + 1] * index_dims) + shape
        if default_value is not None:
            data = torch.full(actual_shape, float(default_value))
        else:
            data = torch.randn(actual_shape)
        psp = PerElementParameter(data, requires_grad=requires_grad)
        psp._index_dims = index_dims
        return psp

    @classmethod
    @torch.no_grad()
    def from_dict(
        cls,
        requires_grad: bool = True,
        default_value: float = 0.0,
        **values: float,
    ) -> PerElementParameter:
        pep = PerElementParameter.of_length(
            1, requires_grad=requires_grad, default_value=default_value
        )
        for element_symbol, value in values.items():
            if element_symbol not in chemical_symbols:
                raise ValueError(f"Unknown element: {element_symbol}")
            Z = chemical_symbols.index(element_symbol)
            pep[Z] = value

        pep.register_elements(atomic_numbers[v] for v in values)

        return pep

    @classmethod
    def of_length(
        cls,
        length: int,
        index_dims: int = 1,
        default_value: float | None = None,
        requires_grad: bool = True,
    ) -> PerElementParameter:
        return PerElementParameter.of_shape(
            (length,), index_dims, default_value, requires_grad
        )

    @classmethod
    @torch.no_grad()
    def covalent_radii(
        cls,
        scaling_factor: float = 1.0,
    ) -> PerElementParameter:
        pep = PerElementParameter.of_length(1, default_value=1.0)
        for Z in range(1, MAX_Z + 1):
            pep[Z] = torch.tensor(covalent_radii[Z]) * scaling_factor
        return pep

    def numel(self) -> int:
        n_elements = len(self._accessed_Zs)
        accessed_parameters = n_elements**self._index_dims
        per_element_size = prod(self.shape[self._index_dims :])
        return accessed_parameters * per_element_size

    # needed for de/serialization
    def __reduce_ex__(self, proto):
        return (
            _rebuild_per_element_parameter,
            (self.data, self.requires_grad, torch._utils._get_obj_state(self)),
        )

    def __instancecheck__(self, instance) -> bool:
        return super().__instancecheck__(instance) or (  # type: ignore[no-untyped-call]
            isinstance(instance, torch.Tensor)
            and getattr(instance, "_is_per_element_param", False)
        )

    @torch.no_grad()
    def __repr__(
        self,
        alias: str | None = None,
        more_info: dict[str, Any] | None = None,
    ) -> str:
        alias = alias or self.__class__.__name__
        more_info = more_info or {}
        if "trainable" not in more_info:
            more_info["trainable"] = self.requires_grad

        if len(self._accessed_Zs) == 0:
            if self._index_dims == 1 and self.shape[1] == 1:
                return uniform_repr(alias, **more_info)

            return uniform_repr(
                alias,
                index_dims=self._index_dims,
                shape=tuple(self.shape[self._index_dims :]),
                **more_info,
            )

        if self._index_dims == 1:
            if self.shape[1] == 1:
                d = {
                    chemical_symbols[Z]: to_significant_figures(self[Z].item())
                    for Z in self._accessed_Zs
                }
                string = f"{alias}({str(d)}, "
                for k, v in more_info.items():
                    string += f"{k}={v}, "
                return string[:-2] + ")"

            elif len(self.shape) == 2:
                d = {
                    chemical_symbols[Z]: self[Z].tolist()
                    for Z in self._accessed_Zs
                }
                string = f"{alias}({str(d)}, "
                for k, v in more_info.items():
                    string += f"{k}={v}, "
                return string[:-2] + ")"

        if self._index_dims == 2 and self.shape[2] == 1:
            columns = []
            columns.append(
                ["Z"] + [chemical_symbols[Z] for Z in self._accessed_Zs]
            )
            for col_Z in self._accessed_Zs:
                row = [col_Z]
                for row_Z in self._accessed_Zs:
                    row.append(
                        to_significant_figures(self[col_Z, row_Z].item())
                    )
                columns.append(row)

            widths = [max(len(str(x)) for x in col) for col in zip(*columns)]
            lines = []
            for row in columns:
                line = ""
                for x, w in zip(row, widths):
                    # right align
                    line += f"{x:>{w}}  "
                lines.append(line)
            table = "\n" + "\n".join(lines)
            return uniform_repr(
                alias,
                table,
                **more_info,
            )

        return uniform_repr(
            alias,
            index_dims=self._index_dims,
            accessed_Zs=sorted(self._accessed_Zs),
            shape=tuple(self.shape[self._index_dims :]),
            **more_info,
        )


def _rebuild_per_element_parameter(data, requires_grad, state):
    psp = PerElementParameter(data, requires_grad)
    psp._accessed_Zs = state["_accessed_Zs"]
    psp._index_dims = state["_index_dims"]
    return psp


class PerElementEmbedding(torch.nn.Module):
    """
    A per-element equivalent of `torch.nn.Embedding`.

    Parameters
    ----------
    dim
        The length of each embedding vector.

    Examples
    --------
    >>> embedding = PerElementEmbedding(10)
    >>> len(graph["atomic_numbers"])  # number of atoms in the graph
    24
    >>> embedding(graph["atomic_numbers"])
    <tensor of shape (24, 10)>
    """

    def __init__(self, dim: int):
        super().__init__()
        self._embeddings = PerElementParameter.of_length(dim)

    def forward(self, Z: Tensor) -> Tensor:
        return self._embeddings[Z]

    def __repr__(self) -> str:
        Zs = sorted(self._embeddings._accessed_Zs)
        return uniform_repr(
            self.__class__.__name__,
            dim=self._embeddings.shape[1],
            elements=[chemical_symbols[Z] for Z in Zs],
        )


class HaddamardProduct(nn.Module):
    def __init__(self, *components: nn.Module, left_aligned: bool = False):
        super().__init__()
        self.components: list[nn.Module] = nn.ModuleList(components)  # type: ignore
        self.left_aligned = left_aligned

    def forward(self, x):
        out = torch.scalar_tensor(1)
        for component in self.components:
            if self.left_aligned:
                out = left_aligned_mul(out, component(x))
            else:
                out = out * component(x)
        return out


# NB: we have to repeat code here somewhat because torchsript doesn't support
# typing for callable


# TODO: sort out this mess
def left_aligned_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Assume:
    x.shape: (n, ...)
    y.shape: (n, )

    We broadcast y to the left of x and add the two tensors elementwise.
    """
    if x.dim() == 1 or x.dim() == 0:
        return x + y
    # add a fake dimension to x to make it (n, 1, ...)
    x = x.unsqueeze(1)
    # transpose x to make it (1, ..., n)
    x = x.transpose(0, -1)
    # apply the operation
    result = x - y  # shape: (1, ..., n)
    # transpose back to the original shape
    result = result.transpose(0, -1)
    # remove the fake dimension
    return result.squeeze(1)


def left_aligned_sub(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1 or x.dim() == 0:
        return x - y
    x = x.unsqueeze(1)
    x = x.transpose(0, -1)
    result = x - y  # shape: (1, ..., n)
    result = result.transpose(0, -1)
    return result.squeeze(1)


def left_aligned_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1 or x.dim() == 0:
        return x * y
    x = x.unsqueeze(1)
    x = x.transpose(0, -1)
    result = x * y  # shape: (1, ..., n)
    result = result.transpose(0, -1)
    return result.squeeze(1)


# TODO: tests for this!
def left_aligned_div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1 or x.dim() == 0:
        return x / y
    x = x.unsqueeze(1)
    x = x.transpose(0, -1)
    result = x / y  # shape: (1, ..., n)
    result = result.transpose(0, -1)
    return result.squeeze(1)


def learnable_parameters(module: nn.Module) -> int:
    """Count the number of **learnable** parameters a module has."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class AtomicOneHot(torch.nn.Module):
    """
    Takes a tensor of atomic numbers Z, and returns a one-hot encoding of
    the atomic numbers.

    Parameters
    ----------
    n_elements
        The total number of expected atomic numbers.
    """

    def __init__(self, elements: list[str]):
        super().__init__()

        self.elements = elements
        self.n_elements = len(elements)

        self.Z_to_idx: Tensor
        self.register_buffer(
            "Z_to_idx",
            # deliberately crazy value to catch errors
            torch.full((MAX_Z + 1,), fill_value=1234),
        )
        for i, symbol in enumerate(elements):
            Z = atomic_numbers[symbol]
            self.Z_to_idx[Z] = i

    def forward(self, Z: Tensor) -> Tensor:
        internal_idx = self.Z_to_idx[Z]

        with torch.no_grad():
            if (internal_idx == 1234).any():
                unknown_Z = torch.unique(Z[internal_idx == 1234])

                raise ValueError(
                    f"Unknown elements: {unknown_Z}. "
                    f"Expected one of {self.elements}"
                )

        return torch.nn.functional.one_hot(
            internal_idx, self.n_elements
        ).float()

    def __repr__(self):
        return uniform_repr(
            self.__class__.__name__,
            elements=self.elements,
        )
