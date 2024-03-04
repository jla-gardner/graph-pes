from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch
import torch.nn as nn
from ase.data import chemical_symbols
from torch import Tensor

from .util import MAX_Z, pairs

# TODO support access to .data via property and setter on ConstrainedParameter
# / cleanup the ConstrainedParameter class


class MLP(nn.Module):
    """
    A multi-layer perceptron model, alternating linear layers and activations.

    Parameters
    ----------
    layers: List[int]
        The number of nodes in each layer.
    activation: str or nn.Module
        The activation function to use: either a named activation function
        from `torch.nn`, or a `torch.nn.Module` instance.
    activate_last: bool
        Whether to apply the activation function after the last linear layer.
    bias: bool
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
        return f"MLP({layers}, activation={self.activation})"


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(x) - self.shift


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


class Product(nn.Module):
    def __init__(self, components: list[nn.Module]):
        super().__init__()
        self.components = nn.ModuleList(components)

    def forward(self, x):
        out = 1
        for component in self.components:
            out = out * component(x)
        return out


class PerSpeciesParameter(torch.nn.Parameter):
    """
    A parameter that is indexed by unqiue atomic numbers.

    Lazily keeps track of which atomic numbers have been accessed
    so as to give accurate counts of trainable parameters.

    Instantiate using the `PerSpeciesParameter.of_dim` class method.

    Examples
    --------
    >>> import torch
    >>> from graph_pes.nn import PerSpeciesParameter
    >>> param = PerSpeciesParameter.of_dim(10)
    >>> # do some computation involving some atomic numbers
    >>> ...
    >>> param[1], param[6]  # access the parameters for H and C at some point
    >>> param.numel()  # get the number of trainable parameters
    20
    >>> param
    PerSpeciesParameter({
         Z : (10,)
         3 : [ 0.6450, -0.1432, -0.5998,  ...,  0.6023, -0.3033,  0.3528],
        12 : [-0.2076, -1.3792, -0.0791,  ..., -1.5645,  1.5325, -0.7080]
    }, requires_grad=True)
    """

    # include these args for type hint reasons, class construction
    # actually takes place in __new__, as defined in PyTorch
    def __init__(self, data: Tensor, requires_grad: bool = True):
        super().__init__()

        # we lazily keep track of which atomic numbers have been accessed
        # so as to provide a correct number of trainable weights
        self._accessed_Zs = set()

    @classmethod
    def of_dim(
        cls,
        dim: int,
        requires_grad: bool = True,
        generator: Callable[[tuple[int, int]], Tensor]
        | int
        | float
        | None = None,
    ):
        """
        Create a `PerSpeciesParameter` of the given dimension.

        Parameters
        ----------
        dim
            The size of the parameter.
        generator
            The generator to use to create the parameter. Defaults to
            `torch.randn`.
        requires_grad
            Whether the parameter should be trainable.
        """
        if isinstance(generator, (int, float)):
            data = torch.full((MAX_Z, dim), generator).float()
        elif generator is None:
            data = torch.randn(MAX_Z, dim)
        else:
            data = generator((MAX_Z, dim))
        return PerSpeciesParameter(data=data, requires_grad=requires_grad)

    def __getitem__(self, Z: int | Tensor) -> Tensor:
        """
        Index the values corresponding to the given atomic number/s.

        Parameters
        ----------
        Z
            The atomic number/s of the parameter to get.
        """

        if isinstance(Z, int):
            self._accessed_Zs.add(Z)
        else:
            for Z_i in torch.unique(Z):
                self._accessed_Zs.add(Z_i.item())
        return super().__getitem__(Z)

    def __setitem__(self, Z: int | Tensor, value: Tensor):
        """
        Index the values corresponding to the given atomic number/s.

        Parameters
        ----------
        Z
            The atomic number/s of the parameter to get.
        """

        if isinstance(Z, int):
            self._accessed_Zs.add(Z)
        else:
            for Z_i in torch.unique(Z):
                self._accessed_Zs.add(Z_i.item())
        return super().__setitem__(Z, value)

    def numel(self) -> int:
        """Get the number of trainable parameters."""

        return sum(self[Z].numel() for Z in self._accessed_Zs)

    def __repr__(self) -> str:
        if len(self._accessed_Zs) == 0:
            return (
                f"PerSpeciesParameter(dim={tuple(self.shape[1:])}, "
                f"requires_grad={self.requires_grad})"
            )

        torch.set_printoptions(threshold=3)
        Zs = sorted(self._accessed_Zs)
        with torch.no_grad():
            matrix = str(self.data[Zs])
        matrix = matrix[8:-2]  # remove "tensor([" and "])"

        # generate a dictionary of atomic numbers to
        # their (nicely formatted) values
        lines = []
        for z, values in zip(Zs, matrix.split("\n")):
            lines.append(f"{chemical_symbols[z]:>2} : {values.strip()}")

        lines = "\n   ".join(lines)

        torch.set_printoptions(profile="default")

        return f"""\
PerSpeciesParameter({{
    {lines}
}}, dim={tuple(self.shape[1:])}, requires_grad={self.requires_grad})"""


class PerSpeciesEmbedding(torch.nn.Module):
    """
    A per-speices equivalent of `torch.nn.Embedding`.

    Parameters
    ----------
    dim
        The dimension of the embedding.

    Examples
    --------
    >>> embedding = PerSpeciesEmbedding(10)
    >>> embedding(graph.Z)  # graph.Z is a tensor of atomic numbers
    <tensor of shape (graph.n_atoms, 10)>
    """

    def __init__(self, dim: int):
        super().__init__()
        self._embeddings = PerSpeciesParameter.of_dim(dim)

    def forward(self, Z: Tensor) -> Tensor:
        return self._embeddings[Z]


class ConstrainedParameter(nn.Module, ABC):
    """
    Abstract base class for constrained parameters.

    Implementations should override the `_constrained_value` property.

    Parameters
    ----------
    x
        The initial value of the parameter.
    requires_grad
        Whether the parameter should be trainable.
    """

    def __init__(self, x: torch.Tensor, requires_grad: bool = True):
        super().__init__()
        self._parameter = nn.Parameter(x, requires_grad)

    @property
    @abstractmethod
    def constrained_value(self) -> torch.Tensor:
        """generate the constrained value"""

    def _do_math(self, other, function, rev=False):
        if isinstance(other, ConstrainedParameter):
            other_value = other.constrained_value
        else:
            other_value = other

        if rev:
            return function(other_value, self.constrained_value)
        else:
            return function(self.constrained_value, other_value)

    def __add__(self, other):
        return self._do_math(other, torch.add)

    def __radd__(self, other):
        return self._do_math(other, torch.add, rev=True)

    def __sub__(self, other):
        return self._do_math(other, torch.sub)

    def __rsub__(self, other):
        return self._do_math(other, torch.sub, rev=True)

    def __mul__(self, other):
        return self._do_math(other, torch.mul)

    def __rmul__(self, other):
        return self._do_math(other, torch.mul, rev=True)

    def __truediv__(self, other):
        return self._do_math(other, torch.true_divide)

    def __rtruediv__(self, other):
        return self._do_math(other, torch.true_divide, rev=True)

    def __pow__(self, other):
        return self._do_math(other, torch.pow)

    def __rpow__(self, other):
        return self._do_math(other, torch.pow, rev=True)

    def log(self):
        return torch.log(self.constrained_value)

    def sqrt(self):
        return torch.sqrt(self.constrained_value)

    def __repr__(self):
        t = self.constrained_value
        if t.numel() == 1:
            return f"{self.__class__.__name__}({t.item():.4f})"
        return f"{self.__class__.__name__}({t})"

    def __neg__(self):
        return self._do_math(0, torch.sub, rev=True)


class PositiveParameter(ConstrainedParameter):
    """
    Drop-in replacement for :class:`torch.nn.Parameter`. An internal
    exponentiation ensures that the parameter is always positive.

    Parameters
    ----------
    x
        The initial value of the parameter. Must be positive.
    requires_grad
        Whether the parameter should be trainable.
    """

    def __init__(self, x: torch.Tensor | float, requires_grad: bool = True):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        super().__init__(torch.log(x), requires_grad)

    @property
    def constrained_value(self):
        return torch.exp(self._parameter)


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


def left_aligned_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Assume:
    x.shape: (n, ...)
    y.shape: (n, )

    We broadcast y to the left of x and apply the operation op.
    """
    if x.dim() == 1 or x.dim() == 0:
        return x + y
    # add a fake dimension to x to make it (n, 1, ...)
    x = x.unsqueeze(1)
    # transpose x to make it (1, ..., n)
    x = x.transpose(0, -1)
    # apply the operation
    result = x + y  # shape: (1, ..., n)
    # transpose back to the original shape
    result = result.transpose(0, -1)
    # remove the fake dimension
    return result.squeeze(1)


def left_aligned_sub(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Assume:
    x.shape: (n, ...)
    y.shape: (n, )

    We broadcast y to the left of x and apply the operation op.
    """
    if x.dim() == 1 or x.dim() == 0:
        return x - y
    x = x.unsqueeze(1)
    x = x.transpose(0, -1)
    result = x - y  # shape: (1, ..., n)
    result = result.transpose(0, -1)
    return result.squeeze(1)


def left_aligned_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Assume:
    x.shape: (n, ...)
    y.shape: (n, )

    We broadcast y to the left of x and apply the operation op.
    """
    if x.dim() == 1 or x.dim() == 0:
        return x * y
    x = x.unsqueeze(1)
    x = x.transpose(0, -1)
    result = x * y  # shape: (1, ..., n)
    result = result.transpose(0, -1)
    return result.squeeze(1)


def left_aligned_div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Assume:
    x.shape: (n, ...)
    y.shape: (n, )

    We broadcast y to the left of x and apply the operation op.
    """
    if x.dim() == 1 or x.dim() == 0:
        return x / y
    x = x.unsqueeze(1)
    x = x.transpose(0, -1)
    result = x / y  # shape: (1, ..., n)
    result = result.transpose(0, -1)
    return result.squeeze(1)
