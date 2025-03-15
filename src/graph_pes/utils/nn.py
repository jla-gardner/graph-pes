from __future__ import annotations

from functools import reduce
from typing import (
    Any,
    Generic,
    Iterable,
    Iterator,
    Sequence,
    TypedDict,
    TypeVar,
)

import torch
import torch.nn
from ase.data import atomic_numbers, chemical_symbols, covalent_radii
from torch import Tensor

from graph_pes.utils.misc import left_aligned_mul

from .misc import (
    MAX_Z,
    pairs,
    to_significant_figures,
    uniform_repr,
)

V = TypeVar("V", bound=torch.nn.Module)


class UniformModuleDict(torch.nn.ModuleDict, Generic[V]):
    """
    A :class:`torch.nn.ModuleDict` sub-class for cases where
    the values are all of the same type.

    Examples
    --------
    >>> from graph_pes.utils.nn import UniformModuleDict
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

    def __setitem__(self, key: str, module: V) -> None:  # type: ignore
        super().__setitem__(key, module)

    def pop(self, key: str) -> V:
        return super().pop(key)  # type: ignore


class UniformModuleList(torch.nn.ModuleList, Sequence[V]):
    """
    A :class:`torch.nn.ModuleList` sub-class for cases where
    the values are all of the same type.

    Examples
    --------
    >>> from graph_pes.utils.nn import UniformModuleList
    >>> from torch.nn import Linear
    >>> linear_list = UniformModuleList(Linear(10, 5), Linear(5, 1))
    """

    def __init__(self, modules: Iterable[V]):
        super().__init__(modules)

    def __getitem__(self, idx: int) -> V:  # type: ignore
        return super().__getitem__(idx)  # type: ignore

    def __setitem__(self, idx: int, value: V) -> None:  # type: ignore
        super().__setitem__(idx, value)

    def append(self, module: V) -> None:  # type: ignore
        super().append(module)

    def extend(self, modules: Iterable[V]) -> None:  # type: ignore
        super().extend(modules)

    def insert(self, idx: int, module: V) -> None:  # type: ignore
        super().insert(idx, module)

    def pop(self, idx: int) -> V:  # type: ignore
        return super().pop(idx)  # type: ignore

    def __iter__(self) -> Iterator[V]:
        return super().__iter__()  # type: ignore


class MLPConfig(TypedDict):
    """
    A TypedDict helper class for configuring an :class:`MLP`.

    Examples
    --------
    Specify this in a config file:

    .. code-block:: yaml

        mlp:
            hidden_depth: 3
            hidden_features: 64
            activation: SiLU
    """

    hidden_depth: int
    """The number of hidden layers in the MLP."""
    hidden_features: int
    """The number of features in the hidden layers."""
    activation: str
    """The activation function to use."""


class MLP(torch.nn.Module):
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
    >>> from graph_pes.utils.nn import MLP
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
        activation: str | torch.nn.Module = "CELU",
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

        self.linear_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(_in, _out, bias=bias)
                for _in, _out in pairs(layers)
            ]
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

    @classmethod
    def from_config(
        cls,
        config: MLPConfig,
        input_features: int,
        output_features: int,
        bias: bool = True,
    ) -> MLP:
        """
        Create an :class:`MLP` from a configuration.
        """
        return cls(
            layers=[input_features]
            + [config["hidden_features"]] * config["hidden_depth"]
            + [output_features],
            activation=config["activation"],
            bias=bias,
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
    """
    A subclass of :class:`torch.nn.Parameter` that is indexed by atomic
    number/s. Crucially, this subclass overrides the :meth:`numel` method,
    for accurately counting the number of relevant and learnable parameters.

    Examples
    --------
    Imagine the case where you have a model parameter with a value for each
    element in the periodic table. If you only train the model on a dataset
    containing a few elements, you don't want to count the total number of
    parameters, as this will be unnecessarily large.

    >>> # don't do this!
    >>> per_element_parameter = torch.nn.Parameter(torch.randn(119))
    >>> per_element_parameter.numel()
    119
    >>> per_element_parameter
    Parameter containing:
    tensor([ 1.2838e-01, -1.4507e+00,  1.3465e-01, -9.5786e-01, ...,
            -1.3329e+00, -1.5515e+00,  2.1106e+00, -9.7268e-01],
       requires_grad=True)

    >>> # do this instead
    >>> per_element_paramter = PerElementParameter.of_shape((1,))
    >>> per_element_parameter.register_elements([1, 6, 8])
    >>> per_element_parameter.numel()
    3
    >>> per_element_parameter
    PerElementParameter({'O': -0.278, 'H': 0.157, 'C': -0.0379}, trainable=True)

    ``graph-pes-train`` automatically registers all elements that a model
    encounters during training, so you rarely need to call
    :meth:`register_elements` yourself.
    """

    def __new__(
        cls, data: Tensor, requires_grad: bool = True
    ) -> PerElementParameter:
        pep = super().__new__(cls, data, requires_grad=requires_grad)
        pep._is_per_element_param = True  # type: ignore
        return pep  # type: ignore

    def __init__(self, data: Tensor, requires_grad: bool = True):
        super().__init__()
        # set extra state
        self._accessed_Zs: set[int] = set()
        # set this to an arbitrary value: this gets updated post-init
        self._index_dims: int = 1

    def register_elements(self, Zs: Iterable[int]) -> None:
        """
        Register the elements that are relevant for the parameter.

        This is typically only used internally - you shouldn't call this
        yourself in any of your model definitions.
        """
        self._accessed_Zs.update(sorted(Zs))

    @classmethod
    def of_shape(
        cls,
        shape: tuple[int, ...] = (),
        index_dims: int = 1,
        default_value: float | None = None,
        requires_grad: bool = True,
    ) -> PerElementParameter:
        """
        Create a :class:`PerElementParameter` with a given shape for each
        element in the periodic table.

        Parameters
        ----------
        shape
            The shape of the parameter for each element.
        index_dims
            The number of dimensions to index by.
        default_value
            The value to initialise the parameter with. If ``None``, the
            parameter is initialised with random values.
        requires_grad
            Whether the parameter should be learnable.

        Returns
        -------
        PerElementParameter
            The created parameter.

        Examples
        --------
        Create a parameter intended to be indexed by a single atomic number,
        i.e. ``pep[Z]``:

        >>> PerElementParameter.of_shape((3,)).shape
        torch.Size([119, 3])
        >>> PerElementParameter.of_shape((3, 4)).shape
        torch.Size([119, 3, 4])

        Create a parameter intended to be indexed by two atomic numbers, i.e.
        ``pep[Z1, Z2]``:

        >>> PerElementParameter.of_shape((3,), index_dims=2).shape
        torch.Size([119, 119, 3])
        """
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
        """
        Create a :class:`PerElementParameter` containing a single value for
        each element in the periodic table from a dictionary of values.

        Parameters
        ----------
        requires_grad
            Whether the parameter should be learnable.
        default_value
            The value to initialise the parameter with. If ``None``, the
            parameter is initialised with random values.
        values
            A dictionary of values, indexed by element symbol.

        Returns
        -------
        PerElementParameter
            The created parameter.

        Examples
        --------
        >>> from graph_pes.utils.nn import PerElementParameter
        >>> pep = PerElementParameter.from_dict(H=1.0, O=2.0)
        >>> pep.register_elements([1, 6, 8])
        >>> pep
        PerElementParameter({'H': 1.0, 'C': 0.0, 'O': 2.0}, trainable=True)
        """
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
        """
        Alias for ``PerElementParameter.of_shape((length,), **kwargs)``.
        """
        return PerElementParameter.of_shape(
            (length,), index_dims, default_value, requires_grad
        )

    @classmethod
    @torch.no_grad()
    def covalent_radii(
        cls,
        scaling_factor: float = 1.0,
    ) -> PerElementParameter:
        """
        Create a :class:`PerElementParameter` containing the covalent radii of
        each element in the periodic table.
        """
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
    def _repr(
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
                row: list[str | float] = [chemical_symbols[col_Z]]
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

    def __repr__(self, *, tensor_contents=None):
        return self._repr()


def _rebuild_per_element_parameter(data, requires_grad, state):
    psp = PerElementParameter(data, requires_grad)
    psp._accessed_Zs = state["_accessed_Zs"]
    psp._index_dims = state["_index_dims"]
    return psp


class PerElementEmbedding(torch.nn.Module):
    """
    A per-element equivalent of :class:`torch.nn.Embedding`.

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

    def dim(self) -> int:
        return self._embeddings.shape[1]

    def __repr__(self) -> str:
        Zs = sorted(self._embeddings._accessed_Zs)
        return uniform_repr(
            self.__class__.__name__,
            dim=self._embeddings.shape[1],
            elements=[chemical_symbols[Z] for Z in Zs],
        )

    def __call__(self, Z: Tensor) -> Tensor:
        return super().__call__(Z)


class HaddamardProduct(torch.nn.Module):
    def __init__(
        self, *components: torch.nn.Module, left_aligned: bool = False
    ):
        super().__init__()
        self.components: list[torch.nn.Module] = torch.nn.ModuleList(components)  # type: ignore
        self.left_aligned = left_aligned

    def forward(self, x):
        out = torch.scalar_tensor(1)
        for component in self.components:
            if self.left_aligned:
                out = left_aligned_mul(out, component(x))
            else:
                out = out * component(x)
        return out


def learnable_parameters(module: torch.nn.Module) -> int:
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

        # make this a local variable to keep torchscript happy
        self.chemical_symbols: list[str] = chemical_symbols

    def forward(self, Z: Tensor) -> Tensor:
        internal_idx = self.Z_to_idx[Z]

        with torch.no_grad():
            if (internal_idx == 1234).any():
                unknown_Z = torch.unique(Z[internal_idx == 1234])
                unknown_symbols = [self.chemical_symbols[Z] for Z in unknown_Z]

                raise ValueError(
                    f"Unknown elements: {unknown_symbols}. "
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
