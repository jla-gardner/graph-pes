from __future__ import annotations

from typing import Union

import torch
from e3nn import o3


class LinearReadOut(o3.Linear):
    def __init__(self, input_irreps: str | o3.Irreps, output_irrep: str = "0e"):
        super().__init__(input_irreps, f"1x{output_irrep}")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)


def _get_activation(name: str) -> torch.nn.Module:
    try:
        return getattr(torch.nn, name)()
    except AttributeError:
        raise ValueError(f"Unknown activation function: {name}") from None


class NonLinearReadOut(torch.nn.Sequential):
    """
    Non-linear readout layer for equivariant neural networks.

    This class implements a non-linear readout layer that takes input features
    with arbitrary irreps and produces a scalar output. It uses a linear layer
    followed by an activation function and another linear layer to produce
    the final scalar output.

    Parameters
    ----------
    input_irreps : str
        The irreps of the input features.
    hidden_dim : int, optional
        The dimension of the hidden layer. If None,
        it defaults to the number of scalar (0e) irreps in the input.
    activation : str or torch.nn.Module, optional
        The activation function to use. Can be specified as a string
        (e.g., 'ReLU', 'SiLU') or as a torch.nn.Module. Defaults to SiLU.
    """

    def __init__(
        self,
        input_irreps: str | o3.Irreps,
        output_irrep: str = "0e",
        hidden_dim: int | None = None,
        activation: str | torch.nn.Module = "SiLU",
    ):
        hidden_dim = (
            o3.Irreps(input_irreps).count(o3.Irrep(output_irrep))
            if hidden_dim is None
            else hidden_dim
        )

        if isinstance(activation, str):
            activation = _get_activation(activation)
        elif not isinstance(activation, torch.nn.Module):
            raise ValueError("activation must be a string or a torch.nn.Module")

        super().__init__(
            o3.Linear(input_irreps, f"{hidden_dim}x{output_irrep}"),
            activation,
            o3.Linear(f"{hidden_dim}x{output_irrep}", f"1x{output_irrep}"),
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


ReadOut = Union[LinearReadOut, NonLinearReadOut]
