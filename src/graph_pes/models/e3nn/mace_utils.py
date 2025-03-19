from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence, cast

import opt_einsum_fx
import torch
import torch.fx
from e3nn import o3
from e3nn.util.codegen import CodeGenMixin

from graph_pes.models.e3nn.utils import to_full_irreps
from graph_pes.utils.nn import UniformModuleList

BATCH_DIM_EG = 20
SPARE = "wxvnzrtyuops"


def _get_optimised_summation(
    _lambda: Callable,
    example_input_sizes: list[Sequence[int]],
) -> torch.fx.GraphModule:
    inputs = tuple(torch.randn(size) for size in example_input_sizes)
    opt = opt_einsum_fx.optimize_einsums_full(
        model=torch.fx.symbolic_trace(_lambda), example_inputs=inputs
    )
    return cast(torch.fx.GraphModule, opt)


@dataclass
class ContractionConfig:
    num_features: int
    irrep_s_in: list[o3.Irrep]
    irrep_out: o3.Irrep
    n_node_attributes: int

    def __repr__(self) -> str:
        _in = to_full_irreps(self.num_features, self.irrep_s_in)
        _out = to_full_irreps(self.num_features, [self.irrep_out])
        return f"{_in} + {self.n_node_attributes}x0e -> {_out}"


class InitialContraction(CodeGenMixin, torch.nn.Module):
    def __init__(self, config: ContractionConfig, correlation: int):
        super().__init__()

        # U is of shape (X, (Y,) * correlation, Z)
        # where X = irrep_out.dim
        self.register_buffer(
            "U", get_U_matrix(config.irrep_s_in, config.irrep_out, correlation)
        )
        Y = self.U.size()[-2]
        Z = self.U.size()[-1]

        self.W = torch.nn.Parameter(
            torch.randn(config.n_node_attributes, Z, config.num_features) / Z
        )

        # the contraction is a summation that takes 4 inputs:
        # U, W, node_embeddings, node_attributes
        instruction = "ik,ekc,bci,be -> bc"
        # we parallelise over the "spare" dimensions in U
        spare_dims = SPARE[:correlation]
        instruction = f"{spare_dims}{instruction}{spare_dims}"

        self.summation = _get_optimised_summation(
            lambda x, y, z, w: torch.einsum(instruction, x, y, z, w),
            [
                (self.U.shape),
                (config.n_node_attributes, Z, config.num_features),
                (BATCH_DIM_EG, config.num_features, Y),
                (BATCH_DIM_EG, config.n_node_attributes),
            ],
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        node_attributes: torch.Tensor,
    ) -> torch.Tensor:
        return self.summation(self.U, self.W, node_embeddings, node_attributes)

    def __call__(
        self,
        node_embeddings: torch.Tensor,
        node_attributes: torch.Tensor,
    ) -> torch.Tensor:
        return super().__call__(node_embeddings, node_attributes)


class FollowingWeightContraction(CodeGenMixin, torch.nn.Module):
    def __init__(self, config: ContractionConfig, correlation: int):
        super().__init__()

        # as above, U is of shape (X, (Y,) * correlation, Z)
        self.register_buffer(
            "U", get_U_matrix(config.irrep_s_in, config.irrep_out, correlation)
        )
        Z = self.U.size()[-1]

        self.W = torch.nn.Parameter(
            torch.randn(config.n_node_attributes, Z, config.num_features) / Z
        )

        # this contraction acts on U, W and the node attributes
        # spare_dims = get_spare_dims(correlation, irrep_out)
        spare_dims = SPARE[: correlation + 1]
        instruction = f"{spare_dims}k,ekc,be->bc{spare_dims}"

        self.summation = _get_optimised_summation(
            lambda x, y, z: torch.einsum(instruction, x, y, z),
            [
                (self.U.shape),
                (config.n_node_attributes, Z, config.num_features),
                (BATCH_DIM_EG, config.n_node_attributes),
            ],
        )

    def forward(
        self,
        node_attributes: torch.Tensor,
    ) -> torch.Tensor:
        return self.summation(self.U, self.W, node_attributes)

    # for mypy
    def __call__(
        self,
        node_attributes: torch.Tensor,
    ) -> torch.Tensor:
        return super().__call__(node_attributes)


class FeatureContraction(CodeGenMixin, torch.nn.Module):
    def __init__(self, config: ContractionConfig, correlation: int):
        super().__init__()

        # as above, U is of shape (X, (Y,) * correlation, Z)
        U = get_U_matrix(config.irrep_s_in, config.irrep_out, correlation)
        X = config.irrep_out.dim
        Y = U.size()[-2]

        # this contraction acts on the output of a weight contraction
        # and the node features
        spare_dims = SPARE[:correlation]
        instruction = f"bc{spare_dims}i,bci->bc{spare_dims}"

        self.summation = _get_optimised_summation(
            lambda x, y: torch.einsum(instruction, x, y),
            [
                [BATCH_DIM_EG, config.num_features, X] + [Y] * correlation,
                (BATCH_DIM_EG, config.num_features, Y),
            ],
        )

    def forward(
        self,
        x: torch.Tensor,
        node_attributes: torch.Tensor,
    ) -> torch.Tensor:
        return self.summation(x, node_attributes)

    # for mypy
    def __call__(
        self,
        x: torch.Tensor,
        node_attributes: torch.Tensor,
    ) -> torch.Tensor:
        return super().__call__(x, node_attributes)


class Contraction(CodeGenMixin, torch.nn.Module):
    def __init__(self, config: ContractionConfig, correlation: int):
        super().__init__()

        self.config = config
        self.correlation = correlation
        self.initial_contraction = InitialContraction(config, correlation)

        self.weight_contractions = UniformModuleList(
            FollowingWeightContraction(config, j)
            for j in reversed(range(1, correlation))
        )

        self.feature_contractions = UniformModuleList(
            FeatureContraction(config, j)
            for j in reversed(range(1, correlation))
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        node_attributes: torch.Tensor,
    ) -> torch.Tensor:
        output = self.initial_contraction(node_embeddings, node_attributes)

        for weight_contraction, feature_contraction in zip(
            self.weight_contractions, self.feature_contractions
        ):
            output = weight_contraction(node_attributes) + output
            output = feature_contraction(output, node_embeddings)

        return output.reshape(output.shape[0], -1)

    # for mypy
    def __call__(
        self,
        node_embeddings: torch.Tensor,
        node_attributes: torch.Tensor,
    ) -> torch.Tensor:
        return super().__call__(node_embeddings, node_attributes)

    def __repr__(self) -> str:
        weights = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"{self.__class__.__name__}({self.config}, "
            f"correlation={self.correlation}, weights={weights})"
        )


_U_cache_sparse: dict[tuple[str, str, int], torch.Tensor] = torch.load(
    Path(__file__).parent / "_high_order_CG_coeff.pt",
    weights_only=True,
)
"""
A pre-computed look-up table for the U matrices used in MACE Contractions.

Keys are tuples of the form ``(in_irreps, out_irreps, correlation)``, where
``in_irreps`` and ``out_irreps`` are strings formed by concatenating the string
representations of the irreducible representations, and ``correlation`` is an
integer.
"""


def get_U_matrix(
    in_irreps: list[o3.Irrep],
    out_irreps: o3.Irrep,
    correlation: int,
) -> torch.Tensor:
    key = (
        " ".join(map(str, in_irreps)),
        str(out_irreps),
        correlation,
    )
    l_max = correlation * int(o3.Irrep(in_irreps[-1]).l)
    if l_max > 11:
        raise ValueError(
            f"l_max > 11 (you supplied {l_max=}) is not supported by e3nn."
        )
    if key not in _U_cache_sparse:
        raise ValueError(
            f"U_matrix for {key} not found in cache - this is surprising! "
            "Please raise an issue at https://github.com/jla-gardner/graph-pes "
            "so we can fix this."
        )

    return _U_cache_sparse[key].to_dense().to(dtype=torch.get_default_dtype())


class UnflattenIrreps(torch.nn.Module):
    """
    Unflattens a tensor of irreps with uniform multiplicity,
    e.g. (N, 16x0e + 16x1o) -> (N, 16, 1x0e + 1x1o)
    """

    def __init__(self, irreps: list[o3.Irrep], channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.irreps = irreps
        self.dims = [ir.dim for ir in irreps]

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        idx = 0
        output = []

        # iterate over the flat tensor, and pull out
        # each channel x irrep
        # e.g.   (N, 16x0e + 16x1o) -> (N, 16, 1x0e + 1x1o)
        # equiv: (N,      16x(1+3)) -> (N, 16, 1+3)
        for dim in self.dims:
            field = tensor[:, idx : idx + self.channels * dim]
            idx += self.channels * dim
            field = field.reshape(-1, self.channels, dim)
            output.append(field)

        return torch.cat(output, dim=-1)

    def __repr__(self) -> str:
        _in = "(N, "
        _in += "+".join([f"{self.channels}x{ir}" for ir in self.irreps])
        _in += ")"

        _out = f"(N, {self.channels}, "
        _out += "+".join([f"1x{ir}" for ir in self.irreps])
        _out += ")"

        return f"{self.__class__.__name__}({_in} -> {_out})"


def parse_irreps(irreps: str | list[str]) -> list[o3.Irrep]:
    if isinstance(irreps, str):
        try:
            return [ir for _, ir in o3.Irreps(irreps)]
        except ValueError:
            raise ValueError(
                f"Unable to parse {irreps} as irreps. "
                "Expected a string of the form '0e + 1o'"
            ) from None
    try:
        return [o3.Irrep(ir) for ir in irreps]
    except ValueError:
        raise ValueError(
            f"Unable to parse {irreps} as irreps. "
            "Expected a list of strings of the form ['0e', '1o']"
        ) from None
