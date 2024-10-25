from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence, Union, cast

import e3nn.util.jit
import opt_einsum_fx
import torch
import torch.fx
from e3nn import o3
from e3nn.util.codegen import CodeGenMixin
from graph_pes.nn import UniformModuleList


class LinearReadOut(o3.Linear):
    """
    Map a set of features with arbitrary irreps to a single irrep with
    a single feature using a linear layer.

    Parameters
    ----------
    input_irreps : str or o3.Irreps
        The irreps of the input features.
    output_irrep : str, optional
        The irrep of the output feature. Defaults to "0e".

    Examples
    --------

    Map an embedding to a scalar output:

    >>> LinearReadOut("16x0e+16x1o+16x2e")
    LinearReadOut(16x0e+16x1o+16x2e -> 1x0e | 16 weights)


    Map an embedding to a vector output:

    >>> LinearReadOut("16x0e+16x1o+16x2e", "1o")
    LinearReadOut(16x0e+16x1o+16x2e -> 1x1o | 16 weights)
    """

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
    output_irrep : str, optional
        The irrep of the output feature. Defaults to "0e".
    hidden_dim : int, optional
        The dimension of the hidden layer. If None,
        it defaults to the number of scalar output irreps in the input.
    activation : str or torch.nn.Module, optional
        The activation function to use. Can be specified as a string
        (e.g., 'ReLU', 'SiLU') or as a torch.nn.Module. Defaults to ``SiLU``
        for even output irreps and ``Tanh`` for odd output irreps.
        **Care must be taken to ensure that the activation is suitable for
        the target irreps!**

    Examples
    --------

    Map an embedding to a scalar output:

    >>> NonLinearReadOut("16x0e+16x1o+16x2e")
    NonLinearReadOut(
      (0): Linear(16x0e+16x1o+16x2e -> 16x0e | 256 weights)
      (1): SiLU()
      (2): Linear(16x0e -> 1x0e | 16 weights)
    )

    Map an embedding to a vector output:

    >>> NonLinearReadOut("16x0e+16x1o+16x2e", "1o")
    NonLinearReadOut(
      (0): Linear(16x0e+16x1o+16x2e -> 16x1o | 256 weights)
      (1): Tanh()
      (2): Linear(16x1o -> 1x1o | 16 weights)
    )
    """

    def __init__(
        self,
        input_irreps: str | o3.Irreps,
        output_irrep: str = "0e",
        hidden_dim: int | None = None,
        activation: str | torch.nn.Module | None = None,
    ):
        if activation is None:
            activation = "SiLU" if "e" in str(output_irrep) else "Tanh"

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


@e3nn.util.jit.compile_mode("script")
class SphericalHarmonics(o3.SphericalHarmonics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return f"SphericalHarmonics(1x1o -> {self.irreps_out})"


def build_limited_tensor_product(
    node_embedding_irreps: o3.Irreps,
    edge_embedding_irreps: o3.Irreps,
    allowed_outputs: list[o3.Irrep],
) -> o3.TensorProduct:
    # we want to build a tensor product that takes the:
    # - node embeddings of each neighbour (node_irreps_in)
    # - spherical-harmonic expansion of the neighbour directions
    #      (o3.Irreps.spherical_harmonics(l_max) = e.g. 1x0e, 1x1o, 1x2e)
    # and generates
    # - message embeddings from each neighbour (node_irreps_out)
    #
    # crucially, rather than using the full tensor product, we limit the
    # output irreps to be of order l_max at most. we do this by defining a
    # sequence of instructions that specify the connectivity between the
    # two input irreps and the output irreps
    #
    # we build this instruction set by naively iterating over all possible
    # combinations of input irreps and spherical-harmonic irreps, and
    # filtering out those that are above the desired order
    #
    # finally, we sort the instructions so that the tensor product generates
    # a tensor where all elements of the same irrep are grouped together
    # this aids normalisation in subsequent operations

    output_irreps = []
    instructions = []

    for i, (channels, ir_in) in enumerate(node_embedding_irreps):
        # the spherical harmonic expansions always have 1 channel per irrep,
        # so we don't care about their channel dimension
        for l, (_, ir_edge) in enumerate(edge_embedding_irreps):
            # get all possible output irreps that this interaction could
            # generate, e.g. 1e x 1e -> 0e + 1e + 2e
            possible_output_irreps = ir_in * ir_edge

            for ir_out in possible_output_irreps:
                # (order, parity) = ir_out
                if ir_out not in allowed_outputs:
                    continue

                # if we want this output from the tensor product, add it to the
                # list of instructions
                k = len(output_irreps)
                output_irreps.append((channels, ir_out))
                # from the i'th irrep of the neighbour embedding
                # and from the l'th irrep of the spherical harmonics
                # to the k'th irrep of the output tensor
                instructions.append((i, l, k, "uvu", True))

    # since many paths can lead to the same output irrep, we sort the
    # instructions so that the tensor product generates tensors in a
    # nice order, e.g. 32x0e + 16x1o, not 16x0e + 16x1o + 16x0e
    output_irreps = o3.Irreps(output_irreps)
    assert isinstance(output_irreps, o3.Irreps)
    output_irreps, permutation, _ = output_irreps.sort()

    # permute the output indexes of the instructions to match the sorted irreps:
    instructions = [
        (i_in1, i_in2, permutation[i_out], mode, train)
        for i_in1, i_in2, i_out, mode, train in instructions
    ]

    return o3.TensorProduct(
        node_embedding_irreps,
        edge_embedding_irreps,
        output_irreps,
        instructions,
        # this tensor product will be parameterised by weights that are learned
        # from neighbour distances, so it has no internal weights
        internal_weights=False,
        shared_weights=False,
    )


BATCH_EXAMPLE = 10
SPARE = "wxvnzrtyuops"


def get_optimised_summation(
    _lambda: Callable,
    example_input_sizes: list[Sequence[int]],
) -> torch.fx.GraphModule:
    inputs = tuple(torch.randn(size) for size in example_input_sizes)
    opt = opt_einsum_fx.optimize_einsums_full(
        model=torch.fx.symbolic_trace(_lambda), example_inputs=inputs
    )
    return cast(torch.fx.GraphModule, opt)


def get_spare_dims(correlation: int, irrep_out: o3.Irrep) -> str:
    n_spare = correlation
    return SPARE[:n_spare]


class InitialContraction(CodeGenMixin, torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        irrep_s_in: list[o3.Irrep],
        irrep_out: o3.Irrep,
        correlation: int,
        n_node_attributes: int,
    ):
        super().__init__()

        # U is of shape (X, (Y,) * correlation, Z)
        # where X = irrep_out.dim
        self.register_buffer(
            "U", get_U_matrix(irrep_s_in, irrep_out, correlation)
        )
        Y = self.U.size()[-2]
        Z = self.U.size()[-1]

        self.W = torch.nn.Parameter(
            torch.randn(n_node_attributes, Z, num_features) / Z
        )

        # the contraction is a summation that takes 4 inputs:
        # U, W, node_embeddings, node_attributes
        instruction = "ik,ekc,bci,be -> bc"
        # we parallelise over the "spare" dimensions in U
        spare_dims = SPARE[:correlation]
        instruction = f"{spare_dims}{instruction}{spare_dims}"

        self.summation = get_optimised_summation(
            lambda x, y, z, w: torch.einsum(instruction, x, y, z, w),
            [
                (self.U.shape),
                (n_node_attributes, Z, num_features),
                (BATCH_EXAMPLE, num_features, Y),
                (BATCH_EXAMPLE, n_node_attributes),
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
    def __init__(
        self,
        num_features: int,
        irrep_s_in: list[o3.Irrep],
        irrep_out: o3.Irrep,
        correlation: int,
        n_node_attributes: int,
    ):
        super().__init__()

        # as above, U is of shape (X, (Y,) * correlation, Z)
        self.register_buffer(
            "U", get_U_matrix(irrep_s_in, irrep_out, correlation)
        )
        Z = self.U.size()[-1]

        self.W = torch.nn.Parameter(
            torch.randn(n_node_attributes, Z, num_features) / Z
        )

        # this contraction acts on U, W and the node attributes
        # spare_dims = get_spare_dims(correlation, irrep_out)
        spare_dims = SPARE[: correlation + 1]
        instruction = f"{spare_dims}k,ekc,be->bc{spare_dims}"

        self.summation = get_optimised_summation(
            lambda x, y, z: torch.einsum(instruction, x, y, z),
            [
                (self.U.shape),
                (n_node_attributes, Z, num_features),
                (BATCH_EXAMPLE, n_node_attributes),
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
    def __init__(
        self,
        num_features: int,
        irrep_s_in: list[o3.Irrep],
        irrep_out: o3.Irrep,
        correlation: int,
        n_node_attributes: int,
    ):
        super().__init__()

        # as above, U is of shape (X, (Y,) * correlation, Z)
        U = get_U_matrix(irrep_s_in, irrep_out, correlation)
        X = irrep_out.dim
        Y = U.size()[-2]

        # this contraction acts on the output of a weight contraction
        # and the node features
        spare_dims = get_spare_dims(correlation, irrep_out)
        instruction = f"bc{spare_dims}i,bci->bc{spare_dims}"

        self.summation = get_optimised_summation(
            lambda x, y: torch.einsum(instruction, x, y),
            [
                [BATCH_EXAMPLE, num_features, X] + [Y] * correlation,
                (BATCH_EXAMPLE, num_features, Y),
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
    def __init__(
        self,
        num_features: int,
        irrep_s_in: list[o3.Irrep],
        irrep_out: o3.Irrep,
        correlation: int,
        n_node_attributes: int,
    ):
        super().__init__()

        self.initial_contraction = InitialContraction(
            num_features,
            irrep_s_in,
            irrep_out,
            correlation,
            n_node_attributes,
        )

        self.weight_contractions = UniformModuleList(
            FollowingWeightContraction(
                num_features,
                irrep_s_in,
                irrep_out,
                j,
                n_node_attributes,
            )
            for j in reversed(range(1, correlation))
        )

        self.feature_contractions = UniformModuleList(
            FeatureContraction(
                num_features,
                irrep_s_in,
                irrep_out,
                j,
                n_node_attributes,
            )
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


def as_irreps(input: Any) -> o3.Irreps:
    # util to precent checking isinstance(o3.Irreps) all the time
    return cast(o3.Irreps, o3.Irreps(input))


def to_full_irreps(n_features: int, irreps: list[o3.Irrep]) -> o3.Irreps:
    # convert a list of irreps to a full irreps object
    return as_irreps([(n_features, ir) for ir in irreps])


_U_cache_sparse: dict[tuple[str, str, int], torch.Tensor] = torch.load(
    Path(__file__).parent / "_high_order_CG_coeff.pt"
)
"""
A pre-computed look-up table for the U matrices.

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

    return _U_cache_sparse[key].to_dense()
