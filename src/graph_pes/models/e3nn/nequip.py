from __future__ import annotations

import warnings

import e3nn
import e3nn.util.jit
import torch
from e3nn import o3
from graph_pes.graphs.graph_typing import AtomicGraph
from graph_pes.graphs.operations import (
    neighbour_distances,
    neighbour_vectors,
    split_over_neighbours,
    sum_over_neighbours,
)
from graph_pes.models import distances
from graph_pes.models.scaling import AutoScaledPESModel
from graph_pes.nn import (
    MLP,
    AtomicOneHot,
    HaddamardProduct,
    PerElementEmbedding,
)
from graph_pes.util import uniform_repr
from torch import Tensor

warnings.filterwarnings(
    "ignore",
    message=(
        "The TorchScript type system doesn't support instance-level "
        "annotations on empty non-base types.*"
    ),
)


def _nequip_message_tensor_product(
    node_embedding_irreps: o3.Irreps,
    edge_embedding_irreps: o3.Irreps,
    l_max: int,
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
                (order, parity) = ir_out

                # if we want this output from the tensor product, add it to the
                # list of instructions
                if order <= l_max:
                    k = len(output_irreps)
                    output_irreps.append((channels, ir_out))
                    # from the i'th irrep of the neighbour embedding
                    # and from the l'th irrep of the spherical harmonics
                    # to the k'th irrep of the output tensor
                    instructions.append((i, l, k, "uvu", True))

    # since many paths can lead to the same output irrep, we sort the
    # instructions so that the tensor product generates tensors in a
    # simplified order, e.g. 32x0e + 16x1o, not 16x0e + 16x1o + 16x0e
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


class PerElementSelfInteraction(torch.nn.Module):
    def __init__(
        self, input_irreps: o3.Irreps, output_irreps: o3.Irreps, n_elements: int
    ):
        super().__init__()

        self.one_hot = AtomicOneHot(n_elements)
        self.tensor_product = o3.FullyConnectedTensorProduct(
            input_irreps, f"{n_elements}x0e", output_irreps
        )

    def register_Zs(self, Zs: list[int]):
        self.one_hot.register_Zs(Zs)

    def forward(
        self,
        node_embeddings: Tensor,  # [n_atoms, irreps_in]
        graph: AtomicGraph,
    ) -> Tensor:  # [n_atoms, irreps_out]
        return self.tensor_product(
            node_embeddings, self.one_hot(graph["atomic_numbers"])
        )

    # type hints
    def __call__(
        self,
        node_embeddings: Tensor,
        graph: AtomicGraph,
    ) -> Tensor:
        return super().__call__(node_embeddings, graph)

    def __repr__(self):
        return uniform_repr(
            self.__class__.__name__,
            self.tensor_product,
            elements=self.one_hot.registered_elements,
        )


def build_gate(output_irreps: o3.Irreps):
    """
    Builds an equivariant non-linearity that produces output_irreps.

    This is done by passing all scalar irreps directly through a non-linearity,
    (tanh for even parity, silu for odd parity).

    All higher order irreps are multiplied by a suitable number of scalar
    irrep values that have also been passed through a non-linearity.

    The gated scalar and higher order irreps are then combined to give the
    final output.
    """

    parity_to_activations = {
        -1: torch.nn.functional.tanh,
        1: torch.nn.functional.silu,
    }

    # separate scalar and higher order irreps
    scalar_inputs = o3.Irreps(
        irreps for irreps in output_irreps if irreps.ir.l == 0
    )
    scalar_activations = [
        parity_to_activations[irrep.ir.p] for irrep in scalar_inputs
    ]

    higher_inputs = o3.Irreps(
        irreps for irreps in output_irreps if irreps.ir.l > 0
    )

    # work out the the `irrep_gates` scalars to be used
    gate_irrep = "0e" if "0e" in output_irreps else "0o"
    gate_parity = o3.Irrep(gate_irrep).p
    gate_value_irreps = o3.Irreps(
        (channels, gate_irrep) for (channels, _) in higher_inputs
    )
    gate_activations = [
        parity_to_activations[gate_parity] for _ in higher_inputs
    ]

    return e3nn.nn.Gate(
        irreps_scalars=scalar_inputs,
        act_scalars=scalar_activations,
        irreps_gates=gate_value_irreps,
        act_gates=gate_activations,
        irreps_gated=higher_inputs,
    )


class NequIPMessagePassingLayer(torch.nn.Module):
    def __init__(
        self,
        starting_node_irreps: o3.Irreps,
        edge_irreps: o3.Irreps,
        l_max: int,
        n_channels: int,
        cutoff: float,
        n_elements: int,
    ):
        super().__init__()

        # NequIP message passing involves the following steps:
        # 1. Create the message from neighbour j to central atom i
        #   - update neighbour node features, h_j, with a linear layer
        #   - mix the neighbour node features with the edge features, using a
        #       tensor product with weights that are learned as a function
        #       of the neighbour distance, to create messages m_j_to_i
        # 2. Aggregate the messages over all neighbours to create the
        #       total message, m_i
        # 3. Simultaneously, allow the central atom embeddings to interact
        #       with themselves via another tensor product
        # 4. Pass the total message through a linear layer to create features
        #       of the same shape as the central atom self interactions, and add
        #       them together
        # 5. Apply a non-linearity to the new central atom embeddings

        # each input irreps should have a multiplicity of n_channels:
        assert all(i.mul == n_channels for i in starting_node_irreps)

        # 1. Message creation
        self.pre_message_linear = o3.Linear(
            starting_node_irreps, starting_node_irreps
        )
        self.message_tensor_product = _nequip_message_tensor_product(
            node_embedding_irreps=starting_node_irreps,
            edge_embedding_irreps=edge_irreps,
            l_max=l_max,
        )
        n_required_weights = self.message_tensor_product.weight_numel
        self.weight_generator = torch.nn.Sequential(
            HaddamardProduct(
                distances.Bessel(8, cutoff, trainable=True),
                distances.PolynomialEnvelope(cutoff, p=6),
            ),
            MLP(
                [8, 10, 10, n_required_weights],
                activation=torch.nn.SiLU(),
            ),
        )

        # 2. Message aggregation
        ...  # no state to save for this

        # 5. Non-linearity
        # NB we do this first so we can work out how many irreps we need
        #    to be generating in step 3
        post_message_irreps: list[o3.Irrep] = sorted(
            set(i.ir for i in self.message_tensor_product.irreps_out)
        )
        desired_output_irreps = o3.Irreps(
            [(n_channels, ir) for ir in post_message_irreps]
        )
        self.non_linearity = build_gate(desired_output_irreps)  # type: ignore

        # 3. Self-interaction
        # create a self interaction that produces the output irreps that
        # are required for the non-linearity
        # TODO: there are wasted weights here in the first layer!
        self.self_interaction = PerElementSelfInteraction(
            starting_node_irreps, self.non_linearity.irreps_in, n_elements
        )

        # 4. Post-message linear
        self.post_message_linear = o3.Linear(
            self.message_tensor_product.irreps_out.simplify(),
            self.non_linearity.irreps_in,
        )

        # bookkeeping
        self.irreps_in = starting_node_irreps
        self.irreps_out = self.non_linearity.irreps_out

    def forward(
        self,
        node_embeddings: Tensor,  # [n_atoms, irreps_in]
        neighbour_distances: Tensor,  # [n_edges,]
        edge_embedding: Tensor,  # [n_edges, edge_irreps]
        graph: AtomicGraph,
    ) -> Tensor:  # [n_atoms, irreps_out]
        # 1. message creation
        neighbour_embeddings = split_over_neighbours(
            self.pre_message_linear(node_embeddings), graph
        )
        weights = self.weight_generator(neighbour_distances.unsqueeze(-1))
        messages = self.message_tensor_product(
            neighbour_embeddings, edge_embedding, weights
        )

        # 2. message aggregation
        total_message = sum_over_neighbours(messages, graph)

        # 3. self-interaction
        self_interaction = self.self_interaction(node_embeddings, graph)

        # 4. post-message linear and adding
        new_node_embeddings = (
            self.post_message_linear(total_message) + self_interaction
        )

        # 5. non-linearity
        return self.non_linearity(new_node_embeddings)

    # type hints for mypy
    def __call__(
        self,
        node_embeddings: Tensor,
        neighbour_distances: Tensor,
        edge_embedding: Tensor,
        graph: AtomicGraph,
    ) -> Tensor:
        return super().__call__(
            node_embeddings, neighbour_distances, edge_embedding, graph
        )


@e3nn.util.jit.compile_mode("script")
class NequIP(AutoScaledPESModel):
    def __init__(
        self,
        n_elements: int,
        n_channels: int = 16,
        n_layers: int = 3,
        cutoff: float = 3.0,
        l_max: int = 2,
        allow_odd_parity: bool = True,
    ):
        super().__init__()

        self.Z_embedding = PerElementEmbedding(n_channels)

        edge_embedding_irreps = o3.Irreps.spherical_harmonics(
            l_max, p=-1 if allow_odd_parity else 1
        )
        self.edge_embedding = o3.SphericalHarmonics(
            edge_embedding_irreps, normalize=True
        )

        # first layer recieves an even parity, scalar embedding
        # of the atomic number
        current_layer_input = o3.Irreps(f"{n_channels}x0e")
        layers = []
        for _ in range(n_layers):
            layer = NequIPMessagePassingLayer(
                starting_node_irreps=current_layer_input,  # type: ignore
                edge_irreps=edge_embedding_irreps,  # type: ignore
                l_max=l_max,
                n_channels=n_channels,
                cutoff=cutoff,
                n_elements=n_elements,
            )
            layers.append(layer)
            current_layer_input = layer.irreps_out

        self.layers = torch.nn.ModuleList(layers)

        self.readout = o3.Linear(current_layer_input, "1x0e")

    def predict_unscaled_energies(self, graph: AtomicGraph) -> Tensor:
        # pre-compute important quantities
        r = neighbour_distances(graph)
        Y = self.edge_embedding(neighbour_vectors(graph))

        # initialise the node embeddings...
        h = self.Z_embedding(graph["atomic_numbers"])

        # ...iteratively update them...
        for layer in self.layers:
            h = layer(h, r, Y, graph)

        # ...and read out the energy
        return self.readout(h)

    def model_specific_pre_fit(self, graphs: AtomicGraph):
        for layer in self.layers:
            layer.self_interaction.register_Zs(
                graphs["atomic_numbers"].tolist()
            )
