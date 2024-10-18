from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Literal

import e3nn
import e3nn.nn
import e3nn.util.jit
import torch
from e3nn import o3
from graph_pes.core import GraphPESModel
from graph_pes.graphs import DEFAULT_CUTOFF, keys
from graph_pes.graphs.graph_typing import AtomicGraph
from graph_pes.graphs.operations import (
    index_over_neighbours,
    neighbour_distances,
    neighbour_vectors,
)
from graph_pes.models.components import distances
from graph_pes.models.components.aggregation import (
    NeighbourAggregation,
    NeighbourAggregationMode,
)
from graph_pes.models.e3nn.utils import LinearReadOut
from graph_pes.nn import (
    MLP,
    AtomicOneHot,
    HaddamardProduct,
    PerElementEmbedding,
    UniformModuleList,
)
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


@e3nn.util.jit.compile_mode("script")
class SphericalHarmonics(o3.SphericalHarmonics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return f"SphericalHarmonics(1x1o -> {self.irreps_out})"


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


class SelfInteraction(ABC, torch.nn.Module):
    @abstractmethod
    def forward(self, node_embeddings: Tensor, Z_embeddings: Tensor) -> Tensor:
        pass


class LinearSelfInteraction(SelfInteraction):
    def __init__(
        self,
        input_irreps: o3.Irreps,
        output_irreps: o3.Irreps,
    ):
        super().__init__()

        self.linear = o3.Linear(input_irreps, output_irreps)

    def forward(self, node_embeddings: Tensor, Z_embeddings: Tensor) -> Tensor:
        return self.linear(node_embeddings)


class TensorProductSelfInteraction(SelfInteraction):
    def __init__(
        self,
        input_irreps: o3.Irreps | str,
        output_irreps: o3.Irreps | str,
        Z_embedding_irreps: o3.Irreps | str,
    ):
        super().__init__()

        self.tensor_product = o3.FullyConnectedTensorProduct(
            input_irreps,
            Z_embedding_irreps,
            output_irreps,
        )

    def forward(self, node_embeddings: Tensor, Z_embeddings: Tensor) -> Tensor:
        return self.tensor_product(node_embeddings, Z_embeddings)


class NequIPMessagePassingLayer(torch.nn.Module):
    def __init__(
        self,
        input_node_irreps: o3.Irreps,
        Z_embedding_dim: int,
        edge_irreps: o3.Irreps,
        l_max: int,
        n_channels: list[int],
        cutoff: float,
        self_interaction: Literal["linear", "tensor_product"] | None,
        prune_weights: bool,
        neighbour_aggregation: NeighbourAggregationMode,
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
        # 3. Pass the total message through a linear layer to create
        #       new central atom embeddings
        # 4. (if configured) Allow the old central atom embeddings to interact
        #       with themselves via another tensor product and residual add
        # 5. Apply a non-linearity to the new central atom embeddings

        # 1. Message creation
        self.pre_message_linear = o3.Linear(
            input_node_irreps, input_node_irreps
        )
        self.message_tensor_product = _nequip_message_tensor_product(
            node_embedding_irreps=input_node_irreps,
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
        self.aggregation = NeighbourAggregation.parse(
            mode=neighbour_aggregation
        )

        # 5. Non-linearity
        # NB we do this first so we can work out how many irreps we need
        #    to be generating in step 3
        post_message_irreps: list[o3.Irrep] = sorted(
            set(i.ir for i in self.message_tensor_product.irreps_out)
        )
        if not prune_weights:
            desired_output_irreps = o3.Irreps(
                [(n_channels[ir.l], ir) for ir in post_message_irreps]
            )
        else:
            desired_output_irreps = o3.Irreps(f"{n_channels[0]}x0e")

        self.non_linearity = build_gate(desired_output_irreps)  # type: ignore

        # 3. Post-message linear
        self.post_message_linear = o3.Linear(
            self.message_tensor_product.irreps_out.simplify(),
            self.non_linearity.irreps_in,
        )

        # 4. Self-interaction
        # create a self interaction that produces the output irreps that
        # are required for the non-linearity
        self.self_interaction: SelfInteraction | None = None

        if self_interaction == "tensor_product":
            self.self_interaction = TensorProductSelfInteraction(
                input_irreps=input_node_irreps,
                Z_embedding_irreps=f"{Z_embedding_dim}x0e",
                output_irreps=self.non_linearity.irreps_in,
            )
        elif self_interaction == "linear":
            self.self_interaction = LinearSelfInteraction(
                input_irreps=input_node_irreps,
                output_irreps=self.non_linearity.irreps_in,
            )

        # bookkeeping
        self.irreps_in = input_node_irreps
        self.irreps_out: o3.Irreps = self.non_linearity.irreps_out

    def forward(
        self,
        node_embeddings: Tensor,  # [n_atoms, irreps_in]
        Z_embeddings: Tensor,  # [n_atoms, Z_embedding_dim]
        neighbour_distances: Tensor,  # [n_edges,]
        edge_embedding: Tensor,  # [n_edges, edge_irreps]
        graph: AtomicGraph,
    ) -> Tensor:  # [n_atoms, irreps_out]
        # 1. message creation
        neighbour_embeddings = index_over_neighbours(
            self.pre_message_linear(node_embeddings), graph
        )
        weights = self.weight_generator(neighbour_distances.unsqueeze(-1))
        messages = self.message_tensor_product(
            neighbour_embeddings, edge_embedding, weights
        )

        # 2. message aggregation
        total_message = self.aggregation(messages, graph)

        # 3. update node embeddings
        new_node_embeddings = self.post_message_linear(total_message)

        # 4. self-interaction
        if self.self_interaction is not None:
            new_node_embeddings = new_node_embeddings + self.self_interaction(
                node_embeddings, Z_embeddings
            )

        # 5. non-linearity
        return self.non_linearity(new_node_embeddings)

    # type hints for mypy
    def __call__(
        self,
        node_embeddings: Tensor,
        Z_embeddings: Tensor,
        neighbour_distances: Tensor,
        edge_embedding: Tensor,
        graph: AtomicGraph,
    ) -> Tensor:
        return super().__call__(
            node_embeddings,
            Z_embeddings,
            neighbour_distances,
            edge_embedding,
            graph,
        )


@e3nn.util.jit.compile_mode("script")
class _BaseNequIP(GraphPESModel):
    def __init__(
        self,
        direct_force_predictions: bool,
        Z_embedding: torch.nn.Module,
        Z_embedding_dim: int,
        n_channels: int | list[int],
        l_max: int,
        n_layers: int,
        cutoff: float,
        allow_odd_parity: bool,
        self_interaction: Literal["linear", "tensor_product"] | None,
        prune_last_layer: bool,
        neighbour_aggregation: NeighbourAggregationMode,
    ):
        props: list[keys.LabelKey] = ["local_energies"]
        if direct_force_predictions:
            props.append("forces")

        super().__init__(
            cutoff=cutoff,
            implemented_properties=props,
            auto_scale_local_energies=True,
        )

        if isinstance(n_channels, int):
            n_channels = [n_channels] * (l_max + 1)

        if len(n_channels) != l_max + 1:
            raise ValueError(
                "n_channels must be an integer or a list of length l_max + 1"
            )

        self.Z_embedding = Z_embedding
        self.initial_node_embedding = PerElementEmbedding(n_channels[0])

        edge_embedding_irreps = o3.Irreps.spherical_harmonics(
            l_max, p=-1 if allow_odd_parity else 1
        )
        self.edge_embedding = SphericalHarmonics(
            edge_embedding_irreps, normalize=True
        )

        # first layer recieves an even parity, scalar embedding
        # of the atomic number
        current_layer_input: o3.Irreps = o3.Irreps(f"{n_channels[0]}x0e")  # type: ignore
        layers: list[NequIPMessagePassingLayer] = []
        for i in range(n_layers):
            layer = NequIPMessagePassingLayer(
                input_node_irreps=current_layer_input,
                edge_irreps=edge_embedding_irreps,  # type: ignore
                l_max=l_max,
                n_channels=n_channels,
                cutoff=cutoff,
                Z_embedding_dim=Z_embedding_dim,
                self_interaction=self_interaction,
                prune_weights=i == n_layers - 1
                and prune_last_layer
                and not direct_force_predictions,
                neighbour_aggregation=neighbour_aggregation,
            )
            layers.append(layer)
            current_layer_input = layer.irreps_out

        self.layers = UniformModuleList(layers)
        self.energy_readout = LinearReadOut(current_layer_input)

        if direct_force_predictions:
            self.force_readout = LinearReadOut(current_layer_input, "1o")
        else:
            self.force_readout = None

    def forward(self, graph: AtomicGraph) -> dict[keys.LabelKey, torch.Tensor]:
        # pre-compute important quantities
        r = neighbour_distances(graph)
        Y = self.edge_embedding(neighbour_vectors(graph))
        Z_embed = self.Z_embedding(graph["atomic_numbers"])

        # initialise the node embeddings...
        node_embed = self.initial_node_embedding(graph["atomic_numbers"])

        # ...iteratively update them...
        for layer in self.layers:
            node_embed = layer(node_embed, Z_embed, r, Y, graph)

        # ...and read out
        preds: dict[keys.LabelKey, torch.Tensor] = {
            "local_energies": self.energy_readout(node_embed).squeeze()
        }
        if self.force_readout is not None:
            preds["forces"] = self.force_readout(node_embed).squeeze()

        return preds


@e3nn.util.jit.compile_mode("script")
class NequIP(_BaseNequIP):
    r"""
    NequIP architecture from `E(3)-equivariant graph neural networks for
    data-efficient and accurate interatomic potentials
    <https://www.nature.com/articles/s41467-022-29939-5>`__.

    If you use this model in your research, please cite the original work:

    .. code:: bibtex

        @article{Batzner-22-05,
            title = {
                      E(3)-Equivariant Graph Neural Networks for
                      Data-Efficient and Accurate Interatomic Potentials
                    },
            author = {
                      Batzner, Simon and Musaelian, Albert and Sun, Lixin
                      and Geiger, Mario and Mailoa, Jonathan P.
                      and Kornbluth, Mordechai and Molinari, Nicola
                      and Smidt, Tess E. and Kozinsky, Boris
                    },
            year = {2022},
            journal = {Nature Communications},
            volume = {13},
            number = {1},
            pages = {2453},
            doi = {10.1038/s41467-022-29939-5},
            copyright = {2022 The Author(s)}
        }

    Parameters
    ----------
    elements
        The elements that the model will encounter in the training data.
        This is used to create the atomic one-hot embedding. If you intend
        to fine-tune this model on additional data, you must ensure that all
        elements you will encounter in both pre-training and fine-tuning are
        present in this list. (See :class:`~graph_pes.models.ZEmbeddingNequIP`
        for an alternative that allows for arbitrary atomic numbers.)
    direct_force_predictions
        Whether to predict forces directly. If ``True``, the model will output
        forces (rather than infer them from the energy) using a
        :class:`~graph_pes.models.e3nn.utils.LinearReadOut` to map the final
        layer node embedding to a set of force predictions.
    cutoff
        The cutoff radius for the model.
    n_channels
        The number of channels for the node embedding. If an integer, all
        :math:`l`-order irreps will have the same number of channels. If a list,
        the entries must be in :math:`l = 0, 1, \ldots, l_{\text{max}}` order.
    n_layers
        The number of layers for the message passing.
    l_max
        The maximum angular momentum for the edge embedding.
    allow_odd_parity
        Whether to allow odd parity for the edge embedding.
    self_interaction
        The kind of self-interaction to use. If ``None``, no self-interaction
        is applied. If ``"linear"``, a linear self-interaction is applied
        to update the node embeddings along the residual path. If
        ``"tensor_product"``, a tensor product combining the old node
        embedding with an embedding of the atomic number is applied.
        As first noticed by the authors of `SevenNet <https://pubs.acs.org/doi/10.1021/acs.jctc.4c00190>`__,
        using linear self-interactions greatly reduces the number of parameters
        in the model, and helps to prevent overfitting.
    prune_last_layer
        Whether to prune irrep communication pathways in the final layer
        that do not contribute to the ``0e`` output embedding.
    neighbour_aggregation
        The neighbour aggregation mode. See
        :class:`~graph_pes.models.components.aggregation.NeighbourAggregationMode`
        for more details. Note that ``"mean"`` or ``"sqrt"`` aggregations lead
        to un-physical discontinuities in the energy function as atoms enter and
        leave the cutoff radius of the model.


    Examples
    --------

    The hidden layer and edge embedding irreps the model generates are
    controlled by the combination of ``n_channels``, ``l_max`` and
    ``allow_odd_parity``:

    .. list-table::
        :header-rows: 1

        * - ``n_channels``
          - ``l_max``
          - ``allow_odd_parity``
          - ``edge_irreps``
          - ``hidden_irreps``
        * - 8
          - 2
          - True
          - ``1x0e + 1x1o + 1x2e``
          - ``8x0e + 8x0o + 8x1e + 8x1o + 8x2e + 8x2o``
        * - 8
          - 2
          - False
          - ``1x0e + 1x1e + 1x2e``
          - ``8x0e + 8x1e + 8x2e``
        * - [16, 8, 4]
          - 2
          - False
          - ``1x0e + 1x1e + 1x2e``
          - ``16x0e + 16x1e + 16x2e``

    Configure a NequIP model for use with
    :doc:`graph-pes-train <../../cli/graph-pes-train>`:

    .. code:: yaml

        model:
          graph_pes.models.NequIP:
            elements: [C, H, O]
            cutoff: 5.0

            # use 2 message passing layers
            n_layers: 2

            # use 64 l=0, 32 l=1 and 8 l=2 node features
            n_channels: [64, 32, 8]
            l_max: 2

            # scale the aggregation by the avg. number of
            # neighbours in the training set
            neighbour_aggregation: constant_fixed


    Observe the drop in parameters as we prune the last layer and
    replace the tensor product interactions with linear layers:

    .. code:: python

        >>> from graph_pes.models import NequIP
        >>> # vanilla NequIP
        >>> vanilla = NequIP(
        ...     elements=["C", "H", "O"],
        ...     cutoff=5.0,
        ...     n_channels=128,
        ...     n_layers=3,
        ...     l_max=2,
        ...     self_interaction="tensor_product",
        ...     prune_last_layer=False,
        ... )
        >>> sum(p.numel() for p in vanilla.parameters())
        2308720
        >>> # SevenNet-flavoured NequIP
        >>> smaller = NequIP(
        ...     elements=["C", "H", "O"],
        ...     cutoff=5.0,
        ...     n_channels=128,
        ...     n_layers=3,
        ...     l_max=2,
        ...     self_interaction="linear",
        ...     prune_last_layer=True,
        ... )
        >>> sum(p.numel() for p in smaller.parameters())
        965232


    """

    def __init__(
        self,
        elements: list[str],
        direct_force_predictions: bool = False,
        cutoff: float = DEFAULT_CUTOFF,
        n_channels: int | list[int] = 16,
        n_layers: int = 3,
        l_max: int = 2,
        allow_odd_parity: bool = True,
        self_interaction: Literal["linear", "tensor_product"]
        | None = "tensor_product",
        prune_last_layer: bool = True,
        neighbour_aggregation: NeighbourAggregationMode = "sum",
    ):
        assert len(set(elements)) == len(elements), "Found duplicate elements"
        Z_embedding = AtomicOneHot(elements)
        Z_embedding_dim = len(elements)
        super().__init__(
            direct_force_predictions=direct_force_predictions,
            Z_embedding=Z_embedding,
            Z_embedding_dim=Z_embedding_dim,
            n_channels=n_channels,
            l_max=l_max,
            n_layers=n_layers,
            cutoff=cutoff,
            allow_odd_parity=allow_odd_parity,
            self_interaction=self_interaction,
            prune_last_layer=prune_last_layer,
            neighbour_aggregation=neighbour_aggregation,
        )


@e3nn.util.jit.compile_mode("script")
class ZEmbeddingNequIP(_BaseNequIP):
    r"""
    A modified version of the :class:`~graph_pes.models.NequIP` architecture
    that embeds atomic numbers into a learnable embedding rather than using an
    atomic one-hot encoding.

    This circumvents the need to know in advance all the elements that the
    model will encounter in any pre-training or fine-tuning datasets.

    **Relevant differences** from :class:`~graph_pes.models.NequIP`\ :

    - The ``elements`` argument is removed.
    - The ``Z_embed_dim`` (:class:`int`) argument controls the size of the
      atomic number embedding (default: 8).

    For all other options, see :class:`~graph_pes.models.NequIP`.
    """

    def __init__(
        self,
        cutoff: float = DEFAULT_CUTOFF,
        direct_force_predictions: bool = False,
        Z_embed_dim: int = 8,
        n_channels: int | list[int] = 16,
        l_max: int = 2,
        n_layers: int = 3,
        allow_odd_parity: bool = True,
        self_interaction: Literal["linear", "tensor_product"]
        | None = "tensor_product",
        prune_last_layer: bool = True,
        neighbour_aggregation: NeighbourAggregationMode = "sum",
    ):
        Z_embedding = PerElementEmbedding(Z_embed_dim)
        super().__init__(
            direct_force_predictions=direct_force_predictions,
            Z_embedding=Z_embedding,
            Z_embedding_dim=Z_embed_dim,
            n_channels=n_channels,
            l_max=l_max,
            n_layers=n_layers,
            cutoff=cutoff,
            allow_odd_parity=allow_odd_parity,
            self_interaction=self_interaction,
            prune_last_layer=prune_last_layer,
            neighbour_aggregation=neighbour_aggregation,
        )
