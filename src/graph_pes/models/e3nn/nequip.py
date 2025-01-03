from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Final, Literal, TypedDict, cast

import e3nn
import e3nn.nn
import e3nn.util.jit
import torch
from e3nn import o3
from torch import Tensor

from graph_pes.atomic_graph import (
    DEFAULT_CUTOFF,
    AtomicGraph,
    PropertyKey,
    index_over_neighbours,
    neighbour_distances,
    neighbour_vectors,
)
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.models.components import distances
from graph_pes.models.components.aggregation import (
    NeighbourAggregation,
    NeighbourAggregationMode,
)
from graph_pes.models.components.scaling import LocalEnergiesScaler
from graph_pes.models.e3nn.utils import (
    LinearReadOut,
    SphericalHarmonics,
    build_limited_tensor_product,
)
from graph_pes.utils.logger import logger
from graph_pes.utils.nn import (
    MLP,
    AtomicOneHot,
    HaddamardProduct,
    PerElementEmbedding,
    UniformModuleList,
)

warnings.filterwarnings(
    "ignore",
    message=(
        "The TorchScript type system doesn't support instance-level "
        "annotations on empty non-base types.*"
    ),
)

### IMPLEMENTATION ###


def build_gate(output_irreps: o3.Irreps):
    """
    Builds an equivariant non-linearity that produces ``output_irreps``.

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
        radial_features: int,
        edge_features: o3.Irreps,
        target_node_features: o3.Irreps,
        cutoff: float,
        self_interaction: Literal["linear", "tensor_product"] | None,
        prune_output_to: list[o3.Irrep] | None,
        neighbour_aggregation: NeighbourAggregationMode,
    ):
        super().__init__()

        target_multiplicities: dict[o3.Irrep, int] = {}
        for mul, ir in target_node_features:
            target_multiplicities[ir] = mul

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
        self.message_tensor_product = build_limited_tensor_product(
            node_embedding_irreps=input_node_irreps,
            edge_embedding_irreps=edge_features,
            allowed_outputs=[ir for _, ir in target_node_features],
        )
        n_required_weights = self.message_tensor_product.weight_numel
        self.weight_generator = HaddamardProduct(
            torch.nn.Sequential(
                distances.Bessel(radial_features, cutoff, trainable=True),
                MLP(
                    [radial_features] * 3 + [n_required_weights],
                    activation=torch.nn.SiLU(),
                ),
            ),
            distances.PolynomialEnvelope(cutoff, p=6),
        )

        # 2. Message aggregation
        self.aggregation = NeighbourAggregation.parse(
            mode=neighbour_aggregation
        )

        # 5. Non-linearity
        # NB we initialise this first so we can work out how many irreps
        #    we need to be generating in step 3
        post_message_irreps: list[o3.Irrep] = sorted(
            set(i.ir for i in self.message_tensor_product.irreps_out)
        )
        if not prune_output_to:
            # desired_output_irreps = post_message_irreps
            irreps = []
            for ir in post_message_irreps:
                irreps.append((target_multiplicities[ir], ir))
            desired_output_irreps = o3.Irreps(irreps)
        else:
            irreps = []
            for ir in post_message_irreps:
                if ir in prune_output_to:
                    irreps.append((target_multiplicities[ir], ir))
            desired_output_irreps = o3.Irreps(irreps)

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
        # model
        cutoff: float,
        direct_force_predictions: bool,
        # input embeddings
        Z_embedding: torch.nn.Module,
        Z_embedding_dim: int,
        radial_features: int,
        # message passing
        layers: int,
        node_features: o3.Irreps,
        edge_features: o3.Irreps,
        self_interaction: Literal["linear", "tensor_product"] | None,
        neighbour_aggregation: NeighbourAggregationMode,
        # optimisation
        prune_last_layer: bool,
    ):
        props: list[PropertyKey] = ["local_energies"]
        if direct_force_predictions:
            props.append("forces")

        super().__init__(
            cutoff=cutoff,
            implemented_properties=props,
        )

        if not prune_last_layer:
            prune_output_to = None
        else:
            if direct_force_predictions:
                prune_output_to = [o3.Irrep("0e"), o3.Irrep("1o")]
            else:
                prune_output_to = [o3.Irrep("0e")]

        self.Z_embedding = Z_embedding

        scalar_even_dim = node_features.count("0e")
        if scalar_even_dim == 0:
            raise ValueError("Hidden irreps must contain a `0e` component.")
        self.initial_node_embedding = PerElementEmbedding(scalar_even_dim)
        # l_max = max(ir.l for ir in hidden_irreps)
        # use_odd_parity = any(ir.ir.p == -1 for ir in hidden_irreps)

        self.edge_embedding = SphericalHarmonics(edge_features, normalize=True)

        # first layer recieves an even parity, scalar embedding
        # of the atomic number
        current_layer_input: o3.Irreps
        current_layer_input = o3.Irreps(f"{scalar_even_dim}x0e")  # type: ignore
        _layers: list[NequIPMessagePassingLayer] = []
        for i in range(layers):
            layer = NequIPMessagePassingLayer(
                input_node_irreps=current_layer_input,
                edge_features=edge_features,
                target_node_features=node_features,
                cutoff=cutoff,
                radial_features=radial_features,
                Z_embedding_dim=Z_embedding_dim,
                self_interaction=self_interaction,
                prune_output_to=None if i < layers - 1 else prune_output_to,
                neighbour_aggregation=neighbour_aggregation,
            )
            _layers.append(layer)
            current_layer_input = layer.irreps_out

        self.layers = UniformModuleList(_layers)
        self.energy_readout = LinearReadOut(current_layer_input)

        if direct_force_predictions:
            if current_layer_input.count("1o") == 0:
                raise ValueError(
                    "Hidden irreps must contain a `1o` component in order "
                    f"to predict forces. Got {current_layer_input}."
                )
            self.force_readout = LinearReadOut(current_layer_input, "1o")

        else:
            self.force_readout = None

        self.scaler = LocalEnergiesScaler()

    def forward(self, graph: AtomicGraph) -> dict[PropertyKey, torch.Tensor]:
        # pre-compute important quantities
        r = neighbour_distances(graph)
        Y = self.edge_embedding(neighbour_vectors(graph))
        Z_embed = self.Z_embedding(graph.Z)

        # initialise the node embeddings...
        node_embed = self.initial_node_embedding(graph.Z)

        # ...iteratively update them...
        for layer in self.layers:
            node_embed = layer(node_embed, Z_embed, r, Y, graph)

        # ...and read out
        local_energies = self.energy_readout(node_embed).squeeze()
        local_energies = self.scaler(local_energies, graph)

        preds: dict[PropertyKey, torch.Tensor] = {
            "local_energies": local_energies
        }
        if self.force_readout is not None:
            preds["forces"] = self.force_readout(node_embed).squeeze()

        return preds


### USER FACING INTERFACE ###


class SimpleIrrepSpec(TypedDict):
    r"""
    A simple specification of the node and edge feature irreps for
    :class:`~graph_pes.models.NequIP`.

    Parameters
    ----------
    channels
        The number of channels for the node embedding. If an integer, all
        :math:`l`-order irreps will have the same number of channels. If a list,
        the :math:`l`-order irreps will have the number of channels specified by
        the :math:`l`-th entry in the list.
    l_max
        The maximum angular momentum for the edge embedding.
    use_odd_parity
        Whether to allow odd parity for the edge embedding.

    Examples
    --------

    .. code:: python

        >>> from graph_pes.models.e3nn.nequip import SimpleIrrepSpec
        >>> SimpleIrrepSpec(channels=16, l_max=2, use_odd_parity=True)
    """

    channels: int | list[int]
    l_max: int
    use_odd_parity: bool


class CompleteIrrepSpec(TypedDict):
    r"""
    A complete specification of the node and edge feature irreps for
    :class:`~graph_pes.models.NequIP`.

    Parameters
    ----------
    node_irreps
        The node feature irreps.
    edge_irreps
        The edge feature irreps.

    Examples
    --------

    .. code:: python

        >>> from graph_pes.models.e3nn.nequip import CompleteIrrepSpec
        >>> CompleteIrrepSpec(
        ...     node_irreps="32x0e + 16x1o + 8x2e",
        ...     edge_irreps="0e + 1o + 2e"
        ... )
    """

    node_irreps: str
    edge_irreps: str


DEFAULT_FEATURES: Final[SimpleIrrepSpec] = {
    "channels": 16,
    "l_max": 2,
    "use_odd_parity": True,
}


def parse_irrep_specification(
    spec: SimpleIrrepSpec | CompleteIrrepSpec,
) -> tuple[o3.Irreps, o3.Irreps]:
    is_simple = set(spec.keys()) == {"channels", "l_max", "use_odd_parity"}
    is_complete = set(spec.keys()) == {"node_irreps", "edge_irreps"}

    if not is_simple and not is_complete:
        raise ValueError(
            "Invalid irrep specification. Expected a dict with keys "
            "`node_irreps` and `edge_irreps` or a dict with keys "
            "`channels`, `l_max`, and `use_odd_parity`."
        )

    if is_simple:
        spec = cast(SimpleIrrepSpec, spec)
        l_max, channels = spec["l_max"], spec["channels"]
        if not isinstance(channels, list):
            channels = [channels] * (l_max + 1)

        if len(channels) != l_max + 1:
            raise ValueError(
                "channels must be an integer or a list of length l_max + 1"
            )

        parities = "oe" if spec["use_odd_parity"] else "e"

        node_irreps = []
        for l, c in zip(range(l_max + 1), channels):
            for p in parities:
                node_irreps.append(f"{c}x{l}{p}")

        edge_irreps = []
        for l in range(l_max + 1):
            p = "e" if not spec["use_odd_parity"] else "eo"[l % 2]
            edge_irreps.append(f"1x{l}{p}")

        node_irreps = o3.Irreps(" + ".join(node_irreps))
        edge_irreps = o3.Irreps(" + ".join(edge_irreps))

    else:
        spec = cast(CompleteIrrepSpec, spec)
        node_irreps = o3.Irreps(spec["node_irreps"])
        edge_irreps = o3.Irreps(spec["edge_irreps"])

    logger.debug(
        f"""\
        Parsed NequIP irrep speficication:
            node_irreps: {node_irreps}
            edge_irreps: {edge_irreps}
        """
    )

    return node_irreps, edge_irreps  # type: ignore


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
    features
        A specification of the irreps to use for the node and edge embeddings.
        Can be either a SimpleIrrepSpec or a CompleteIrrepSpec.
    layers
        The number of layers for the message passing.
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
    radial_features
        The number of features to expand the radial distances into. These
        features are then passed through an :class:`~graph_pes.utils.nn.MLP` to
        generate distance-conditioned weights for the message tensor product.

    Examples
    --------

    Configure a NequIP model for use with
    :doc:`graph-pes-train <../../cli/graph-pes-train/root>`:

    .. code:: yaml

        model:
          +NequIP:
            elements: [C, H, O]
            cutoff: 5.0

            # use 2 message passing layers
            layers: 2

            # using SimpleIrrepSpec
            features:
              channels: [64, 32, 8]
              l_max: 2
              use_odd_parity: true

            # scale the aggregation by the avg. number of
            # neighbours in the training set
            neighbour_aggregation: constant_fixed

    The hidden layer and edge embedding irreps the model generates can be
    controlled using either a :class:`~graph_pes.models.e3nn.nequip.SimpleIrrepSpec`
    or a :class:`~graph_pes.models.e3nn.nequip.CompleteIrrepSpec`:

    .. code:: python

        >>> from graph_pes.models import NequIP
        >>> model = NequIP(
        ...     elements=["C", "H", "O"],
        ...     cutoff=5.0,
        ...     features={
        ...         "channels": [16, 8, 4],
        ...         "l_max": 2,
        ...         "use_odd_parity": True
        ...     },
        ...     layers=3,
        ... )
        >>> for layer in model.layers:
        ...     print(layer.irreps_in, "->", layer.irreps_out)
        16x0e -> 16x0e+8x1o+4x2e
        16x0e+8x1o+4x2e -> 16x0e+8x1o+8x1e+4x2o+4x2e
        16x0e+8x1o+8x1e+4x2o+4x2e -> 16x0e

    .. code:: python

        >>> from graph_pes.models import NequIP
        >>> model = NequIP(
        ...     elements=["C", "H", "O"],
        ...     cutoff=5.0,
        ...     features={
        ...         "node_irreps": "32x0e + 16x1o + 8x2e",
        ...         "edge_irreps": "1x0e + 1x1o + 1x2e"
        ...     },
        ...     layers=3,
        ... )
        >>> for layer in model.layers:
        ...     print(layer.irreps_in, "->", layer.irreps_out)
        32x0e -> 32x0e+16x1o+8x2e
        32x0e+16x1o+8x2e -> 32x0e+16x1o+8x2e
        32x0e+16x1o+8x2e -> 32x0e

    Observe the drop in parameters as we prune the last layer and
    replace the tensor product interactions with linear layers:

    .. code:: python

        >>> from graph_pes.models import NequIP
        >>> # vanilla NequIP
        >>> vanilla = NequIP(
        ...     elements=["C", "H", "O"],
        ...     cutoff=5.0,
        ...     features={"channels": 128, "l_max": 2, "use_odd_parity": True},
        ...     layers=3,
        ...     self_interaction="tensor_product",
        ...     prune_last_layer=False,
        ... )
        >>> sum(p.numel() for p in vanilla.parameters())
        2308720
        >>> # SevenNet-flavoured NequIP
        >>> smaller = NequIP(
        ...     elements=["C", "H", "O"],
        ...     cutoff=5.0,
        ...     features={"channels": 128, "l_max": 2, "use_odd_parity": True},
        ...     layers=3,
        ...     self_interaction="linear",
        ...     prune_last_layer=True,
        ... )
        >>> sum(p.numel() for p in smaller.parameters())
        965232
    """  # noqa: E501

    def __init__(
        self,
        elements: list[str],
        direct_force_predictions: bool = False,
        cutoff: float = DEFAULT_CUTOFF,
        layers: int = 3,
        features: SimpleIrrepSpec | CompleteIrrepSpec = DEFAULT_FEATURES,
        self_interaction: Literal["linear", "tensor_product"]
        | None = "tensor_product",
        prune_last_layer: bool = True,
        neighbour_aggregation: NeighbourAggregationMode = "sum",
        radial_features: int = 8,
    ):
        assert len(set(elements)) == len(elements), "Found duplicate elements"
        Z_embedding = AtomicOneHot(elements)
        Z_embedding_dim = len(elements)

        node_features, edge_features = parse_irrep_specification(features)

        super().__init__(
            direct_force_predictions=direct_force_predictions,
            Z_embedding=Z_embedding,
            Z_embedding_dim=Z_embedding_dim,
            node_features=node_features,
            edge_features=edge_features,
            layers=layers,
            cutoff=cutoff,
            self_interaction=self_interaction,
            prune_last_layer=prune_last_layer,
            neighbour_aggregation=neighbour_aggregation,
            radial_features=radial_features,
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
        features: SimpleIrrepSpec | CompleteIrrepSpec = DEFAULT_FEATURES,
        layers: int = 3,
        self_interaction: Literal["linear", "tensor_product"]
        | None = "tensor_product",
        prune_last_layer: bool = True,
        neighbour_aggregation: NeighbourAggregationMode = "sum",
        radial_features: int = 8,
    ):
        Z_embedding = PerElementEmbedding(Z_embed_dim)
        node_features, edge_features = parse_irrep_specification(features)
        super().__init__(
            direct_force_predictions=direct_force_predictions,
            Z_embedding=Z_embedding,
            Z_embedding_dim=Z_embed_dim,
            node_features=node_features,
            edge_features=edge_features,
            layers=layers,
            cutoff=cutoff,
            self_interaction=self_interaction,
            prune_last_layer=prune_last_layer,
            neighbour_aggregation=neighbour_aggregation,
            radial_features=radial_features,
        )
