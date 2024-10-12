from __future__ import annotations

from typing import Callable

import e3nn.util.jit
import graph_pes.models.distances
import torch
from e3nn import o3
from graph_pes.core import LocalEnergyModel
from graph_pes.graphs import DEFAULT_CUTOFF
from graph_pes.graphs.graph_typing import AtomicGraph
from graph_pes.graphs.operations import neighbour_distances, neighbour_vectors
from graph_pes.models.distances import (
    DistanceExpansion,
    PolynomialEnvelope,
)
from graph_pes.models.e3nn.utils import LinearReadOut, NonLinearReadOut, ReadOut
from graph_pes.nn import (
    AtomicOneHot,
    HaddamardProduct,
    PerElementEmbedding,
    UniformModuleList,
)
from mace_layer import MACE_layer


def _get_distance_expansion(name: str) -> type[DistanceExpansion]:
    try:
        return getattr(graph_pes.models.distances, name)
    except AttributeError:
        raise ValueError(f"Unknown distance expansion type: {name}") from None


@e3nn.util.jit.compile_mode("script")
class _BaseMACE(LocalEnergyModel):
    """
    Base class for MACE models.
    """

    def __init__(
        self,
        # radial things
        cutoff: float,
        n_radial: int,
        radial_expansion_type: type[DistanceExpansion] | str,
        # node attributes
        z_embed_dim: int,
        z_embedding: Callable[[torch.Tensor], torch.Tensor],
        # message passing
        layers: int,
        max_ell: int,
        correlation: int,
        hidden_irreps: str,
        neighbour_scaling: float,
        use_self_connection: bool,
    ):
        super().__init__(cutoff=cutoff, auto_scale=True)

        if isinstance(radial_expansion_type, str):
            radial_expansion_type = _get_distance_expansion(
                radial_expansion_type
            )

        self.radial_expansion = HaddamardProduct(
            radial_expansion_type(
                n_features=n_radial, cutoff=cutoff, trainable=True
            ),
            PolynomialEnvelope(cutoff=cutoff),
        )

        self.z_embedding = z_embedding
        self.starting_linear = o3.Linear(
            irreps_in=f"{z_embed_dim}x0e", irreps_out=hidden_irreps
        )

        self.layers = UniformModuleList(
            MACE_layer(
                max_ell=max_ell,
                correlation=correlation,
                n_dims_in=z_embed_dim,
                hidden_irreps=hidden_irreps,
                node_feats_irreps=hidden_irreps,
                edge_feats_irreps=f"{n_radial}x0e",
                avg_num_neighbors=neighbour_scaling,
                use_sc=use_self_connection,
            )
            for _ in range(layers)
        )

        self.readouts: UniformModuleList[ReadOut] = UniformModuleList(
            [LinearReadOut(hidden_irreps) for _ in range(layers - 1)]
            + [NonLinearReadOut(hidden_irreps)]
        )

    def predict_raw_energies(self, graph: AtomicGraph) -> torch.Tensor:
        vectors = neighbour_vectors(graph)
        Z_embedding = self.z_embedding(graph["atomic_numbers"])

        node_features = self.starting_linear(Z_embedding)
        edge_features = self.radial_expansion(
            neighbour_distances(graph).view(-1, 1)
        )

        per_atom_energies = []
        for layer, readout in zip(self.layers, self.readouts):
            node_features = layer(
                vectors,
                node_features,
                Z_embedding,
                edge_features,
                graph["neighbour_index"],
            )
            per_atom_energies.append(readout(node_features))

        return torch.sum(torch.stack(per_atom_energies), dim=0)


@e3nn.util.jit.compile_mode("script")
class MACE(_BaseMACE):
    r"""
    Vanilla MACE model.

    One-hot encodings of the atomic numbers are used to condition the
    ``TensorProduct`` update in the residual connection of the message passing
    layers.

    During the ``pre_fit`` stage, this model guesses an initial scale for
    the energies of each element in the training set.
    To disable this feature, turn off pre-fitting.

    Internally, we rely on the `mace-layer <https://github.com/ACEsuit/mace-layer>`_
    implementation for the message passing layers, using the
    ``RealAgnosticResidualInteractionBlock``.

    Parameters
    ----------
    elements
        list of elements that this MACE model will be able to handle.
    cutoff
        radial cutoff (in Å) for the radial expansion (and message passing)
    n_radial
        number of bases to expand the radial distances into
    radial_expansion_type
        type of radial expansion to use. See :class:`~graph_pes.models.distances.DistanceExpansion`
        for available options
    layers
        number of message passing layers
    max_ell
        :math:`l_\max` for the spherical harmonics
    correlation
        maximum correlation order of the messages
    hidden_irreps
        :class:`e3nn.o3.Irreps` string for the node features at each
        message passing layer
    neighbour_scaling
        normalisation factor, :math:`\lambda`, for use in message aggregation:
        :math:`m_i = \frac{1}{\lambda} \sum_j m_{j \rightarrow i}`. Typically
        set to the average number of neighbours
    use_self_connection
        whether to use self-connections in the message passing layers

    Examples
    --------
    Basic usage:

    .. code-block:: python

        >>> from graph_pes.models import MACE
        >>> from graph_pes.models.distances import Bessel
        >>> model = MACE(
        ...     elements=["H", "C", "N", "O"],
        ...     cutoff=5.0,
        ...     radial_expansion_type=Bessel,
        ... )

    Specification in a YAML file:

    .. code-block:: yaml

        model:
            graph_pes.models.MACE:
                elements: [H, C, N, O]
                cutoff: 5.0
                hidden_irreps: "128x0e + 128x1o"
                radial_expansion_type: GaussianSmearing

    Citation
    --------
    Please cite the following papers if you use this model in your research:

    .. code-block:: bibtex

        @misc{Batatia2022MACE,
            title = {MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields},
            author = {Batatia, Ilyes and Kov{\'a}cs, D{\'a}vid P{\'e}ter and Simm, Gregor N. C. and Ortner, Christoph and Cs{\'a}nyi, G{\'a}bor},
            year = {2022},
            number = {arXiv:2206.07697},
            eprint = {2206.07697},
            eprinttype = {arxiv},
            doi = {10.48550/ARXIV.2206.07697},
            archiveprefix = {arXiv}
        }
        @misc{Batatia2022Design,
            title = {The Design Space of E(3)-Equivariant Atom-Centered Interatomic Potentials},
            author = {Batatia, Ilyes and Batzner, Simon and Kov{\'a}cs, D{\'a}vid P{\'e}ter and Musaelian, Albert and Simm, Gregor N. C. and Drautz, Ralf and Ortner, Christoph and Kozinsky, Boris and Cs{\'a}nyi, G{\'a}bor},
            year = {2022},
            number = {arXiv:2205.06643},
            eprint = {2205.06643},
            eprinttype = {arxiv},
            doi = {10.48550/arXiv.2205.06643},
            archiveprefix = {arXiv}
        }
    """  # noqa: E501

    def __init__(
        self,
        elements: list[str],
        # radial things
        cutoff: float = DEFAULT_CUTOFF,
        n_radial: int = 8,
        radial_expansion_type: type[DistanceExpansion] | str = "Bessel",
        # message passing
        layers: int = 2,
        max_ell: int = 3,
        correlation: int = 3,
        hidden_irreps: str = "128x0e + 128x1o",
        neighbour_scaling: float = 10.0,
        use_self_connection: bool = True,
    ):
        z_embed_dim = len(elements)
        z_embedding = AtomicOneHot(elements)

        super().__init__(
            cutoff=cutoff,
            n_radial=n_radial,
            radial_expansion_type=radial_expansion_type,
            z_embed_dim=z_embed_dim,
            z_embedding=z_embedding,
            layers=layers,
            max_ell=max_ell,
            correlation=correlation,
            hidden_irreps=hidden_irreps,
            neighbour_scaling=neighbour_scaling,
            use_self_connection=use_self_connection,
        )


@e3nn.util.jit.compile_mode("script")
class ZEmbeddingMACE(_BaseMACE):
    r"""
    MACE model that uses a learnable embedding of atomic number to
    condition the ``TensorProduct`` update in the residual connection of the
    message passing layers.

    During the ``pre_fit`` stage, this model guesses an initial scale for
    the energies of each element in the training set.
    To disable this feature, turn off pre-fitting.

    Internally, we rely on the `mace-layer <https://github.com/ACEsuit/mace-layer>`_
    implementation for the message passing layers, using the
    ``RealAgnosticResidualInteractionBlock``.

    Parameters
    ----------
    cutoff
        radial cutoff (in Å) for the radial expansion (and message passing)
    n_radial
        number of bases to expand the radial distances into
    radial_expansion_type
        type of radial expansion to use. See :class:`~graph_pes.models.distances.DistanceExpansion`
        for available options
    z_embed_dim
        dimension of the atomic number embedding
    layers
        number of message passing layers
    max_ell
        :math:`l_\max` for the spherical harmonics
    correlation
        maximum correlation order of the messages
    hidden_irreps
        :class:`~e3nn.o3.Irreps` string for the node features at each
        message passing layer
    neighbour_scaling
        normalisation factor, :math:`\lambda`, for use in message aggregation:
        :math:`m_i = \frac{1}{\lambda} \sum_j m_{j \rightarrow i}`. Typically
        set to the average number of neighbours
    use_self_connection
        whether to use self-connections in the message passing layers

    Examples
    --------
    Basic usage:

    .. code-block:: python

        >>> model = ZEmbeddingMACE(
        ...     cutoff=5.0,
        ... )

    Specification in a YAML file:

    .. code-block:: yaml

        model:
            graph_pes.models.ZEmbeddingMACE:
                cutoff: 5.0
                hidden_irreps: "128x0e + 128x1o"


    Citation
    --------
    Please cite the following papers if you use this model in your research:

    .. code-block:: bibtex

        @misc{Batatia2022MACE,
            title = {MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields},
            author = {Batatia, Ilyes and Kov{\'a}cs, D{\'a}vid P{\'e}ter and Simm, Gregor N. C. and Ortner, Christoph and Cs{\'a}nyi, G{\'a}bor},
            year = {2022},
            number = {arXiv:2206.07697},
            eprint = {2206.07697},
            eprinttype = {arxiv},
            doi = {10.48550/ARXIV.2206.07697},
            archiveprefix = {arXiv}
        }
        @misc{Batatia2022Design,
            title = {The Design Space of E(3)-Equivariant Atom-Centered Interatomic Potentials},
            author = {Batatia, Ilyes and Batzner, Simon and Kov{\'a}cs, D{\'a}vid P{\'e}ter and Musaelian, Albert and Simm, Gregor N. C. and Drautz, Ralf and Ortner, Christoph and Kozinsky, Boris and Cs{\'a}nyi, G{\'a}bor},
            year = {2022},
            number = {arXiv:2205.06643},
            eprint = {2205.06643},
            eprinttype = {arxiv},
            doi = {10.48550/arXiv.2205.06643},
            archiveprefix = {arXiv}
        }
    """  # noqa: E501

    def __init__(
        self,
        # radial things
        cutoff: float = DEFAULT_CUTOFF,
        n_radial: int = 8,
        radial_expansion_type: type[DistanceExpansion] | str = "Bessel",
        # node attributes
        z_embed_dim: int = 16,
        # message passing
        layers: int = 2,
        max_ell: int = 3,
        correlation: int = 3,
        hidden_irreps: str = "128x0e + 128x1o",
        neighbour_scaling: float = 10.0,
        use_self_connection: bool = True,
    ):
        super().__init__(
            cutoff=cutoff,
            n_radial=n_radial,
            radial_expansion_type=radial_expansion_type,
            z_embed_dim=z_embed_dim,
            z_embedding=PerElementEmbedding(z_embed_dim),
            layers=layers,
            max_ell=max_ell,
            correlation=correlation,
            hidden_irreps=hidden_irreps,
            neighbour_scaling=neighbour_scaling,
            use_self_connection=use_self_connection,
        )
