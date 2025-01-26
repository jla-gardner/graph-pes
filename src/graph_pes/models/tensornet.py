from __future__ import annotations

import torch
from torch import Tensor, nn

from graph_pes.atomic_graph import (
    DEFAULT_CUTOFF,
    AtomicGraph,
    PropertyKey,
    index_over_neighbours,
    neighbour_distances,
    neighbour_vectors,
    number_of_edges,
    sum_over_neighbours,
)
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.models.components.scaling import LocalEnergiesScaler
from graph_pes.utils.nn import (
    MLP,
    HaddamardProduct,
    PerElementEmbedding,
    UniformModuleList,
)

from .components.distances import (
    CosineEnvelope,
    DistanceExpansion,
    ExponentialRBF,
    get_distance_expansion,
)


class TensorNet(GraphPESModel):
    r"""
    The `TensorNet <http://arxiv.org/abs/2306.06482>`_ architecture.

    Citation:

    .. code:: bibtex

        @misc{Simeon-23-06,
            title = {
                TensorNet: Cartesian Tensor Representations for
                Efficient Learning of Molecular Potentials
            },
            author = {Simeon, Guillem and {de Fabritiis}, Gianni},
            year = {2023},
            number = {arXiv:2306.06482},
        }

    Parameters
    ----------
    cutoff
        The cutoff radius to use for the model.
    radial_features
        The number of radial features to use for the model.
    radial_expansion
        The type of radial basis function to use for the model.
        For more examples, see
        :class:`~graph_pes.models.components.distances.DistanceExpansion`.
    channels
        The size of the embedding for each atom.
    layers
        The number of interaction layers to use for the model.
    direct_force_predictions
        Whether to predict forces directly. If ``True``, the model will
        generate force predictions by passing the final
        layer's node embeddings through a
        :class:`~graph_pes.models.tensornet.VectorOutput` read out.
        Otherwise, ``graph-pes`` automatically infers the forces as the
        derivative of the energy with respect to the atomic positions.

    Examples
    --------

    Configure a TensorNet model for use with ``graph-pes-train``:

    .. code:: yaml

        model:
          +TensorNet:
            radial_features: 8
            radial_expansion: Bessel
            channels: 32
            cutoff: 5.0
    """

    def __init__(
        self,
        cutoff: float = DEFAULT_CUTOFF,
        radial_features: int = 32,
        radial_expansion: str | type[DistanceExpansion] = ExponentialRBF,
        channels: int = 32,
        layers: int = 2,
        direct_force_predictions: bool = False,
    ):
        properties: list[PropertyKey] = ["local_energies"]
        if direct_force_predictions:
            properties.append("forces")

        super().__init__(
            cutoff=cutoff,
            implemented_properties=properties,
        )

        self.embedding = Embedding(
            radial_features, radial_expansion, channels, cutoff
        )

        self.interactions = UniformModuleList(
            Interaction(radial_features, channels, cutoff)
            for _ in range(layers)
        )

        self.energy_read_out = ScalarOutput(channels)
        if direct_force_predictions:
            self.force_read_out = VectorOutput(channels)
        else:
            self.force_read_out = None

        self.scaler = LocalEnergiesScaler()

    def forward(self, graph: AtomicGraph) -> dict[PropertyKey, torch.Tensor]:
        X = self.embedding(graph)  # (N, C, 3, 3)

        for interaction in self.interactions:
            # normalise -> interaction -> residual connection
            X = X / (frobenius_norm(X)[..., None, None] + 1)
            dX = interaction(X, graph)
            X = X + dX

        local_energies = self.energy_read_out(X).squeeze()
        local_energies = self.scaler(local_energies, graph)

        results: dict[PropertyKey, torch.Tensor] = {
            "local_energies": local_energies
        }

        if self.force_read_out is not None:
            results["forces"] = self.force_read_out(X)

        return results


### components ###


class EdgeEmbedding(nn.Module):
    r"""
    Generates embeddings for each (directed) edge in the graph, incorporating
    the species of each atom, and the vector between them.

    1. generate initial edge embedding components:

       * :math:`I_0^{(ij)} = \text{Id}`
       * :math:`A_0^{(ij)} = \begin{pmatrix} 
         0 & \hat{r}_{ij}^z & - \hat{r}_{ij}^y \\
         - \hat{r}_{ij}^z & 0 & \hat{r}_{ij}^x \\
         \hat{r}_{ij}^y & - \hat{r}_{ij}^x & 0
         \end{pmatrix}`
       * :math:`S_0^{(ij)} = \hat{r}_{ij} \cdot \hat{r}_{ij}^T -
         \text{Tr}(\hat{r}_{ij} \cdot \hat{r}_{ij}^T) \text{Id}`

    2. generate an embedding of the ordered pair of species:

    .. math::
    
       h_Z^{(ij)} = f(z_i, z_j) = \text{Linear}(\text{embed}(z_i) 
       || \text{embed}(z_j))

    3. expand the edge vectors in to an exponential radial basis:

    .. math::
      
         h_{R,n}^{(ij)} = \exp(\beta_n \cdot (\exp(- r_{ij}) - \mu_k)^2)

    4. combine all edge embeddings:

    .. math::

        X^{ij} = \phi(r_{ij}) \cdot h_Z^{(ij)} \cdot \left ( 
        \text{Dense}(h_R^{(ij)}) \cdot I_0^{(ij)} +
        \text{Dense}(h_R^{(ij)}) \cdot A_0^{(ij)} +
        \text{Dense}(h_R^{(ij)}) \cdot S_0^{(ij)}
        \right)

    where :math:`\phi(r_{ij})` is the cosine envelope function.
    """

    def __init__(
        self,
        radial_features: int,
        radial_expansion: str | type[DistanceExpansion],
        channels: int,
        cutoff: float,
    ):
        super().__init__()

        self.z_embedding = PerElementEmbedding(channels)
        self.z_map = nn.Linear(2 * channels, channels, bias=False)
        expansion_klass = get_distance_expansion(radial_expansion)
        self.distance_embedding = HaddamardProduct(
            nn.Sequential(
                expansion_klass(radial_features, cutoff),
                nn.Linear(radial_features, 3 * channels),
            ),
            CosineEnvelope(cutoff),
            left_aligned=True,
        )

    def forward(self, graph: AtomicGraph) -> tuple[Tensor, Tensor, Tensor]:
        C = self.z_embedding.dim()
        E = number_of_edges(graph)

        # 1. generate initial edge embedding components:
        I_0, A_0, S_0 = self._initial_edge_embeddings(graph)  # (E, 1, 3, 3)

        # 2. encode atomic species of ordered neighbour pairs:
        h_z_atom = self.z_embedding(graph.Z)  # (N, C)
        h_z_edge = h_z_atom[graph.neighbour_list]  # (2, E, C)
        h_z_edge = h_z_edge.permute(1, 0, 2).reshape(E, 2 * C)
        h_z_edge = self.z_map(h_z_edge)  # (E, C)

        # 3. embed edge distances
        h_r = self.distance_embedding(neighbour_distances(graph))  # (E, 3C)

        # 4. combine information into coefficients
        c = (h_r * h_z_edge.repeat(1, 3))[..., None, None]  # (E, 3C, 1, 1)
        c_I, c_A, c_S = torch.chunk(c, 3, dim=1)  # (E, C, 1, 1)

        return c_I * I_0, c_A * A_0, c_S * S_0  # 3x (E, C, 3, 3)

    def _initial_edge_embeddings(
        self, graph: AtomicGraph
    ) -> tuple[Tensor, Tensor, Tensor]:
        E = number_of_edges(graph)
        r_hat = neighbour_vectors(graph) / neighbour_distances(graph)[..., None]
        eye = torch.eye(3, device=graph.Z.device)
        I_ij = torch.repeat_interleave(eye[None, ...], E, dim=0)  # (E, 3, 3)
        A_ij = vector_to_skew_symmetric_matrix(r_hat)
        S_ij = vector_to_symmetric_traceless_matrix(r_hat)
        return (
            I_ij.view(E, 1, 3, 3),
            A_ij.view(E, 1, 3, 3),
            S_ij.view(E, 1, 3, 3),
        )


class Embedding(nn.Module):
    """
    Embed the local environment of each atom into a ``(C, 3, 3)`` tensor.
    """

    def __init__(
        self,
        radial_features: int,
        radial_expansion: str | type[DistanceExpansion],
        channels: int,
        cutoff: float,
    ):
        super().__init__()
        self.edge_embedding = EdgeEmbedding(
            radial_features, radial_expansion, channels, cutoff
        )
        self.layer_norm = nn.LayerNorm(channels)
        self.mlp = MLP(
            layers=[
                channels,
                2 * channels,
                3 * channels,
            ],
            activation=nn.SiLU(),
            activate_last=True,
        )
        self.W_I = TensorLinear(channels, channels)
        self.W_A = TensorLinear(channels, channels)
        self.W_S = TensorLinear(channels, channels)

    def forward(self, graph: AtomicGraph) -> Tensor:
        # embed edges
        I_edge, A_edge, S_edge = self.edge_embedding(graph)  # (E, C, 3, 3)

        # sum over neighbours to get atom embeddings
        I_atom = sum_over_neighbours(I_edge, graph)  # (N, C, 3, 3)
        A_atom = sum_over_neighbours(A_edge, graph)  # (N, C, 3, 3)
        S_atom = sum_over_neighbours(S_edge, graph)  # (N, C, 3, 3)

        # generate coefficients from tensor representations
        # (mixes irreps)
        norms = frobenius_norm(I_atom + A_atom + S_atom)  # (N, C)
        norms = self.layer_norm(norms)
        coefficients = self.mlp(norms)[..., None, None]  # (N, 3C, 1, 1)
        c_I, c_A, c_S = torch.chunk(coefficients, 3, dim=1)  # (N, C, 1, 1)

        # ...and combine with mixed coefficients with linear
        # mixing of features
        return (
            c_I * self.W_I(I_atom)
            + c_A * self.W_A(A_atom)
            + c_S * self.W_S(S_atom)
        )  # (N, C, 3, 3)


class Interaction(nn.Module):
    def __init__(
        self,
        radial_features: int,
        channels: int,
        cutoff: float,
    ):
        super().__init__()

        # unfortunately, we need to be explicit to satisfy torchscript
        self.W_I_pre = TensorLinear(channels, channels)
        self.W_A_pre = TensorLinear(channels, channels)
        self.W_S_pre = TensorLinear(channels, channels)
        self.W_I_post = TensorLinear(channels, channels)
        self.W_A_post = TensorLinear(channels, channels)
        self.W_S_post = TensorLinear(channels, channels)

        self.distance_embedding = HaddamardProduct(
            nn.Sequential(
                ExponentialRBF(radial_features, cutoff),
                MLP(
                    layers=[radial_features, 2 * channels, 3 * channels],
                    activation=nn.SiLU(),
                    activate_last=True,
                ),
            ),
            CosineEnvelope(cutoff),
            left_aligned=True,
        )

    def forward(self, X: Tensor, graph: AtomicGraph) -> Tensor:
        # decompose matrix representations
        I, A, S = decompose_matrix(X)  # (N, C, 3, 3)

        # update I, A, S
        Y = self.W_I_pre(I) + self.W_A_pre(A) + self.W_S_pre(S)  # (N, C, 3, 3)

        # get coefficients from neighbour distances
        c = self.distance_embedding(neighbour_distances(graph))  # (E, 3C)
        f_I, f_A, f_S = torch.chunk(c[..., None, None], 3, dim=1)

        # message passing
        m_ij = (
            f_I * index_over_neighbours(I, graph)
            + f_A * index_over_neighbours(A, graph)
            + f_S * index_over_neighbours(S, graph)
        )
        # total message
        M_i = sum_over_neighbours(m_ij, graph)  # (N, C, 3, 3)

        # scalar/vector/tensor mixing
        Y = Y @ M_i + M_i @ Y  # (N, C, 3, 3)

        # renormalise and decompose
        norm = (frobenius_norm(Y) + 1)[..., None, None]  # (N, C, 1, 1)
        I, A, S = decompose_matrix(Y / norm)  # (N, C, 3, 3)

        # mix features again
        Y = self.W_I_post(I) + self.W_A_post(A) + self.W_S_post(S)
        return Y + torch.matrix_power(Y, 2)  # (N, C, 3, 3)


class ScalarOutput(nn.Module):
    """
    A non-linear read-out function:

    ``X`` with shape ``(N, C, 3, 3)`` is decomposed into ``I``, ``A``, and
    ``S`` components. The concatenation of the Frobenius norms of these
    components are passed through an MLP to generate a scalar.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(3 * channels)
        self.mlp = MLP(
            layers=[3 * channels, 2 * channels, 1],
            activation=nn.SiLU(),
        )

    def forward(self, X: Tensor) -> Tensor:
        """X: (N, C, 3, 3) --> (N, 1)"""
        I, A, S = decompose_matrix(X)  # (N, C, 3, 3)
        norm_I = frobenius_norm(I)  # (N, C)
        norm_A = frobenius_norm(A)
        norm_S = frobenius_norm(S)

        X = torch.cat((norm_I, norm_A, norm_S), dim=-1)  # (N, 3C)
        X = self.layer_norm(X)
        return self.mlp(X)  # (N, 1)


class VectorOutput(nn.Module):
    """
    A non-linear read-out function:

    The ``A`` component of ``X`` with shape ``(N, C, 3, 3)`` is passed through
    a linear layer, before extracting the ``x``, ``y``, and ``z`` components
    of the resulting vector.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.linear = TensorLinear(channels, 1)

    def forward(self, X: Tensor) -> Tensor:
        """X: (N, C, 3, 3) --> (N, 3)"""
        _, A, _ = decompose_matrix(X)  # (N, C, 3, 3)
        A_final = self.linear(A).squeeze()  # (N, 3, 3)
        # A_final[b, c] = [
        #     [  0, -vz,  vy],
        #     [ vz,   0, -vx],
        #     [-vy,  vx,   0],
        # ]
        x = A_final[..., 2, 1]
        y = A_final[..., 0, 2]
        z = A_final[..., 1, 0]
        return torch.stack((x, y, z), dim=-1)  # (N, 3)


### utils ###


def one_third_trace(X: Tensor) -> Tensor:
    """
    Calculate the one third trace of a (optionally batched) matrix, ``X``, of
    shape ``(...B, 3, 3)``.
    """
    return X.diagonal(offset=0, dim1=-1, dim2=-2).mean(-1)


def frobenius_norm(X: Tensor) -> Tensor:
    """
    Calculate the Frobenius norm of a (optionally batched) matrix, ``X``, of
    shape ``(...B, 3, 3)``.
    """
    return (X**2).sum((-2, -1))


def decompose_matrix(X: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """
    Take a (optionally batched) matrix, ``X``, of shape ``(...B, 3, 3)`` and
    decompose it into irreducible components, ``I``, ``A``, and ``S``:

    * ``I[b] = 1/3 * trace(X[b]) * Id``
    * ``A[b] = 0.5 * (X[b] - X[b].T)``
    * ``S[b] = 0.5 * (X[b] + X[b].T) - I[b]``

    where ``Id`` is the ``3x3`` identity matrix and ``b`` is some
    batch dimension(s)

    Parameters
    ----------
    x
        The matrix to decompose, of shape ``(...B, 3, 3)``.
    """
    trace = one_third_trace(X)[..., None, None]  # (...B, 1, 1)
    I = trace * torch.eye(3, 3, device=X.device, dtype=X.dtype)  # (...B, 3, 3)
    A = 0.5 * (X - X.transpose(-2, -1))
    S = 0.5 * (X + X.transpose(-2, -1)) - I
    return I, A, S  # 3x (...B, 3, 3)


def vector_to_skew_symmetric_matrix(v: Tensor) -> Tensor:
    """
    Creates a skew-symmetric tensor from a (optionally batched) vector.

    v: ([B], 3) --> sst: ([B], 3, 3)
    sst[b] = [
        [     0,   -v[b].z,   v[b].y],
        [ v[b].z,        0,  -v[b].x],
        [-v[b].y,   v[b].x,        0],
    ]

    """
    x, y, z = v.unbind(dim=-1)
    zero = torch.zeros_like(x)
    tensor = torch.stack(
        (
            zero,
            -z,
            y,
            z,
            zero,
            -x,
            -y,
            x,
            zero,
        ),
        dim=1,
    )  # (B, 9)
    return tensor.reshape(tensor.shape[:-1] + (3, 3))


def vector_to_symmetric_traceless_matrix(v: Tensor) -> Tensor:
    """
    Creates a symmetric traceless matrix from a vector.

    v: (..., 3) --> stm: (..., 3, 3)
    stm[...b] = 0.5 * (v[...b] v[...b].T) - 1/3 * trace(v[...b] v[...b].T) * Id
    """
    v_vT = torch.matmul(v.unsqueeze(-1), v.unsqueeze(-2))
    Id = torch.eye(3, 3, device=v_vT.device, dtype=v_vT.dtype)
    I = one_third_trace(v_vT)[..., None, None] * Id
    return 0.5 * (v_vT + v_vT.transpose(-2, -1)) - I


class TensorLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """x: (N, in, 3, 3) --> (N, out, 3, 3)"""
        return self.linear(x.transpose(-1, -3)).transpose(-1, -3)

    def __repr__(self):
        _in, _out = self.linear.in_features, self.linear.out_features
        return (
            f"{self.__class__.__name__}("
            f"[N, {_in}, 3, 3] --> [N, {_out}, 3, 3])"
        )
