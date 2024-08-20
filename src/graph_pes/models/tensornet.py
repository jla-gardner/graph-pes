from __future__ import annotations

import torch
from torch import Tensor, nn

from graph_pes.core import ConservativePESModel
from graph_pes.graphs import AtomicGraph
from graph_pes.graphs.operations import (
    neighbour_distances,
    neighbour_vectors,
    number_of_atoms,
    number_of_edges,
)
from graph_pes.nn import (
    MLP,
    HaddamardProduct,
    PerElementEmbedding,
    UniformModuleList,
)

from .distances import CosineEnvelope, ExponentialRBF


def decompose_tensor(x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Full tensor decomposition into irreducible components."""
    I = (x.diagonal(offset=0, dim1=-1, dim2=-2)).mean(-1)[
        ..., None, None
    ] * torch.eye(3, 3, device=x.device, dtype=x.dtype)
    A = 0.5 * (x - x.transpose(-2, -1))
    S = 0.5 * (x + x.transpose(-2, -1)) - I
    return I, A, S


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
    """

    def __init__(
        self,
        radial_features: int,
        embedding_size: int,
        cutoff: float,
    ):
        super().__init__()
        self.z_embedding = PerElementEmbedding(embedding_size)
        self.z_map = nn.Linear(2 * embedding_size, embedding_size, bias=False)
        self.distance_embedding = HaddamardProduct(
            nn.Sequential(
                ExponentialRBF(radial_features, cutoff),
                nn.Linear(radial_features, 3 * embedding_size),
            ),
            CosineEnvelope(cutoff),
            left_aligned=True,
        )

    def forward(self, graph: AtomicGraph) -> tuple[Tensor, Tensor, Tensor]:
        # 1. generate initial edge embedding components:
        I_0, A_0, S_0 = self._initial_edge_embeddings(graph)  # (E, 1, 3, 3)

        # 2. encode atomic species
        h_z_atom = self.z_embedding(graph["atomic_numbers"])  # (N, C)
        C = h_z_atom.shape[-1]
        h_z_edge = h_z_atom[graph["neighbour_index"]].reshape(
            -1, 2 * C
        )  # (E, 2C)
        h_z_edge = self.z_map(h_z_edge)  # (E, C)

        # 3. embed edge distances
        h_r = self.distance_embedding(neighbour_distances(graph))  # (E, 3C)

        # 4. combine information into coefficients
        c = (h_r * h_z_edge.repeat(1, 3))[..., None, None]  # (E, 3C, 1, 1)
        c_I, c_A, c_S = torch.chunk(c, 3, dim=1)  # (E, C, 1, 1)

        return c_I * I_0, c_A * A_0, c_S * S_0  # 3x (E, 1, 3, 3)

    def _initial_edge_embeddings(
        self, graph: AtomicGraph
    ) -> tuple[Tensor, Tensor, Tensor]:
        n_edges = number_of_edges(graph)
        r_hat = neighbour_vectors(graph) / neighbour_distances(graph)[..., None]
        eye = torch.eye(3, device=graph["atomic_numbers"].device)
        I_ij = torch.repeat_interleave(eye[None, ...], n_edges, dim=0)
        A_ij = vector_to_skew_vector(r_hat)
        S_ij = vector_to_symmetric_tensor(r_hat)
        return (
            I_ij.view(n_edges, 1, 3, 3),
            A_ij.view(n_edges, 1, 3, 3),
            S_ij.view(n_edges, 1, 3, 3),
        )


def vector_to_skew_vector(v: Tensor) -> Tensor:
    """
    Creates a skew-symmetric tensor from a vector.

    v: (..., 3) --> (..., 3, 3)
    """
    batch_size = v.size(0)
    zero = torch.zeros(batch_size, device=v.device, dtype=v.dtype)
    tensor = torch.stack(
        (
            zero,
            -v[:, 2],
            v[:, 1],
            v[:, 2],
            zero,
            -v[:, 0],
            -v[:, 1],
            v[:, 0],
            zero,
        ),
        dim=1,
    )
    tensor = tensor.view(-1, 3, 3)
    return tensor.squeeze(0)


def vector_to_symmetric_tensor(v: Tensor) -> Tensor:
    """
    Creates a symmetric tensor from a vector.

    v: (..., 3) --> (..., 3, 3)
    """
    tensor = torch.matmul(v.unsqueeze(-1), v.unsqueeze(-2))
    Id = (tensor.diagonal(offset=0, dim1=-1, dim2=-2)).mean(-1)[
        ..., None, None
    ] * torch.eye(3, 3, device=tensor.device, dtype=tensor.dtype)
    S = 0.5 * (tensor + tensor.transpose(-2, -1)) - Id
    return S


def frobenius_norm(t):
    return (t**2).sum((-2, -1))


class TensorLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """x: (N, in, 3, 3) --> (N, out, 3, 3)"""
        return self.linear(x.transpose(-1, -3)).transpose(-1, -3)


def sum_per_atom(edge_tensor: Tensor, graph: AtomicGraph) -> Tensor:
    N = number_of_atoms(graph)
    C = edge_tensor.shape[1]
    central_atom = graph["neighbour_index"][0]

    atom_tensor = torch.zeros(N, C, 3, 3, device=graph["cell"].device)
    atom_tensor.scatter_add_(
        0,
        central_atom[:, None, None, None].expand_as(edge_tensor),
        edge_tensor,
    )
    return atom_tensor


class Embedding(nn.Module):
    def __init__(
        self,
        radial_features: int,
        embedding_size: int,
        cutoff: float,
    ):
        super().__init__()
        self.edge_embedding = EdgeEmbedding(
            radial_features, embedding_size, cutoff
        )
        self.layer_norm = nn.LayerNorm(embedding_size)
        self.mlp = MLP(
            layers=[
                embedding_size,
                2 * embedding_size,
                3 * embedding_size,
            ],
            activation=nn.SiLU(),
            activate_last=True,
        )
        self.W_I = TensorLinear(embedding_size, embedding_size)
        self.W_A = TensorLinear(embedding_size, embedding_size)
        self.W_S = TensorLinear(embedding_size, embedding_size)

    def forward(self, graph: AtomicGraph) -> Tensor:
        I_edge, A_edge, S_edge = self.edge_embedding(graph)  # (E, C, 3, 3)

        I_atom = sum_per_atom(I_edge, graph)  # (N, C, 3, 3)
        A_atom = sum_per_atom(A_edge, graph)  # (N, C, 3, 3)
        S_atom = sum_per_atom(S_edge, graph)  # (N, C, 3, 3)

        norms = frobenius_norm(I_atom + A_atom + S_atom)  # (N, C)
        norms = self.layer_norm(norms)
        coefficients = self.mlp(norms)[..., None, None]  # (N, 3C, 1, 1)
        c_I, c_A, c_S = torch.chunk(coefficients, 3, dim=1)  # (N, C, 1, 1)

        return (
            c_I * self.W_I(I_atom)
            + c_A * self.W_A(A_atom)
            + c_S * self.W_S(S_atom)
        )  # (N, C, 3, 3)


class Interaction(nn.Module):
    def __init__(
        self,
        radial_features: int,
        embedding_size: int,
        cutoff: float,
    ):
        super().__init__()
        self.mlp = MLP(
            layers=[embedding_size, 2 * embedding_size, 3 * embedding_size],
            activation=nn.SiLU(),
            activate_last=True,
        )

        # unfortunately, we need to be explicit to satisfy torchscript
        self.W_I_pre = TensorLinear(embedding_size, embedding_size)
        self.W_A_pre = TensorLinear(embedding_size, embedding_size)
        self.W_S_pre = TensorLinear(embedding_size, embedding_size)
        self.W_I_post = TensorLinear(embedding_size, embedding_size)
        self.W_A_post = TensorLinear(embedding_size, embedding_size)
        self.W_S_post = TensorLinear(embedding_size, embedding_size)

        self.distance_embedding = HaddamardProduct(
            nn.Sequential(
                ExponentialRBF(radial_features, cutoff),
                nn.Linear(radial_features, 3 * embedding_size),
            ),
            CosineEnvelope(cutoff),
            left_aligned=True,
        )

    def forward(self, X: Tensor, graph: AtomicGraph) -> Tensor:
        X = X / (frobenius_norm(X)[..., None, None] + 1)  # (N, C, 3, 3)
        I, A, S = decompose_tensor(X)  # (N, C, 3, 3)

        # update I, A, S
        Y = self.W_I_pre(I) + self.W_A_pre(A) + self.W_S_pre(S)  # (N, C, 3, 3)

        # edge-wise mixing
        neighbours = graph["neighbour_index"][1]  # (E,)
        coeffecients = self.distance_embedding(
            neighbour_distances(graph)
        )  # (E, 3C)
        f_I, f_A, f_S = torch.chunk(
            coeffecients[..., None, None], 3, dim=1
        )  # (E, C, 1, 1)

        M_ij = f_I * I[neighbours] + f_A * A[neighbours] + f_S * S[neighbours]
        M_i = torch.zeros_like(X)  # (N, C, 3, 3)
        M_i.scatter_add_(
            0, neighbours[:, None, None, None].expand_as(M_ij), M_ij
        )

        # scalar/vector/tensor mixing
        Y = Y @ M_i + M_i @ Y  # (N, C, 3, 3)
        norm = frobenius_norm(Y + 1)[..., None, None]  # (N, C, 1, 1)
        I, A, S = decompose_tensor(Y / norm)  # (N, C, 3, 3)
        Y = self.W_I_post(I) + self.W_A_post(A) + self.W_S_post(S)

        return Y + torch.matrix_power(Y, 2)  # (N, C, 3, 3)


class ScalarOutput(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.mlp = MLP(
            layers=[3 * embedding_size, 2 * embedding_size, 1],
            activation=nn.SiLU(),
        )

    def forward(self, X: Tensor) -> Tensor:
        """X: (N, C, 3, 3) --> (N, 1)"""
        I, A, S = decompose_tensor(X)  # (N, C, 3, 3)
        norm_I = frobenius_norm(I)
        norm_A = frobenius_norm(A)
        norm_S = frobenius_norm(S)

        X = torch.cat((norm_I, norm_A, norm_S), dim=-1)  # (N, 3C)
        return self.mlp(X)  # (N, 1)


class TensorNet(ConservativePESModel):
    def __init__(
        self,
        radial_features: int = 32,
        embedding_size: int = 32,
        cutoff: float = 5.0,
        layers: int = 2,
    ):
        super().__init__(cutoff, auto_scale=True)
        self.embedding = Embedding(radial_features, embedding_size, cutoff)
        self.interactions = UniformModuleList(
            Interaction(radial_features, embedding_size, cutoff)
            for _ in range(layers)
        )
        self.read_out = ScalarOutput(embedding_size)

    def predict_local_energies(self, graph: AtomicGraph):
        X = self.embedding(graph)  # (N, C, 3, 3)
        for interaction in self.interactions:
            X = interaction(X, graph) + X  # residual connection
        return self.read_out(X)  # (N, 1)
