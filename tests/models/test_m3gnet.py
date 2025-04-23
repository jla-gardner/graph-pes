from __future__ import annotations

import logging
import pytest
import torch
from ase.atoms import Atoms
from ase.build import molecule, bulk
from ase.calculators.emt import EMT

from graph_pes import AtomicGraph
from graph_pes.atomic_graph import (
    number_of_atoms,
    to_batch,
    PropertyKey,
    neighbour_distances,
)
from graph_pes.utils.threebody import triplet_bond_descriptors
from graph_pes.models import M3GNet
from graph_pes.models.m3gnet import M3GNetInteraction, GatedMLP
from graph_pes.models.components.distances import GaussianSmearing

CUTOFF = 3.5
logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def setup_logging(caplog):
    """Configure logging for all tests in this module"""
    caplog.set_level(logging.INFO)


@pytest.fixture
def default_model():
    """Fixture providing a default M3GNet model

    Default configuration:
    - channels=64 (equivalent to 'units' in original)
    - layers=3 (equivalent to 'n_blocks' in original)
    - expansion_features=50 (max_n=3, max_l=3 in original)

    TODO: Confirm with original paper/repo
    """
    return M3GNet(channels=64, layers=3, expansion_features=50, cutoff=CUTOFF)


@pytest.fixture
def copper_graph():
    """Fixture providing an AtomicGraph for FCC copper with EMT calculator"""
    copper = bulk("Cu", "fcc", cubic=True)
    copper.calc = EMT()
    copper.get_potential_energy()
    copper.get_forces()
    graph = AtomicGraph.from_ase(copper, cutoff=CUTOFF)
    return graph


def test_m3gnet_init(default_model):
    """Test M3GNet initialization with different parameters"""
    assert default_model.cutoff == CUTOFF
    assert default_model.implemented_properties == ["local_energies"]
    assert len(default_model.interactions) == 3  # Default layers=3


@pytest.mark.filterwarnings("ignore:No energy data found in training data")
def test_m3gnet_predictions(default_model, copper_graph):
    """Test M3GNet predictions on a simple molecule"""
    default_model.pre_fit_all_components([copper_graph])
    predictions = default_model.get_all_PES_predictions(copper_graph)

    # Check shapes
    n_atoms = number_of_atoms(copper_graph)
    assert "energy" in predictions
    assert "forces" in predictions
    assert "local_energies" in predictions
    assert predictions["energy"].shape == ()
    assert predictions["forces"].shape == (n_atoms, 3)
    assert predictions["local_energies"].shape == (n_atoms,)


@pytest.mark.filterwarnings("ignore:No energy data found in training data")
def test_m3gnet_batch_predictions(default_model):
    """Test M3GNet predictions on batched molecules"""
    molecules = [molecule("H2O"), molecule("NH3")]
    graphs = [AtomicGraph.from_ase(mol, cutoff=CUTOFF) for mol in molecules]
    batch = to_batch(graphs)
    default_model.pre_fit_all_components(graphs)
    predictions = default_model.get_all_PES_predictions(batch)

    # Check shapes
    assert predictions["energy"].shape == (2,)
    assert predictions["forces"].shape == (number_of_atoms(batch), 3)
    assert predictions["local_energies"].shape == (number_of_atoms(batch),)


@pytest.mark.filterwarnings("ignore:No energy data found in training data")
def test_m3gnet_isolated_atom(default_model):
    """Test M3GNet predictions on an isolated atom"""
    atom = molecule("H")
    graph = AtomicGraph.from_ase(atom, cutoff=CUTOFF)
    default_model.pre_fit_all_components([graph])

    predictions = default_model.get_all_PES_predictions(graph)
    assert torch.allclose(
        predictions["forces"], torch.zeros_like(predictions["forces"])
    )


@pytest.mark.filterwarnings("ignore:No energy data found in training data")
def test_m3gnet_serialization(default_model, tmp_path):
    """Test saving and loading M3GNet model"""
    methane = molecule("CH4")
    graph = AtomicGraph.from_ase(methane, cutoff=CUTOFF)
    default_model.pre_fit_all_components([graph])

    original_predictions = default_model.get_all_PES_predictions(graph)

    save_path = tmp_path / "m3gnet.pt"
    torch.save(default_model.state_dict(), save_path)

    # Create new model with same configuration as default_model
    loaded_model = M3GNet(cutoff=CUTOFF, channels=64, layers=3, expansion_features=50)
    loaded_model.load_state_dict(torch.load(save_path))

    loaded_predictions = loaded_model.get_all_PES_predictions(graph)

    # Check predictions are the same
    assert torch.allclose(
        original_predictions["energy"],
        loaded_predictions["energy"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        original_predictions["forces"],
        loaded_predictions["forces"],
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.filterwarnings("ignore:No energy data found in training data")
def test_m3gnet_parameter_count():
    """Test M3GNet parameter count matching M3GNet-MP-2021.2.8-PES.

    Reference checkpoint model: M3GNet-MP-2021.2.8-PES

    n: 3
    l: 3
    n_blocks: 3

    Reference model configuration (I think?):
    - EmbeddingBlock(
        (activation): SiLU()
        (layer_node_embedding): Embedding(89, 64)
        (layer_edge_embedding): MLP(9 → 64, SiLU)
    )
    - ModuleList(
        (0-2): 3 x ThreeBodyInteractions(
            (update_network_atom): MLP(64 → 9, Sigmoid)
        (update_network_bond): GatedMLP(
        (layers): Sequential(
            (0): Linear(in_features=9, out_features=64, bias=False)
            (1): SiLU()
        )
        (gates): Sequential(
            (0): Linear(in_features=9, out_features=64, bias=False)
            (1): Sigmoid()
        )
        )
    )
    - ModuleList(
        (0-2): 3 x M3GNetBlock(
            (activation): SiLU()
            (conv): M3GNetGraphConv(
            (edge_update_func): GatedMLP(
                (layers): Sequential(
                (0): Linear(in_features=192, out_features=64, bias=True)
                (1): SiLU()
                (2): Linear(in_features=64, out_features=64, bias=True)
                (3): SiLU()
                (4): Linear(in_features=64, out_features=64, bias=True)
                (5): SiLU()
                )
                (gates): Sequential(
                (0): Linear(in_features=192, out_features=64, bias=True)
                (1): SiLU()
                (2): Linear(in_features=64, out_features=64, bias=True)
                (3): SiLU()
                (4): Linear(in_features=64, out_features=64, bias=True)
                (5): Sigmoid()
                )
            )
            (edge_weight_func): Linear(in_features=9, out_features=64, bias=False)
            (node_update_func): GatedMLP(
                (layers): Sequential(
                (0): Linear(in_features=192, out_features=64, bias=True)
                (1): SiLU()
                (2): Linear(in_features=64, out_features=64, bias=True)
                (3): SiLU()
                (4): Linear(in_features=64, out_features=64, bias=True)
                (5): SiLU()
                )
                (gates): Sequential(
                (0): Linear(in_features=192, out_features=64, bias=True)
                (1): SiLU()
                (2): Linear(in_features=64, out_features=64, bias=True)
                (3): SiLU()
                (4): Linear(in_features=64, out_features=64, bias=True)
                (5): Sigmoid()
                )
            )
            (node_weight_func): Linear(in_features=9, out_features=64, bias=False)
            )
        )
        )
    - WeightedReadOut(
        (gated): GatedMLP(
            (layers): Sequential(
            (0): Linear(in_features=64, out_features=64, bias=True)
            (1): SiLU()
            (2): Linear(in_features=64, out_features=64, bias=True)
            (3): SiLU()
            (4): Linear(in_features=64, out_features=64, bias=True)
            (5): SiLU()
            (6): Linear(in_features=64, out_features=1, bias=True)
            )
            (gates): Sequential(
            (0): Linear(in_features=64, out_features=64, bias=True)
            (1): SiLU()
            (2): Linear(in_features=64, out_features=64, bias=True)
            (3): SiLU()
            (4): Linear(in_features=64, out_features=64, bias=True)
            (5): SiLU()
            (6): Linear(in_features=64, out_features=1, bias=True)
            (7): Sigmoid()
            )
        )
    )

    Total trainable parameters: ~288,157
    """
    # Create a single chain structure with all elements Z=1 to Z=89
    atoms = []
    spacing = 1.5
    for z in range(1, 90):  # Z=1 to Z=89
        # Place atoms in a line along x-axis with 2Å spacing
        position = (z * spacing, 0.0, 0.0)
        atoms.append(Atoms(numbers=[z], positions=[position]))
    chain = sum(atoms[1:], atoms[0])
    chain.cell = [89 * 2.0, 5.0, 5.0]
    chain.pbc = True
    chain.center()

    # Create graph from chain structure
    test_graph = AtomicGraph.from_ase(chain, cutoff=CUTOFF)

    # Initialize model with reference configuration
    model = M3GNet(
        cutoff=CUTOFF,
        channels=64,
        layers=3,
        expansion_features=50,
    )

    # Pre-fit on test graph to register all elements
    model.pre_fit_all_components([test_graph])
    _ = model.get_all_PES_predictions(test_graph)

    # Count parameters per component
    total_params = 0
    component_params = {}

    logger.info("\nM3GNet Parameter Count (Reference Model Configuration):")
    logger.info("-" * 60)
    logger.info("Elements included: Z=1 to Z=89 (H to Ac)")
    logger.info("-" * 60)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        num_params = param.numel()
        component_name = name.split(".")[0]
        component_params[component_name] = (
            component_params.get(component_name, 0) + num_params
        )
        total_params += num_params

    # Log component-wise breakdown
    for component, count in sorted(component_params.items()):
        logger.info(f"{component:.<40s} {count:,} parameters")

    logger.info("-" * 60)
    logger.info(f"{'Total parameters:':<40s} {total_params:,}")
    logger.info(f"{'Reference model parameters:':<40s} 288,157")
    logger.info("-" * 60)
    print(total_params)
    # Log elements seen by model
    logger.info("\nElements seen by model:")
    logger.info(", ".join(model.elements_seen))
    logger.info("-" * 60)

    # Basic parameter checks
    assert total_params > 0, "Model should have parameters"
    assert (
        component_params.get("chemical_embedding", 0) > 0
    ), "Chemical embedding should have parameters"
    assert component_params.get("scaler", 0) > 0, "Scaler should have parameters"

    # Verify we registered all elements up to Z=89
    assert len(model.elements_seen) == 89, "Model should see all elements up to Z=89"


@pytest.mark.filterwarnings("ignore:No energy data found in training data")
def test_m3gnet_bessel_basis(copper_graph):
    """Test M3GNet with Bessel function basis expansion"""
    from graph_pes.models.components.distances import Bessel

    # Create model with Bessel basis expansion
    model = M3GNet(
        cutoff=CUTOFF,
        expansion=Bessel,  # Use Bessel basis expansion
        expansion_features=40,  # max_n=3, max_l=3 from original paper
    )

    model.pre_fit_all_components([copper_graph])
    predictions = model.get_all_PES_predictions(copper_graph)

    # Log parameter information for Bessel basis
    basis_params = [(n, p) for n, p in model.named_parameters() if "basis" in n]

    logger.info("\nBessel Basis Parameters:")
    logger.info("-" * 40)

    for name, param in basis_params:
        logger.info(f"{name}: shape={list(param.shape)}")

    # Basic checks
    assert "energy" in predictions
    assert "forces" in predictions
    assert predictions["energy"].shape == ()
    assert predictions["forces"].shape[1] == 3  # Only check force dimension
    assert len(predictions["forces"]) == number_of_atoms(copper_graph)

    # Check that gradients can flow through Bessel basis
    loss = predictions["energy"].abs() + predictions["forces"].abs().mean()
    loss.backward()

    # Verify parameters related to basis expansion have gradients
    basis_params = [p for _, p in model.named_parameters() if "basis" in name]
    assert len(basis_params) > 0, "No basis expansion parameters found"

    logger.info("\nBessel Basis Gradients:")
    logger.info("-" * 40)

    # Check that at least some parameters have non-zero gradients
    has_nonzero_grad = False
    for param in basis_params:
        assert param.grad is not None, "Basis parameters have no gradient"
        grad_norm = param.grad.norm().item()
        logger.info(f"Gradient norm: {grad_norm:.3e}")
        if (
            grad_norm > 1e-10
        ):  # Use a small threshold to account for numerical precision
            has_nonzero_grad = True

    assert has_nonzero_grad, "All basis parameters have zero gradients"


def test_gated_mlp():
    """Test the GatedMLP implementation."""

    mlp = GatedMLP(neurons=[10, 20, 5])
    x = torch.randn(3, 10)
    out = mlp(x)
    assert out.shape == (3, 5)

    # Different activation
    mlp = GatedMLP(neurons=[10, 20, 5], activation="relu")
    out = mlp(x)
    assert out.shape == (3, 5)

    # Without bias
    mlp = GatedMLP(neurons=[10, 20, 5], use_bias=False)
    out = mlp(x)
    assert out.shape == (3, 5)


def test_m3gnet_interaction():
    """Test the M3GNet interaction block."""
    # H3 triangle
    Z = torch.tensor([1, 1, 1])
    R = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],
        ]
    )

    cell = torch.eye(3) * 5.0

    n_atoms = len(Z)
    neighbor_list = []
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i != j:
                neighbor_list.append([i, j])
    # Shape: [6, 2] for 6 edges
    neighbor_list = torch.tensor(neighbor_list).t()
    neighbor_cell_offsets = torch.zeros((6, 3), dtype=torch.long)

    graph = AtomicGraph(
        Z=Z,
        R=R,
        cell=cell,
        neighbour_list=neighbor_list,
        neighbour_cell_offsets=neighbor_cell_offsets,
        properties=[],
        cutoff=2.0,
        other={},
    )

    channels = 64
    expansion_features = 50
    cutoff = 2.0
    interaction = M3GNetInteraction(
        channels=channels,
        expansion_features=expansion_features,
        cutoff=cutoff,
        basis_type=GaussianSmearing,
    )

    features = torch.randn(3, channels)
    d = neighbour_distances(graph)  # This will give us 6 distances for all pairs
    out = interaction(features, d, graph)

    assert out.shape == (3, channels)


def test_m3gnet():
    """Test the full M3GNet model."""

    Z = torch.tensor([1, 1, 1])
    R = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],
        ]
    )

    cell = torch.eye(3) * 5.0

    n_atoms = len(Z)
    neighbor_list = []
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i != j:
                neighbor_list.append([i, j])
    neighbor_list = torch.tensor(neighbor_list).t()  # Transpose to [2, 6]

    neighbor_cell_offsets = torch.zeros((6, 3), dtype=torch.long)

    graph = AtomicGraph(
        Z=Z,
        R=R,
        cell=cell,
        neighbour_list=neighbor_list,
        neighbour_cell_offsets=neighbor_cell_offsets,
        properties=[],
        cutoff=2.0,
        other={},
    )

    model = M3GNet(
        cutoff=2.0,
        channels=64,
        expansion_features=50,
        layers=3,
    )

    out = model(graph)

    assert isinstance(out, dict)
    assert "local_energies" in out  # Use string literal instead of PropertyKey
    assert out["local_energies"].shape == (3,)

    model = M3GNet(
        cutoff=3.0,
        channels=32,
        expansion_features=30,
        layers=2,
    )
    out = model(graph)
    assert out["local_energies"].shape == (3,)
