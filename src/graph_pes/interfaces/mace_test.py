import torch
from e3nn import o3
from mace.modules import ScaleShiftMACE, gate_dict, interaction_classes

ELEMENTS = [1, 6, 8]
CUTOFF = 5.0


def create_default_scaleshift_mace(
    atomic_numbers: list,
    r_max: float = 5.0,
    num_bessel: int = 8,
    num_polynomial_cutoff: int = 5,
    max_ell: int = 3,
    num_interactions: int = 3,
    hidden_irreps: str = "32x0e + 32x1o",
    mlp_irreps: str = "16x0e",
    atomic_energies: torch.Tensor = None,
    avg_num_neighbors: float = 1.0,
    atomic_inter_scale: float = 1.0,
    atomic_inter_shift: float = 0.0,
) -> ScaleShiftMACE:
    """Creates a ScaleShiftMACE model with default parameters.

    Args:
        atomic_numbers: List of atomic numbers to include in the model
        r_max: Cutoff radius in Angstroms
        num_bessel: Number of radial basis functions
        num_polynomial_cutoff: Number of polynomial functions for cutoff
        max_ell: Maximum L for spherical harmonics
        num_interactions: Number of interaction blocks
        hidden_irreps: Irreps string for hidden features
        mlp_irreps: Irreps string for MLP in final layer
        atomic_energies: Per-atom energy shifts (defaults to zeros)
        avg_num_neighbors: Average number of neighbors per atom
        atomic_inter_scale: Scale factor for interaction energies
        atomic_inter_shift: Shift for interaction energies

    Returns:
        ScaleShiftMACE model
    """
    # Set up default atomic energies if not provided
    if atomic_energies is None:
        atomic_energies = torch.zeros(len(atomic_numbers))

    # Default interaction classes
    interaction_cls = interaction_classes[
        "RealAgnosticResidualInteractionBlock"
    ]
    interaction_cls_first = interaction_classes["RealAgnosticInteractionBlock"]

    # Default gate function
    gate = gate_dict["silu"]

    # Create model
    model = ScaleShiftMACE(
        r_max=r_max,
        num_bessel=num_bessel,
        num_polynomial_cutoff=num_polynomial_cutoff,
        max_ell=max_ell,
        num_interactions=num_interactions,
        num_elements=len(atomic_numbers),
        hidden_irreps=o3.Irreps(hidden_irreps),
        MLP_irreps=o3.Irreps(mlp_irreps),
        atomic_energies=atomic_energies,
        avg_num_neighbors=avg_num_neighbors,
        atomic_numbers=atomic_numbers,
        correlation=3,
        atomic_inter_scale=atomic_inter_scale,
        atomic_inter_shift=atomic_inter_shift,
        # Added required parameters
        interaction_cls=interaction_cls,
        interaction_cls_first=interaction_cls_first,
        gate=gate,
        # Optional parameters with defaults
        pair_repulsion=False,
        distance_transform="None",
        radial_MLP=[64, 64, 64],
        radial_type="bessel",
        heads=None,
        cueq_config=None,
    )

    return model


MACE_MODEL = create_default_scaleshift_mace(ELEMENTS, CUTOFF)


def test_interface():
    pass
