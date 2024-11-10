/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

// Contributing authors:
//   - John Gardner (Oxford)

// Writing was greatly aided by:
//   - "Writing new pair styles", LAMMPS (https://docs.lammps.org/Developer_write_pair.html)
//   - Anders Johansson's torchscript integration from pair_nequip and pair_allegro
//     (https://github.com/mir-group/pair_nequip/blob/3dda11b972f1cdf196215ed92ee5bea8d576d37b/pair_nequip.cpp)
//     (https://github.com/mir-group/pair_allegro/blob/20538c9fd308bd0d066a716805f6f085a979c741/compute_allegro.cpp)
//   - Yutack Park's pair_e3gnn (https://github.com/MDIL-SNU/SevenNet/blob/main/sevenn/pair_e3gnn/pair_e3gnn.cpp)

// Assumptions:
//   - LAMMPS is being run in serial mode: there are no "ghost" atoms here!

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "potential_file_reader.h"
#include "tokenizer.h"
#include <pair_graph_pes.h>

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/script.h>
#include <torch/torch.h>

// freezing is broken from C++ in <=1.10
#if (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR <= 10)
#error "PyTorch version < 1.11 is not supported"
#endif

using namespace LAMMPS_NS;

// utilities -------------------------------------------------------------------
torch::Tensor bool_tensor(bool flag, torch::Device device)
{
  return torch::tensor(flag, torch::TensorOptions().dtype(torch::kBool).device(device));
}

// constructor -----------------------------------------------------------------
PairGraphPES::PairGraphPES(LAMMPS *lmp) : Pair(lmp)
{
  this->writedata = 0;     // don't write model parameters to any data file
  this->restartinfo = 0;   // don't write model parameters to restart file
  this->single_enable = 0; // no implementation for a single pairwise interaction
  this->manybody_flag = 1; // in the general case, GraphPES models are many-body
  this->one_coeff = 1;     // only one pair_coeff command is allowed

  // by default, use the GPU if available
  // this can be overriden by the user by passing "cpu" as an argument
  // to the pair_style command
  if (torch::cuda::is_available())
    this->device = torch::kCUDA;
  else
    this->device = torch::kCPU;
}

void PairGraphPES::allocate()
{
  this->allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  memory->create(lammps_type_to_Z, n + 1, "pair:lammps_type_to_Z");
}

// destructor -----------------------------------------------------------------
PairGraphPES::~PairGraphPES()
{
  if (this->allocated)
  {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(lammps_type_to_Z);
  }
}

// tell LAMMPS how to use this style -------------------------------------------
void PairGraphPES::init_style()
{
  if (atom->tag_enable == 0)
    error->all(FLERR, "Pair style graph_pes requires atom IDs");
  if (force->newton_pair == 1)
    error->all(FLERR, "Pair style graph_pes requires newton pair off");

  // request a full neighbourlist
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

// provide the pair-wise cutoff for each pair of atom types:
// for GraphPES, this is a uniform and global cutoff
// -----------------------------------------------------------------------------
double PairGraphPES::init_one(int i, int j)
{
  return cutoff;
}

// tell LAMMPS how to read in this style via the pair_style command ------------
void PairGraphPES::settings(int narg, char **args)
{
  // allowed declarations:
  //    pair_style graph_pes
  //    pair_style graph_pes debug
  //    pair_style graph_pes cpu
  //    pair_style graph_pes debug cpu

  for (int i = 0; i < narg; i++)
  {
    if (strcmp(args[i], "debug") == 0)
    {
      this->debug_mode = 1;
      std::cout << "GraphPES is in debug mode\n";
    }
    else if (strcmp(args[i], "cpu") == 0)
    {
      this->device = torch::kCPU;
    }
    else
    {
      std::stringstream ss;
      ss << "Unknown argument to pair_style graph_pes: " << args[i];
      error->all(FLERR, ss.str().c_str());
    }
  }

  std::cout << "GraphPES is using device " << this->device << "\n";
}

// tell LAMMPS how to read the single pair_coeff command -----------------------
void PairGraphPES::coeff(int narg, char **arg)
{
  if (!allocated)
    allocate();

  // create a list of the element symbols, H, He etc.
  const char *element_symbols[] = {
      "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si",
      "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
      "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru",
      "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr",
      "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
      "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac",
      "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf",
      "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};

  int ntypes = atom->ntypes;
  if (narg != (3 + ntypes))
    error->all(FLERR,
               "Incorrect args for pair_coeff. Correct usage: pair_coeff * * <model.pt> <symbol of "
               "LAMMPS type 1> <symbol of LAMMPS type 2> ... <symbol of LAMMPS type N>");

  // ensure first two args are "* *".
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "Expected pair_coeff * * ...");

  // signal to LAMMPS to use init_one to set the cutoff (pretty archane)
  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++)
      setflag[i][j] = 1;

  // parse passed element symbols into the lammps_type_to_Z map
  for (int type = 1; type <= ntypes; type++)
  {
    int found = 0;
    for (int Z = 1; Z < 119; Z++)
    {
      if (strcmp(arg[3 + type - 1], element_symbols[Z]) == 0)
      {
        lammps_type_to_Z[type] = Z;
        found = 1;
        break;
      }
    }
    if (!found)
    {
      // show the user which element symbol was not recognised
      std::stringstream ss;
      ss << "Unrecognised element symbol in pair_coeff: " << arg[3 + type - 1];
      error->all(FLERR, ss.str().c_str());
    }
  }

  // load the model
  std::cout << "Loading model from " << arg[2] << "\n";
  model = torch::jit::load(std::string(arg[2]), device);

  // load cutoff
  std::vector<torch::jit::IValue> stack;
  cutoff = model.get_method("get_cutoff")(stack).toTensor().item<double>();

  if (debug_mode)
    std::cout << "Model cutoff: " << cutoff << "\n";

  // eval and freeze the model: speeds things up
  model.eval();
  if (model.hasattr("training"))
  {
    std::cout << "Freezing TorchScript model...\n";
    model = torch::jit::freeze(model);
  }

  torch::jit::FusionStrategy strategy = {{torch::jit::FusionBehavior::DYNAMIC, 10}};
  torch::jit::setFusionStrategy(strategy);

  // Set whether to allow TF32:
  bool allow_tf32 = false;
  // See https://pytorch.org/docs/stable/notes/cuda.html
  at::globalContext().setAllowTF32CuBLAS(allow_tf32);
  at::globalContext().setAllowTF32CuDNN(allow_tf32);
}

// energy, force and virial calculation ---------------------------------------
void PairGraphPES::compute(int eflag, int vflag)
{

  if (debug_mode)
    std::cout << "Computing properties with GraphPES\n";

  ev_init(eflag, vflag);

  if (force->newton_pair == 1)
    error->all(FLERR, "Pair style GraphPES requires 'newton off'");

  if (vflag_atom)
    error->all(FLERR, "Pair style GraphPES does not support per-atom virial");

  // Useful info:
  // 
  // How LAMMPS stores atoms:
  // - each atom WITHIN THE UNIT CELL has a unique, 1-based tag that is persistent across time
  // - LAMMPS will calculate a neighbourlist for each of these atoms
  // - LAMMPS doesn't necessarily store information about these atoms in tag order: I refer
  //      to the order that LAMMPS actually uses to access these atoms as the "frame_index".
  //      this frame_index order is not persisted across timesteps, and also includes atoms that
  //      are replicates due to PBCs. (see below)
  // - LAMMPS duplicates atoms that appear in other unit cells, but which are relevant 
  //      due to PBCs.
  // - the positions of ALL atoms (including duplicates) are stored in the atom->x and atom->f arrays.
  //    i.e. len(atom->x) >> # atoms in the structure
  //
  // For ease of debugging, we construct all input tensors in "tag" order

  double **x = atom->x;                   // positions of each atom (includes replicated atoms due to PBC)
  double **f = atom->f;                   // forces on each atom
  tagint *frame_index_to_tag = atom->tag; // each atom's tag is a 1-based identifier that is unique
                                          //   across the whole system and through time. Atom replicates share the same tag
                                          //   as the atom in the unit cell from which they were created.
                                          //   This is a map from the atom's index in the position/force/etc.
                                          //   arrays to the atom's tag
  int *type = atom->type;                 // map from frame index to the atom's type (as defined in the data file)
  int nlocal = atom->nlocal;              // total number of atoms (only using a single processor here, so nlocal == natoms)
  int inum = list->inum;                  // number of atoms that LAMMPS has calculated neighbours for
  
  // check that LAMMPS calculated neighbours for all atoms
  assert(inum == nlocal);

  int nghost = list->gnum;
  assert(nghost == 0 && "This is currently a serial implementation. Ghost atoms are not supported.");
  
  int *numneigh = list->numneigh;         // number of neighbours for each atom
  int **neighbour_list = list->firstneigh;   // neighbours for each atom

  int *nl_to_frame_index = list->ilist;   // mapping from index in the neighbourlist to the frame index

  // Step 1:
  // convert the unit cell of the structure to a tensor
  torch::Tensor cell_tensor = extract_cell_tensor();


  // Step 2:
  // loop over the atoms, and store their positions and atomic numbers 
  // as tensors. Also build up useful reverse maps from:
  //   atom tag to real frame index
  //   frame index to atom tag
  int frame_index_to_atom_tag[nlocal];
  int atom_tag_to_real_atom_frame_index[nlocal + 1];
  torch::Tensor pos_tensor = torch::zeros({nlocal, 3});
  auto pos = pos_tensor.accessor<float, 2>();
  torch::Tensor Z_tensor = torch::zeros({nlocal}, torch::TensorOptions().dtype(torch::kInt64));
  auto Z = Z_tensor.accessor<long, 1>();

  // x can be in any order, i.e. some replicates may appear before the atom in the unit cell
  // BUT lammps only calculates neighbourlists for atoms within the unit cell
  // therefore we need to loop over the total number of atoms, get the frame_index from the
  // neighbourlist, and use this to index into the position array
  for (int nl_index = 0; nl_index < nlocal; nl_index++)
  {
    int frame_index = nl_to_frame_index[nl_index];
    int atom_tag = frame_index_to_tag[frame_index];

    frame_index_to_atom_tag[frame_index] = atom_tag;
    atom_tag_to_real_atom_frame_index[atom_tag] = frame_index;
  
    pos[atom_tag - 1][0] = x[frame_index][0];
    pos[atom_tag - 1][1] = x[frame_index][1];
    pos[atom_tag - 1][2] = x[frame_index][2];

    int t = type[frame_index];
    Z[atom_tag - 1] = lammps_type_to_Z[t];
  }


  // Step 3:
  // loop over the neighbourlist to build up 
  // the edge index and cell shift tensors
  int nedges = std::accumulate(numneigh, numneigh + nlocal, 0);

  torch::Tensor periodic_shift_tensor = torch::zeros({3});
  auto periodic_shift = periodic_shift_tensor.accessor<float, 1>();
  long edges[2 * nedges];
  float edge_cell_shifts[3 * nedges];
  auto cell_inv = cell_tensor.inverse().transpose(0, 1);
  
  int edge_counter = 0;

  for (int central_atom_nl_index = 0; central_atom_nl_index < nlocal; central_atom_nl_index++)
  {
    int central_atom_frame_index = nl_to_frame_index[central_atom_nl_index];
    int central_atom_tag = frame_index_to_tag[central_atom_frame_index];

    int n_neighbours = numneigh[central_atom_frame_index];
    int *central_atom_neighbours = neighbour_list[central_atom_frame_index];
    for (int neighbour_nl_index = 0; neighbour_nl_index < n_neighbours; neighbour_nl_index++)
    {
      int neighbour_frame_index = central_atom_neighbours[neighbour_nl_index];
      neighbour_frame_index &= NEIGHMASK;
      int neighbour_tag = frame_index_to_tag[neighbour_frame_index];

      double dx = x[central_atom_frame_index][0] - x[neighbour_frame_index][0];
      double dy = x[central_atom_frame_index][1] - x[neighbour_frame_index][1];
      double dz = x[central_atom_frame_index][2] - x[neighbour_frame_index][2];

      double distance_squared = dx * dx + dy * dy + dz * dz;
      if (distance_squared < cutoff * cutoff)
      {
        // if this neighbour is actually within the cutoff:
        //   1. store the edge cell shift
        //   2. store the edge indices
        periodic_shift[0] = x[neighbour_frame_index][0] - pos[neighbour_tag - 1][0];
        periodic_shift[1] = x[neighbour_frame_index][1] - pos[neighbour_tag - 1][1];
        periodic_shift[2] = x[neighbour_frame_index][2] - pos[neighbour_tag - 1][2];
        torch::Tensor cell_shift_tensor = cell_inv.matmul(periodic_shift_tensor);
        auto cell_shift = cell_shift_tensor.accessor<float, 1>();

        float *e_vec = &edge_cell_shifts[edge_counter * 3];
        e_vec[0] = std::round(cell_shift[0]);
        e_vec[1] = std::round(cell_shift[1]);
        e_vec[2] = std::round(cell_shift[2]);

        edges[edge_counter * 2] = central_atom_tag - 1;
        edges[edge_counter * 2 + 1] = neighbour_tag - 1;
        edge_counter++;
      }
    }
  }

  if (debug_mode)
  {
    // print difference between the number of edges from LAMMPS, and those
    // that we counted as actually less than the cutoff
    std::cout << "Number of edges from LAMMPS: " << nedges << "\n";
    std::cout << "Number of edges counted: " << edge_counter << "\n";
  }


  // Step 4:
  // send the data to the model

  // shorten the list before sending to graph-pes
  torch::Tensor edges_tensor =
      torch::zeros({2, edge_counter}, torch::TensorOptions().dtype(torch::kInt64));
  torch::Tensor edge_cell_shifts_tensor = torch::zeros({edge_counter, 3});
  auto new_edges = edges_tensor.accessor<long, 2>();
  auto new_edge_cell_shifts = edge_cell_shifts_tensor.accessor<float, 2>();
  for (int i = 0; i < edge_counter; i++)
  {
    long *e = &edges[i * 2];
    new_edges[0][i] = e[0];
    new_edges[1][i] = e[1];

    float *ev = &edge_cell_shifts[i * 3];
    new_edge_cell_shifts[i][0] = ev[0];
    new_edge_cell_shifts[i][1] = ev[1];
    new_edge_cell_shifts[i][2] = ev[2];
  }

  c10::Dict<std::string, torch::Tensor> input;
  input.insert("positions", pos_tensor.to(device));
  input.insert("neighbour_list", edges_tensor.to(device));
  input.insert("neighbour_cell_offsets", edge_cell_shifts_tensor.to(device));
  input.insert("cell", cell_tensor.to(device));
  input.insert("atomic_numbers", Z_tensor.to(device));
  input.insert("compute_virial", bool_tensor(vflag, device));
  input.insert("debug", bool_tensor(debug_mode, device));
  std::vector<torch::IValue> input_vector(1, input);

  auto output = model.forward(input_vector).toGenericDict();

  if (debug_mode)
    std::cout << "Output from model:\n" << output << "\n";


  // Step 5:
  // extract the forces, atomic energies and virial from the model output

  torch::Tensor forces_tensor = output.at("forces").toTensor().cpu();
  auto forces = forces_tensor.accessor<double, 2>();

  torch::Tensor atomic_energy_tensor = output.at("local_energies").toTensor().cpu();
  auto atomic_energies = atomic_energy_tensor.accessor<double, 1>();

  // write results of computations back to LAMMPS
  if (vflag)
  {
    torch::Tensor v_tensor = output.at("virial").toTensor().cpu();
    auto v = v_tensor.accessor<double, 1>();
    
    // Directly copy the values to the LAMMPS virial array
    for (int i = 0; i < 6; i++)
      virial[i] = v[i];
  }

  // store the total energy where LAMMPS wants it
  eng_vdwl = 0.0;

  // we get back energies and forces for the real 
  // (non-replicated) atoms only and in tag order.
  // We need to convert these back to frame_index order in order to write
  // them to the correct place in the force array
  for (int atom_tag = 1; atom_tag <= nlocal; atom_tag++)
  {
    int frame_index = atom_tag_to_real_atom_frame_index[atom_tag];
    f[frame_index][0] = forces[atom_tag - 1][0];
    f[frame_index][1] = forces[atom_tag - 1][1];
    f[frame_index][2] = forces[atom_tag - 1][2];
    eng_vdwl += atomic_energies[atom_tag - 1];
    if (eflag_atom)
      eatom[frame_index] = atomic_energies[atom_tag - 1];
  }
}

// Helpers
// -----------------------------------------------------------------------------

torch::Tensor PairGraphPES::extract_cell_tensor()
{
  torch::Tensor cell = torch::zeros({3, 3});
  auto access = cell.accessor<float, 2>();

  access[0][0] = domain->boxhi[0] - domain->boxlo[0];

  access[1][0] = domain->xy;
  access[1][1] = domain->boxhi[1] - domain->boxlo[1];

  access[2][0] = domain->xz;
  access[2][1] = domain->yz;
  access[2][2] = domain->boxhi[2] - domain->boxlo[2];

  return cell;
}
