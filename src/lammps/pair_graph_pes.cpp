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
  // this can be over-riden by the user by passing "cpu" as an argument
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

// provide the pair-wise cutoff for each pair of atom types
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
  // TODO: check these are correct
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

  // TODO: what actually is this?
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

  // load metadata
  std::vector<torch::jit::IValue> stack;
  // cutoff = model.get_method("get_cutoff")(stack).toTensor().item<double>();
  cutoff = 2;

  if (debug_mode)
    std::cout << "Model metadata:\n"
              << "   cutoff: " << cutoff << "\n";

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

// TODO: ensure that we're wrapping all positions into the box?
// add tests in the python package for this too
void PairGraphPES::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  if (force->newton_pair == 1)
  {
    error->all(FLERR, "Pair style GraphPES requires 'newton off'");
  }

  double **x = atom->x;    // positions of all atoms: real atoms together with
                           //   ghost atoms due to periodic boundary conditions
                           //   and ghost atoms due to LAMMPS parallelization
  double **f = atom->f;    // forces on atoms: unsure if these are just real ones
                           //   or also ghost atoms too. In either case, we only write
                           //   to indices corresponding to real atoms
  tagint *tag = atom->tag; // map from real/ghost atom index to the real atom's tag (1-based)

  int *type = atom->type;    // map from real atom index to the atom's type (as defined in the data file)
  int nlocal = atom->nlocal; // number of real atoms on this processor

  int inum = list->inum; // number of atoms that LAMMPS has calculated neighbours for
  // check that LAMMPS calculated neighbours for all real atoms
  assert(inum == nlocal);

  int nghost = list->gnum;      // number of ghost atoms
  int ntotal = nlocal + nghost; // number of ghost atoms

  int *numneigh = list->numneigh;      // number of neighbours for each real atom
  int **firstneigh = list->firstneigh; // neighbours for each real atom

  int *ilist = list->ilist; // mapping from index in the neighbourlist to the real atom index

  int nedges = std::accumulate(numneigh, numneigh + ntotal, 0);

  torch::Tensor pos_tensor = torch::zeros({nlocal, 3});
  torch::Tensor Z_tensor = torch::zeros({nlocal}, torch::TensorOptions().dtype(torch::kInt64));
  torch::Tensor periodic_shift_tensor = torch::zeros({3});

  auto pos = pos_tensor.accessor<float, 2>();
  long edges[2 * nedges];
  float edge_cell_shifts[3 * nedges];
  auto Z = Z_tensor.accessor<long, 1>();
  auto periodic_shift = periodic_shift_tensor.accessor<float, 1>();

  // Step 1:
  // loop over the real atoms, and store their positions and atomic numbers
  // TODO: parallelisation: this assumes that:
  //  1. the real atoms are the entire system
  //  2. the ghost atoms are just replications of the real atoms
  for (int ii = 0; ii < nlocal; ii++)
  {
    int i = ilist[ii];
    pos[i][0] = x[i][0];
    pos[i][1] = x[i][1];
    pos[i][2] = x[i][2];

    int itype = type[i];
    Z[i] = lammps_type_to_Z[itype];
  }

  torch::Tensor cell_tensor = extract_cell_tensor();
  auto cell = cell_tensor.accessor<float, 2>();
  auto cell_inv = cell_tensor.inverse().transpose(0, 1);
  // Loop over atoms and neighbors,
  // store edges and _cell_shifts
  // ii follows the order of the neighbor lists,
  // i follows the order of x, f, etc.
  int edge_counter = 0;

  for (int ii = 0; ii < nlocal; ii++)
  {
    // loop over all real atoms, i, ...
    int i = ilist[ii];
    int itag = tag[i];

    int jnum = numneigh[i];
    int *jlist = firstneigh[i];
    for (int jj = 0; jj < jnum; jj++)
    {
      // ... and get all of i's real/ghost neighbours, j
      int j = jlist[jj];
      j &= NEIGHMASK;
      int jtag = tag[j];

      double dx = x[i][0] - x[j][0];
      double dy = x[i][1] - x[j][1];
      double dz = x[i][2] - x[j][2];

      double distance_squared = dx * dx + dy * dy + dz * dz;
      if (distance_squared < cutoff * cutoff)
      {
        // if this neighbour is actually within the cutoff:
        //   1. store the edge cell shift
        //   2. store the edge indices
        periodic_shift[0] = x[j][0] - pos[jtag - 1][0];
        periodic_shift[1] = x[j][1] - pos[jtag - 1][1];
        periodic_shift[2] = x[j][2] - pos[jtag - 1][2];
        torch::Tensor cell_shift_tensor = cell_inv.matmul(periodic_shift_tensor);
        auto cell_shift = cell_shift_tensor.accessor<float, 1>();

        float *e_vec = &edge_cell_shifts[edge_counter * 3];
        e_vec[0] = std::round(cell_shift[0]);
        e_vec[1] = std::round(cell_shift[1]);
        e_vec[2] = std::round(cell_shift[2]);

        edges[edge_counter * 2] = itag - 1;
        edges[edge_counter * 2 + 1] = jtag - 1;
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
  input.insert("_positions", pos_tensor.to(device));
  input.insert("neighbour_index", edges_tensor.to(device));
  input.insert("_neighbour_cell_offsets", edge_cell_shifts_tensor.to(device));
  input.insert("cell", cell_tensor.to(device));
  input.insert("atomic_numbers", Z_tensor.to(device));
  input.insert("compute_virial", bool_tensor(vflag, device));
  input.insert("debug", bool_tensor(debug_mode, device));
  std::vector<torch::IValue> input_vector(1, input);

  auto output = model.forward(input_vector).toGenericDict();

  torch::Tensor forces_tensor = output.at("forces").toTensor().cpu();
  auto forces = forces_tensor.accessor<double, 2>();

  torch::Tensor atomic_energy_tensor = output.at("local_energies").toTensor().cpu();
  auto atomic_energies = atomic_energy_tensor.accessor<double, 1>();

  // write results of computations back to LAMMPS
  if (vflag)
  {
    torch::Tensor v_tensor = output.at("virial").toTensor().cpu();
    auto v = v_tensor.accessor<double, 2>();
    // Convert from 3x3 symmetric tensor format to the flattened form LAMMPS expects
    virial[0] = v[0][0];
    virial[1] = v[1][1];
    virial[2] = v[2][2];
    virial[3] = v[0][1];
    virial[4] = v[0][2];
    virial[5] = v[1][2];
  }
  if (vflag_atom)
    error->all(FLERR, "Pair style GraphPES does not support per-atom virial");

  // store the total energy where LAMMPS wants it by summing over non-ghost atoms
  eng_vdwl = 0.0;

  for (int ii = 0; ii < inum; ii++)
  {
    int i = ilist[ii];
    f[i][0] = forces[i][0];
    f[i][1] = forces[i][1];
    f[i][2] = forces[i][2];
    eng_vdwl += atomic_energies[i];
    if (eflag_atom)
      eatom[i] = atomic_energies[i];
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
