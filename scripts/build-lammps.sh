#!/bin/bash

# This script builds a LAMMPS executable with graph-pes support.
# It sets up a conda environment, clones LAMMPS, patches it with graph-pes,
# and builds the executable for either CPU or GPU use.

# Fail if any command fails
set -e

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "This script builds a LAMMPS executable with support for 'pair_style graph_pes'."
    echo
    echo "It performs the following tasks:"
    echo "  1. Locates the graph-pes installation"
    echo "  2. Creates a conda environment with necessary dependencies"
    echo "  3. Clones the LAMMPS repository"
    echo "  4. Patches LAMMPS with graph-pes source code"
    echo "  5. Builds LAMMPS with graph-pes support"
    echo
    echo "The final executable will be located at:"
    echo "  ./graph_pes_lmp_cpu_only (if --cpu-only is used)"
    echo "  ./graph_pes_lmp (default, with GPU support)"
    echo
    echo "Requirements:"
    echo "  - conda"
    echo
    echo "Options:"
    echo "  --help           Display this help message and exit"
    echo "  --cpu-only       Build LAMMPS for CPU only (default: GPU enabled)"
    echo "  --force-rebuild  Force rebuilding of conda environment and LAMMPS"
    echo
}

###################
## PARSE OPTIONS ##
###################

CPU_ONLY=false
FORCE_REBUILD=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --cpu-only) CPU_ONLY=true ;;
        --force-rebuild) FORCE_REBUILD=true ;;
        --help) show_help; exit 0 ;;
        *)
            echo "Unknown parameter passed: $1"
            echo "Use --help for usage information"
            exit 1
        ;;
    esac
    shift
done


CURRENT_DIR=$(pwd)
if [ "$CPU_ONLY" = true ]; then
    FINAL_EXE="$CURRENT_DIR/graph_pes_lmp_cpu_only"
else
    FINAL_EXE="$CURRENT_DIR/graph_pes_lmp"
fi

echo "Running build-lammps.sh with the following parameters:"
echo "          CPU_ONLY: $CPU_ONLY"
echo "    FORCE_REBUILD : $FORCE_REBUILD"

#################################
## FIND GRAPH-PES INSTALLATION ##
#################################

script="
try:
    import graph_pes
    from pathlib import Path

    # at graph_pes/__init__.py
    init_file = Path(graph_pes.__file__)
    pair_style_dir = init_file.parent / 'pair_style'
    print(pair_style_dir)
except ImportError:
    print('')
"

PAIR_STYLE_DIR=$(python -c "$script")
if [ -z "$PAIR_STYLE_DIR" ]; then
    echo "Error: Failed to find the graph_pes install path: 
    do you have it installed in your current python environment?"
    exit 1
fi
PAIR_STYLE_DIR=$(realpath "$PAIR_STYLE_DIR")
if [ ! -d "$PAIR_STYLE_DIR" ]; then
    echo "Error: Pair style directory not found at $PAIR_STYLE_DIR"
    exit 1
fi
echo "Found graph-pes pair style at $PAIR_STYLE_DIR"


###################
## GET CONDA ENV ##
###################

if [ "$CPU_ONLY" = true ]; then
    ENV_NAME="lammps-env-cpu-throwaway"
    ENV_YAML="
channels:
  - conda-forge
  - defaults
dependencies:
  - cmake
  - pytorch-cpu=2.3.0
  - python=3.10
  - mkl-include
  - gcc=12.4.0
  - gxx=12.4.0
  - compilers
"
else
    ENV_NAME="lammps-env-gpu-throwaway"
    ENV_YAML="
channels:
  - conda-forge
  - defaults
dependencies:
  - cmake
  - python=3.10
  - pip
  - mkl-include
  - gcc=12.4.0
  - gxx=12.4.0
  - curl
  - rhash
  - git
  - pip:
    - torch
    - numpy
"
fi

# fail if conda not installed
if ! command -v conda &> /dev/null; then
    echo "Conda not found, please install conda and try again"
    exit 1
fi

# activate base conda environment
eval "$(conda shell.bash hook)"

# check if environment exists
_env_exists=$(conda env list | grep -q "$ENV_NAME" && echo "true" || echo "false")
if [ "$_env_exists" = true ]; then
    echo "Conda environment $ENV_NAME already exists"
fi

# delete if FORCE_REBUILD is true and environment exists
if [ "$FORCE_REBUILD" = true ] && [ "$_env_exists" = true ]; then
    echo "Deleting existing conda environment $ENV_NAME"
    conda env remove -n "$ENV_NAME" -y
fi

_env_exists=$(conda env list | grep -q "$ENV_NAME" && echo "true" || echo "false")

# create environment if it doesn't exist
if [ "$_env_exists" = false ]; then
    echo "Creating conda environment $ENV_NAME"
    echo "$ENV_YAML" > temp_env.yaml
    conda env create -f temp_env.yaml -n "$ENV_NAME"
    rm temp_env.yaml
fi

# activate environment
conda activate "$ENV_NAME"
echo "Conda environment $ENV_NAME successfully activated"


############################
## CLONE AND PATCH LAMMPS ##
############################

mkdir -p ignore
cd ignore

if [ "$FORCE_REBUILD" = true ]; then
    echo "Deleting existing lammps repository"
    rm -rf lammps
fi

if [ ! -d "lammps" ]; then
    git clone -b release https://github.com/lammps/lammps
else
    echo "LAMMPS repository already exists, skipping clone."
fi

# add graph-pes source code to lammps
cp $PAIR_STYLE_DIR/*.cpp $PAIR_STYLE_DIR/*.h lammps/src/

# patch CMakeLists.txt
if ! grep -q "find_package(Torch REQUIRED)" lammps/cmake/CMakeLists.txt; then
    echo "
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS \"\${CMAKE_CXX_FLAGS} \${TORCH_CXX_FLAGS}\")
target_link_libraries(lammps PUBLIC \"\${TORCH_LIBRARIES}\")
" >> lammps/cmake/CMakeLists.txt
fi

##################
## BUILD LAMMPS ##
##################

cd lammps
rm -rf build
mkdir -p build
cd build
cmake ../cmake \
    -DCMAKE_PREFIX_PATH="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')" \
    -DCMAKE_CXX_STANDARD=17 \
    -DMKL_INCLUDE_DIR="$CONDA_PREFIX/include"

echo "Building LAMMPS executable"
make -j$(nproc)

####################
## CREATE SYMLINK ##
####################

rm -f "$FINAL_EXE"
ln -s "$(pwd)/lmp" "$FINAL_EXE"

echo "LAMMPS executable successfully built with graph-pes, and is available at $FINAL_EXE"
