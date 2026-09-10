#!/bin/bash
#SBATCH --job-name=ec_earth_build_container
#SBATCH --account=project_465002687
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:20:00
#SBATCH --output=./hpc_outputs/%x-%j.out
#SBATCH --error=./hpc_outputs/%x-%j.err

set -euo pipefail

module purge
module load CrayEnv
module load PRoot

mkdir -p "/tmp/$USER/singularity-build"
export SINGULARITY_TMPDIR="/tmp/$USER/singularity-build"
export SINGULARITY_CACHEDIR="/tmp/$USER/singularity-build"

mkdir -p "/tmp/$USER/mpl"
export MPLCONFIGDIR="/tmp/$USER/mpl"

mkdir -p "/tmp/$USER/numba_cache"
export NUMBA_CACHE_DIR="/tmp/$USER/numba_cache"

cd ~/Desktop/sirius

srun --account=project_465002687 singularity build /scratch/project_465002687/ec_earth/containers/dockerv3.sif singularity/update.def
