#!/bin/bash
#SBATCH --job-name=ec_earth_data_preprocess
#SBATCH --account=project_465002687
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=./hpc_outputs/%x-%j.out
#SBATCH --error=./hpc_outputs/%x-%j.err

set -euo pipefail

cd "/users/haackesk/Desktop/sirius/"

SIF="/users/haackesk/Desktop/sirius/singularity/docker.sif"

pwd

srun --account=project_465002687 singularity exec \
    --bind /scratch/project_465002687/ec_earth:/scratch/project_465002687/ec_earth \
    "$SIF" \
    python ./src/data_builders/preproces.py