#!/bin/bash
#SBATCH --job-name=ec_earth_train
#SBATCH --account=project_465002687
#SBATCH --partition=small-g
#SBATCH --nodes=1-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=./hpc_outputs/train/%x-%j.out
#SBATCH --error=./hpc_outputs/train/%x-%j.err

set -euo pipefail

module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm

# --------------------------------------------------
# Job-local cache directory setup
# --------------------------------------------------

# Job-local temporary directory
export TMPDIR="/tmp/$USER/$SLURM_JOB_ID"
export SINGULARITY_TMPDIR="$TMPDIR"
export SINGULARITY_CACHEDIR="$TMPDIR"

mkdir -p "$TMPDIR"
chmod 700 "$TMPDIR"

echo "TMPDIR=$TMPDIR"
echo "SINGULARITY_TMPDIR=$SINGULARITY_TMPDIR"
echo "SINGULARITY_CACHEDIR=$SINGULARITY_CACHEDIR"

export MIOPEN_USER_DB_PATH="$TMPDIR/miopen-db"
export MIOPEN_CUSTOM_CACHE_DIR="$TMPDIR/miopen-cache"

mkdir -p "$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_CUSTOM_CACHE_DIR"

echo "MIOPEN_USER_DB_PATH=$MIOPEN_USER_DB_PATH"
echo "MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_CUSTOM_CACHE_DIR"

# --------------------------------------------------
# Job validation
# --------------------------------------------------

amd-smi version
srun rocm-smi

# --------------------------------------------------
# Run
# --------------------------------------------------

cd "/scratch/project_465002687/sirius/"
SIF="/scratch/project_465002687/ec_earth/containers/dockerv5.sif"
CONFIG_PATH="./src/configs/sample_config.toml"

# --cpu-bind=mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000 \
srun --account=project_465002687 singularity exec \
    --bind /scratch/project_465002687 \
    "$SIF" \
    python -m src.train \
        --config_path "$CONFIG_PATH" \
        --run_id "$SLURM_JOB_ID"