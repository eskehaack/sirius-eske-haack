#!/bin/bash
#SBATCH --job-name=ec_earth_train
#SBATCH --account=project_465002687
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=./hpc_outputs/%x-%j.out
#SBATCH --error=./hpc_outputs/%x-%j.err

set -euo pipefail

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

# --------------------------------------------------
# Run
# --------------------------------------------------

cd "/users/haackesk/Desktop/sirius/"
SIF="/scratch/project_465002687/ec_earth/containers/docker.sif"

srun --account=project_465002687 singularity exec \
    --bind /scratch/project_465002687/ec_earth:/scratch/project_465002687/ec_earth \
    "$SIF" \
    python -m src.train \
        --config_path ./src/configs/sample_config.toml \
        --run_id "$SLURM_JOB_ID"

srun --account=project_465002687 singularity exec \
    --bind /scratch/project_465002687/ec_earth:/scratch/project_465002687/ec_earth \
    "$SIF" \
    python -m src.sample \
        --run_id "$SLURM_JOB_ID"
