#!/bin/bash
#SBATCH --job-name=ec_earth_inversion_test
#SBATCH --account=project_465002687
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=./hpc_outputs/tests/%x-%j.out
#SBATCH --error=./hpc_outputs/tests/%x-%j.err

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

cd "/scratch/project_465002687/sirius/"
SIF="/scratch/project_465002687/ec_earth/containers/dockerv5.sif"

srun --account=project_465002687 singularity exec \
    --bind /scratch/project_465002687 \
    "$SIF" \
    python -m src.postprocessing.inference.inversion_test \
        --run_id "22871248" \
        --date "2020-01-01" \
        --scenario "ssp126" \
        --num_steps 1000 \
        --config_path "./src/configs/larger_config.toml"
