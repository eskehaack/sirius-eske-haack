#!/bin/bash
#SBATCH --job-name=ec_earth_train
#SBATCH --account=project_465002687
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#SBATCH --output=./hpc_outputs/train/%x-%j.out
#SBATCH --error=./hpc_outputs/train/%x-%j.err

set -euo pipefail

module load rocm   # only for host-side rocm-smi / amd-smi

export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
export OMP_NUM_THREADS=1
export WORLD_SIZE=$SLURM_NTASKS

# ---------------- Job-local temp/cache ----------------
export TMPDIR="/tmp/$USER/$SLURM_JOB_ID"
export SINGULARITY_TMPDIR="$TMPDIR"
export SINGULARITY_CACHEDIR="$TMPDIR"
export MIOPEN_USER_DB_PATH="$TMPDIR/miopen-db"
export MIOPEN_CUSTOM_CACHE_DIR="$TMPDIR/miopen-cache"
mkdir -p "$TMPDIR" "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR"
chmod 700 "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

echo "TMPDIR=$TMPDIR"
echo "MIOPEN_USER_DB_PATH=$MIOPEN_USER_DB_PATH"

# ---------------- Validation (printed once) ----------------
srun --ntasks=1 amd-smi version
srun --ntasks=1 rocm-smi

# ---------------- Run ----------------
cd "/scratch/project_465002687/sirius/"
SIF="/scratch/project_465002687/ec_earth/containers/dockerv5.sif"
CONFIG_PATH="./src/configs/sample_config.toml"   # <- check: same config for train and sample?

CPU_BIND="mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000"

# Training: 8 ranks, one per GCD
srun --kill-on-bad-exit=1 --label --cpu-bind="$CPU_BIND" \
    singularity exec --bind /scratch/project_465002687 "$SIF" \
    python -m src.hp_tuning \
        --config_path "$CONFIG_PATH" \
        --n-trials 20 \
        --max-epochs-hpo 30 \
        --study-name "edm_hpo"