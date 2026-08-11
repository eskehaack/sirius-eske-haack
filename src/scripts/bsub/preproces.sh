#!/bin/bash
#BSUB -q hpc
#BSUB -J preproces_icyalert
#BSUB -n 1
#BSUB -R "rusage[mem=5GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 1:00
#BSUB -o hpc_outputs/cfg_download_%J.out
#BSUB -e hpc_outputs/cfg_download_%J.err
#BSUB -B
#BSUB -N

# module load python3/3.12.4
# module load cuda/13.0.0

cd /work3/s214643/sirius
source /work3/s214643/sirius/.venv/bin/activate

python ./src/data_builders/preproces.py