#!/bin/bash
#BSUB -q gpuv100
#BSUB -J pilot_icyalert_4
#BSUB -n 8
#BSUB -R "rusage[mem=5GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 3:00
#BSUB -o hpc_outputs/cfg_%J.out
#BSUB -e hpc_outputs/cfg_%J.err
#BSUB -B
#BSUB -N

# module load python3/3.12.4
# module load cuda/12.6.0

cd /work3/s214643/sirius
source /work3/s214643/sirius/.venv/bin/activate
ts="$(date +%Y%m%d%H%M%S)"

python -m src.train \
    --config_path src/configs/training_config.yaml \
    --run_id ${LSB_JOBID} \

python -m src.sample \
    --run_id ${LSB_JOBID}
