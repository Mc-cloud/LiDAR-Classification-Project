#!/bin/bash
#


#SBATCH --job-name=PointNet++
#SBATCH --partition=prod40
#SBATCH --gres=gpu:nvidia_a100_3g.40gb:1 
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=./slurmlogs/slurm-%j.out
#SBATCH --error=./slurmlogs/slurm-%j.err

WORKDIR=$(pwd)

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total \
           --format=csv -lms 500 > "${WORKDIR}/jobs_output/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_gpu.csv" & GPU_MON=$!
trap 'kill ${GPU_MON:-} 2>/dev/null || true' EXIT

echo "activation de l'environnement"

source .venv/bin/activate

echo "clé wandb"

export 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

echo 'lancement du script'

python3 -u train.py
