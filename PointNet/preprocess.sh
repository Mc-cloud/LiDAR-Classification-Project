#!/bin/bash
#


#SBATCH --job-name=pointnet
#SBATCH --partition=prod40
#SBATCH --gres=gpu:nvidia_a100_3g.40gb:1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=./slurmlogs/slurm-%j.out
#SBATCH --error=./slurmlogs/slurm-%j.err

source .venv/bin/activate

python3 -u testGBM.py