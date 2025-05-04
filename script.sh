# Evan Stark - May 2nd 2025 - ITCS 4145 001
# SLURM Script to allocate a GPU node and to run NBody-CUDA implementation.

#!/bin/bash
#SBATCH --job-name=nbody_cuda
#SBATCH --partition=GPU
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

module load cuda/12.4

salloc --partition=GPU --time=01:00:00 --gres=gpu:1
srun --pty bash
