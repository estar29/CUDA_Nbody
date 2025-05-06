#!/bin/bash
#SBATCH --job-name=nbody_cuda
#SBATCH --partition=GPU
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

module load cuda/12.4
salloc --partition=GPU --time=01:00:00 --gres=gpu:1
nvcc -o nbody_cuda nbody_cuda.cu planet 200 5000000 10000
srun --pty bash
