#!/bin/bash
#SBATCH -A m3341_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -t 24:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

export SLURM_CPU_BIND="cores"
export JULIA_CUDA_SOFT_MEMORY_LIMIT=50%

julia $1 $2 $3 $4 $5 $6 > logs_jl/tebd_coarsegrained_gpu_L${2}_chi${3}_beta${4}_dt${5}_Jprime${6}.txt

exit 0
