#!/bin/bash
#SBATCH -A m4863_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -t 24:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -o ./logs_slurm/slurm-%j.out

export SLURM_CPU_BIND="cores"
export JULIA_CUDA_SOFT_MEMORY_LIMIT=50%

julia $1 $2 $3 $4 $5 $6 $7 $8> logs_jl/IBM_sims_L${2}_chi${3}_dt${4}_offset${5}_rungX${6}_rungY${7}_rungZ${8}.txt

exit 0
