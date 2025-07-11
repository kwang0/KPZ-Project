#!/bin/bash
#SBATCH -A m4863
#SBATCH -C cpu
#SBATCH -c 256
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -o ./logs_slurm/slurm-%j.out

julia -t 8 --heap-size-hint=400G $1 $2 $3 $4 $5 > logs_jl/tebd_coarsegrained_L${2}_chi${3}_Jprime${4}_mu${5}.txt

exit 0
