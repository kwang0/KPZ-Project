#!/bin/bash
#SBATCH -A m4863
#SBATCH -C cpu
#SBATCH -c 256
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -o ./logs_slurm/slurm-%j.out

export OMP_NUM_THREADS=256
export MKL_NUM_THREADS=256
julia --heap-size-hint=400G $1 $2 $3 $4 $5 $6 $7 $8 > logs_jl/tdvp_coarsegrained_dw_L${2}_chi${3}_beta${4}_dt_ramped0.1_20.0_0.5_Jprime${5}_U${6}_Uprime${7}_mu${8}.txt

exit 0
