#!/bin/bash
#SBATCH -A m3341
#SBATCH -C cpu
#SBATCH -c 256
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 24:00:00

export OMP_NUM_THREADS=256
export MKL_NUM_THREADS=256
julia --heap-size-hint=400G $1 $2 $3 $4 $5 $6 $7 > logs_jl/tdvp_coarsegrained_dw_L${2}_chi${3}_beta${4}_dt${5}_Jprime${6}_mu${7}.txt

exit 0
