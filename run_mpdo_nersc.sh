#!/bin/bash
#SBATCH -A m3341
#SBATCH -C cpu
#SBATCH -c 256
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 24:00:00

export OMP_NUM_THREADS=256
export MKL_NUM_THREADS=256
julia --heap-size-hint=400G $1 $2 $3 $4 $5 $6 > logs_jl/tebd_mpdo_L${2}_chi${3}_dt${4}_Jprime${5}_mu${6}.txt

exit 0
